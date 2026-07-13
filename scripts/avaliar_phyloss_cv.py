#!/usr/bin/env python3
"""Evaluate PhyLoss single-target with GroupKFold CV + hold-out."""
import os, sys, warnings, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DISABLE_RETRACING_LOG"] = "1"
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from config import SEED, FEATURE_COLS, TARGET_COLS, GROUP_COL, MODEL_MAP, N_TARGETS, N_FEATURES_BASE, DATA_PATH
from data import read_tabular_csv, build_XY_groups_with_model_map

class LucHybridLoss(tf.keras.losses.Loss):
    def __init__(self, m, omega=0.5, huber_delta=1.0):
        super().__init__(name="luc_hybrid_loss")
        self.m = m; self.omega = omega
        self._huber = tf.keras.losses.Huber(delta=huber_delta, reduction=tf.keras.losses.Reduction.NONE)
    def call(self, y_true_aug, y_pred_2m):
        m = self.m; y_true = y_true_aug[:, :m]; y_phy = y_true_aug[:, m:2*m]; y_pred = y_pred_2m[:, :m]
        l_data = self._huber(y_true, y_pred)
        l_phys = tf.reduce_mean(tf.square(y_phy - y_pred), axis=1)
        w = tf.clip_by_value(tf.cast(self.omega, tf.float32), 0.0, 1.0)
        return tf.reduce_mean((1.0-w)*l_data + w*l_phys)

df = read_tabular_csv(str(DATA_PATH), decimal_comma=True)
X, Y_true, groups, _, Y_model, _ = build_XY_groups_with_model_map(df, FEATURE_COLS, TARGET_COLS, GROUP_COL, True, MODEL_MAP)

baselines = {
    "alim": {"hidden": [512], "act": "relu", "lr": 0.003, "l2": 1e-6, "bs": 64},
    "ref":  {"hidden": [256, 128], "act": "tanh", "lr": 0.003, "l2": 0.001, "bs": 16},
    "flux": {"hidden": [512], "act": "relu", "lr": 0.003, "l2": 1e-6, "bs": 64},
}
target_map = {"alim": (0, "Alim_T_out"), "ref": (1, "Ref_T_out"), "flux": (2, "Flux")}

results = {}
for tkey, (tidx, tname) in target_map.items():
    hp = baselines[tkey]
    y_true_1d = Y_true[:, tidx]
    y_phy_1d = Y_model[:, tidx]
    Y_aug_1d = np.column_stack([y_true_1d, y_phy_1d])  # 2 cols: [exp, phy]
    print(f"\n=== {tname} (idx={tidx}) ===")
    print(f"  HP: hidden={hp['hidden']} act={hp['act']} lr={hp['lr']} l2={hp['l2']} bs={hp['bs']}")

    def build_model(l2):
        tf.keras.backend.clear_session()
        inp = tf.keras.Input(shape=(N_FEATURES_BASE,))
        h = inp
        for u in hp["hidden"]:
            h = tf.keras.layers.Dense(int(u), activation=hp["act"], kernel_regularizer=tf.keras.regularizers.l2(l2))(h)
        y_hat = tf.keras.layers.Dense(1, activation="linear")(h)
        y_out = tf.keras.layers.Concatenate()([y_hat, y_hat])
        m = tf.keras.Model(inp, y_out)
        opt_cls = tf.keras.optimizers.AdamW if hp.get("opt") == "adamw" else tf.keras.optimizers.Adam
        m.compile(optimizer=opt_cls(learning_rate=hp["lr"]), loss=LucHybridLoss(m=1, omega=0.1))
        return m

    cbs = [tf.keras.callbacks.EarlyStopping(patience=12, min_delta=1e-4, restore_best_weights=True),
           tf.keras.callbacks.TerminateOnNaN()]

    # ---- Hold-out test ----
    X_tr, X_te, y_aug_tr, y_aug_te = train_test_split(X, Y_aug_1d, test_size=0.2, random_state=SEED)
    m_test = build_model(hp["l2"])
    sx = StandardScaler(); sy = StandardScaler()
    m_test.fit(sx.fit_transform(X_tr), sy.fit_transform(y_aug_tr),
               validation_split=0.2, shuffle=True, epochs=200, batch_size=hp["bs"],
               verbose=0, callbacks=cbs)
    pred_te = sy.inverse_transform(m_test.predict(sx.transform(X_te), verbose=0))[:, 0]
    test_rmse = float(np.sqrt(mean_squared_error(y_aug_te[:, 0], pred_te)))

    # ---- CV ----
    gkf = GroupKFold(3)
    cv_preds = np.zeros(len(Y_true))
    for fold, (tr, te) in enumerate(gkf.split(X, y_true_1d, groups)):
        m = build_model(hp["l2"])
        sx2 = StandardScaler(); sy2 = StandardScaler()
        m.fit(sx2.fit_transform(X[tr]), sy2.fit_transform(Y_aug_1d[tr]),
              validation_split=0.2, shuffle=True, epochs=200, batch_size=hp["bs"],
              verbose=0, callbacks=cbs)
        cv_preds[te] = sy2.inverse_transform(m.predict(sx2.transform(X[te]), verbose=0))[:, 0]
    cv_rmse = float(np.sqrt(mean_squared_error(y_true_1d, cv_preds)))
    pct = (test_rmse / cv_rmse - 1) * 100
    print(f"  CV={cv_rmse:.4f}  Test={test_rmse:.4f}  Dif={pct:+.1f}%")
    results[tname] = {"cv_rmse": cv_rmse, "test_rmse": test_rmse, "dif_pct": pct}

print("\n\n=== PhyLoss Single-target Results ===")
for tn, r in results.items():
    print(f"  {tn}: CV={r['cv_rmse']:.4f}, Test={r['test_rmse']:.4f}, Dif={r['dif_pct']:+.1f}%")

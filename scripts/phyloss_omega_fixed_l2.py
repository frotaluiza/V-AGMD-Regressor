#!/usr/bin/env python3
"""PhyLoss: w optimo + L2 fixo na baseline. Extrai CV + OOF por target."""
import os, sys, json, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DISABLE_RETRACING_LOG"] = "1"
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from config import SEED, FEATURE_COLS, TARGET_COLS, GROUP_COL, MODEL_MAP, N_TARGETS, N_FEATURES_BASE, DATA_PATH
from data import read_tabular_csv, build_XY_groups_with_model_map

np.random.seed(SEED); tf.random.set_seed(SEED)

class PhyLoss(tf.keras.losses.Loss):
    def __init__(self, m, omega=0.5, huber_delta=1.0):
        super().__init__(); self.m = m; self.omega = omega
        self._huber = tf.keras.losses.Huber(delta=huber_delta, reduction=tf.keras.losses.Reduction.NONE)
    def call(self, y_aug, y_2m):
        m = self.m
        d = self._huber(y_aug[:,:m], y_2m[:,:m])
        p = tf.reduce_mean(tf.square(y_aug[:,m:2*m] - y_2m[:,:m]), axis=1)
        w = tf.clip_by_value(tf.cast(self.omega, tf.float32), 0.0, 1.0)
        return tf.reduce_mean((1.0-w)*d + w*p)

baselines = {
    "alim": {"hidden":[512], "act":"relu", "lr":0.003, "l2":1e-6, "bs":64, "opt":"adam", "best_w": 0.0},
    "ref":  {"hidden":[256,128], "act":"tanh", "lr":0.003, "l2":0.001, "bs":16, "opt":"adam", "best_w": 0.7},
    "flux": {"hidden":[512], "act":"relu", "lr":0.003, "l2":1e-6, "bs":64, "opt":"adam", "best_w": 0.0},
}
tnames = ["Alim_T_out", "Ref_T_out", "Flux"]

df = read_tabular_csv(str(DATA_PATH), decimal_comma=True)
X, Y_true, groups, _, Y_model, _ = build_XY_groups_with_model_map(df, FEATURE_COLS, TARGET_COLS, GROUP_COL, True, MODEL_MAP)
Y_aug = np.concatenate([Y_true, Y_model], axis=1)

results = {}
for bkey, hp in baselines.items():
    w = hp["best_w"]
    print(f"\n=== PhyLoss (base {bkey}) w={w} L2={hp['l2']:.0e} ===")
    gkf = GroupKFold(3)
    fold_rmses = [[] for _ in range(N_TARGETS)]
    oof_preds = np.zeros_like(Y_true)

    for fold, (tr, te) in enumerate(gkf.split(X, Y_true, groups)):
        tf.keras.backend.clear_session()
        sx = StandardScaler(); sy = StandardScaler()
        X_tr = sx.fit_transform(X[tr]); X_te = sx.transform(X[te])
        y_tr = sy.fit_transform(Y_aug[tr])
        inp = tf.keras.Input(shape=(N_FEATURES_BASE,)); x = inp
        for u in hp["hidden"]:
            x = tf.keras.layers.Dense(int(u), activation=hp["act"],
                kernel_regularizer=tf.keras.regularizers.l2(hp["l2"]))(x)
        y_hat = tf.keras.layers.Dense(N_TARGETS, activation="linear")(x)
        y_out = tf.keras.layers.Concatenate()([y_hat, y_hat])
        m = tf.keras.Model(inp, y_out)
        opt = tf.keras.optimizers.AdamW if hp.get("opt")=="adamw" else tf.keras.optimizers.Adam
        m.compile(optimizer=opt(hp["lr"]), loss=PhyLoss(m=N_TARGETS, omega=w))
        m.fit(X_tr, y_tr, validation_data=(X_te, sy.transform(Y_aug[te])),
              epochs=200, batch_size=hp["bs"], verbose=0, shuffle=True,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=12, min_delta=1e-4, restore_best_weights=True),
                         tf.keras.callbacks.TerminateOnNaN()])
        pred = sy.inverse_transform(m.predict(X_te, verbose=0))[:, :N_TARGETS]
        oof_preds[te] = pred
        for t in range(N_TARGETS):
            fold_rmses[t].append(float(np.sqrt(mean_squared_error(Y_true[te][:,t], pred[:,t]))))

    cv_rmse = [round(float(np.mean(fold_rmses[t])), 4) for t in range(N_TARGETS)]
    oof_rmse = [round(float(np.sqrt(mean_squared_error(Y_true[:,t], oof_preds[:,t]))), 4) for t in range(N_TARGETS)]
    oof_r2 = [round(float(r2_score(Y_true[:,t], oof_preds[:,t])), 4) for t in range(N_TARGETS)]

    print(f"  CV  RMSE: A={cv_rmse[0]:.3f}  R={cv_rmse[1]:.3f}  F={cv_rmse[2]:.3f}")
    print(f"  OOF RMSE: A={oof_rmse[0]:.3f}  R={oof_rmse[1]:.3f}  F={oof_rmse[2]:.3f}")
    print(f"  OOF R2:   A={oof_r2[0]:.4f}  R={oof_r2[1]:.4f}  F={oof_r2[2]:.4f}")

    results[bkey] = {"best_omega": w, "cv_rmse": cv_rmse, "oof_rmse": oof_rmse, "oof_r2": oof_r2}

print("\n=== FINAL ===")
for bk, r in results.items():
    print(f"  {bk} (w={r['best_omega']}): CV={r['cv_rmse']} OOF={r['oof_rmse']}")

out = os.path.join(os.path.dirname(__file__), "..", "teste-inicial-frozenBaseline-tcc", "phyloss_fixed_l2.json")
json.dump(results, open(out, "w"), indent=2)
print(f"\nSalvo em: {out}")

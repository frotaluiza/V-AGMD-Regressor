#!/usr/bin/env python3
"""PhyLoss: avalia config otima (w + L2) de cada baseline com hold-out test."""
import os, sys, json, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DISABLE_RETRACING_LOG"] = "1"
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, tensorflow as tf
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from config import SEED, FEATURE_COLS, TARGET_COLS, GROUP_COL, MODEL_MAP, N_TARGETS, N_FEATURES_BASE, DATA_PATH
from data import read_tabular_csv, build_XY_groups_with_model_map

np.random.seed(SEED); tf.random.set_seed(SEED)

class PhyLoss(tf.keras.losses.Loss):
    def __init__(self, m, omega=0.5, huber_delta=1.0):
        super().__init__(name="phyloss"); self.m = m; self.omega = omega
        self._huber = tf.keras.losses.Huber(delta=huber_delta, reduction=tf.keras.losses.Reduction.NONE)
    def call(self, y_true_aug, y_pred_2m):
        m = self.m; d = self._huber(y_true_aug[:,:m], y_pred_2m[:,:m])
        p = tf.reduce_mean(tf.square(y_true_aug[:,m:2*m] - y_pred_2m[:,:m]), axis=1)
        w = tf.clip_by_value(tf.cast(self.omega, tf.float32), 0.0, 1.0)
        return tf.reduce_mean((1.0-w)*d + w*p)

def fit_and_eval(X_tr, Y_aug_tr, X_te, Y_aug_te, Y_true_te, hp, l2, omega):
    tf.keras.backend.clear_session()
    inp = tf.keras.Input(shape=(N_FEATURES_BASE,)); x = inp
    for u in hp["hidden"]:
        x = tf.keras.layers.Dense(int(u), activation=hp["act"], kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    y_hat = tf.keras.layers.Dense(N_TARGETS, activation="linear")(x)
    y_out = tf.keras.layers.Concatenate()([y_hat, y_hat])
    m = tf.keras.Model(inp, y_out)
    opt = tf.keras.optimizers.AdamW if hp.get("opt")=="adamw" else tf.keras.optimizers.Adam
    m.compile(optimizer=opt(learning_rate=hp["lr"]), loss=PhyLoss(m=N_TARGETS, omega=omega))
    sx = StandardScaler(); sy = StandardScaler()
    m.fit(sx.fit_transform(X_tr), sy.fit_transform(Y_aug_tr), validation_split=0.2,
          shuffle=True, epochs=200, batch_size=hp["bs"], verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=12, min_delta=1e-4, restore_best_weights=True),
                     tf.keras.callbacks.TerminateOnNaN()])
    pred = sy.inverse_transform(m.predict(sx.transform(X_te), verbose=0))[:, :N_TARGETS]
    return [float(np.sqrt(mean_squared_error(Y_true_te[:,t], pred[:,t]))) for t in range(N_TARGETS)]

best_configs = {
    "alim": {"omega": 0.0, "l2": 1e-4},
    "ref":  {"omega": 0.7, "l2": 1e-3},
    "flux": {"omega": 0.0, "l2": 1e-3},
}
baselines = {
    "alim": {"hidden":[512], "act":"relu", "lr":0.003, "l2":1e-6, "bs":64, "opt":"adam"},
    "ref":  {"hidden":[256,128], "act":"tanh", "lr":0.003, "l2":0.001, "bs":16, "opt":"adam"},
    "flux": {"hidden":[512], "act":"relu", "lr":0.003, "l2":1e-6, "bs":64, "opt":"adam"},
}

df = read_tabular_csv(str(DATA_PATH), decimal_comma=True)
X, Y_true, groups, _, Y_model, _ = build_XY_groups_with_model_map(
    df, FEATURE_COLS, TARGET_COLS, GROUP_COL, True, MODEL_MAP)
Y_aug = np.concatenate([Y_true, Y_model], axis=1)

# Hold-out split
X_tr, X_te, Y_aug_tr, Y_aug_te, Y_true_tr, Y_true_te, g_tr, _ = train_test_split(
    X, Y_aug, Y_true, groups, test_size=0.2, random_state=SEED)

results = {}
for bkey, hp in baselines.items():
    cfg = best_configs[bkey]
    print(f"\n=== PhyLoss (base {bkey}) w={cfg['omega']} L2={cfg['l2']:.0e} ===")

    # CV com GroupKFold
    gkf = GroupKFold(3)
    fold_rmse = [[] for _ in range(N_TARGETS)]
    for tr, cv in gkf.split(X_tr, Y_true_tr, g_tr):
        rmse = fit_and_eval(X_tr[tr], Y_aug_tr[tr], X_tr[cv], Y_aug_tr[cv], Y_true_tr[cv], hp, cfg["l2"], cfg["omega"])
        for t in range(N_TARGETS): fold_rmse[t].append(rmse[t])
    cv_rmse = [float(np.mean(fold_rmse[t])) for t in range(N_TARGETS)]

    # Test com hold-out
    test_rmse = fit_and_eval(X_tr, Y_aug_tr, X_te, Y_aug_te, Y_true_te, hp, cfg["l2"], cfg["omega"])
    dif_pct = [((test_rmse[t] - cv_rmse[t]) / max(cv_rmse[t], 1e-8))*100 for t in range(N_TARGETS)]

    print(f"  CV:   A={cv_rmse[0]:.3f}  R={cv_rmse[1]:.3f}  F={cv_rmse[2]:.3f}")
    print(f"  Test: A={test_rmse[0]:.3f}  R={test_rmse[1]:.3f}  F={test_rmse[2]:.3f}")
    print(f"  Dif%: A={dif_pct[0]:.1f}  R={dif_pct[1]:.1f}  F={dif_pct[2]:.1f}")

    results[bkey] = {"cv_rmse":[round(x,3) for x in cv_rmse], "test_rmse":[round(x,3) for x in test_rmse], "dif_pct":[round(x,1) for x in dif_pct]}

print("\n=== RESUMO PhyLoss ===")
for bkey, r in results.items():
    print(f"  {bkey}: CV={r['cv_rmse']} Test={r['test_rmse']} Dif%={r['dif_pct']}")

out = os.path.join(os.path.dirname(__file__), "..", "teste-inicial-frozenBaseline-tcc", "phyloss_tabela.json")
json.dump(results, open(out, "w"), indent=2)
print(f"\nSalvo em: {out}")

#!/usr/bin/env python3
"""PhyLoss sweep apenas de omega (L2 fixo na baseline)."""
import os, sys, json, logging, warnings
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
        super().__init__(name="phyloss"); self.m = m; self.omega = omega
        self._huber = tf.keras.losses.Huber(delta=huber_delta, reduction=tf.keras.losses.Reduction.NONE)
    def call(self, y_true_aug, y_pred_2m):
        m = self.m; d = self._huber(y_true_aug[:,:m], y_pred_2m[:,:m])
        p = tf.reduce_mean(tf.square(y_true_aug[:,m:2*m] - y_pred_2m[:,:m]), axis=1)
        w = tf.clip_by_value(tf.cast(self.omega, tf.float32), 0.0, 1.0)
        return tf.reduce_mean((1.0-w)*d + w*p)

baselines = {
    "alim": {"hidden":[512], "act":"relu", "lr":0.003, "l2":1e-6, "bs":64, "opt":"adam"},
    "ref":  {"hidden":[256,128], "act":"tanh", "lr":0.003, "l2":0.001, "bs":16, "opt":"adam"},
    "flux": {"hidden":[512], "act":"relu", "lr":0.003, "l2":1e-6, "bs":64, "opt":"adam"},
}
target_names = ["Alim_T_out", "Ref_T_out", "Flux"]
omega_grid = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

df = read_tabular_csv(str(DATA_PATH), decimal_comma=True)
X, Y_true, groups, _, Y_model, _ = build_XY_groups_with_model_map(
    df, FEATURE_COLS, TARGET_COLS, GROUP_COL, True, MODEL_MAP)
Y_aug = np.concatenate([Y_true, Y_model], axis=1)

results_all = {}
for bkey in ["alim", "ref", "flux"]:
    hp = baselines[bkey]
    print(f"\n{'='*60}")
    print(f"PhyLoss (base {bkey}) | HP: hidden={hp['hidden']} l2={hp['l2']:.0e} bs={hp['bs']}")
    print(f"{'='*60}")

    best_omega = None; best_mean = np.inf
    omega_summary = {}

    for w in omega_grid:
        tf.keras.backend.clear_session()
        gkf = GroupKFold(3)
        fold_rmses = [[] for _ in range(N_TARGETS)]
        fold_preds = np.zeros_like(Y_true)

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
            m.compile(optimizer=opt(learning_rate=hp["lr"]),
                      loss=PhyLoss(m=N_TARGETS, omega=w))
            m.fit(X_tr, y_tr, validation_data=(X_te, sy.transform(Y_aug[te])),
                  epochs=200, batch_size=hp["bs"], verbose=0, shuffle=True,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=12, min_delta=1e-4, restore_best_weights=True),
                             tf.keras.callbacks.TerminateOnNaN()])
            pred = sy.inverse_transform(m.predict(X_te, verbose=0))[:, :N_TARGETS]
            fold_preds[te] = pred
            for t in range(N_TARGETS):
                fold_rmses[t].append(float(np.sqrt(mean_squared_error(Y_true[te][:,t], pred[:,t]))))

        cv_per_target = [float(np.mean(fold_rmses[t])) for t in range(N_TARGETS)]
        mean_cv = float(np.mean(cv_per_target))
        omega_summary[w] = {"cv_rmse_per_target": [round(x,4) for x in cv_per_target], "mean_cv": round(mean_cv,4)}
        print(f"  w={w:.1f}: CV=[{cv_per_target[0]:.3f},{cv_per_target[1]:.3f},{cv_per_target[2]:.3f}] mean={mean_cv:.4f}")

        if mean_cv < best_mean:
            best_mean = mean_cv
            best_omega = w
            best_preds = fold_preds.copy()

    oof_rmse = [float(np.sqrt(mean_squared_error(Y_true[:,t], best_preds[:,t]))) for t in range(N_TARGETS)]
    oof_r2 = [float(r2_score(Y_true[:,t], best_preds[:,t])) for t in range(N_TARGETS)]

    print(f"\n  >> Melhor w={best_omega:.1f} (mean CV={best_mean:.4f})")
    print(f"  >> OOF RMSE: A={oof_rmse[0]:.4f} R={oof_rmse[1]:.4f} F={oof_rmse[2]:.4f}")
    print(f"  >> OOF R2:   A={oof_r2[0]:.4f} R={oof_r2[1]:.4f} F={oof_r2[2]:.4f}")

    results_all[bkey] = {
        "best_omega": best_omega,
        "cv_rmse_per_target": [round(x,4) for x in omega_summary[best_omega]["cv_rmse_per_target"]],
        "oof_rmse": [round(x,4) for x in oof_rmse],
        "oof_r2": [round(x,4) for x in oof_r2],
        "omega_sweep": {str(k): v for k,v in omega_summary.items()},
    }

print("\n\n=== FINAL ===")
for bk, r in results_all.items():
    print(f"  {bk}: w={r['best_omega']} CV={r['cv_rmse_per_target']} OOF={r['oof_rmse']} R2={r['oof_r2']}")

out = os.path.join(os.path.dirname(__file__), "..", "teste-inicial-frozenBaseline-tcc", "phyloss_omega_only.json")
json.dump(results_all, open(out, "w"), indent=2)
print(f"\nSalvo em: {out}")

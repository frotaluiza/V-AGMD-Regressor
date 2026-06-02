#!/usr/bin/env python3
"""Generate OOF scatter plots for ZohanResidual winner."""
import os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from config import SEED, N_SPLITS, FEATURE_COLS, TARGET_COLS, GROUP_COL, ORIGIN_COL
from config import N_FEATURES_PLUS_PHY, N_TARGETS, MODEL_MAP, KERAS_CALLBACKS
from data import read_tabular_csv, build_XY_groups_with_model_map
from cv import make_cv
from sklearn.preprocessing import StandardScaler
from models.keras_builders import build_keras_mlp_residual_add_phy_x_only

DATA_CSV_PATH = r"C:\Users\frota\OneDrive\Documentos\TCC\dados_att_com_var_com_phy.csv"
OUT_DIR = r"C:\Users\frota\OneDrive\Documentos\TCC\Codigos\Codigo-revisado\stage2_only_dados_att_com_var_com_phy_20260601_151146\oof_plots_winner"
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
tf.keras.backend.set_floatx("float32")
tf.config.optimizer.set_jit(True)

df = read_tabular_csv(DATA_CSV_PATH, decimal_comma=True)
origin_col = ORIGIN_COL if ORIGIN_COL in df.columns else None
X, Y_true, groups, used_df, Y_model, origin_values = build_XY_groups_with_model_map(
    df=df, feature_cols=FEATURE_COLS, target_cols=TARGET_COLS,
    group_col=GROUP_COL, dropna=True, model_map=MODEL_MAP,
    origin_col=origin_col)
cv, eval_mask = make_cv(N_SPLITS, origin_values=origin_values)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=KERAS_CALLBACKS["early_stop_patience"],
        min_delta=KERAS_CALLBACKS["early_stop_min_delta"], restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=KERAS_CALLBACKS["reduce_lr_factor"],
        patience=KERAS_CALLBACKS["reduce_lr_patience"],
        min_delta=KERAS_CALLBACKS["early_stop_min_delta"],
        cooldown=0, min_lr=KERAS_CALLBACKS["reduce_lr_min_lr"], verbose=0),
    tf.keras.callbacks.TerminateOnNaN(),
]

X_in = np.concatenate([X, Y_model], axis=1)
y_pred_oof = np.full_like(Y_true, np.nan)

for fold, (tr, te) in enumerate(cv.split(X_in, Y_true, groups), start=1):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_tr_s = x_scaler.fit_transform(X_in[tr])
    x_te_s = x_scaler.transform(X_in[te])
    y_tr_s = y_scaler.fit_transform(Y_true[tr])

    tf.keras.backend.clear_session()
    model = build_keras_mlp_residual_add_phy_x_only(
        n_features=N_FEATURES_PLUS_PHY, n_targets=N_TARGETS,
        hidden_layer_sizes=(256,), activation="tanh",
        optimizer="adam", learning_rate=0.001, loss="huber",
        huber_delta=1.0, l2=0.0)
    model.fit(x_tr_s, y_tr_s, validation_split=0.2, epochs=200,
              batch_size=32, shuffle=True, verbose=0, callbacks=callbacks)
    y_pred_oof[te] = y_scaler.inverse_transform(model.predict(x_te_s, verbose=0))
    print(f"Fold {fold}/3 done", flush=True)

rmse_per = [float(np.sqrt(np.mean((Y_true[:, i] - y_pred_oof[:, i]) ** 2))) for i in range(N_TARGETS)]
for name, rmse in zip(TARGET_COLS, rmse_per):
    print(f"OOF {name}: RMSE = {rmse:.4f}", flush=True)

fig, axes = plt.subplots(1, N_TARGETS, figsize=(5 * N_TARGETS, 5))
for i, (name, ax) in enumerate(zip(TARGET_COLS, axes)):
    ax.scatter(Y_true[:, i], y_pred_oof[:, i], alpha=0.7, edgecolors="k", linewidth=0.5)
    lo = min(Y_true[:, i].min(), y_pred_oof[:, i].min())
    hi = max(Y_true[:, i].max(), y_pred_oof[:, i].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Observado"); ax.set_ylabel("Predito (OOF)")
    ax.set_title(f"{name} RMSE={rmse_per[i]:.4f}")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
plt.suptitle("ZohanResidual - Predições Out-of-Fold", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "oof_ZohanResidual_all_targets.png"), dpi=300, bbox_inches="tight")
plt.close()

for i, name in enumerate(TARGET_COLS):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(Y_true[:, i], y_pred_oof[:, i], alpha=0.7, edgecolors="k", linewidth=0.5)
    lo = min(Y_true[:, i].min(), y_pred_oof[:, i].min())
    hi = max(Y_true[:, i].max(), y_pred_oof[:, i].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Observado"); ax.set_ylabel("Predito (OOF)")
    ax.set_title(f"{name} RMSE={rmse_per[i]:.4f}")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"oof_ZohanResidual_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

print(f"Done! Plots in {OUT_DIR}", flush=True)

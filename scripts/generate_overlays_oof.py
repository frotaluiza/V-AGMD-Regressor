"""OOF-based 0D vs winner overlays: re-fit CV folds from saved best_params."""
import json, os, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import read_tabular_csv, build_XY_groups_with_model_map
from config import FEATURE_COLS, TARGET_COLS, GROUP_COL, ORIGIN_COL, MODEL_MAP, N_SPLITS, KERAS_CALLBACKS
from file_io import save_overlay_comparison_table
from plots import plot_oof_vs_phy_overlay_per_target, plot_oof_vs_phy_overlay_panel
from models.keras_builders import (
    make_keras_mlp_estimator, make_keras_mlp_hrnn_estimator,
    KERAS_AVAILABLE,
)
from models.wrappers import YScalerRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import keras, tensorflow as tf

tf.random.set_seed(42)
keras.backend.set_floatx("float32")

DATA = r"C:\Users\frota\OneDrive\Documentos\TCC\dados_att_com_var_com_phy.csv"
RESULTS = Path(r"C:\Users\frota\OneDrive\Documentos\TCC\Codigos\Codigo-revisado\results_dados_att_com_var_com_phy_20260531_120240")
OUT = RESULTS / "overlay_oof_vs_0d"
OUT.mkdir(parents=True, exist_ok=True)

df = read_tabular_csv(DATA, decimal_comma=True)
origin_col = ORIGIN_COL if ORIGIN_COL in df.columns else None
X, Y_true, groups, _, Y_model, _ = build_XY_groups_with_model_map(
    df, feature_cols=FEATURE_COLS, target_cols=TARGET_COLS,
    group_col=GROUP_COL, dropna=True, model_map=MODEL_MAP,
    origin_col=origin_col,
)
target_cols = list(TARGET_COLS)
cv = GroupKFold(n_splits=N_SPLITS)

with open(RESULTS / "stage2_restricted_hybrid_comparison" /
    "agmd_stage2_restricted_hybrids_dados_att_com_var_com_phy_best_params.json") as f:
    best_params_all = json.load(f)

class SimpleWinner:
    def __init__(self, family, oof_y_true, oof_y_pred):
        self.family = family
        self.oof_y_true = oof_y_true
        self.oof_y_pred = oof_y_pred

def _make_pipeline(base_est, scale_y, augmented_2m, m):
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", YScalerRegressor(
            base_estimator=base_est, scale_y=scale_y,
            augmented_2m=augmented_2m, m_targets=m,
        )),
    ])

def _clean_best_params(bp):
    """Clean best_params for set_params: remove callbacks, fix hidden_layer_sizes."""
    if bp is None:
        return {}
    cleaned = {}
    for k, v in bp.items():
        if k == "model__fit__callbacks":
            continue
        if k == "model__model__hidden_layer_sizes" and isinstance(v, list):
            cleaned[k] = tuple(v)
        elif isinstance(v, list) and k.endswith("hidden_layer_sizes"):
            cleaned[k] = tuple(v)
        else:
            cleaned[k] = v
    return cleaned

def build_pipeline_from_best(estimator_fn, best_params, x_mode, y_mode):
    m = int(Y_true.shape[1])
    augmented_2m = (y_mode == "true_plus_model")
    scale_y = True
    base = estimator_fn(m)

    if x_mode == "x":
        X_use = X
    elif x_mode == "x_plus_model":
        X_use = np.concatenate([X, Y_model], axis=1)
    else:
        raise ValueError(f"Unknown x_mode: {x_mode}")

    if y_mode == "true":
        Y_use = Y_true
    elif y_mode == "true_plus_model":
        Y_use = np.concatenate([Y_true, Y_model], axis=1)
    else:
        raise ValueError(f"Unknown y_mode: {y_mode}")

    pipe = _make_pipeline(base, scale_y, augmented_2m, m)
    cleaned = _clean_best_params(best_params)
    # Use same callbacks as original pipeline
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=KERAS_CALLBACKS["early_stop_patience"],
        min_delta=KERAS_CALLBACKS["early_stop_min_delta"], restore_best_weights=True,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=KERAS_CALLBACKS["reduce_lr_factor"],
        patience=KERAS_CALLBACKS["reduce_lr_patience"],
        min_delta=KERAS_CALLBACKS["early_stop_min_delta"],
        cooldown=0, min_lr=KERAS_CALLBACKS["reduce_lr_min_lr"], verbose=0,
    )
    term_nan = tf.keras.callbacks.TerminateOnNaN()
    cleaned["model__fit__callbacks"] = [early_stop, reduce_lr, term_nan]
    if cleaned:
        pipe.set_params(**cleaned)

    oof_pred = np.zeros_like(Y_true)
    oof_true = np.zeros_like(Y_true)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_use, Y_use, groups)):
        fold_pipe = clone(pipe)
        X_tr, X_te = X_use[train_idx], X_use[test_idx]
        Y_tr, Y_te = Y_use[train_idx], Y_use[test_idx]

        fold_pipe.fit(X_tr, Y_tr)
        pred = fold_pipe.predict(X_te)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        oof_pred[test_idx] = pred
        oof_true[test_idx] = Y_te[:, :pred.shape[1]]
        print(f"  Fold {fold_idx+1}/{N_SPLITS}: train={len(train_idx)} test={len(test_idx)} rmse={np.sqrt(np.mean((Y_te[:, :pred.shape[1]] - pred)**2)):.4f}")

    return oof_true, oof_pred

# --- FrozenBaseline (0D model) ---
print("OOF FrozenBaseline...")
fb_params = best_params_all.get("KerasMLP_FrozenBaseline", {})
fb_true, fb_pred = build_pipeline_from_best(make_keras_mlp_estimator, fb_params, "x", "true")
fb_rmse = np.sqrt(np.mean((fb_true - fb_pred)**2))
print(f"  OOF RMSE total = {fb_rmse:.4f}")

# --- HRNN (winner) ---
print("OOF HRNN...")
hrnn_params = best_params_all.get("KerasMLP_ZohanHRNN_Restricted", {})
hrnn_true, hrnn_pred = build_pipeline_from_best(make_keras_mlp_hrnn_estimator, hrnn_params, "x_plus_model", "true")
hrnn_rmse = np.sqrt(np.mean((hrnn_true - hrnn_pred)**2))
print(f"  OOF RMSE total = {hrnn_rmse:.4f}")

# --- Generate overlays ---
print("Generating overlay plots...")
hrnn_winner = SimpleWinner("KerasMLP_ZohanHRNN_Restricted", hrnn_true, hrnn_pred)
plot_oof_vs_phy_overlay_per_target(
    winner=hrnn_winner, target_cols=target_cols, y_model=Y_model,
    out_dir=str(OUT), prefix="agmd_oof_overlay",
)
plot_oof_vs_phy_overlay_panel(
    winner=hrnn_winner, target_cols=target_cols, y_model=Y_model,
    out_path=str(OUT / "agmd_oof_overlay_panel_HRNN_vs_0D.png"),
    figure_title="Comparacao OOF: modelo 0D vs HRNN vencedor",
)
save_overlay_comparison_table(
    winner=hrnn_winner, target_cols=target_cols, y_model=Y_model,
    out_path=str(OUT / "agmd_oof_overlay_points_HRNN_vs_0D.csv"),
)

print(f"\nDone! OOF overlays in: {OUT}")
print(f"0D RMSE={fb_rmse:.4f} | HRNN OOF RMSE={hrnn_rmse:.4f}")

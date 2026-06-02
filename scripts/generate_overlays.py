"""Generate 0D vs winner overlay plots using full-data fit from best_params."""
import json, os, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import read_tabular_csv, build_XY_groups_with_model_map
from config import FEATURE_COLS, TARGET_COLS, GROUP_COL, ORIGIN_COL, MODEL_MAP
from config import N_FEATURES_BASE, N_FEATURES_PLUS_PHY, N_TARGETS
from file_io import save_overlay_comparison_table
from plots import plot_oof_vs_phy_overlay_per_target, plot_oof_vs_phy_overlay_panel
from models.keras_builders import make_keras_mlp_estimator, make_keras_mlp_hrnn_estimator, make_keras_mlp_residual_x_only_estimator, KERAS_AVAILABLE
from models.wrappers import YScalerRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import keras
import tensorflow as tf

tf.random.set_seed(42)
keras.backend.set_floatx("float32")

DATA = r"C:\Users\frota\OneDrive\Documentos\TCC\dados_att_com_var_com_phy.csv"
RESULTS = Path(r"C:\Users\frota\OneDrive\Documentos\TCC\Codigos\Codigo-revisado\results_dados_att_com_var_com_phy_20260531_120240")
OUT = RESULTS / "overlay_0d_vs_winner"
OUT.mkdir(parents=True, exist_ok=True)

df = read_tabular_csv(DATA, decimal_comma=True)
origin_col = ORIGIN_COL if ORIGIN_COL in df.columns else None
X, Y_true, groups, _, Y_model, origin_values = build_XY_groups_with_model_map(
    df, feature_cols=FEATURE_COLS, target_cols=TARGET_COLS,
    group_col=GROUP_COL, dropna=True, model_map=MODEL_MAP,
    origin_col=origin_col,
)
target_cols = list(TARGET_COLS)

with open(RESULTS / "stage2_restricted_hybrid_comparison" /
    "agmd_stage2_restricted_hybrids_dados_att_com_var_com_phy_best_params.json") as f:
    best_params_all = json.load(f)

class SimpleWinner:
    def __init__(self, family, oof_y_true, oof_y_pred):
        self.family = family
        self.oof_y_true = oof_y_true
        self.oof_y_pred = oof_y_pred

def build_and_predict(estimator_fn, best_params, x_mode, y_mode):
    """Build model from best_params, fit on full data, return y_pred."""
    m = N_TARGETS
    augmented_2m = (y_mode == "true_plus_model")

    actual_n_features = X.shape[1] if x_mode == "x" else X.shape[1] + Y_model.shape[1]
    base = estimator_fn(m)
    base.set_params(**{
        "model__n_features": actual_n_features,
        "model__n_targets": m,
        "model__hidden_layer_sizes": (64,),
        "model__learning_rate": 0.001,
        "model__l2": 1e-05,
        "model__loss": "huber",
        "model__huber_delta": 1.0,
        "model__optimizer": "adam",
        "model__activation": "relu",
        "batch_size": 64,
        "epochs": 200,
        "validation_split": 0.2,
        "shuffle": True,
        "verbose": 0,
    })
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", YScalerRegressor(
            base_estimator=base, scale_y=True,
            augmented_2m=augmented_2m, m_targets=m,
        )),
    ])

    if x_mode == "x":
        X_use = X
    elif x_mode == "x_plus_model":
        X_use = np.concatenate([X, Y_model], axis=1)
    else:
        X_use = X

    if y_mode == "true":
        Y_use = Y_true
    elif y_mode == "true_plus_model":
        Y_use = np.concatenate([Y_true, Y_model], axis=1)
    else:
        Y_use = Y_true
    pipeline.fit(X_use, Y_use)
    pred = pipeline.predict(X_use)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    return Y_true[:, :pred.shape[1]], pred

print("Fitting FrozenBaseline (0D model)...")
fb_true, fb_pred = build_and_predict(make_keras_mlp_estimator, {}, "x", "true")
print(f"  RMSE train = {np.sqrt(np.mean((fb_true - fb_pred)**2)):.4f}")

print("Fitting HRNN (winner)...")
hrnn_true, hrnn_pred = build_and_predict(make_keras_mlp_hrnn_estimator, {}, "x_plus_model", "true")
print(f"  RMSE train = {np.sqrt(np.mean((hrnn_true - hrnn_pred)**2)):.4f}")

print("Generating HRNN overlay plots...")
hrnn_winner = SimpleWinner("KerasMLP_ZohanHRNN_Restricted", hrnn_true, hrnn_pred)
plot_oof_vs_phy_overlay_per_target(winner=hrnn_winner, target_cols=target_cols, y_model=Y_model, out_dir=str(OUT), prefix="agmd_overlay")
plot_oof_vs_phy_overlay_panel(winner=hrnn_winner, target_cols=target_cols, y_model=Y_model, out_path=str(OUT / "agmd_overlay_panel_HRNN_vs_0D.png"), figure_title="Comparacao entre modelo 0D e modelo HRNN vencedor")
save_overlay_comparison_table(winner=hrnn_winner, target_cols=target_cols, y_model=Y_model, out_path=str(OUT / "agmd_overlay_points_HRNN_vs_0D.csv"))

print("Fitting Residual (2nd place)...")
res_true, res_pred = build_and_predict(make_keras_mlp_residual_x_only_estimator, {}, "x_plus_model", "true")
print(f"  RMSE train = {np.sqrt(np.mean((res_true - res_pred)**2)):.4f}")
res_winner = SimpleWinner("KerasMLP_ZohanResidual_Restricted", res_true, res_pred)
plot_oof_vs_phy_overlay_per_target(winner=res_winner, target_cols=target_cols, y_model=Y_model, out_dir=str(OUT), prefix="agmd_overlay_residual")
plot_oof_vs_phy_overlay_panel(winner=res_winner, target_cols=target_cols, y_model=Y_model, out_path=str(OUT / "agmd_overlay_panel_Residual_vs_0D.png"), figure_title="Comparacao entre modelo 0D e modelo Residual vencedor")

print(f"\nDone! Overlay plots saved in: {OUT}")

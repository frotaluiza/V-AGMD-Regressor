#!/usr/bin/env python3
"""Compute OOF RMSE for all 3 outputs for every model using verified approach."""
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from config import SEED, N_SPLITS, FEATURE_COLS, TARGET_COLS, GROUP_COL, ORIGIN_COL
from config import N_FEATURES_BASE, N_FEATURES_PLUS_PHY, N_TARGETS, MODEL_MAP, KERAS_CALLBACKS
from data import read_tabular_csv, build_XY_groups_with_model_map
from cv import make_cv
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
tf.keras.backend.set_floatx("float32")
tf.config.optimizer.set_jit(True)

DATA_CSV_PATH = r"C:\Users\frota\OneDrive\Documentos\TCC\dados_att_com_var_com_phy.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "oof_all_models"
OUT_DIR.mkdir(exist_ok=True)

df = read_tabular_csv(DATA_CSV_PATH, decimal_comma=True)
origin_col = ORIGIN_COL if ORIGIN_COL in df.columns else None
X, Y_true, groups, used_df, Y_model, origin_values = build_XY_groups_with_model_map(
    df=df, feature_cols=FEATURE_COLS, target_cols=TARGET_COLS,
    group_col=GROUP_COL, dropna=True, model_map=MODEL_MAP,
    origin_col=origin_col)
cv, eval_mask = make_cv(N_SPLITS, origin_values=origin_values)

CBS = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=KERAS_CALLBACKS["early_stop_patience"],
                                     min_delta=1e-4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                                          min_delta=1e-4, cooldown=0, min_lr=1e-6, verbose=0),
    tf.keras.callbacks.TerminateOnNaN(),
]

def fit_predict_sklearn(est, X, Y, groups, cv):
    """Standard sklearn: no YScalerRegressor issues."""
    pred = np.full_like(Y, np.nan)
    for tr, te in cv.split(X, Y, groups):
        c = clone(est)
        c.fit(X[tr], Y[tr])
        pred[te] = c.predict(X[te])
    return pred

def fit_predict_keras_reg(X_fit, Y_fit, groups, cv, build_fn, kw, cb=True):
    """Manual scaling + direct Keras model (no Pipeline, no YScalerRegressor)."""
    pred = np.full_like(Y_fit, np.nan)
    for tr, te in cv.split(X_fit, Y_fit, groups):
        tf.keras.backend.clear_session()
        xs = StandardScaler()
        x_tr_s = xs.fit_transform(X_fit[tr])
        x_te_s = xs.transform(X_fit[te])
        ys = StandardScaler()
        y_tr_s = ys.fit_transform(Y_fit[tr])
        model = build_fn(**kw)
        cbs_in = CBS if cb else None
        model.fit(x_tr_s, y_tr_s, validation_split=0.2, epochs=200,
                  batch_size=kw.get("batch_size", 32), shuffle=True,
                  verbose=0, callbacks=cbs_in)
        pred[te] = ys.inverse_transform(model.predict(x_te_s, verbose=0))
    return pred

def fit_predict_luc(X_fit, Y_true, Y_model, groups, cv, build_fn, kw):
    """Luc: Y = [Y_true_scaled, Y_model_scaled], 6-col output."""
    pred = np.full_like(Y_true, np.nan)
    for tr, te in cv.split(X_fit, Y_true, groups):
        tf.keras.backend.clear_session()
        xs = StandardScaler()
        x_tr_s = xs.fit_transform(X_fit[tr])
        x_te_s = xs.transform(X_fit[te])
        ys = StandardScaler()
        y_true_s = ys.fit_transform(Y_true[tr])
        y_phy_s = ys.transform(Y_model[tr])
        y_6col = np.concatenate([y_true_s, y_phy_s], axis=1)
        model = build_fn(**kw)
        model.fit(x_tr_s, y_6col, validation_split=0.2, epochs=200,
                  batch_size=kw.get("batch_size", 32), shuffle=True,
                  verbose=0, callbacks=CBS)
        p6 = model.predict(x_te_s, verbose=0)
        pred[te] = ys.inverse_transform(p6[:, :3])
    return pred

results = {}

def store(name, pred):
    results[name] = {TARGET_COLS[i]: float(np.sqrt(np.mean((Y_true[:, i] - pred[:, i])**2))) for i in range(3)}
    r = results[name]
    print(f"  {name}: Alim={r['Alim_T_out']:.4f} Ref={r['Ref_T_out']:.4f} Flux={r['Flux']:.4f}", flush=True)

# ============ STAGE 0 - CLASSICAL ============
print("Stage 0: sklearn models...", flush=True)
Xb = X

from sklearn.neural_network import MLPRegressor
p = fit_predict_sklearn(Pipeline([("scaler", StandardScaler()), ("model",
    MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(64,), learning_rate_init=0.001, alpha=0.0001,
                 max_iter=5000, random_state=SEED,
                 early_stopping=True, validation_fraction=0.2)))])), Xb, Y_true, groups, cv)
store("MLP_sklearn", p)

from sklearn.linear_model import LinearRegression
p = fit_predict_sklearn(Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]), Xb, Y_true, groups, cv)
store("OLS", p)

from sklearn.linear_model import Ridge
p = fit_predict_sklearn(Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10))]), Xb, Y_true, groups, cv)
store("Ridge", p)

from sklearn.linear_model import MultiTaskLasso
p = fit_predict_sklearn(Pipeline([("scaler", StandardScaler()), ("model", MultiTaskLasso(alpha=0.01, max_iter=10000, random_state=SEED))]), Xb, Y_true, groups, cv)
store("Lasso_MultiTask", p)

from sklearn.linear_model import Lasso
p = fit_predict_sklearn(Pipeline([("scaler", StandardScaler()), ("model", MultiOutputRegressor(Lasso(alpha=0.01, max_iter=10000, random_state=SEED)))]), Xb, Y_true, groups, cv)
store("Lasso_Indep", p)

from sklearn.linear_model import MultiTaskElasticNet
p = fit_predict_sklearn(Pipeline([("scaler", StandardScaler()), ("model", MultiTaskElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=SEED))]), Xb, Y_true, groups, cv)
store("ElasticNet_MultiTask", p)

from sklearn.linear_model import ElasticNet
p = fit_predict_sklearn(Pipeline([("scaler", StandardScaler()), ("model", MultiOutputRegressor(ElasticNet(alpha=0.1, l1_ratio=0.3, max_iter=10000, random_state=SEED)))]), Xb, Y_true, groups, cv)
store("ElasticNet_Indep", p)

from sklearn.ensemble import GradientBoostingRegressor
p = fit_predict_sklearn(Pipeline([("scaler", StandardScaler()), ("model",
    MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, max_depth=1, min_samples_leaf=5,
                              learning_rate=0.1, subsample=0.6, random_state=SEED)))]), Xb, Y_true, groups, cv)
store("GB", p)

from sklearn.ensemble import RandomForestRegressor
p = fit_predict_sklearn(Pipeline([("scaler", StandardScaler()), ("model",
    RandomForestRegressor(n_estimators=200, max_depth=2, max_features=0.8,
                          min_samples_leaf=10, min_samples_split=2, random_state=SEED))]), Xb, Y_true, groups, cv)
store("RF", p)

from sklearn.tree import DecisionTreeRegressor
p = fit_predict_sklearn(Pipeline([("scaler", StandardScaler()), ("model",
    DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, min_samples_split=2,
                          ccp_alpha=0.0, random_state=SEED))]), Xb, Y_true, groups, cv)
store("DT", p)

# ============ STAGE 1 - BASELINE ============
print("Stage 1: KerasMLP Baseline...", flush=True)
from models.keras_builders import build_keras_mlp
kw = dict(n_features=N_FEATURES_BASE, n_targets=N_TARGETS,
    hidden_layer_sizes=(256,), activation="tanh",
    learning_rate=0.001, l2=0.0001, loss="huber", huber_delta=1.0,
    optimizer="adam", batch_size=32)
p = fit_predict_keras_reg(Xb, Y_true, groups, cv, build_keras_mlp, kw)
store("KerasMLP_Baseline", p)

# ============ STAGE 2 - HYBRIDS ============
print("Stage 2: Hybrids...", flush=True)
Xp = np.concatenate([X, Y_model], axis=1)

from models.keras_builders import build_keras_mlp_residual_add_phy_x_only, build_keras_mlp_luc_loss_2m_output

kw = dict(n_features=N_FEATURES_BASE, n_targets=N_TARGETS,
    hidden_layer_sizes=(256,), activation="tanh",
    learning_rate=0.001, l2=0.0001, loss="huber", huber_delta=1.0,
    optimizer="adam", batch_size=32)
p = fit_predict_keras_reg(Xb, Y_true, groups, cv, build_keras_mlp, kw)
store("KerasMLP_FrozenBaseline", p)

kw = dict(n_features=N_FEATURES_BASE, n_targets=N_TARGETS,
    hidden_layer_sizes=(256,), activation="tanh",
    learning_rate=0.001, l2=0.0001, data_loss="huber", huber_delta=1.0,
    omega=0.0, physics_norm="mse", optimizer="adam", batch_size=32)
p = fit_predict_luc(Xb, Y_true, Y_model, groups, cv, build_keras_mlp_luc_loss_2m_output, kw)
store("KerasMLP_Luc_Restricted", p)

kw = dict(n_features=N_FEATURES_PLUS_PHY, n_targets=N_TARGETS,
    hidden_layer_sizes=(256,), activation="tanh",
    learning_rate=0.001, l2=1e-5, loss="huber", huber_delta=1.0,
    optimizer="adam", batch_size=32)
p = fit_predict_keras_reg(Xp, Y_true, groups, cv, build_keras_mlp, kw)
store("KerasMLP_ZohanHPD_Restricted", p)

kw = dict(n_features=N_FEATURES_PLUS_PHY, n_targets=N_TARGETS,
    hidden_layer_sizes=(256,), activation="tanh",
    learning_rate=0.001, l2=0.0, loss="huber", huber_delta=1.0,
    optimizer="adam", batch_size=32)
p = fit_predict_keras_reg(Xp, Y_true, groups, cv, build_keras_mlp_residual_add_phy_x_only, kw)
store("KerasMLP_ZohanResidual_Restricted", p)

# Build and save
rows = []
for name, r in results.items():
    rows.append({"Modelo": name, "RMSE_Alim_T_out": r["Alim_T_out"],
                 "RMSE_Ref_T_out": r["Ref_T_out"], "RMSE_Flux": r["Flux"]})
df_out = pd.DataFrame(rows).sort_values("RMSE_Flux").reset_index(drop=True)
df_out.to_csv(str(OUT_DIR / "oof_rmse_all_models.csv"), index=False)
df_out.to_csv(str(OUT_DIR / "oof_rmse_all_models_pt.csv"), index=False, sep=";", decimal=",")
print("\n=== OOF RMSE POR SAÍDA ===", flush=True)
print(df_out.to_string(index=False, float_format="%.4f"), flush=True)
print(f"\nSalvo em: {OUT_DIR}", flush=True)

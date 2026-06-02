#!/usr/bin/env python3
"""Quick OOF for Keras models only."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings; warnings.filterwarnings("ignore")
import numpy as np; np.random.seed(42)
import tensorflow as tf; tf.random.set_seed(42); tf.keras.backend.set_floatx("float32")

from config import N_SPLITS, FEATURE_COLS, TARGET_COLS, GROUP_COL, ORIGIN_COL, MODEL_MAP
from config import N_FEATURES_BASE, N_FEATURES_PLUS_PHY, N_TARGETS, KERAS_CALLBACKS
from data import read_tabular_csv, build_XY_groups_with_model_map
from cv import make_cv
from sklearn.preprocessing import StandardScaler
from models.keras_builders import build_keras_mlp, build_keras_mlp_residual_add_phy_x_only, build_keras_mlp_luc_loss_2m_output

DATA = r"C:\Users\frota\OneDrive\Documentos\TCC\dados_att_com_var_com_phy.csv"
df = read_tabular_csv(DATA, decimal_comma=True)
oc = ORIGIN_COL if ORIGIN_COL in df.columns else None
X, Y_true, groups, _, Y_model, _ = build_XY_groups_with_model_map(
    df=df, feature_cols=FEATURE_COLS, target_cols=TARGET_COLS,
    group_col=GROUP_COL, dropna=True, model_map=MODEL_MAP, origin_col=oc)
cv, _ = make_cv(N_SPLITS, origin_values=None)

CBS = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, min_delta=1e-4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_delta=1e-4, cooldown=0, min_lr=1e-6, verbose=0),
    tf.keras.callbacks.TerminateOnNaN(),
]

def run(name, build_fn, X_fit, Y_fit, y_true_ref, groups, cv, kw):
    pred = np.full_like(Y_fit, np.nan)
    for tr, te in cv.split(X_fit, Y_fit, groups):
        tf.keras.backend.clear_session()
        xs = StandardScaler(); ys = StandardScaler()
        x_tr_s = xs.fit_transform(X_fit[tr]); x_te_s = xs.transform(X_fit[te])
        y_tr_s = ys.fit_transform(Y_fit[tr])
        model = build_fn(**kw)
        model.fit(x_tr_s, y_tr_s, validation_split=0.2, epochs=200, batch_size=32, shuffle=True, verbose=0, callbacks=CBS)
        pred[te] = ys.inverse_transform(model.predict(x_te_s, verbose=0))
    for i, nm in enumerate(TARGET_COLS):
        rmse = float(np.sqrt(np.mean((y_true_ref[:, i] - pred[:, i])**2)))
        print(f"  {name} {nm}: RMSE = {rmse:.4f}", flush=True)

def run_luc(name, X_fit, Y_fit, Y_model, y_true_ref, groups, cv, kw):
    pred = np.full_like(Y_fit, np.nan)
    for tr, te in cv.split(X_fit, Y_fit, groups):
        tf.keras.backend.clear_session()
        xs = StandardScaler(); ys = StandardScaler()
        x_tr_s = xs.fit_transform(X_fit[tr]); x_te_s = xs.transform(X_fit[te])
        y_true_s = ys.fit_transform(Y_fit[tr]); y_phy_s = ys.transform(Y_model[tr])
        y_6 = np.concatenate([y_true_s, y_phy_s], axis=1)
        model = build_keras_mlp_luc_loss_2m_output(**kw)
        model.fit(x_tr_s, y_6, validation_split=0.2, epochs=200, batch_size=32, shuffle=True, verbose=0, callbacks=CBS)
        p6 = model.predict(x_te_s, verbose=0)
        pred[te] = ys.inverse_transform(p6[:, :3])
    for i, nm in enumerate(TARGET_COLS):
        rmse = float(np.sqrt(np.mean((y_true_ref[:, i] - pred[:, i])**2)))
        print(f"  {name} {nm}: RMSE = {rmse:.4f}", flush=True)

Xb = X
Xp = np.concatenate([X, Y_model], axis=1)

run("KerasMLP_Baseline", build_keras_mlp, Xb, Y_true, Y_true, groups, cv,
    dict(n_features=5, n_targets=3, hidden_layer_sizes=(256,), activation="tanh",
         learning_rate=0.001, l2=0.0001, loss="huber", huber_delta=1.0, optimizer="adam"))

run("FrozenBaseline", build_keras_mlp, Xb, Y_true, Y_true, groups, cv,
    dict(n_features=5, n_targets=3, hidden_layer_sizes=(256,), activation="tanh",
         learning_rate=0.001, l2=0.0001, loss="huber", huber_delta=1.0, optimizer="adam"))

run_luc("Luc_Restricted", Xb, Y_true, Y_model, Y_true, groups, cv,
    dict(n_features=5, n_targets=3, hidden_layer_sizes=(256,), activation="tanh",
         learning_rate=0.001, l2=0.0001, data_loss="huber", huber_delta=1.0,
         omega=0.0, physics_norm="mse", optimizer="adam"))

run("ZohanHPD_Restricted", build_keras_mlp, Xp, Y_true, Y_true, groups, cv,
    dict(n_features=8, n_targets=3, hidden_layer_sizes=(256,), activation="tanh",
         learning_rate=0.001, l2=1e-5, loss="huber", huber_delta=1.0, optimizer="adam"))

run("ZohanResidual_Restricted", build_keras_mlp_residual_add_phy_x_only, Xp, Y_true, Y_true, groups, cv,
    dict(n_features=8, n_targets=3, hidden_layer_sizes=(256,), activation="tanh",
         learning_rate=0.001, l2=0.0, loss="huber", huber_delta=1.0, optimizer="adam"))
print("Done!", flush=True)

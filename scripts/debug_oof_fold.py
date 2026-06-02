#!/usr/bin/env python3
"""Debug 3-fold OOF for ZohanResidual to compare with CV score 0.0678."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
tf.keras.backend.set_floatx('float32')
import warnings
warnings.filterwarnings('ignore')

from config import N_FEATURES_PLUS_PHY, N_TARGETS, N_SPLITS, KERAS_CALLBACKS
from config import FEATURE_COLS, TARGET_COLS, GROUP_COL, ORIGIN_COL, MODEL_MAP, FLUX_INDEX
from data import read_tabular_csv, build_XY_groups_with_model_map
from cv import make_cv
from sklearn.preprocessing import StandardScaler
from models.keras_builders import build_keras_mlp_residual_add_phy_x_only
from sklearn.model_selection import GroupKFold

DATA_CSV_PATH = r'C:\Users\frota\OneDrive\Documentos\TCC\dados_att_com_var_com_phy.csv'
df = read_tabular_csv(DATA_CSV_PATH, decimal_comma=True)
origin_col = ORIGIN_COL if ORIGIN_COL in df.columns else None
X, Y_true, groups, used_df, Y_model, origin_values = build_XY_groups_with_model_map(
    df=df, feature_cols=FEATURE_COLS, target_cols=TARGET_COLS,
    group_col=GROUP_COL, dropna=True, model_map=MODEL_MAP,
    origin_col=origin_col)
cv, eval_mask = make_cv(N_SPLITS, origin_values=origin_values)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, min_delta=1e-4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_delta=1e-4, cooldown=0, min_lr=1e-6, verbose=0),
    tf.keras.callbacks.TerminateOnNaN(),
]

X_in = np.concatenate([X, Y_model], axis=1)
all_rmse_flux = []

for fold, (tr, te) in enumerate(cv.split(X_in, Y_true, groups)):
    xs = StandardScaler()
    x_tr_s = xs.fit_transform(X_in[tr])
    x_te_s = xs.transform(X_in[te])
    ys = StandardScaler()
    y_tr_s = ys.fit_transform(Y_true[tr])

    tf.keras.backend.clear_session()
    model = build_keras_mlp_residual_add_phy_x_only(
        n_features=8, n_targets=3, hidden_layer_sizes=(256,),
        activation='tanh', learning_rate=0.001, l2=0.0,
        loss='huber', huber_delta=1.0, optimizer='adam')
    hist = model.fit(x_tr_s, y_tr_s, validation_split=0.2, epochs=200,
                     batch_size=32, shuffle=True, verbose=0, callbacks=callbacks)
    pred_s = model.predict(x_te_s, verbose=0)
    pred = ys.inverse_transform(pred_s)
    for k in range(3):
        rmse = float(np.sqrt(np.mean((Y_true[te, k] - pred[:, k])**2)))
        if k == FLUX_INDEX:
            all_rmse_flux.append(rmse)
        print(f'Fold {fold}, output {k}: RMSE = {rmse:.4f}', flush=True)
    print(f'Fold {fold}: epochs = {len(hist.history["loss"])}', flush=True)

if all_rmse_flux:
    mean_rmse_flux = float(np.mean(all_rmse_flux))
    print(f'\nMean Flux RMSE across {len(all_rmse_flux)} folds: {mean_rmse_flux:.6f}', flush=True)
print('Done', flush=True)

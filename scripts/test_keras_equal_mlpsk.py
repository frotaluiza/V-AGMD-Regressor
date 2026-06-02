#!/usr/bin/env python3
"""Test KerasMLP with exact MLP_sklearn configuration (batch=full, no clipnorm, mse, constant lr)."""
import argparse, json, os, sys, logging
from pathlib import Path
from datetime import datetime
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SEED, N_SPLITS, FEATURE_COLS, TARGET_COLS, GROUP_COL, ORIGIN_COL
from config import N_FEATURES_BASE, N_TARGETS, FLUX_INDEX, MODEL_MAP
from data import read_tabular_csv, build_XY_groups_with_model_map
from cv import make_cv
from selection import make_neg_rmse_single_true_target_scorer, C_keras_mlp
from runner import FamilySpec, run_family, winners_table
from file_io import save_outputs

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=r"C:\Users\frota\OneDrive\Documentos\TCC\dados_att_com_var_com_phy.csv")
    return p.parse_args()

def main():
    args = parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    DATA_CSV_PATH = args.csv
    dataset_stem = Path(DATA_CSV_PATH).stem.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR = os.path.normpath(os.path.join(Path(__file__).resolve().parent.parent, f"test_keras_equal_mlpsk_{dataset_stem}_{timestamp}"))
    os.makedirs(OUT_DIR, exist_ok=True)
    logger = logging.getLogger("test_keras_mlpsk")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(OUT_DIR, "run.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    np.random.seed(SEED)
    import tensorflow as tf
    tf.random.set_seed(SEED)
    tf.keras.backend.set_floatx("float32")
    tf.config.optimizer.set_jit(True)
    flux_scorer = make_neg_rmse_single_true_target_scorer(FLUX_INDEX, N_TARGETS, "Flux")
    df = read_tabular_csv(DATA_CSV_PATH, decimal_comma=True, logger=logger)
    origin_col = ORIGIN_COL if ORIGIN_COL in df.columns else None
    X, Y_true, groups, used_df, Y_model, origin_values = build_XY_groups_with_model_map(
        df=df, feature_cols=FEATURE_COLS, target_cols=TARGET_COLS,
        group_col=GROUP_COL, dropna=True, model_map=MODEL_MAP,
        origin_col=origin_col, logger=logger,
    )
    cv, eval_mask = make_cv(N_SPLITS, origin_values=origin_values, logger=logger)
    from models.keras_builders import make_keras_mlp_estimator

    n_samples = X.shape[0]
    full_batch = n_samples  # 174 = sklearn's auto batch size (= min(200, n))

    # Early stopping on training loss (like sklearn's n_iter_no_change)
    early_stop_loss = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=10,
        min_delta=1e-4, restore_best_weights=True,
    )
    term_nan = tf.keras.callbacks.TerminateOnNaN()

    grid_mlpsk_exact = [dict(
        model__model__activation=["relu"],
        model__model__optimizer=["adam"],
        model__model__learning_rate=[3e-3, 1e-3, 3e-4, 1e-4],  # same as MLP_sklearn grid
        model__model__l2=[0.0, 1e-6, 1e-5, 1e-4, 1e-3],        # same as alpha grid
        model__model__hidden_layer_sizes=[(64,), (128,)],        # (64,) was best
        model__model__n_features=[N_FEATURES_BASE],
        model__model__n_targets=[N_TARGETS],
        model__model__loss=["mse"],      # MLP_sklearn uses MSE
        model__model__clipnorm=[None],   # No gradient clipping (MLP_sklearn default)
        model__model__batch_size=[full_batch],  # Full batch like sklearn
        model__model__epochs=[10000],           # High max (sklearn has n_iter_no_change)
        model__fit__validation_split=[0.0],  # No validation split
        model__fit__shuffle=[True],
        model__fit__verbose=[0],
        model__fit__callbacks=[[early_stop_loss, term_nan]],  # Early stop on train loss only
    )]
    spec_mlpsk_exact = FamilySpec(
        name="KerasMLP_eq_MLPsklearn", estimator=make_keras_mlp_estimator,
        search="grid", param_grid=grid_mlpsk_exact,
        x_mode="x", y_mode="true", complexity_fn=C_keras_mlp,
        n_jobs=1, scale_y=True,
    )
    # Also test with early stopping + reduce lr (our original Keras setup) but with batch=full, no clipnorm, mse loss
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=20,
        min_delta=1e-4, restore_best_weights=True,
    )
    term_nan = tf.keras.callbacks.TerminateOnNaN()
    grid_keras_fullbatch = [dict(
        model__model__activation=["relu"],
        model__model__optimizer=["adam"],
        model__model__learning_rate=[3e-3, 1e-3],
        model__model__l2=[1e-4, 1e-5, 0.0],
        model__model__hidden_layer_sizes=[(64,), (128,)],
        model__model__n_features=[N_FEATURES_BASE],
        model__model__n_targets=[N_TARGETS],
        model__model__loss=["mse"],
        model__model__clipnorm=[None],
        model__model__batch_size=[full_batch],
        model__model__epochs=[2000],
        model__fit__validation_split=[0.2],
        model__fit__shuffle=[True],
        model__fit__verbose=[0],
        model__fit__callbacks=[[early_stop, term_nan]],
    )]
    spec_keras_fullbatch = FamilySpec(
        name="KerasMLP_fullbatch", estimator=make_keras_mlp_estimator,
        search="grid", param_grid=grid_keras_fullbatch,
        x_mode="x", y_mode="true", complexity_fn=C_keras_mlp,
        n_jobs=1, scale_y=True,
    )
    families = [spec_mlpsk_exact, spec_keras_fullbatch]
    out_dirs = [os.path.join(OUT_DIR, "mlpsk_exact"), os.path.join(OUT_DIR, "keras_fullbatch")]
    winners = []
    for spec, odir in zip(families, out_dirs):
        os.makedirs(odir, exist_ok=True)
        winner = run_family(spec=spec, X=X, Y_true=Y_true, Y_model=Y_model,
                            groups=groups, cv=cv, scoring=flux_scorer,
                            eval_mask=eval_mask, logger=logger)
        winners.append(winner)
    summary = winners_table(winners, target_cols=TARGET_COLS)
    save_outputs(out_dir=OUT_DIR, winners=winners, summary=summary,
                 target_cols=TARGET_COLS, X=X, Y_true=Y_true, groups=groups,
                 Y_model=Y_model, cv=cv, prefix=f"test_keras_mlpsk_{dataset_stem}",
                 make_plots=True, save_family_tables=True, logger=logger,
                 make_oof_plots=False, make_winners_rmse_plot=True,
                 make_sweeps=True, fit_final_keras_histories=False)
    print(f"\n{'='*60}")
    for w in winners:
        print(f"{w.name}: RMSE_Flux = {abs(w.best_score):.4f}")
    print(f"MLP_sklearn reference: 0.0735")
    print(f"Results in: {OUT_DIR}")

if __name__ == "__main__":
    main()

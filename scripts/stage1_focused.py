#!/usr/bin/env python3
"""Focused Stage 1 only: search KerasMLP baseline to beat MLP_sklearn (0.0735)."""
import argparse, json, os, sys, time, logging, threading
from pathlib import Path
from datetime import datetime
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SEED, N_SPLITS, FEATURE_COLS, TARGET_COLS, GROUP_COL, ORIGIN_COL, DATA_PATH
from config import N_FEATURES_BASE, N_TARGETS, FLUX_INDEX, MODEL_MAP, KERAS_CALLBACKS
from data import read_tabular_csv, build_XY_groups_with_model_map
from cv import make_cv, resolve_n_splits
from selection import make_neg_rmse_single_true_target_scorer, C_keras_mlp
from runner import FamilySpec, run_family, winners_table
from file_io import save_outputs

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(DATA_PATH))
    return p.parse_args()

def setup_logger(name, level=logging.INFO, out_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if out_dir:
        fh = logging.FileHandler(os.path.join(out_dir, "run.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def heartbeat(interval=10):
    t0 = time.time()
    while True:
        time.sleep(interval)
        elapsed = time.time() - t0
        print(f"HEARTBEAT: alive_for={int(elapsed)}s", flush=True)

def main():
    args = parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    DATA_CSV_PATH = args.csv
    dataset_stem = Path(DATA_CSV_PATH).stem.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR = os.path.normpath(os.path.join(Path(__file__).resolve().parent.parent, f"stage1_only_{dataset_stem}_{timestamp}"))
    os.makedirs(OUT_DIR, exist_ok=True)
    logger = setup_logger("stage1_only", level=logging.INFO, out_dir=OUT_DIR)

    t_heart = threading.Thread(target=heartbeat, args=(10,), daemon=True)
    t_heart.start()

    np.random.seed(SEED)
    import tensorflow as tf
    tf.random.set_seed(SEED)
    tf.keras.backend.set_floatx("float32")
    tf.config.optimizer.set_jit(True)

    flux_scorer = make_neg_rmse_single_true_target_scorer(FLUX_INDEX, N_TARGETS, "Flux")

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

    from models.keras_builders import make_keras_mlp_estimator

    # Massive search space
    keras_mlp_grid = [dict(
        model__model__activation=["relu", "tanh"],
        model__model__optimizer=["adam", "adamw"],
        model__model__learning_rate=[3e-3, 1e-3, 3e-4, 1e-4],
        model__model__l2=[0.0, 1e-6, 1e-5, 1e-4, 1e-3],
        model__model__hidden_layer_sizes=[
            (32,), (64,), (128,), (256,), (512,),
            (64, 32), (128, 64), (256, 128), (512, 256),
            (128, 64, 32), (256, 128, 64), (512, 256, 128),
        ],
        model__model__n_features=[N_FEATURES_BASE],
        model__model__n_targets=[N_TARGETS],
        model__model__loss=["huber"],
        model__model__huber_delta=[1.0],
        model__batch_size=[16, 32, 64],
        model__epochs=[200],
        model__fit__validation_split=[0.2],
        model__fit__shuffle=[True],
        model__fit__verbose=[0],
        model__fit__callbacks=[[early_stop, reduce_lr, term_nan]],
    )]

    baseline_family = FamilySpec(
        name="KerasMLP_BaselineSearch", estimator=make_keras_mlp_estimator,
        search="random", param_grid=keras_mlp_grid, n_iter=80,
        x_mode="x", y_mode="true", complexity_fn=C_keras_mlp,
        n_jobs=1, scale_y=True,
    )

    logger.info("=" * 72)
    logger.info(f"FOCUSED STAGE 1 - searching KerasMLP to beat MLP_sklearn (0.0735)")
    logger.info(f"n_iter=80, epochs=200")
    logger.info(f"Grid: {len(keras_mlp_grid[0])} params")
    logger.info(f"  learning_rate x l2 x hidden x batch_size x activation x optimizer = "
                f"{len(keras_mlp_grid[0]['model__model__learning_rate'])} x "
                f"{len(keras_mlp_grid[0]['model__model__l2'])} x "
                f"{len(keras_mlp_grid[0]['model__model__hidden_layer_sizes'])} x "
                f"{len(keras_mlp_grid[0]['model__batch_size'])} x "
                f"{len(keras_mlp_grid[0]['model__model__activation'])} x "
                f"{len(keras_mlp_grid[0]['model__model__optimizer'])}")

    stage1_dir = os.path.join(OUT_DIR, "stage1_baseline_search")
    os.makedirs(stage1_dir, exist_ok=True)

    # Inline run_model_selection_from_csv
    df = read_tabular_csv(DATA_CSV_PATH, decimal_comma=True, logger=logger)
    origin_col = ORIGIN_COL if ORIGIN_COL in df.columns else None
    X, Y_true, groups, used_df, Y_model, origin_values = build_XY_groups_with_model_map(
        df=df, feature_cols=FEATURE_COLS, target_cols=TARGET_COLS,
        group_col=GROUP_COL, dropna=True, model_map=MODEL_MAP,
        origin_col=origin_col, logger=logger,
    )
    n_splits = resolve_n_splits(groups, config_n_splits=N_SPLITS)
    cv, eval_mask = make_cv(n_splits, origin_values=origin_values, logger=logger)

    winners = []
    for spec in [baseline_family]:
        scoring = spec.scoring if spec.scoring is not None else flux_scorer
        logger.info(f"RUN FAMILY: {spec.name}")
        winners.append(run_family(
            spec=spec, X=X, Y_true=Y_true, Y_model=Y_model,
            groups=groups, cv=cv, scoring=scoring,
            eval_mask=eval_mask, logger=logger,
        ))

    summary = winners_table(winners, target_cols=TARGET_COLS)
    save_outputs(
        out_dir=stage1_dir, winners=winners, summary=summary,
        target_cols=TARGET_COLS, X=X, Y_true=Y_true, groups=groups,
        Y_model=Y_model, cv=cv,
        include_splits_in_cv_results=False,
        prefix=f"agmd_stage1_baseline_{dataset_stem}",
        make_plots=True, save_family_tables=True, logger=logger,
        make_oof_plots=True, make_winners_rmse_plot=True,
        make_sweeps=True, fit_final_keras_histories=False,
    )

    winner = winners[0]
    rmse = abs(winner.best_score)
    target = 0.0735
    logger.info(f"Best KerasMLP RMSE_Flux = {rmse:.4f} (target {target}, diff {rmse - target:+.4f})")
    if rmse < target:
        logger.info("*** KERASMLP BEAT MLP_SKLEARN! ***")
    else:
        logger.info("KerasMLP still behind MLP_sklearn. Need more search.")

    print(f"\n=== BEST ===")
    print(f"RMSE_Flux = {rmse:.4f}")
    print(f"Params: {json.dumps({k: str(v) for k, v in winner.best_params.items()}, indent=2)}")
    print(f"\nResults in: {OUT_DIR}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Roda apenas a rede híbrida KerasMLP_ZohanHRNN_Restricted com hiperparâmetros
baseline fixos (extraídos do gridSearchCV do Stage 1).

Uso:
  python scripts/run_hrnn_only.py --csv <caminho_do_csv>

Equivalente a: t3.py
"""

import argparse
import json
import os
import sys
import time
import logging
import threading
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    SEED, N_SPLITS, FEATURE_COLS, TARGET_COLS, GROUP_COL,
    N_FEATURES_BASE, N_TARGETS, N_FEATURES_PLUS_PHY, FLUX_INDEX, MODEL_MAP,
    DEFAULT_GAP_FILTER_TOL, DEFAULT_USE_GAP_FILTER, DEFAULT_USE_GAP_TIEBREAK,
    KERAS_CALLBACKS,
)
from data import read_tabular_csv, build_XY_groups_with_model_map
from selection import make_neg_rmse_single_true_target_scorer, C_keras_mlp
from runner import FamilySpec, run_family, winners_table
from sweep import build_hrnn_restricted_grid_from_baseline
from file_io import save_outputs
from models.keras_builders import KERAS_AVAILABLE, make_keras_mlp_hrnn_estimator
from sklearn.model_selection import GroupKFold


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run only KerasMLP_ZohanHRNN_Restricted with fixed baseline params.")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV dataset.")
    return parser.parse_args()


def setup_logger(name, level=logging.INFO, log_to_file=True, out_dir=None, filename="run.log"):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_to_file and out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(out_dir, filename), mode="w", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def start_heartbeat(seconds=10):
    def _hb():
        t0 = time.time()
        k = 0
        while True:
            time.sleep(seconds)
            k += 1
            print(f"HEARTBEAT {k}: alive_for={int(time.time() - t0)}s", flush=True)
    threading.Thread(target=_hb, daemon=True).start()


def main():
    args = parse_cli_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    start_heartbeat(seconds=10)

    print("SCRIPT STARTED", flush=True)

    if not KERAS_AVAILABLE:
        raise RuntimeError("Keras/SciKeras is not available in this environment.")

    import tensorflow as tf

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_CSV_PATH = args.csv
    DATA_CSV_NAME = os.path.basename(DATA_CSV_PATH)
    dataset_stem = Path(DATA_CSV_NAME).stem.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR = os.path.join(SCRIPT_DIR, "..", f"results_hrnn_only_{dataset_stem}_{timestamp}")
    OUT_DIR = os.path.normpath(OUT_DIR)

    logger = setup_logger("hrnn_only", level=logging.INFO, log_to_file=True, out_dir=OUT_DIR, filename="run.log")
    logger.info(f"DATA_CSV_PATH={DATA_CSV_PATH}")
    logger.info(f"Output directory: {OUT_DIR}")

    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.keras.backend.set_floatx("float32")
    tf.config.optimizer.set_jit(True)

    # Fixed baseline best params (from Stage 1 GridSearchCV)
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

    baseline_best_params = {
        "model__batch_size": 64,
        "model__epochs": 200,
        "model__fit__validation_split": 0.2,
        "model__fit__shuffle": True,
        "model__fit__verbose": 0,
        "model__fit__callbacks": [early_stop, reduce_lr, term_nan],
        "model__model__activation": "relu",
        "model__model__hidden_layer_sizes": (128,),
        "model__model__huber_delta": 1.0,
        "model__model__l2": 0.0,
        "model__model__learning_rate": 1e-3,
        "model__model__loss": "huber",
        "model__model__n_features": N_FEATURES_BASE,
        "model__model__n_targets": N_TARGETS,
        "model__model__optimizer": "adam",
    }

    hrnn_restricted_grid = build_hrnn_restricted_grid_from_baseline(
        baseline_best_params=baseline_best_params,
        n_features_plus_phy=N_FEATURES_PLUS_PHY,
        n_targets=N_TARGETS,
    )

    flux_rmse_scorer = make_neg_rmse_single_true_target_scorer(
        target_index=FLUX_INDEX, n_true_targets=N_TARGETS, target_name="Flux",
    )

    families = [
        FamilySpec(
            name="KerasMLP_ZohanHRNN_Restricted",
            estimator=make_keras_mlp_hrnn_estimator,
            search="grid",
            param_grid=hrnn_restricted_grid,
            x_mode="x_plus_model",
            y_mode="true",
            complexity_fn=C_keras_mlp,
            n_jobs=1,
            scale_y=True,
            use_gap_filter=DEFAULT_USE_GAP_FILTER,
            gap_filter_tol=DEFAULT_GAP_FILTER_TOL,
            use_gap_tiebreak=DEFAULT_USE_GAP_TIEBREAK,
        ),
    ]

    df = read_tabular_csv(DATA_CSV_PATH, decimal_comma=True, sep=",", logger=logger)
    X, Y_true, groups, used_df, Y_model, _ = build_XY_groups_with_model_map(
        df=df, feature_cols=FEATURE_COLS, target_cols=TARGET_COLS,
        group_col=GROUP_COL, dropna=True, model_map=MODEL_MAP, logger=logger,
    )

    cv = GroupKFold(n_splits=N_SPLITS)

    winners = []
    for spec in families:
        scoring = spec.scoring if spec.scoring is not None else flux_rmse_scorer
        logger.info("=" * 72)
        logger.info(f"RUN FAMILY: {spec.name}")

        w = run_family(
            spec=spec, X=X, Y_true=Y_true, Y_model=Y_model,
            groups=groups, cv=cv, scoring=scoring, logger=logger,
        )
        winners.append(w)

    stage2_dir = os.path.join(OUT_DIR, "stage2_hrnn_only")
    summary = winners_table(winners, target_cols=TARGET_COLS)

    save_outputs(
        out_dir=stage2_dir, winners=winners, summary=summary,
        target_cols=TARGET_COLS, X=X, Y_true=Y_true, groups=groups,
        Y_model=Y_model, cv=cv,
        prefix=f"agmd_stage2_hrnn_only_{dataset_stem}",
        logger=logger,
    )

    cols_show = [
        "family", "scoring", "rmse_cv_Flux", "rmse_train_Flux", "gap_rmse_Flux",
        "complexity", "r2_Flux", "rmse_Flux", "rmse_Alim_T_out", "rmse_Ref_T_out",
        "rank_by_score_space", "rank_by_flux_rmse",
    ]
    existing = [c for c in cols_show if c in summary.columns]

    print("\n=== FINAL SUMMARY ===", flush=True)
    print(summary[existing].to_string(index=False), flush=True)
    print(f"\nSaved outputs in: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

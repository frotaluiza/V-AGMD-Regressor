#!/usr/bin/env python3
"""
Pipeline completo de 3 estágios:
  Stage 0: Modelos clássicos
  Stage 1: Busca baseline KerasMLP
  Stage 2: Híbridos restritos (FrozenBaseline, Luc, HPD, Residual, HRNN)

Uso:
  python scripts/run_pipeline.py --csv <caminho_do_csv>

Equivalente a: t2.py
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    SEED, N_SPLITS, FEATURE_COLS, TARGET_COLS, GROUP_COL, ORIGIN_COL,
    N_FEATURES_BASE, N_TARGETS, N_FEATURES_PLUS_PHY, FLUX_INDEX, MODEL_MAP,
    DEFAULT_GAP_FILTER_TOL, DEFAULT_USE_GAP_FILTER, DEFAULT_USE_GAP_TIEBREAK,
    KERAS_CALLBACKS,
)
from data import read_tabular_csv, build_XY_groups_with_model_map
from cv import make_cv
from selection import make_neg_rmse_single_true_target_scorer, C_keras_mlp
from selection import COMPLEXITY_MAP
from runner import FamilySpec, run_family, winners_table, _make_scaled_pipeline, _resolve_estimator
from sweep import (
    build_frozen_baseline_grid, build_luc_restricted_grid_from_baseline,
    build_hpd_restricted_grid_from_baseline, build_residual_restricted_grid_from_baseline,
    build_hrnn_restricted_grid_from_baseline,
)
from file_io import save_outputs

from models.classical import (
    make_ols, make_ridge, make_lasso_multitask, make_elasticnet_multitask,
    make_lasso_indep, make_elasticnet_indep, make_dt, make_rf, make_gb, make_mlp_sklearn,
)
from models.keras_builders import (
    make_keras_mlp_estimator, make_keras_mlp_luc_estimator,
    make_keras_mlp_residual_x_only_estimator, make_keras_mlp_hrnn_estimator,
    KERAS_AVAILABLE,
)


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Unified model selection runner for AGMD datasets.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the input CSV dataset.")
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


def run_model_selection_from_csv(
    csv_path,
    feature_cols,
    target_cols,
    group_col,
    families,
    n_splits,
    global_scoring,
    decimal_comma=True,
    sep=",",
    encoding=None,
    dropna=True,
    out_dir=None,
    include_splits_in_cv_results=False,
    save_prefix="model_selection",
    make_plots=True,
    save_family_tables=True,
    model_map=None,
    run_only_families=None,
    logger=None,
    make_oof_plots=True,
    make_winners_rmse_plot=True,
    make_sweeps=True,
    fit_final_keras_histories=True,
    origin_col=ORIGIN_COL,
):
    df = read_tabular_csv(csv_path, decimal_comma=decimal_comma, sep=sep, encoding=encoding, logger=logger)

    effective_origin_col = origin_col if origin_col in df.columns else None

    X, Y_true, groups, used_df, Y_model, origin_values = build_XY_groups_with_model_map(
        df=df, feature_cols=feature_cols, target_cols=target_cols,
        group_col=group_col, dropna=dropna, model_map=model_map,
        origin_col=effective_origin_col, logger=logger,
    )

    cv, eval_mask = make_cv(n_splits, origin_values=origin_values, logger=logger)

    only_set = set(run_only_families) if run_only_families is not None else None

    winners = []
    for spec in families:
        if only_set is not None and spec.name not in only_set:
            continue
        scoring = spec.scoring if spec.scoring is not None else global_scoring
        if logger:
            logger.info("=" * 72)
            logger.info(f"RUN FAMILY: {spec.name}")

        winners.append(run_family(
            spec=spec, X=X, Y_true=Y_true, Y_model=Y_model,
            groups=groups, cv=cv, scoring=scoring,
            eval_mask=eval_mask, logger=logger,
        ))

    summary = winners_table(winners, target_cols=target_cols)

    if out_dir is not None:
        if logger:
            logger.info(f"Saving outputs to: {out_dir}")
        save_outputs(
            out_dir=out_dir, winners=winners, summary=summary,
            target_cols=target_cols, X=X, Y_true=Y_true, groups=groups,
            Y_model=Y_model, cv=cv,
            include_splits_in_cv_results=include_splits_in_cv_results,
            prefix=save_prefix, make_plots=make_plots,
            save_family_tables=save_family_tables, logger=logger,
            make_oof_plots=make_oof_plots, make_winners_rmse_plot=make_winners_rmse_plot,
            make_sweeps=make_sweeps, fit_final_keras_histories=fit_final_keras_histories,
        )

    return dict(used_df=used_df, X=X, Y_true=Y_true, Y_model=Y_model, groups=groups,
                origin_values=origin_values, cv=cv, winners=winners, summary=summary)


def main():
    args = parse_cli_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    start_heartbeat(seconds=10)

    print("SCRIPT STARTED", flush=True)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_CSV_PATH = args.csv
    DATA_CSV_NAME = os.path.basename(DATA_CSV_PATH)
    dataset_stem = Path(DATA_CSV_NAME).stem.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR = os.path.join(SCRIPT_DIR, "..", f"results_{dataset_stem}_{timestamp}")
    OUT_DIR = os.path.normpath(OUT_DIR)

    logger = setup_logger("model_selection", level=logging.INFO, log_to_file=True, out_dir=OUT_DIR, filename="run.log")
    logger.info(f"Script directory: {SCRIPT_DIR}")
    logger.info(f"Output directory: {OUT_DIR}")
    logger.info(f"KERAS_AVAILABLE={KERAS_AVAILABLE}")
    logger.info(f"DATA_CSV_PATH={DATA_CSV_PATH}")
    logger.info(f"dataset_stem={dataset_stem}")

    np.random.seed(SEED)

    if KERAS_AVAILABLE:
        import tensorflow as tf
        try:
            tf.random.set_seed(SEED)
            tf.keras.backend.set_floatx("float32")
            tf.config.optimizer.set_jit(True)
        except Exception:
            pass

    flux_rmse_scorer = make_neg_rmse_single_true_target_scorer(
        target_index=FLUX_INDEX, n_true_targets=N_TARGETS, target_name="Flux",
    )

    # --- Classical families ---
    classical_families = [
        FamilySpec(name="OLS", estimator=make_ols, search="grid", param_grid={},
                   complexity_fn=COMPLEXITY_MAP.get("OLS"), n_jobs=1, scale_y=True,
                   use_gap_filter=DEFAULT_USE_GAP_FILTER, gap_filter_tol=DEFAULT_GAP_FILTER_TOL,
                   use_gap_tiebreak=DEFAULT_USE_GAP_TIEBREAK),
        FamilySpec(name="Ridge", estimator=make_ridge, search="grid",
                   param_grid={"model__alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100]},
                   complexity_fn=COMPLEXITY_MAP.get("Ridge"), n_jobs=1, scale_y=True,
                   use_gap_filter=DEFAULT_USE_GAP_FILTER, gap_filter_tol=DEFAULT_GAP_FILTER_TOL,
                   use_gap_tiebreak=DEFAULT_USE_GAP_TIEBREAK),
        FamilySpec(name="Lasso_MultiTask", estimator=make_lasso_multitask, search="grid",
                   param_grid={"model__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1]},
                   complexity_fn=COMPLEXITY_MAP.get("Lasso_MultiTask"), n_jobs=1, scale_y=True,
                   use_gap_filter=DEFAULT_USE_GAP_FILTER, gap_filter_tol=DEFAULT_GAP_FILTER_TOL,
                   use_gap_tiebreak=DEFAULT_USE_GAP_TIEBREAK),
        FamilySpec(name="ElasticNet_MultiTask", estimator=make_elasticnet_multitask, search="grid",
                   param_grid={"model__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1],
                                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
                   complexity_fn=COMPLEXITY_MAP.get("ElasticNet_MultiTask"), n_jobs=1, scale_y=True,
                   use_gap_filter=DEFAULT_USE_GAP_FILTER, gap_filter_tol=DEFAULT_GAP_FILTER_TOL,
                   use_gap_tiebreak=DEFAULT_USE_GAP_TIEBREAK),
        FamilySpec(name="Lasso_Indep(MultiOutputRegressor)", estimator=make_lasso_indep, search="grid",
                   param_grid={"model__estimator__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1]},
                   complexity_fn=COMPLEXITY_MAP.get("Lasso_Indep(MultiOutputRegressor)"), n_jobs=1, scale_y=True,
                   use_gap_filter=DEFAULT_USE_GAP_FILTER, gap_filter_tol=DEFAULT_GAP_FILTER_TOL,
                   use_gap_tiebreak=DEFAULT_USE_GAP_TIEBREAK),
        FamilySpec(name="ElasticNet_Indep(MultiOutputRegressor)", estimator=make_elasticnet_indep, search="grid",
                   param_grid={"model__estimator__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1],
                                "model__estimator__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
                   complexity_fn=COMPLEXITY_MAP.get("ElasticNet_Indep(MultiOutputRegressor)"), n_jobs=1, scale_y=True,
                   use_gap_filter=DEFAULT_USE_GAP_FILTER, gap_filter_tol=DEFAULT_GAP_FILTER_TOL,
                   use_gap_tiebreak=DEFAULT_USE_GAP_TIEBREAK),
        FamilySpec(name="DT", estimator=make_dt, search="grid",
                   param_grid={"model__max_depth": [2, 3, 4, 5, 6, None],
                                "model__min_samples_leaf": [1, 2, 5, 8, 10],
                                "model__min_samples_split": [2, 5, 10, 15],
                                "model__ccp_alpha": [0.0, 1e-4, 1e-3, 1e-2]},
                   complexity_fn=COMPLEXITY_MAP.get("DT"), n_jobs=1, scale_y=True,
                   use_gap_filter=DEFAULT_USE_GAP_FILTER, gap_filter_tol=DEFAULT_GAP_FILTER_TOL,
                   use_gap_tiebreak=DEFAULT_USE_GAP_TIEBREAK),
        FamilySpec(name="RF", estimator=make_rf, search="grid",
                   param_grid={"model__n_estimators": [200, 400, 600],
                                "model__max_depth": [2, 3, 4, 5, None],
                                "model__min_samples_leaf": [1, 2, 5, 8, 10],
                                "model__min_samples_split": [2, 5, 10],
                                "model__max_features": [0.5, 0.8, 1.0]},
                   complexity_fn=COMPLEXITY_MAP.get("RF"), n_jobs=1, scale_y=True,
                   use_gap_filter=DEFAULT_USE_GAP_FILTER, gap_filter_tol=DEFAULT_GAP_FILTER_TOL,
                   use_gap_tiebreak=DEFAULT_USE_GAP_TIEBREAK),
        FamilySpec(name="GB", estimator=make_gb, search="grid",
                   param_grid={"model__estimator__n_estimators": [50, 100, 200, 400],
                                "model__estimator__learning_rate": [0.01, 0.03, 0.05, 0.1],
                                "model__estimator__max_depth": [1, 2, 3],
                                "model__estimator__min_samples_leaf": [1, 2, 5, 10],
                                "model__estimator__subsample": [0.6, 0.8, 1.0]},
                   complexity_fn=COMPLEXITY_MAP.get("GB"), n_jobs=1,
                   force_multioutput_wrapper=True, scale_y=True,
                   use_gap_filter=DEFAULT_USE_GAP_FILTER, gap_filter_tol=DEFAULT_GAP_FILTER_TOL,
                   use_gap_tiebreak=DEFAULT_USE_GAP_TIEBREAK),
        FamilySpec(name="MLP_sklearn", estimator=make_mlp_sklearn, search="grid",
                   param_grid={"model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                                "model__alpha": [0.0, 1e-6, 1e-5, 1e-4],
                                "model__learning_rate_init": [3e-3, 1e-3, 3e-4, 1e-4]},
                   complexity_fn=COMPLEXITY_MAP.get("MLP_sklearn"), n_jobs=1, scale_y=True,
                   use_gap_filter=DEFAULT_USE_GAP_FILTER, gap_filter_tol=DEFAULT_GAP_FILTER_TOL,
                   use_gap_tiebreak=DEFAULT_USE_GAP_TIEBREAK),
    ]

    if not KERAS_AVAILABLE:
        logger.warning("Keras/SciKeras not available -> only non-hybrid classical families will run.")
        final_result = run_model_selection_from_csv(
            csv_path=DATA_CSV_PATH, feature_cols=FEATURE_COLS, target_cols=TARGET_COLS,
            group_col=GROUP_COL, families=classical_families, n_splits=N_SPLITS,
            global_scoring=flux_rmse_scorer, model_map=MODEL_MAP,
            out_dir=OUT_DIR, save_prefix=f"agmd_classical_only_{dataset_stem}",
            logger=logger, make_oof_plots=True, make_winners_rmse_plot=True,
            make_sweeps=True, fit_final_keras_histories=False, origin_col=ORIGIN_COL,
        )
        print("\n=== FINAL SUMMARY ===", flush=True)
        cols_show = ["family", "scoring", "rmse_cv_Flux", "rmse_train_Flux", "gap_rmse_Flux",
                      "complexity", "r2_Flux", "rmse_Flux", "rank_by_flux_rmse"]
        existing = [c for c in cols_show if c in final_result["summary"].columns]
        print(final_result["summary"][existing].to_string(index=False), flush=True)
        print(f"\nSaved outputs in: {OUT_DIR}", flush=True)
        raise SystemExit(0)

    import tensorflow as tf

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

    # --- Stage 0 ---
    logger.info("=" * 72)
    logger.info("STAGE 0 - CLASSICAL NON-HYBRID MODELS")
    stage0_dir = os.path.join(OUT_DIR, "stage0_classical_models")
    stage0_result = run_model_selection_from_csv(
        csv_path=DATA_CSV_PATH, feature_cols=FEATURE_COLS, target_cols=TARGET_COLS,
        group_col=GROUP_COL, families=classical_families, n_splits=N_SPLITS,
        global_scoring=flux_rmse_scorer, model_map=MODEL_MAP,
        out_dir=stage0_dir, save_prefix=f"agmd_stage0_classical_{dataset_stem}",
        logger=logger, make_oof_plots=True, make_winners_rmse_plot=True,
        make_sweeps=True, fit_final_keras_histories=False, origin_col=ORIGIN_COL,
    )

    # --- Stage 1 ---
    base_common = {
        "model__model__activation": ["relu"],
        "model__model__optimizer": ["adam"],
        "model__model__learning_rate": [1e-3, 3e-4],
        "model__model__l2": [0.0, 1e-6, 1e-5],
        "model__model__hidden_layer_sizes": [
            (32,), (64,), (128,), (256,), (64, 32), (128, 64),
            (256, 128), (128, 64, 32), (256, 128, 64),
        ],
        "model__batch_size": [64],
        "model__epochs": [200],
        "model__fit__validation_split": [0.2],
        "model__fit__shuffle": [True],
        "model__fit__verbose": [0],
        "model__fit__callbacks": [[early_stop, reduce_lr, term_nan]],
    }

    keras_mlp_grid = [dict(
        **base_common,
        **{
            "model__model__n_features": [N_FEATURES_BASE],
            "model__model__n_targets": [N_TARGETS],
            "model__model__loss": ["huber"],
            "model__model__huber_delta": [1.0],
        },
    )]

    baseline_family = FamilySpec(
        name="KerasMLP_BaselineSearch", estimator=make_keras_mlp_estimator,
        search="random", param_grid=keras_mlp_grid, n_iter=10,
        x_mode="x", y_mode="true", complexity_fn=C_keras_mlp,
        n_jobs=1, scale_y=True,
        use_gap_filter=DEFAULT_USE_GAP_FILTER, gap_filter_tol=DEFAULT_GAP_FILTER_TOL,
        use_gap_tiebreak=DEFAULT_USE_GAP_TIEBREAK,
    )

    logger.info("=" * 72)
    logger.info("STAGE 1 - SEARCH BASELINE KerasMLP")
    stage1_dir = os.path.join(OUT_DIR, "stage1_baseline_search")
    baseline_result = run_model_selection_from_csv(
        csv_path=DATA_CSV_PATH, feature_cols=FEATURE_COLS, target_cols=TARGET_COLS,
        group_col=GROUP_COL, families=[baseline_family], n_splits=N_SPLITS,
        global_scoring=flux_rmse_scorer, model_map=MODEL_MAP,
        out_dir=stage1_dir, save_prefix=f"agmd_stage1_baseline_{dataset_stem}",
        logger=logger, make_oof_plots=True, make_winners_rmse_plot=True,
        make_sweeps=True, fit_final_keras_histories=True, origin_col=ORIGIN_COL,
    )

    baseline_winner = baseline_result["winners"][0]
    logger.info(f"Stage 1 baseline best params: {baseline_winner.best_params}")

    # --- Stage 2 grids ---
    frozen_baseline_grid = build_frozen_baseline_grid(baseline_winner.best_params, N_FEATURES_BASE, N_TARGETS)
    luc_restricted_grid = build_luc_restricted_grid_from_baseline(baseline_winner.best_params, N_FEATURES_BASE, N_TARGETS)
    hpd_restricted_grid = build_hpd_restricted_grid_from_baseline(baseline_winner.best_params, N_FEATURES_PLUS_PHY, N_TARGETS)
    residual_restricted_grid = build_residual_restricted_grid_from_baseline(baseline_winner.best_params, N_FEATURES_PLUS_PHY, N_TARGETS)
    hrnn_restricted_grid = build_hrnn_restricted_grid_from_baseline(baseline_winner.best_params, N_FEATURES_PLUS_PHY, N_TARGETS)

    luc_rmse_scorer = make_neg_rmse_single_true_target_scorer(
        target_index=FLUX_INDEX, n_true_targets=N_TARGETS, target_name="Flux",
    )

    stage2_families = [
        FamilySpec(name="KerasMLP_FrozenBaseline", estimator=make_keras_mlp_estimator,
                   search="random", param_grid=frozen_baseline_grid, n_iter=5,
                   x_mode="x", y_mode="true", complexity_fn=C_keras_mlp,
                   n_jobs=1, scale_y=True,
                   use_gap_filter=DEFAULT_USE_GAP_FILTER, gap_filter_tol=DEFAULT_GAP_FILTER_TOL,
                   use_gap_tiebreak=DEFAULT_USE_GAP_TIEBREAK),
        FamilySpec(name="KerasMLP_Luc_Restricted", estimator=make_keras_mlp_luc_estimator,
                   search="random", param_grid=luc_restricted_grid, n_iter=5,
                   x_mode="x", y_mode="true_plus_model", scoring=luc_rmse_scorer,
                   complexity_fn=C_keras_mlp, n_jobs=1, scale_y=True,
                   use_gap_filter=DEFAULT_USE_GAP_FILTER, gap_filter_tol=DEFAULT_GAP_FILTER_TOL,
                   use_gap_tiebreak=DEFAULT_USE_GAP_TIEBREAK),
        FamilySpec(name="KerasMLP_ZohanHPD_Restricted", estimator=make_keras_mlp_estimator,
                   search="random", param_grid=hpd_restricted_grid, n_iter=5,
                   x_mode="x_plus_model", y_mode="true", complexity_fn=C_keras_mlp,
                   n_jobs=1, scale_y=True,
                   use_gap_filter=DEFAULT_USE_GAP_FILTER, gap_filter_tol=DEFAULT_GAP_FILTER_TOL,
                   use_gap_tiebreak=DEFAULT_USE_GAP_TIEBREAK),
        FamilySpec(name="KerasMLP_ZohanResidual_Restricted", estimator=make_keras_mlp_residual_x_only_estimator,
                   search="random", param_grid=residual_restricted_grid, n_iter=5,
                   x_mode="x_plus_model", y_mode="true", complexity_fn=C_keras_mlp,
                   n_jobs=1, scale_y=True,
                   use_gap_filter=DEFAULT_USE_GAP_FILTER, gap_filter_tol=DEFAULT_GAP_FILTER_TOL,
                   use_gap_tiebreak=DEFAULT_USE_GAP_TIEBREAK),
        FamilySpec(name="KerasMLP_ZohanHRNN_Restricted", estimator=make_keras_mlp_hrnn_estimator,
                   search="random", param_grid=hrnn_restricted_grid, n_iter=5,
                   x_mode="x_plus_model", y_mode="true", complexity_fn=C_keras_mlp,
                   n_jobs=1, scale_y=True,
                   use_gap_filter=DEFAULT_USE_GAP_FILTER, gap_filter_tol=DEFAULT_GAP_FILTER_TOL,
                   use_gap_tiebreak=DEFAULT_USE_GAP_TIEBREAK),
    ]

    logger.info("=" * 72)
    logger.info("STAGE 2 - RESTRICTED HYBRID STRATEGY COMPARISON")
    stage2_dir = os.path.join(OUT_DIR, "stage2_restricted_hybrid_comparison")
    stage2_result = run_model_selection_from_csv(
        csv_path=DATA_CSV_PATH, feature_cols=FEATURE_COLS, target_cols=TARGET_COLS,
        group_col=GROUP_COL, families=stage2_families, n_splits=N_SPLITS,
        global_scoring=flux_rmse_scorer, model_map=MODEL_MAP,
        out_dir=stage2_dir, save_prefix=f"agmd_stage2_restricted_hybrids_{dataset_stem}",
        logger=logger, make_oof_plots=True, make_winners_rmse_plot=True,
        make_sweeps=True, fit_final_keras_histories=True, origin_col=ORIGIN_COL,
    )

    # --- Final aggregation ---
    all_winners = stage0_result["winners"] + baseline_result["winners"] + stage2_result["winners"]
    final_summary = winners_table(all_winners, target_cols=TARGET_COLS)

    final_dir = os.path.join(OUT_DIR, "final_comparison")
    save_outputs(
        out_dir=final_dir, winners=all_winners, summary=final_summary,
        target_cols=TARGET_COLS,
        X=stage0_result["X"], Y_true=stage0_result["Y_true"],
        groups=stage0_result["groups"], Y_model=stage0_result["Y_model"],
        cv=stage0_result["cv"],
        include_splits_in_cv_results=False,
        prefix=f"agmd_final_full_comparison_{dataset_stem}",
        make_plots=True, save_family_tables=True, logger=logger,
        make_oof_plots=True, make_winners_rmse_plot=True,
        make_sweeps=True, fit_final_keras_histories=True,
    )

    cols_show = [
        "family", "scoring", "rmse_cv_Flux", "rmse_train_Flux", "gap_rmse_Flux",
        "complexity", "r2_Flux", "rmse_Flux", "rmse_Alim_T_out", "rmse_Ref_T_out",
        "rank_by_score_space", "rank_by_flux_rmse",
    ]
    existing = [c for c in cols_show if c in final_summary.columns]

    print("\n=== FINAL SUMMARY ===", flush=True)
    print(final_summary[existing].to_string(index=False), flush=True)
    print(f"\nSaved outputs in: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

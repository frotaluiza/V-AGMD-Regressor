#!/usr/bin/env python3
"""Stage 2 only: run hybrid models using best baseline from focused Stage 1."""
import json, os, sys, logging
from pathlib import Path
from datetime import datetime
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SEED, N_SPLITS, FEATURE_COLS, TARGET_COLS, GROUP_COL, ORIGIN_COL
from config import N_FEATURES_BASE, N_FEATURES_PLUS_PHY, N_TARGETS, FLUX_INDEX, MODEL_MAP
from config import KERAS_CALLBACKS, DATA_PATH
from data import read_tabular_csv, build_XY_groups_with_model_map
from cv import make_cv, resolve_n_splits
from selection import make_neg_rmse_single_true_target_scorer, C_keras_mlp
from sweep import (build_frozen_baseline_grid, build_luc_restricted_grid_from_baseline,
                   build_hpd_restricted_grid_from_baseline,
                   build_residual_restricted_grid_from_baseline,
                   build_hrnn_restricted_grid_from_baseline)
from runner import FamilySpec, run_family, winners_table
from file_io import save_outputs
from consolidate import consolidate_results

def main():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_CSV_PATH = str(DATA_PATH)
    dataset_stem = Path(DATA_CSV_PATH).stem.replace(" ", "_")
    OUT_DIR = os.path.normpath(os.path.join(Path(__file__).resolve().parent.parent,
                                             f"stage2_only_{dataset_stem}_{timestamp}"))
    os.makedirs(OUT_DIR, exist_ok=True)
    logger = logging.getLogger("stage2")
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

    # Best params from focused Stage 1 (RMSE 0.0845)
    BEST_PARAMS_PATH = Path(__file__).resolve().parent.parent / "stage1_only_dados_att_com_var_com_phy_20260601_012129" / "stage1_baseline_search" / "agmd_stage1_baseline_dados_att_com_var_com_phy_best_params.json"
    best_params = json.loads(Path(BEST_PARAMS_PATH).read_text(encoding="utf-8"))["KerasMLP_BaselineSearch"]
    logger.info(f"Loaded baseline params: rmse_cv_Flux=0.0845")
    logger.info(f"  hidden={best_params['model__model__hidden_layer_sizes']}, lr={best_params['model__model__learning_rate']}, l2={best_params['model__model__l2']}, act={best_params['model__model__activation']}, batch={best_params['model__batch_size']}")

    flux_rmse_scorer = make_neg_rmse_single_true_target_scorer(FLUX_INDEX, N_TARGETS, "Flux")
    luc_rmse_scorer = make_neg_rmse_single_true_target_scorer(
        target_index=FLUX_INDEX, n_true_targets=N_TARGETS, target_name="Flux",
    )

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

    # Rebuild grid with proper callbacks (they can't be serialized in JSON)
    best_params["model__fit__callbacks"] = [early_stop, reduce_lr, term_nan]

    from models.keras_builders import (make_keras_mlp_estimator, make_keras_mlp_luc_estimator,
                                        make_keras_mlp_residual_x_only_estimator,
                                        make_keras_mlp_hrnn_estimator)

    frozen_baseline_grid = build_frozen_baseline_grid(best_params, N_FEATURES_BASE, N_TARGETS)
    luc_restricted_grid = build_luc_restricted_grid_from_baseline(best_params, N_FEATURES_BASE, N_TARGETS)
    hpd_restricted_grid = build_hpd_restricted_grid_from_baseline(best_params, N_FEATURES_PLUS_PHY, N_TARGETS)
    residual_restricted_grid = build_residual_restricted_grid_from_baseline(best_params, N_FEATURES_PLUS_PHY, N_TARGETS)

    stage2_families = [
        FamilySpec(name="KerasMLP_FrozenBaseline", estimator=make_keras_mlp_estimator,
                   search="random", param_grid=frozen_baseline_grid, n_iter=5,
                   x_mode="x", y_mode="true", complexity_fn=C_keras_mlp,
                   n_jobs=1, scale_y=True),
        FamilySpec(name="KerasMLP_Luc_Restricted", estimator=make_keras_mlp_luc_estimator,
                   search="random", param_grid=luc_restricted_grid, n_iter=5,
                   x_mode="x", y_mode="true_plus_model", scoring=luc_rmse_scorer,
                   complexity_fn=C_keras_mlp, n_jobs=1, scale_y=True),
        FamilySpec(name="KerasMLP_ZohanHPD_Restricted", estimator=make_keras_mlp_estimator,
                   search="random", param_grid=hpd_restricted_grid, n_iter=5,
                   x_mode="x_plus_model", y_mode="true", complexity_fn=C_keras_mlp,
                   n_jobs=1, scale_y=True),
        FamilySpec(name="KerasMLP_ZohanResidual_Restricted", estimator=make_keras_mlp_residual_x_only_estimator,
                   search="random", param_grid=residual_restricted_grid, n_iter=5,
                   x_mode="x_plus_model", y_mode="true", complexity_fn=C_keras_mlp,
                   n_jobs=1, scale_y=True),
    ]

    df = read_tabular_csv(DATA_CSV_PATH, decimal_comma=True, logger=logger)
    origin_col = ORIGIN_COL if ORIGIN_COL in df.columns else None
    X, Y_true, groups, used_df, Y_model, origin_values = build_XY_groups_with_model_map(
        df=df, feature_cols=FEATURE_COLS, target_cols=TARGET_COLS,
        group_col=GROUP_COL, dropna=True, model_map=MODEL_MAP,
        origin_col=origin_col, logger=logger,
    )
    n_splits = resolve_n_splits(groups, config_n_splits=N_SPLITS)
    cv, eval_mask = make_cv(n_splits, origin_values=origin_values, logger=logger)

    logger.info("=" * 72)
    logger.info("STAGE 2 - RESTRICTED HYBRID STRATEGY COMPARISON (random search)")
    stage2_dir = os.path.join(OUT_DIR, "stage2_restricted_hybrid_comparison")
    os.makedirs(stage2_dir, exist_ok=True)

    winners = []
    for spec in stage2_families:
        scorer = spec.scoring if spec.scoring is not None else flux_rmse_scorer
        logger.info(f"RUN FAMILY: {spec.name}")
        winners.append(run_family(spec=spec, X=X, Y_true=Y_true, Y_model=Y_model,
                                  groups=groups, cv=cv, scoring=scorer,
                                  eval_mask=eval_mask, logger=logger))

    summary = winners_table(winners, target_cols=TARGET_COLS)
    save_outputs(out_dir=stage2_dir, winners=winners, summary=summary,
                 target_cols=TARGET_COLS, X=X, Y_true=Y_true, groups=groups,
                 Y_model=Y_model, cv=cv, prefix=f"agmd_stage2_hybrids_{dataset_stem}",
                 make_plots=True, save_family_tables=True, logger=logger,
                 make_oof_plots=True, make_winners_rmse_plot=True,
                 make_sweeps=True, fit_final_keras_histories=False)

    print("\n=== STAGE 2 HYBRID RESULTS ===")
    for w in winners:
        print(f"  {w.family}: RMSE_Flux = {abs(w.best_score):.4f}")

    # Consolidate
    try:
        consol_dir = os.path.join(OUT_DIR, "consolidated_analysis")
        consolidate_results(base_dir=Path(OUT_DIR), out_dir=Path(consol_dir),
                           dataset_stem=dataset_stem)
    except Exception as e:
        logger.error(f"Consolidation failed: {e}")

    print(f"\nResults in: {OUT_DIR}")

if __name__ == "__main__":
    main()

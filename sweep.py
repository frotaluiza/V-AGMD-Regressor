from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.wrappers import YScalerRegressor
from selection import ProgressGridSearchCV, evaluate_oof_per_output
from config import N_FEATURES_BASE, N_FEATURES_PLUS_PHY, N_TARGETS, FLUX_INDEX, MODEL_MAP


def _safe_filename(s: str) -> str:
    keep = []
    for ch in str(s):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _dedup_keep_order(seq: Sequence[Any]) -> List[Any]:
    out = []
    seen = set()
    for x in seq:
        key = repr(x)
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def extract_baseline_keras_fixed_params(best_params: Dict[str, Any]) -> Dict[str, List[Any]]:
    keys_to_freeze = [
        "model__model__n_features",
        "model__model__n_targets",
        "model__model__hidden_layer_sizes",
        "model__model__activation",
        "model__model__optimizer",
        "model__model__learning_rate",
        "model__model__l2",
        "model__batch_size",
        "model__epochs",
        "model__fit__validation_split",
        "model__fit__shuffle",
        "model__fit__verbose",
        "model__fit__callbacks",
    ]

    optional_keys = [
        "model__model__loss",
        "model__model__huber_delta",
    ]

    frozen = {}
    for k in keys_to_freeze + optional_keys:
        if k in best_params:
            frozen[k] = [best_params[k]]

    return frozen


def build_frozen_baseline_grid(
    baseline_best_params: Dict[str, Any],
    n_features_base: int,
    n_targets: int,
) -> Dict[str, List[Any]]:
    frozen = extract_baseline_keras_fixed_params(baseline_best_params)
    frozen["model__model__n_features"] = [n_features_base]
    frozen["model__model__n_targets"] = [n_targets]
    return frozen


def _small_l2_grid_from_baseline(baseline_best_params: Dict[str, Any]) -> List[float]:
    baseline_l2 = float(baseline_best_params.get("model__model__l2", 0.0))
    cand = [baseline_l2, 0.0, 1e-6, 1e-5, 1e-4]
    cand = [float(x) for x in cand if float(x) >= 0.0]
    return _dedup_keep_order(cand)


def build_luc_restricted_grid_from_baseline(
    baseline_best_params: Dict[str, Any],
    n_features_base: int,
    n_targets: int,
) -> Dict[str, List[Any]]:
    frozen = extract_baseline_keras_fixed_params(baseline_best_params)
    frozen["model__model__n_features"] = [n_features_base]
    frozen["model__model__n_targets"] = [n_targets]

    baseline_loss = baseline_best_params.get("model__model__loss", "mse")
    baseline_huber_delta = baseline_best_params.get("model__model__huber_delta", 1.0)

    if "model__model__loss" in frozen:
        del frozen["model__model__loss"]

    frozen["model__model__data_loss"] = [baseline_loss]
    frozen["model__model__huber_delta"] = [baseline_huber_delta]
    frozen["model__model__physics_norm"] = ["mse", "mae"]
    frozen["model__model__omega"] = [0.0, 0.1, 0.3, 0.5, 0.7]
    return frozen


def build_hpd_restricted_grid_from_baseline(
    baseline_best_params: Dict[str, Any],
    n_features_plus_phy: int,
    n_targets: int,
) -> Dict[str, List[Any]]:
    frozen = extract_baseline_keras_fixed_params(baseline_best_params)
    frozen["model__model__n_features"] = [n_features_plus_phy]
    frozen["model__model__n_targets"] = [n_targets]
    frozen["model__model__l2"] = _small_l2_grid_from_baseline(baseline_best_params)
    return frozen


def build_residual_restricted_grid_from_baseline(
    baseline_best_params: Dict[str, Any],
    n_features_plus_phy: int,
    n_targets: int,
) -> Dict[str, List[Any]]:
    frozen = extract_baseline_keras_fixed_params(baseline_best_params)
    frozen["model__model__n_features"] = [n_features_plus_phy]
    frozen["model__model__n_targets"] = [n_targets]
    frozen["model__model__l2"] = _small_l2_grid_from_baseline(baseline_best_params)
    return frozen


def build_dropout_baseline_grid(
    baseline_best_params: Dict[str, Any],
    n_features_base: int,
    n_targets: int,
) -> Dict[str, List[Any]]:
    frozen = extract_baseline_keras_fixed_params(baseline_best_params)
    frozen["model__model__n_features"] = [n_features_base]
    frozen["model__model__n_targets"] = [n_targets]
    frozen["model__model__hidden_layer_sizes"] = [(64, 32)]
    frozen["model__model__dropout_rate"] = [0.0, 0.05, 0.10, 0.15, 0.20]
    frozen["model__model__l2"] = [0.0, 1e-6, 1e-5, 1e-4]
    return frozen


def build_residual_dropout_grid_from_baseline(
    baseline_best_params: Dict[str, Any],
    n_features_plus_phy: int,
    n_targets: int,
) -> Dict[str, List[Any]]:
    frozen = extract_baseline_keras_fixed_params(baseline_best_params)
    frozen["model__model__n_features"] = [n_features_plus_phy]
    frozen["model__model__n_targets"] = [n_targets]
    frozen["model__model__hidden_layer_sizes"] = [(64, 32)]
    frozen["model__model__dropout_rate"] = [0.0, 0.05, 0.10, 0.15, 0.20]
    frozen["model__model__l2"] = [0.0, 1e-6, 1e-5, 1e-4]
    return frozen


def build_hrnn_restricted_grid_from_baseline(
    baseline_best_params: Dict[str, Any],
    n_features_plus_phy: int,
    n_targets: int,
) -> Dict[str, List[Any]]:
    frozen = extract_baseline_keras_fixed_params(baseline_best_params)
    frozen["model__model__n_features"] = [n_features_plus_phy]
    frozen["model__model__n_targets"] = [n_targets]
    frozen["model__model__l2"] = _small_l2_grid_from_baseline(baseline_best_params)
    return frozen


def _sweep_key_and_values_from_winner(winner) -> Tuple[Optional[str], Optional[List[Any]]]:
    name = winner.family
    p = winner.best_params

    sweep_map = {
        "Ridge": ("model__alpha", [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]),
        "Lasso_MultiTask": ("model__alpha", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]),
        "ElasticNet_MultiTask": ("model__alpha", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]),
        "Lasso_Indep(MultiOutputRegressor)": ("model__estimator__alpha", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]),
        "ElasticNet_Indep(MultiOutputRegressor)": ("model__estimator__alpha", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]),
        "DT": ("model__ccp_alpha", [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
        "RF": ("model__max_depth", [2, 3, 4, 5, 6, 8, 10]),
        "GB": ("model__estimator__max_depth", [1, 2, 3, 4, 5]),
        "MLP_sklearn": ("model__alpha", [0.0, 1e-6, 1e-5, 1e-4, 1e-3]),
    }

    if name in sweep_map:
        return sweep_map[name]

    keras_l2_names = [
        "KerasMLP_BaselineSearch",
        "KerasMLP_FrozenBaseline",
        "KerasMLP_ZohanHPD_Restricted",
        "KerasMLP_ZohanResidual_Restricted",
        "KerasMLP_ZohanHRNN_Restricted",
    ]

    if name in keras_l2_names:
        base = float(p.get("model__model__l2", 0.0))
        grid = sorted({0.0, 1e-6, 1e-5, 1e-4, 1e-3, base})
        return "model__model__l2", grid

    if name == "KerasMLP_Luc_Restricted":
        return "model__model__omega", [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    return None, None


def run_sweep_from_winner(
    winner,
    X: np.ndarray,
    Y_true: np.ndarray,
    groups: np.ndarray,
    cv,
    *,
    Y_model=None,
    logger=None,
) -> Optional[pd.DataFrame]:
    sweep_key, sweep_values = _sweep_key_and_values_from_winner(winner)
    if sweep_key is None or sweep_values is None or len(sweep_values) == 0:
        return None

    spec = winner.spec
    scoring = winner.scoring_used

    from runner import _select_XY_for_spec, _resolve_estimator, _make_scaled_pipeline

    X_fit, Y_fit = _select_XY_for_spec(spec, X, Y_true, Y_model)
    m_true = int(Y_true.shape[1])
    augmented_2m = (spec.y_mode == "true_plus_model")

    base_est = _resolve_estimator(spec, n_targets=m_true)
    estimator = _make_scaled_pipeline(
        base_estimator=base_est,
        scale_y=bool(spec.scale_y),
        augmented_2m=bool(augmented_2m),
        m_targets=m_true,
    )

    fixed_grid = {k: [v] for k, v in dict(winner.best_params).items()}
    fixed_grid[sweep_key] = list(sweep_values)

    if logger:
        logger.info(f"[{winner.family}] Sweep on key={sweep_key} with {len(sweep_values)} values")

    sweep_search = ProgressGridSearchCV(
        estimator=estimator,
        param_grid=fixed_grid,
        scoring=scoring,
        cv=cv,
        refit=False,
        return_train_score=True,
        n_jobs=1,
        verbose=0,
        error_score="raise",
        progress_prefix=f"[{winner.family}][SWEEP]",
        progress_every_candidates=1,
        progress_chunk_size=1,
        logger=logger,
    )
    sweep_search.fit(X_fit, Y_fit, groups=groups)

    df = pd.DataFrame(sweep_search.cv_results_).copy()
    df["sweep_key"] = sweep_key
    df["sweep_value"] = [p.get(sweep_key, np.nan) for p in df["params"]]
    df["metric_test_natural"] = -df["mean_test_score"].astype(float)
    if "mean_train_score" in df.columns:
        df["metric_train_natural"] = -df["mean_train_score"].astype(float)
        df["gap_natural_cv"] = df["metric_test_natural"] - df["metric_train_natural"]

    best_idx = int(np.argmin(np.asarray(df["metric_test_natural"], dtype=float)))
    df["best_raw_by_score"] = False
    df.loc[best_idx, "best_raw_by_score"] = True

    return df

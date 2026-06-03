import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.wrappers import YScalerRegressor
from selection import (
    ProgressGridSearchCV,
    ProgressRandomizedSearchCV,
    make_refit_1se_gapfilter_min_complexity_then_min_gap,
    make_neg_rmse_single_true_target_scorer,
    evaluate_oof_per_output,
    COMPLEXITY_MAP,
    C_keras_mlp,
)
from config import KERAS_CALLBACKS


EstimatorLike = Any
EstimatorFactory = Callable[[int], EstimatorLike]


@dataclass
class FamilySpec:
    name: str
    estimator: Union[EstimatorLike, EstimatorFactory]
    search: str = "grid"
    param_grid: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    n_iter: int = 10

    complexity_fn: Callable[[Dict[str, Any]], float] = C_keras_mlp
    scoring: Optional[Union[str, Callable]] = None
    n_jobs: int = 1
    verbose: int = 0
    force_multioutput_wrapper: bool = False

    y_mode: str = "true"
    x_mode: str = "x"
    scale_y: bool = False

    progress_every_candidates: int = 1
    progress_chunk_size: int = 1

    use_gap_tiebreak: bool = False
    use_gap_filter: bool = False
    gap_filter_tol: float = 0.02


@dataclass
class FamilyWinner:
    family: str
    best_index: int
    best_params: Dict[str, Any]
    best_score: float
    best_std_test: float
    best_mean_train: Optional[float]
    best_gap: Optional[float]
    complexity: float
    scoring_used: Union[str, Callable]
    complexity_fn: Callable[[Dict[str, Any]], float]
    grid: Any
    spec: FamilySpec
    family_table: Optional[pd.DataFrame] = None
    r2_per_output: Optional[List[float]] = None
    rmse_per_output: Optional[List[float]] = None
    oof_y_true: Optional[np.ndarray] = None
    oof_y_pred: Optional[np.ndarray] = None
    X_fit_used: Optional[np.ndarray] = None
    Y_fit_used: Optional[np.ndarray] = None
    history_df: Optional[pd.DataFrame] = None


def _scoring_name(scoring) -> str:
    return scoring if isinstance(scoring, str) else getattr(scoring, "__name__", "callable_scoring")


def score_to_natural(score_mean, scoring):
    if score_mean is None:
        return None
    if isinstance(scoring, str) and scoring.startswith("neg_"):
        return -float(score_mean)
    return -float(score_mean)


def _extract_split_columns(cv_results):
    split_test_cols = sorted([k for k in cv_results.keys() if k.startswith("split") and k.endswith("_test_score")])
    split_train_cols = sorted([k for k in cv_results.keys() if k.startswith("split") and k.endswith("_train_score")])
    return split_test_cols, split_train_cols


def _select_XY_for_spec(spec: FamilySpec, X, Y_true, Y_model):
    if spec.x_mode == "x":
        X_fit = X
    elif spec.x_mode == "x_plus_model":
        if Y_model is None:
            raise ValueError(f"Family '{spec.name}' requires Y_model (x_plus_model).")
        X_fit = np.concatenate([X, Y_model], axis=1)
    else:
        raise ValueError(f"Unknown x_mode='{spec.x_mode}'")

    if spec.y_mode == "true":
        Y_fit = Y_true
    elif spec.y_mode == "true_plus_model":
        if Y_model is None:
            raise ValueError(f"Family '{spec.name}' requires Y_model (true_plus_model).")
        Y_fit = np.concatenate([Y_true, Y_model], axis=1)
    else:
        raise ValueError(f"Unknown y_mode='{spec.y_mode}'")

    return X_fit, Y_fit


def _resolve_estimator(spec: FamilySpec, n_targets: int):
    est = spec.estimator(n_targets) if callable(spec.estimator) else clone(spec.estimator)
    if spec.force_multioutput_wrapper and n_targets > 1:
        est = MultiOutputRegressor(est)
    return est


def _make_scaled_pipeline(base_estimator, *, scale_y, augmented_2m, m_targets):
    model_step = base_estimator
    if scale_y:
        model_step = YScalerRegressor(
            base_estimator=base_estimator,
            scale_y=True,
            augmented_2m=bool(augmented_2m),
            m_targets=int(m_targets),
        )

    return Pipeline(steps=[("scaler", StandardScaler()), ("model", model_step)])


def family_compact_table(family_name, cv_results, scoring, complexity_fn, selected_index):
    df = pd.DataFrame(cv_results).copy()
    df["complexity"] = [float(complexity_fn(p)) for p in df["params"]]

    df["model_type"] = "Regular"
    if 0 <= selected_index < len(df):
        df.loc[selected_index, "model_type"] = "Selected"

    best_idx = int(np.argmax(np.asarray(df["mean_test_score"], dtype=float)))
    df["best_raw_by_score"] = False
    df.loc[best_idx, "best_raw_by_score"] = True

    df["selected_by_refit"] = False
    if 0 <= selected_index < len(df):
        df.loc[selected_index, "selected_by_refit"] = True

    df["selection_target"] = "Flux"
    df["selection_metric"] = "RMSE"
    df["rmse_cv_Flux"] = -df["mean_test_score"].astype(float)
    df["std_rmse_cv_Flux"] = df["std_test_score"].astype(float)
    if "mean_train_score" in df:
        df["rmse_train_Flux"] = -df["mean_train_score"].astype(float)
        df["gap_rmse_Flux_cv"] = df["rmse_cv_Flux"] - df["rmse_train_Flux"]
    if "std_train_score" in df:
        df["std_rmse_train_Flux"] = df["std_train_score"].astype(float)

    wanted = [
        "selection_target", "selection_metric",
        "complexity", "mean_test_score", "std_test_score",
        "mean_train_score", "std_train_score",
        "rank_test_score", "mean_fit_time",
        "model_type", "best_raw_by_score", "selected_by_refit",
        "params", "rmse_cv_Flux", "std_rmse_cv_Flux",
        "rmse_train_Flux", "std_rmse_train_Flux", "gap_rmse_Flux_cv",
    ]
    cols = [c for c in wanted if c in df.columns]
    out = df[cols].copy()
    out = out.sort_values("complexity", ascending=True).reset_index(drop=True)
    out.insert(0, "family", family_name)
    out.insert(1, "scoring", _scoring_name(scoring))
    return out


def _oof_direct(
    spec, X_fit, Y_fit, groups, cv, m_true, augmented_2m, eval_mask, params_i, logger,
):
    """OOF evaluation building fresh pipelines per fold (bypasses scikeras clone bug)."""
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from models.wrappers import YScalerRegressor

    Y_fit_arr = np.asarray(Y_fit)
    n = Y_fit_arr.shape[0]
    mfit = Y_fit_arr.shape[1]
    y_pred_oof = np.full((n, mfit), np.nan, dtype=float)

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, min_delta=1e-4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_delta=1e-4, cooldown=0, min_lr=1e-6, verbose=0),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    for fold, (tr, te) in enumerate(cv.split(X_fit, Y_fit_arr, groups=groups), start=1):
        tf.keras.backend.clear_session()
        # Fresh callbacks per fold to avoid stale state (sklearn clone cannot deep-copy TF callbacks)
        fold_cbs = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, min_delta=1e-4, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_delta=1e-4, cooldown=0, min_lr=1e-6, verbose=0),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
        base_est = spec.estimator(n_targets=m_true)
        if spec.scale_y:
            model_step = YScalerRegressor(base_estimator=base_est, scale_y=True,
                                          augmented_2m=bool(augmented_2m), m_targets=int(m_true))
        else:
            model_step = base_est
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model_step)])
        to_set = {}
        for k, v in params_i.items():
            if "callbacks" in k:
                continue
            to_set[k] = v
        to_set["model__fit__callbacks"] = fold_cbs
        pipe.set_params(**to_set)
        pipe.fit(X_fit[tr], Y_fit_arr[tr])
        pred = np.asarray(pipe.predict(X_fit[te]))
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if pred.shape[1] > mfit:
            pred = pred[:, :mfit]
        y_pred_oof[te] = pred

    if eval_mask is None or (isinstance(eval_mask, np.ndarray) and len(eval_mask) == 1):
        eval_mask_arr = np.ones(n, dtype=bool)
    else:
        eval_mask_arr = np.asarray(eval_mask, dtype=bool)
    y_true = Y_fit_arr[eval_mask_arr, :m_true]
    y_pred = y_pred_oof[eval_mask_arr, :m_true]

    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_true, y_pred, multioutput="raw_values").astype(float).tolist()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="raw_values")).astype(float).tolist()
    return r2, rmse, y_true, y_pred


def _make_fresh_callbacks():
    import tensorflow as tf
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=KERAS_CALLBACKS["early_stop_patience"],
            min_delta=KERAS_CALLBACKS["early_stop_min_delta"], restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=KERAS_CALLBACKS["reduce_lr_factor"],
            patience=KERAS_CALLBACKS["reduce_lr_patience"],
            min_delta=KERAS_CALLBACKS["early_stop_min_delta"],
            cooldown=0, min_lr=KERAS_CALLBACKS["reduce_lr_min_lr"], verbose=0,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def _filter_callbacks_from_grid(grid):
    if isinstance(grid, dict):
        return {k: v for k, v in grid.items() if "callbacks" not in k}
    elif isinstance(grid, list):
        return [{k: v for k, v in g.items() if "callbacks" not in k} for g in grid]
    return grid


def _filter_callbacks(params):
    return {k: v for k, v in params.items() if "callbacks" not in k}


def run_family(
    spec: FamilySpec,
    X: np.ndarray,
    Y_true: np.ndarray,
    groups: np.ndarray,
    cv,
    scoring: Union[str, Callable],
    Y_model=None,
    eval_mask=None,
    logger=None,
) -> FamilyWinner:
    refit_rule = make_refit_1se_gapfilter_min_complexity_then_min_gap(
        complexity_fn=spec.complexity_fn,
        use_gap_tiebreak=spec.use_gap_tiebreak,
        use_gap_filter=spec.use_gap_filter,
        gap_filter_tol=spec.gap_filter_tol,
    )

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

    if logger:
        logger.info(
            f"[{spec.name}] Search={spec.search} | n_splits={cv.get_n_splits()} | scoring={_scoring_name(scoring)} | scale_y={spec.scale_y}"
        )
        if spec.search == "grid":
            ncomb = len(list(ParameterGrid(spec.param_grid or {})))
            logger.info(f"[{spec.name}] Grid combos ~{ncomb}")
        elif spec.search == "random":
            logger.info(f"[{spec.name}] Random search n_iter={spec.n_iter}")
        if spec.use_gap_filter:
            logger.info(f"[{spec.name}] Gap policy | use_gap_filter={spec.use_gap_filter} | gap_filter_tol={spec.gap_filter_tol}")

    if spec.param_grid is None:
        raise ValueError(f"Family '{spec.name}' requires param_grid.")

    # Ensure fresh callbacks per family (sklearn clone cannot deep-copy TF callbacks)
    try:
        estimator.set_params(model__fit__callbacks=_make_fresh_callbacks())
    except ValueError:
        pass  # non-Keras models don't have model__fit__callbacks
    clean_grid = _filter_callbacks_from_grid(spec.param_grid)

    if spec.search == "random":
        searchcv = ProgressRandomizedSearchCV(
            estimator=estimator,
            param_distributions=clean_grid,
            n_iter=spec.n_iter,
            scoring=scoring,
            cv=cv,
            refit=False,
            return_train_score=True,
            n_jobs=spec.n_jobs,
            verbose=0,
            error_score="raise",
            progress_prefix=f"[{spec.name}]",
            progress_every_candidates=1,
            logger=logger,
        )
    else:
        searchcv = ProgressGridSearchCV(
        estimator=estimator,
        param_grid=clean_grid,
        scoring=scoring,
        cv=cv,
        refit=False,
        return_train_score=True,
        n_jobs=spec.n_jobs,
        verbose=0,
        error_score="raise",
        progress_prefix=f"[{spec.name}]",
        progress_every_candidates=spec.progress_every_candidates,
        progress_chunk_size=spec.progress_chunk_size,
        logger=logger,
    )

    t0 = time.time()
    searchcv.fit(X_fit, Y_fit, groups=groups)
    t1 = time.time()

    cvres = searchcv.cv_results_
    i = int(refit_rule(cvres)) if callable(refit_rule) else int(searchcv.best_index_)
    params_i = cvres["params"][i]

    mean_test = float(cvres["mean_test_score"][i])
    std_test = float(cvres["std_test_score"][i])
    mean_train = float(cvres["mean_train_score"][i]) if "mean_train_score" in cvres else None
    gap = (mean_train - mean_test) if (mean_train is not None) else None
    comp = float(spec.complexity_fn(params_i))

    fam_tbl = family_compact_table(family_name=spec.name, cv_results=cvres, scoring=scoring, complexity_fn=spec.complexity_fn, selected_index=i)

    winner = FamilyWinner(
        family=spec.name,
        best_index=i,
        best_params=params_i,
        best_score=mean_test,
        best_std_test=std_test,
        best_mean_train=mean_train,
        best_gap=gap,
        complexity=comp,
        scoring_used=scoring,
        complexity_fn=spec.complexity_fn,
        grid=searchcv,
        spec=spec,
        family_table=fam_tbl,
    )

    winner.X_fit_used = X_fit
    winner.Y_fit_used = Y_fit

    if logger:
        logger.info(f"[{spec.name}] DONE in {t1 - t0:.1f}s | best_score={mean_test:.6f} | complexity={comp:.3f} | gap={gap}")

    # OOF via clone of fresh pipeline (sklearn clone + evaluate_oof_per_output works correctly)
    params_i_clean = _filter_callbacks(params_i)

    if logger:
        logger.info(f"[{spec.name}] OOF per-output evaluation (fresh pipeline + clone per fold)")
    try:
        fresh_base = _resolve_estimator(spec, n_targets=m_true)
        fresh_est = _make_scaled_pipeline(
            base_estimator=fresh_base,
            scale_y=bool(spec.scale_y),
            augmented_2m=bool(augmented_2m),
            m_targets=m_true,
        )
        fresh_est.set_params(**params_i_clean)
        try:
            fresh_est.set_params(model__fit__callbacks=_make_fresh_callbacks())
        except ValueError:
            pass
        r2_list, rmse_list, y_true_oof, y_pred_oof = evaluate_oof_per_output(
            fresh_est, X_fit, Y_fit, groups, cv,
            m_true=m_true, augmented_2m=augmented_2m,
            eval_mask=eval_mask, logger=logger,
        )
        winner.r2_per_output = r2_list
        winner.rmse_per_output = rmse_list
        winner.oof_y_true = y_true_oof
        winner.oof_y_pred = y_pred_oof
        if logger:
            logger.info(f"[{spec.name}] OOF succeeded")
    except Exception as e:
        if logger:
            logger.warning(f"[{spec.name}] OOF failed (no fallback): {repr(e)}")

    # Manual refit on full data
    if logger:
        logger.info(f"[{spec.name}] Manual refit on full data")
    try:
        refit_base = _resolve_estimator(spec, n_targets=m_true)
        refit_est = _make_scaled_pipeline(
            base_estimator=refit_base,
            scale_y=bool(spec.scale_y),
            augmented_2m=bool(augmented_2m),
            m_targets=m_true,
        )
        refit_est.set_params(**params_i_clean)
        try:
            refit_est.set_params(model__fit__callbacks=_make_fresh_callbacks())
        except ValueError:
            pass
        refit_est.fit(X_fit, Y_fit)
        searchcv.best_estimator_ = refit_est
        searchcv.best_index_ = i
        searchcv.best_params_ = params_i
    except Exception as e:
        if logger:
            logger.warning(f"[{spec.name}] Manual refit failed: {repr(e)}")

    return winner


def winners_table(winners: List[FamilyWinner], target_cols) -> pd.DataFrame:
    rows = []
    for w in winners:
        sname = _scoring_name(w.scoring_used)

        cvres = w.grid.cv_results_
        best_idx = int(np.argmax(cvres["mean_test_score"]))
        best_overall = float(cvres["mean_test_score"][best_idx])
        thr_1se = float(cvres["mean_test_score"][best_idx] - cvres["std_test_score"][best_idx])

        selected_score = float(w.best_score)
        within_1se = bool(selected_score >= thr_1se)
        margin = float(selected_score - thr_1se)

        test_nat = score_to_natural(w.best_score, w.scoring_used)
        train_nat = score_to_natural(w.best_mean_train, w.scoring_used)
        gap_nat = float(test_nat - train_nat) if (train_nat is not None and test_nat is not None) else None

        r2_dict = {}
        rmse_dict = {}
        if w.r2_per_output is not None:
            for j, col in enumerate(target_cols):
                if j < len(w.r2_per_output):
                    r2_dict[f"r2_{col}"] = float(w.r2_per_output[j])
        if w.rmse_per_output is not None:
            for j, col in enumerate(target_cols):
                if j < len(w.rmse_per_output):
                    rmse_dict[f"rmse_{col}"] = float(w.rmse_per_output[j])

        rows.append(dict(
            family=w.family,
            scoring=sname,
            selection_target="Flux",
            selection_metric="RMSE",
            selection_rule="1SE_mincomplexity_on_Flux",
            best_overall_mean_test_score=best_overall,
            threshold_1se_score_space=thr_1se,
            selected_is_within_1se=within_1se,
            selected_score_minus_threshold=margin,
            best_score_mean_test=w.best_score,
            best_std_test=w.best_std_test,
            best_mean_train=w.best_mean_train,
            gap_train_minus_test_score_space=w.best_gap,
            metric_test_natural=test_nat,
            metric_train_natural=train_nat,
            gap_natural=gap_nat,
            complexity=w.complexity,
            best_params=w.best_params,
            gap_filter_tol=w.spec.gap_filter_tol,
            use_gap_filter=w.spec.use_gap_filter,
            **r2_dict,
            **rmse_dict,
        ))

    df = pd.DataFrame(rows)

    if "metric_test_natural" in df.columns:
        df = df.rename(columns={"metric_test_natural": "rmse_cv_Flux"})
    if "metric_train_natural" in df.columns:
        df = df.rename(columns={"metric_train_natural": "rmse_train_Flux"})
    if "gap_natural" in df.columns:
        df = df.rename(columns={"gap_natural": "gap_rmse_Flux"})

    df = df.sort_values("best_score_mean_test", ascending=False).reset_index(drop=True)
    df["rank_by_score_space"] = np.arange(1, len(df) + 1)

    df = df.sort_values("rmse_cv_Flux", ascending=True).reset_index(drop=True)
    df["rank_by_flux_rmse"] = np.arange(1, len(df) + 1)

    return df


def cv_results_to_frame(cv_results, include_splits=False):
    df = pd.DataFrame(cv_results).copy()
    df["best_raw_by_score"] = False
    best_idx = int(np.argmax(np.asarray(df["mean_test_score"], dtype=float)))
    df.loc[best_idx, "best_raw_by_score"] = True

    if "mean_train_score" in df.columns:
        df["metric_test_natural"] = -df["mean_test_score"].astype(float)
        df["metric_train_natural"] = -df["mean_train_score"].astype(float)
        df["gap_natural_cv"] = df["metric_test_natural"] - df["metric_train_natural"]

    if not include_splits:
        drop_cols = [c for c in df.columns if c.startswith("split") and c.endswith("_test_score")]
        drop_cols += [c for c in df.columns if c.startswith("split") and c.endswith("_train_score")]
        if drop_cols:
            df = df.drop(columns=drop_cols)
    return df


def _winner_is_keras(w: FamilyWinner) -> bool:
    try:
        pipe = w.grid.best_estimator_
        model_step = pipe.named_steps["model"]
        if isinstance(model_step, YScalerRegressor):
            inner = model_step.base_estimator_
        else:
            inner = model_step
        from scikeras.wrappers import KerasRegressor
        from models.keras_builders import KERAS_AVAILABLE
        return KERAS_AVAILABLE and isinstance(inner, KerasRegressor)
    except Exception:
        return False


def fit_keras_history_for_winner(winner: FamilyWinner, logger=None):
    if not _winner_is_keras(winner):
        return None
    if winner.X_fit_used is None or winner.Y_fit_used is None:
        return None

    try:
        est = clone(winner.grid.best_estimator_)
        est.fit(winner.X_fit_used, winner.Y_fit_used)

        model_step = est.named_steps["model"]
        if isinstance(model_step, YScalerRegressor):
            inner = model_step.base_estimator_
        else:
            inner = model_step

        hist = getattr(inner, "history_", None)
        if hist is None:
            return None

        hist_df = pd.DataFrame(hist).copy()
        hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
        winner.history_df = hist_df

        if logger:
            logger.info(f"[{winner.family}] Stored final Keras history with {len(hist_df)} epochs.")
        return hist_df
    except Exception as e:
        if logger:
            logger.warning(f"[{winner.family}] Could not fit final Keras history: {repr(e)}")
        return None

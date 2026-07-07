import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import logging
import time
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV


# ---- Helpers ----
def _get(params: Dict[str, Any], *keys: str, default=None):
    for k in keys:
        if k in params:
            return params[k]
    return default


# ---- Refit rule: 1-SE + gap filter + min complexity + min gap ----
def lower_bound_1se(cv_results: Dict[str, Any]) -> float:
    best_idx = int(np.argmax(cv_results["mean_test_score"]))
    return float(cv_results["mean_test_score"][best_idx] - cv_results["std_test_score"][best_idx])


def gap_from_cv_results(cv_results: Dict[str, Any], i: int) -> float:
    return float(cv_results["mean_train_score"][i] - cv_results["mean_test_score"][i])


def make_refit_1se_gapfilter_min_complexity_then_min_gap(
    complexity_fn: Callable[[Dict[str, Any]], float],
    use_gap_tiebreak: bool = True,
    use_gap_filter: bool = True,
    gap_filter_tol: float = 0.02,
) -> Callable[[Dict[str, Any]], int]:
    def _refit(cv_results: Dict[str, Any]) -> int:
        thr = lower_bound_1se(cv_results)
        cand = np.flatnonzero(cv_results["mean_test_score"] >= thr)

        if len(cand) == 0:
            return int(np.argmax(cv_results["mean_test_score"]))

        if use_gap_filter and "mean_train_score" in cv_results and len(cand) > 1:
            G = np.array([gap_from_cv_results(cv_results, int(i)) for i in cand], dtype=float)
            minG = float(np.min(G))
            keep = G <= (minG + float(gap_filter_tol))
            cand_gap = cand[keep]
            if len(cand_gap) > 0:
                cand = cand_gap

        C = np.array([complexity_fn(cv_results["params"][i]) for i in cand], dtype=float)
        minC = float(np.min(C))
        cand2 = cand[C == minC]

        if use_gap_tiebreak and len(cand2) > 1 and "mean_train_score" in cv_results:
            G2 = np.array([gap_from_cv_results(cv_results, int(i)) for i in cand2], dtype=float)
            minG2 = float(np.min(G2))
            cand2 = cand2[G2 == minG2]

        best_local = cand2[np.argmax(cv_results["mean_test_score"][cand2])]
        return int(best_local)

    return _refit


# ---- Complexity functions ----
# Baseiam-se no princípio de que complexidade cresce com a capacidade do modelo
# (Hastie et al., 2009, Cap. 7): mais regularizacao -> menos complexo;
# maior profundidade / mais neuronios / mais estimadores -> mais complexo.

def _neg_log10_alpha(alpha, floor=1e-20):
    """Quanto maior alpha (mais regularizacao), menor a complexidade."""
    a = max(float(alpha), floor)
    return -math.log10(a)


def C_ols(params: Dict[str, Any]) -> float:
    return 0.0


def C_ridge(params: Dict[str, Any]) -> float:
    alpha = _get(params, "model__alpha", default=1.0)
    return _neg_log10_alpha(alpha)


def C_lasso(params: Dict[str, Any]) -> float:
    alpha = _get(params, "model__alpha", default=1.0)
    return _neg_log10_alpha(alpha)


def C_elasticnet(params: Dict[str, Any]) -> float:
    alpha = _get(params, "model__alpha", default=1.0)
    return _neg_log10_alpha(alpha)


def C_lasso_mo(params: Dict[str, Any]) -> float:
    alpha = _get(params, "model__estimator__alpha", default=1.0)
    return _neg_log10_alpha(alpha)


def C_elasticnet_mo(params: Dict[str, Any]) -> float:
    alpha = _get(params, "model__estimator__alpha", default=1.0)
    return _neg_log10_alpha(alpha)


def C_dt(params: Dict[str, Any]) -> float:
    """Complexidade = profundidade maxima. None = arvore sem limite."""
    max_depth = _get(params, "model__max_depth", default=None)
    if max_depth is None:
        return 1e6
    return float(max_depth)


def C_rf(params: Dict[str, Any]) -> float:
    """Complexidade = profundidade + contribuicao do numero de arvores."""
    n_estimators = float(_get(params, "model__n_estimators", default=100))
    max_depth = _get(params, "model__max_depth", default=None)
    d = 1e3 if max_depth is None else float(max_depth)
    return d + float(n_estimators) * 1e-3


def C_gb(params: Dict[str, Any]) -> float:
    """Complexidade = profundidade + contribuicao do numero de arvores."""
    n_estimators = float(_get(params, "model__estimator__n_estimators", default=100))
    max_depth = _get(params, "model__estimator__max_depth", default=None)
    d = 1e3 if max_depth is None else float(max_depth)
    return d + n_estimators * 1e-3


def C_mlp_proxy(params: Dict[str, Any]) -> float:
    """Complexidade = numero total de neuronios + pequena penalidade por camada."""
    hls = _get(params, "model__hidden_layer_sizes", default=None)
    if hls is None:
        return 1e9
    layers = (hls,) if isinstance(hls, int) else tuple(hls)
    return float(sum(layers) + 0.1 * len(layers))


def C_keras_mlp(params: Dict[str, Any]) -> float:
    """Complexidade = total de neuronios + penalidade por camada - log10(l2).
    
    Combina capacidade da arquitetura com regularizacao L2: para uma mesma
    arquitetura, maior L2 resulta em menor complexidade (modelo mais simples).
    """
    hls = _get(params, "model__model__hidden_layer_sizes", default=(64, 32))
    if hls is None:
        return 1e9
    layers = (hls,) if isinstance(hls, int) else tuple(hls)
    n_neurons = float(sum(int(u) for u in layers) + 0.1 * len(layers))
    l2 = float(_get(params, "model__model__l2", default=0.0))
    return n_neurons - math.log10(max(l2, 1e-20))


COMPLEXITY_MAP = {
    "OLS": C_ols,
    "Ridge": C_ridge,
    "Lasso_MultiTask": C_lasso,
    "ElasticNet_MultiTask": C_elasticnet,
    "Lasso_Indep(MultiOutputRegressor)": C_lasso_mo,
    "ElasticNet_Indep(MultiOutputRegressor)": C_elasticnet_mo,
    "DT": C_dt,
    "RF": C_rf,
    "GB": C_gb,
    "MLP_sklearn": C_mlp_proxy,
}


# ---- Progress GridSearchCV ----
class ProgressGridSearchCV(GridSearchCV):
    def __init__(
        self,
        estimator,
        param_grid,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score="raise",
        return_train_score=False,
        progress_prefix: str = "[GridSearch]",
        progress_every_candidates: int = 1,
        progress_chunk_size: int = 1,
        logger=None,
    ):
        super().__init__(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.progress_prefix = str(progress_prefix)
        self.progress_every_candidates = int(progress_every_candidates)
        self.progress_chunk_size = int(progress_chunk_size)
        self.logger = logger

    def _logp(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg, flush=True)

    def _run_search(self, evaluate_candidates):
        grid = list(ParameterGrid(self.param_grid))
        total = len(grid)
        chunk = max(1, int(self.progress_chunk_size))
        every = max(1, int(self.progress_every_candidates))

        self._logp(f"{self.progress_prefix} TOTAL: candidates={total} | chunk={chunk}")

        i = 0
        while i < total:
            j = min(i + chunk, total)
            if ((i + 1) == 1) or ((i + 1) % every == 0) or (j == total):
                self._logp(f"{self.progress_prefix} combos {i + 1}/{total}")
            evaluate_candidates(grid[i:j])
            i = j


class ProgressRandomizedSearchCV(RandomizedSearchCV):
    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score="raise",
        return_train_score=False,
        progress_prefix: str = "[RandomSearch]",
        progress_every_candidates: int = 1,
        logger=None,
    ):
        super().__init__(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.progress_prefix = str(progress_prefix)
        self.progress_every_candidates = int(progress_every_candidates)
        self.logger = logger

    def _logp(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg, flush=True)

    def _run_search(self, evaluate_candidates):
        total = int(self.n_iter)
        every = max(1, int(self.progress_every_candidates))
        self._logp(f"{self.progress_prefix} TOTAL: n_iter={total}")

        class _Sampler:
            def __init__(self, parent, total, every):
                self.parent = parent
                self.total = total
                self.every = every
                self.count = 0

            def __call__(self, candidates):
                self.count += 1
                if (self.count == 1) or (self.count % self.every == 0) or (self.count == self.total):
                    self.parent._logp(f"{self.parent.progress_prefix} iter {self.count}/{self.total}")
                evaluate_candidates(candidates)

        sampler = _Sampler(self, total, every)
        super()._run_search(sampler)


# ---- Scorers ----
def make_neg_rmse_mean_all_targets_scorer(
    n_true_targets: int,
):
    def _scorer(estimator, X, y):
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y_pred = estimator.predict(X)
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        m = int(n_true_targets)
        rmse_per = np.array([
            float(np.sqrt(np.mean((y[:len(y_pred), k] - y_pred[:len(y_pred), k]) ** 2)))
            for k in range(m)
        ])
        return -float(np.mean(rmse_per))
    _scorer.__name__ = "neg_rmse_mean_all_targets"
    return _scorer


def make_neg_rmse_single_true_target_scorer(
    target_index: int,
    n_true_targets: int,
    target_name: Optional[str] = None,
):
    def _scorer(estimator, X, y):
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        y_pred = estimator.predict(X)
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        k = int(target_index)
        m = int(n_true_targets)

        if y.shape[1] < m:
            raise ValueError(f"y has only {y.shape[1]} columns, but expected at least {m}")
        if y_pred.shape[1] < m:
            raise ValueError(f"y_pred has only {y_pred.shape[1]} columns, but expected at least {m}")

        y_true_part = y[:, :m]
        y_pred_part = y_pred[:, :m]

        y_true_k = y_true_part[:, k]
        y_pred_k = y_pred_part[:, k]

        n = min(len(y_true_k), len(y_pred_k))
        y_true_k = y_true_k[:n]
        y_pred_k = y_pred_k[:n]

        rmse = float(np.sqrt(np.mean((y_true_k - y_pred_k) ** 2)))
        return -rmse

    metric_name = f"neg_rmse_true_target_{target_name}" if target_name is not None else f"neg_rmse_true_target_{target_index}"
    _scorer.__name__ = metric_name
    return _scorer


# ---- OOF evaluation ----
def evaluate_oof_per_output(
    estimator,
    X: np.ndarray,
    Y_fit: np.ndarray,
    groups: np.ndarray,
    cv,
    *,
    m_true: int,
    augmented_2m: bool,
    eval_mask=None,
    logger=None,
):
    Y_fit = np.asarray(Y_fit)
    if Y_fit.ndim == 1:
        Y_fit = Y_fit.reshape(-1, 1)

    n = Y_fit.shape[0]
    mfit = Y_fit.shape[1]
    y_pred_oof_fit = np.full((n, mfit), np.nan, dtype=float)

    for fold, (tr, te) in enumerate(cv.split(X, Y_fit, groups=groups), start=1):
        if logger:
            logger.info(f"  Eval OOF fold {fold}: fit on {len(tr)}, predict on {len(te)}")
        from sklearn.base import clone
        est = clone(estimator)
        est.fit(X[tr], Y_fit[tr])
        pred = est.predict(X[te])
        pred = np.asarray(pred)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)

        if pred.shape[1] > mfit:
            pred = pred[:, :mfit]
        if pred.shape[1] < mfit:
            pad = np.full((pred.shape[0], mfit - pred.shape[1]), np.nan, dtype=float)
            pred = np.concatenate([pred, pad], axis=1)

        y_pred_oof_fit[te] = pred

    if eval_mask is None:
        eval_mask = np.ones(n, dtype=bool)
    else:
        eval_mask = np.asarray(eval_mask, dtype=bool)
        if len(eval_mask) == 1:
            eval_mask = np.ones(n, dtype=bool)

    if np.isnan(y_pred_oof_fit[eval_mask]).any():
        raise RuntimeError("OOF prediction contains NaNs on evaluation subset.")

    y_true_oof = Y_fit[eval_mask, :m_true]
    y_pred_oof = y_pred_oof_fit[eval_mask, :m_true]

    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_true_oof, y_pred_oof, multioutput="raw_values").astype(float).tolist()
    rmse = np.sqrt(mean_squared_error(y_true_oof, y_pred_oof, multioutput="raw_values")).astype(float).tolist()
    return r2, rmse, y_true_oof, y_pred_oof

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from runner import FamilyWinner, _extract_split_columns, _scoring_name
from sklearn.metrics import r2_score, mean_squared_error


def _use_log_x_for_sweep_key(key: str) -> bool:
    key = str(key)
    return ("alpha" in key.lower()) or ("l2" in key.lower())


def _prepare_sweep_x(sweep_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, str]:
    key = str(sweep_df["sweep_key"].iloc[0])
    raw = np.asarray(sweep_df["sweep_value"], dtype=object)

    if _use_log_x_for_sweep_key(key):
        x_num = np.array([float(v) for v in raw], dtype=float)
        mask = np.isfinite(x_num) & (x_num > 0)
        x_plot = np.log10(x_num[mask])
        keep_idx = np.flatnonzero(mask)
        xlabel = f"log10({key})"
        return x_plot, keep_idx, xlabel

    x_num = np.array([float(v) for v in raw], dtype=float)
    mask = np.isfinite(x_num)
    keep_idx = np.flatnonzero(mask)
    xlabel = key
    return x_num[mask], keep_idx, xlabel


def plot_family_selection(winner: FamilyWinner, out_path: str, title_prefix: str = "") -> None:
    cvres = winner.grid.cv_results_
    params_list = cvres["params"]

    C = np.array([winner.complexity_fn(p) for p in params_list], dtype=float)

    split_test_cols, split_train_cols = _extract_split_columns(cvres)
    n_splits = len(split_test_cols)

    mean_test = np.array(cvres["mean_test_score"], dtype=float)
    std_test = np.array(cvres["std_test_score"], dtype=float)

    has_train = ("mean_train_score" in cvres) and ("std_train_score" in cvres) and (len(split_train_cols) == n_splits)
    mean_train = np.array(cvres["mean_train_score"], dtype=float) if has_train else None
    std_train = np.array(cvres["std_train_score"], dtype=float) if has_train else None

    best_idx = int(np.argmax(mean_test))
    thr = float(mean_test[best_idx] - std_test[best_idx])
    sel_idx = int(winner.best_index)

    order = np.argsort(C)
    C_s = C[order]
    mean_test_s = mean_test[order]
    std_test_s = std_test[order]

    plt.figure(figsize=(12, 8))

    if n_splits > 0:
        test_scores = np.array([[cvres[k][j] for k in split_test_cols] for j in range(len(params_list))], dtype=float)
        for j in range(len(params_list)):
            plt.scatter([C[j]] * n_splits, test_scores[j], alpha=0.15, s=18, label="Fold test scores" if j == 0 else "")
        if has_train and mean_train is not None:
            train_scores = np.array([[cvres[k][j] for k in split_train_cols] for j in range(len(params_list))], dtype=float)
            for j in range(len(params_list)):
                plt.scatter([C[j]] * n_splits, train_scores[j], alpha=0.15, s=18, label="Fold train scores" if j == 0 else "")

    plt.errorbar(C_s, mean_test_s, yerr=std_test_s, fmt="-o", linewidth=2,
                 capsize=4, capthick=1.5, markersize=4, label="Mean test score")

    if has_train and mean_train is not None and std_train is not None:
        mean_train_s = mean_train[order]
        std_train_s = std_train[order]
        plt.errorbar(C_s, mean_train_s, yerr=std_train_s, fmt="-o", linewidth=2,
                     capsize=4, capthick=1.5, markersize=4, label="Mean train score")

    plt.axhline(mean_test[best_idx], linestyle="--", linewidth=2, label="Best mean_test_score")
    plt.axhline(thr, linestyle="--", linewidth=2, label="1-SE threshold (best - std)")
    plt.axvline(C[sel_idx], alpha=0.2, linewidth=8, label="Selected (refit)")
    plt.scatter([C[sel_idx]], [mean_test[sel_idx]], s=140, marker="X", label="Selected point")

    plt.xlabel("Complexity C(theta)")
    plt.ylabel("mean_test_score (score-space)")
    plt.title(f"{title_prefix}{winner.family} | 1-SE + min complexity")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _exp_yerr(yt: np.ndarray, target_name: str) -> np.ndarray:
    if "Flux" in target_name:
        return 0.10 * np.abs(yt)
    return np.full_like(yt, 1.0)


def plot_oof_pred_vs_true_per_target(winner: FamilyWinner, target_cols, out_dir: str, prefix: str) -> None:
    if winner.oof_y_true is None or winner.oof_y_pred is None:
        return

    y_true = np.asarray(winner.oof_y_true)
    y_pred = np.asarray(winner.oof_y_pred)

    m = min(y_true.shape[1], y_pred.shape[1], len(target_cols))
    fam = winner.family.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")

    for j in range(m):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        tname = str(target_cols[j]).replace(" ", "_")

        finite = np.isfinite(yt) & np.isfinite(yp)
        if not np.any(finite):
            continue
        yt = yt[finite]
        yp = yp[finite]

        vmin = float(min(np.min(yt), np.min(yp)))
        vmax = float(max(np.max(yt), np.max(yp)))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin -= 1.0
            vmax += 1.0

        rmse_j = float(winner.rmse_per_output[j]) if (winner.rmse_per_output is not None and j < len(winner.rmse_per_output)) else None
        yerr = _exp_yerr(yt, str(target_cols[j]))

        plt.figure(figsize=(7.5, 6.5))
        plt.errorbar(yp, yt, yerr=yerr, fmt="none", ecolor="gray", alpha=0.5, capsize=3, elinewidth=1)
        plt.scatter(yp, yt, s=18, alpha=0.7, label="OOF predictions")
        plt.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=2, color="gray")

        if rmse_j is not None and rmse_j > 0:
            plt.fill_between([vmin, vmax],
                             [vmin - rmse_j, vmax - rmse_j],
                             [vmin + rmse_j, vmax + rmse_j],
                             alpha=0.08, color="red", label=f"\u00b1RMSE={rmse_j:.4f}")
        plt.legend(loc="lower right")

        plt.xlabel("Predicted (OOF)")
        plt.ylabel("Experimental (OOF)")
        plt.title(f"{winner.family} | {target_cols[j]} | Pred vs Experimental")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        fname = f"{prefix}_oof_pred_vs_true_{fam}_{tname}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close()


def plot_winners_rmse_by_target(winners: List[FamilyWinner], target_cols, out_path: str,
                                title: str = "RMSE by target (OOF) - selected models") -> None:
    usable = [w for w in winners if w.rmse_per_output is not None]
    if not usable:
        return

    m = len(target_cols)
    fam_names = [w.family for w in usable]
    rmse_mat = np.full((len(usable), m), np.nan, dtype=float)
    for i, w in enumerate(usable):
        for j in range(min(m, len(w.rmse_per_output or []))):
            rmse_mat[i, j] = float(w.rmse_per_output[j])

    if not np.isfinite(rmse_mat).any():
        return

    x = np.arange(len(usable), dtype=float)
    width = 0.8 / max(1, m)

    plt.figure(figsize=(max(10, 1.1 * len(usable)), 6.5))
    for j in range(m):
        y = rmse_mat[:, j]
        plt.bar(x + (j - (m - 1) / 2) * width, y, width=width, label=str(target_cols[j]))

    plt.xticks(x, fam_names, rotation=45, ha="right")
    plt.ylabel("RMSE (OOF)")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_keras_learning_curve(history_df: pd.DataFrame, out_path: str, title: str) -> None:
    if history_df is None or history_df.empty or "loss" not in history_df.columns:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(history_df["epoch"], history_df["loss"], linewidth=2, label="Train loss")
    if "val_loss" in history_df.columns and history_df["val_loss"].notna().any():
        plt.plot(history_df["epoch"], history_df["val_loss"], linewidth=2, label="Validation loss")
        best_epoch = int(history_df.loc[history_df["val_loss"].astype(float).idxmin(), "epoch"])
        plt.axvline(best_epoch, linestyle="--", linewidth=2, label=f"Best val epoch = {best_epoch}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_sweep_error(sweep_df: pd.DataFrame, winner: FamilyWinner, out_path: str) -> None:
    if sweep_df is None or sweep_df.empty:
        return

    x_plot, keep_idx, xlabel = _prepare_sweep_x(sweep_df)
    y_test = np.asarray(sweep_df["metric_test_natural"], dtype=float)[keep_idx]
    y_train = np.asarray(sweep_df["metric_train_natural"], dtype=float)[keep_idx] if "metric_train_natural" in sweep_df.columns else None

    has_std = "std_test_score" in sweep_df.columns
    if has_std:
        std_test = np.asarray(sweep_df["std_test_score"], dtype=float)[keep_idx]
        std_train = np.asarray(sweep_df["std_train_score"], dtype=float)[keep_idx] if y_train is not None and "std_train_score" in sweep_df.columns else None

    order = np.argsort(x_plot)
    x_plot = x_plot[order]
    y_test = y_test[order]
    y_train = y_train[order] if y_train is not None else None
    if has_std:
        std_test = std_test[order]
        std_train = std_train[order] if std_train is not None else None

    best_idx = int(np.argmin(y_test))

    key = str(sweep_df["sweep_key"].iloc[0])
    ref_raw = winner.best_params.get(key, np.nan)
    try:
        ref_val = float(ref_raw)
    except Exception:
        ref_val = np.nan

    if _use_log_x_for_sweep_key(key):
        ref_x = np.log10(ref_val) if np.isfinite(ref_val) and ref_val > 0 else np.nan
    else:
        ref_x = ref_val if np.isfinite(ref_val) else np.nan

    plt.figure(figsize=(10, 6))
    if has_std:
        plt.errorbar(x_plot, y_test, yerr=std_test, fmt="-o", linewidth=2,
                     capsize=4, capthick=1.5, markersize=4, label="Validation/CV error")
    else:
        plt.plot(x_plot, y_test, "-o", linewidth=2, markersize=4, label="Validation/CV error")
    if y_train is not None:
        if std_train is not None:
            plt.errorbar(x_plot, y_train, yerr=std_train, fmt="-o", linewidth=2,
                         capsize=4, capthick=1.5, markersize=4, label="Train error")
        else:
            plt.plot(x_plot, y_train, "-o", linewidth=2, markersize=4, label="Train error")

    plt.axvline(x_plot[best_idx], linestyle="--", linewidth=2, label="Minimum sweep error")
    if np.isfinite(ref_x):
        plt.axvline(ref_x, linestyle="--", linewidth=2, label="Selected winner value")

    plt.xlabel(xlabel)
    plt.ylabel("RMSE")
    plt.title(f"{winner.family}: error vs sweep parameter")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_sweep_gap(sweep_df: pd.DataFrame, winner: FamilyWinner, out_path: str) -> None:
    if sweep_df is None or sweep_df.empty or "gap_natural_cv" not in sweep_df.columns:
        return

    x_plot, keep_idx, xlabel = _prepare_sweep_x(sweep_df)
    y_gap = np.asarray(sweep_df["gap_natural_cv"], dtype=float)[keep_idx]

    has_std = ("std_test_score" in sweep_df.columns and "std_train_score" in sweep_df.columns)
    if has_std:
        std_test = np.asarray(sweep_df["std_test_score"], dtype=float)[keep_idx]
        std_train = np.asarray(sweep_df["std_train_score"], dtype=float)[keep_idx]
        std_gap = np.sqrt(std_test ** 2 + std_train ** 2)

    order = np.argsort(x_plot)
    x_plot = x_plot[order]
    y_gap = y_gap[order]
    if has_std:
        std_gap = std_gap[order]

    best_idx = int(np.argmin(np.abs(y_gap)))

    key = str(sweep_df["sweep_key"].iloc[0])
    ref_raw = winner.best_params.get(key, np.nan)
    try:
        ref_val = float(ref_raw)
    except Exception:
        ref_val = np.nan

    if _use_log_x_for_sweep_key(key):
        ref_x = np.log10(ref_val) if np.isfinite(ref_val) and ref_val > 0 else np.nan
    else:
        ref_x = ref_val if np.isfinite(ref_val) else np.nan

    plt.figure(figsize=(10, 6))
    if has_std:
        plt.errorbar(x_plot, y_gap, yerr=std_gap, fmt="-o", linewidth=2,
                     capsize=4, capthick=1.5, markersize=4,
                     label="Gap = RMSE_test - RMSE_train")
    else:
        plt.plot(x_plot, y_gap, "-o", linewidth=2, markersize=4, label="Gap = RMSE_test - RMSE_train")
    plt.axvline(x_plot[best_idx], linestyle="--", linewidth=2, label="Smallest |gap| in sweep")
    if np.isfinite(ref_x):
        plt.axvline(ref_x, linestyle="--", linewidth=2, label="Selected winner value")

    plt.xlabel(xlabel)
    plt.ylabel("Generalization gap")
    plt.title(f"{winner.family}: gap vs sweep parameter")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_lasso_coefficient_path(
    *,
    base_estimator_factory,
    X: np.ndarray,
    Y: np.ndarray,
    alphas: np.ndarray,
    feature_names: Sequence[str],
    selected_alpha,
    out_path: str,
    title: str = "Lasso coefficient path",
) -> None:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    alphas = np.array(alphas, dtype=float)
    alphas = alphas[np.isfinite(alphas) & (alphas > 0)]
    alphas = np.unique(alphas)
    alphas = np.sort(alphas)[::-1]

    n_features = Xs.shape[1]
    paths = np.zeros((len(alphas), n_features), dtype=float)

    for i, a in enumerate(alphas):
        est = base_estimator_factory()
        est.set_params(alpha=float(a))
        est.fit(Xs, Y)

        coef = getattr(est, "coef_", None)
        if coef is None:
            raise RuntimeError("Estimator has no coef_ after fit.")
        coef = np.array(coef, dtype=float)

        if coef.ndim == 1:
            paths[i, :] = np.abs(coef)
        else:
            paths[i, :] = np.sqrt(np.sum(coef ** 2, axis=0))

    x = np.log10(alphas)

    plt.figure(figsize=(11, 7))
    for j in range(n_features):
        label = str(feature_names[j]) if feature_names is not None else f"x{j}"
        plt.plot(x, paths[:, j], linewidth=2, label=label)

    if selected_alpha is not None and np.isfinite(selected_alpha) and selected_alpha > 0:
        plt.axvline(np.log10(float(selected_alpha)), linestyle="--", linewidth=2, label="Selected alpha")

    plt.xlabel("log10(lambda)")
    plt.ylabel("||coef_j|| (per feature)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_oof_vs_phy_overlay_per_target(
    winner: FamilyWinner,
    target_cols,
    y_model: np.ndarray,
    out_dir: str,
    prefix: str = "model_compare",
) -> None:
    if winner.oof_y_true is None or winner.oof_y_pred is None or y_model is None:
        return

    y_true = np.asarray(winner.oof_y_true, dtype=float)
    y_pred_hrnn = np.asarray(winner.oof_y_pred, dtype=float)
    y_pred_phy = np.asarray(y_model, dtype=float)

    m = min(y_true.shape[1], y_pred_hrnn.shape[1], y_pred_phy.shape[1], len(target_cols))
    fam = winner.family.replace(" ", "_").replace("(", "").replace(")", "")

    for j in range(m):
        yt = y_true[:, j]
        yp_hrnn = y_pred_hrnn[:, j]
        yp_phy = y_pred_phy[:, j]

        finite = np.isfinite(yt) & np.isfinite(yp_hrnn) & np.isfinite(yp_phy)
        yt = yt[finite]
        yp_hrnn = yp_hrnn[finite]
        yp_phy = yp_phy[finite]

        if len(yt) == 0:
            continue

        vmin = float(min(np.min(yt), np.min(yp_hrnn), np.min(yp_phy)))
        vmax = float(max(np.max(yt), np.max(yp_hrnn), np.max(yp_phy)))

        rmse_phy = float(np.sqrt(np.mean((yt - yp_phy) ** 2)))
        rmse_hrnn = float(np.sqrt(np.mean((yt - yp_hrnn) ** 2)))

        r2_phy = float(r2_score(yt, yp_phy))
        r2_hrnn = float(r2_score(yt, yp_hrnn))

        yerr = _exp_yerr(yt, str(target_cols[j]))

        plt.figure(figsize=(7.8, 6.8))
        plt.errorbar(yp_phy, yt, yerr=yerr, fmt="none", ecolor="gray", alpha=0.4, capsize=2, elinewidth=0.8)
        plt.errorbar(yp_hrnn, yt, yerr=yerr, fmt="none", ecolor="gray", alpha=0.4, capsize=2, elinewidth=0.8)
        plt.scatter(
            yp_phy, yt, s=24, alpha=0.70,
            label=f"Modelo 0D (_phy) | RMSE={rmse_phy:.3f} | R\u00b2={r2_phy:.3f}",
        )
        plt.scatter(
            yp_hrnn, yt, s=24, alpha=0.70,
            label=f"{winner.family} (OOF) | RMSE={rmse_hrnn:.3f} | R\u00b2={r2_hrnn:.3f}",
        )
        plt.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=2, label="y = x")
        plt.fill_between([vmin, vmax],
                         [vmin - rmse_hrnn, vmax - rmse_hrnn],
                         [vmin + rmse_hrnn, vmax + rmse_hrnn],
                         alpha=0.06, color="orange", label=f"\u00b1RMSE={rmse_hrnn:.3f}")

        plt.xlabel("Predito")
        plt.ylabel("Experimental")
        plt.title(f"Compara\u00e7\u00e3o de desempenho | {target_cols[j]}")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()

        fname = f"{prefix}_overlay_{fam}_{str(target_cols[j]).replace(' ', '_')}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=220)
        plt.close()


def plot_oof_vs_phy_overlay_panel(
    winner: FamilyWinner,
    target_cols,
    y_model: np.ndarray,
    out_path: str,
    figure_title: Optional[str] = None,
) -> None:
    if winner.oof_y_true is None or winner.oof_y_pred is None or y_model is None:
        return

    y_true = np.asarray(winner.oof_y_true, dtype=float)
    y_pred_hrnn = np.asarray(winner.oof_y_pred, dtype=float)
    y_pred_phy = np.asarray(y_model, dtype=float)

    m = min(y_true.shape[1], y_pred_hrnn.shape[1], y_pred_phy.shape[1], len(target_cols))
    if m == 0:
        return

    fig, axes = plt.subplots(1, m, figsize=(6.1 * m, 5.8))
    if m == 1:
        axes = [axes]

    for j, ax in enumerate(axes[:m]):
        yt = y_true[:, j]
        yp_hrnn = y_pred_hrnn[:, j]
        yp_phy = y_pred_phy[:, j]

        finite = np.isfinite(yt) & np.isfinite(yp_hrnn) & np.isfinite(yp_phy)
        yt = yt[finite]
        yp_hrnn = yp_hrnn[finite]
        yp_phy = yp_phy[finite]

        if len(yt) == 0:
            continue

        vmin = float(min(np.min(yt), np.min(yp_hrnn), np.min(yp_phy)))
        vmax = float(max(np.max(yt), np.max(yp_hrnn), np.max(yp_phy)))

        rmse_phy = float(np.sqrt(np.mean((yt - yp_phy) ** 2)))
        rmse_hrnn = float(np.sqrt(np.mean((yt - yp_hrnn) ** 2)))

        yerr = _exp_yerr(yt, str(target_cols[j]))

        ax.errorbar(yp_phy, yt, yerr=yerr, fmt="none", ecolor="gray", alpha=0.4, capsize=2, elinewidth=0.8)
        ax.errorbar(yp_hrnn, yt, yerr=yerr, fmt="none", ecolor="gray", alpha=0.4, capsize=2, elinewidth=0.8)
        ax.scatter(yp_phy, yt, s=22, alpha=0.70, label=f"0D | RMSE={rmse_phy:.3f}")
        ax.scatter(yp_hrnn, yt, s=22, alpha=0.70, label=f"{winner.family} | RMSE={rmse_hrnn:.3f}")
        ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=2)
        ax.fill_between([vmin, vmax],
                        [vmin - rmse_hrnn, vmax - rmse_hrnn],
                        [vmin + rmse_hrnn, vmax + rmse_hrnn],
                        alpha=0.06, color="orange", label=f"\u00b1RMSE={rmse_hrnn:.3f}")

        ax.set_xlabel("Predito")
        ax.set_ylabel("Experimental")
        ax.set_title(str(target_cols[j]))
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    if figure_title:
        fig.suptitle(figure_title, y=1.02)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

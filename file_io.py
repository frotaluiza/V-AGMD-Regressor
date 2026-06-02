import json
import os
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

from runner import (
    FamilyWinner,
    cv_results_to_frame,
    fit_keras_history_for_winner,
)
from sweep import run_sweep_from_winner
from plots import (
    plot_family_selection,
    plot_oof_pred_vs_true_per_target,
    plot_winners_rmse_by_target,
    plot_keras_learning_curve,
    plot_sweep_error,
    plot_sweep_gap,
    plot_oof_vs_phy_overlay_per_target,
    plot_oof_vs_phy_overlay_panel,
)


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


def _jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, tuple):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()

    if hasattr(obj, "get_config") and callable(getattr(obj, "get_config")):
        try:
            return {"class": obj.__class__.__name__, "config": _jsonable(obj.get_config())}
        except Exception:
            return {"class": obj.__class__.__name__}

    return str(obj)


def sanitize_params_for_json(params: Dict[str, Any]) -> Dict[str, Any]:
    return _jsonable(params)


def save_overlay_comparison_table(winner, target_cols, y_model, out_path):
    if winner.oof_y_true is None or winner.oof_y_pred is None or y_model is None:
        return

    y_true = np.asarray(winner.oof_y_true, dtype=float)
    y_pred_hrnn = np.asarray(winner.oof_y_pred, dtype=float)
    y_pred_phy = np.asarray(y_model, dtype=float)

    m = min(y_true.shape[1], y_pred_hrnn.shape[1], y_pred_phy.shape[1], len(target_cols))

    data = {}
    for j in range(m):
        name = str(target_cols[j])
        data[f"{name}_exp"] = y_true[:, j]
        data[f"{name}_phy"] = y_pred_phy[:, j]
        data[f"{name}_hrnn_oof"] = y_pred_hrnn[:, j]

    pd.DataFrame(data).to_csv(out_path, index=False)


def save_outputs(
    out_dir: str,
    winners: List[FamilyWinner],
    summary: pd.DataFrame,
    *,
    target_cols: Sequence[str],
    X: np.ndarray,
    Y_true: np.ndarray,
    groups: np.ndarray,
    Y_model: Optional[np.ndarray],
    cv,
    include_splits_in_cv_results: bool = False,
    prefix: str = "model_selection",
    make_plots: bool = True,
    save_family_tables: bool = True,
    logger=None,
    make_oof_plots: bool = True,
    make_winners_rmse_plot: bool = True,
    make_sweeps: bool = True,
    fit_final_keras_histories: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    summary.to_csv(os.path.join(out_dir, f"{prefix}_summary.csv"), index=False)

    best_params_clean = {w.family: sanitize_params_for_json(w.best_params) for w in winners}
    with open(os.path.join(out_dir, f"{prefix}_best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params_clean, f, ensure_ascii=False, indent=2)

    for w in winners:
        cv_df = cv_results_to_frame(w.grid.cv_results_, include_splits=include_splits_in_cv_results)
        cv_df.to_csv(os.path.join(out_dir, f"{prefix}_cv_results_{_safe_filename(w.family)}.csv"), index=False)

        if save_family_tables and w.family_table is not None:
            w.family_table.to_csv(os.path.join(out_dir, f"{prefix}_family_table_{_safe_filename(w.family)}.csv"), index=False)

        if fit_final_keras_histories:
            try:
                hist_df = fit_keras_history_for_winner(w, logger=logger)
            except Exception as e:
                if logger:
                    logger.warning(f"[{w.family}] Could not fit final Keras history: {e}")
                hist_df = None
            if hist_df is not None:
                try:
                    hist_csv = os.path.join(out_dir, f"{prefix}_training_history_{_safe_filename(w.family)}.csv")
                    hist_df.to_csv(hist_csv, index=False)
                except Exception as e:
                    if logger:
                        logger.warning(f"[{w.family}] Could not save training history CSV: {e}")
                    hist_csv = None
                if make_plots and hist_csv is not None:
                    try:
                        plot_keras_learning_curve(
                            history_df=hist_df,
                            out_path=os.path.join(out_dir, f"{prefix}_learning_curve_{_safe_filename(w.family)}.png"),
                            title=f"{w.family}: train vs validation loss",
                        )
                    except Exception as e:
                        if logger:
                            logger.warning(f"[{w.family}] Could not save learning curve plot: {e}")

        if make_plots:
            plot_family_selection(w, os.path.join(out_dir, f"{prefix}_selection_plot_{_safe_filename(w.family)}.png"))

        if make_oof_plots:
            plot_oof_pred_vs_true_per_target(w, target_cols=target_cols, out_dir=out_dir, prefix=prefix)

            if Y_model is not None:
                plot_oof_vs_phy_overlay_per_target(
                    winner=w, target_cols=target_cols, y_model=Y_model,
                    out_dir=out_dir, prefix=prefix,
                )

                plot_oof_vs_phy_overlay_panel(
                    winner=w, target_cols=target_cols, y_model=Y_model,
                    out_path=os.path.join(out_dir, f"{prefix}_overlay_panel_{_safe_filename(w.family)}.png"),
                    figure_title="Compara\u00e7\u00e3o entre modelo 0D e modelo HRNN vencedor",
                )

                save_overlay_comparison_table(
                    winner=w, target_cols=target_cols, y_model=Y_model,
                    out_path=os.path.join(out_dir, f"{prefix}_overlay_points_{_safe_filename(w.family)}.csv"),
                )

        if make_sweeps:
            sweep_df = run_sweep_from_winner(
                winner=w, X=X, Y_true=Y_true, groups=groups, cv=cv,
                Y_model=Y_model, logger=logger,
            )
            if sweep_df is not None and not sweep_df.empty:
                sweep_df.to_csv(
                    os.path.join(out_dir, f"{prefix}_sweep_{_safe_filename(w.family)}.csv"),
                    index=False,
                )
                if make_plots:
                    plot_sweep_error(
                        sweep_df=sweep_df, winner=w,
                        out_path=os.path.join(out_dir, f"{prefix}_error_vs_sweepparam_{_safe_filename(w.family)}.png"),
                    )
                    plot_sweep_gap(
                        sweep_df=sweep_df, winner=w,
                        out_path=os.path.join(out_dir, f"{prefix}_gap_vs_sweepparam_{_safe_filename(w.family)}.png"),
                    )

    if make_winners_rmse_plot:
        plot_winners_rmse_by_target(
            winners=winners, target_cols=target_cols,
            out_path=os.path.join(out_dir, f"{prefix}_winners_rmse_by_target.png"),
        )

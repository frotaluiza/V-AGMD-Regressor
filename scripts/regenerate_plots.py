import os, sys, json, ast
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sklearn.metrics import r2_score

import matplotlib
matplotlib.use("Agg")

if len(sys.argv) > 1:
    OUT = sys.argv[1]
else:
    OUT = r"C:\Users\frota\OneDrive\Documentos\TCC\Codigos\Codigo-revisado\stage2_only_dados_att_com_var_com_phy_20260602_012037\stage2_restricted_hybrid_comparison"

overlay_files = sorted(f for f in os.listdir(OUT) if "_overlay_points_" in f and os.path.isfile(os.path.join(OUT, f)))
if not overlay_files:
    raise FileNotFoundError(f"No _overlay_points_ files found in {OUT}")
PREFIX = overlay_files[0].rsplit("_overlay_points_", 1)[0]

first_overlay = pd.read_csv(os.path.join(OUT, overlay_files[0]))
target_cols_raw = [c.replace("_exp", "").replace("_phy", "").replace("_hrnn_oof", "")
                   for c in first_overlay.columns]
target_cols = sorted(set(target_cols_raw))

def _parse_best_params(row):
    raw = row["best_params"]
    return ast.literal_eval(raw)

@dataclass
class PseudoFamilyWinner:
    family: str
    best_score: float
    best_std_test: float
    best_mean_train: float
    best_gap: float
    complexity: float
    rmse_per_output: Optional[List[float]] = None
    r2_per_output: Optional[List[float]] = None
    oof_y_true: Optional[np.ndarray] = None
    oof_y_pred: Optional[np.ndarray] = None

def _r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))

def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

summary = pd.read_csv(os.path.join(OUT, f"{PREFIX}_summary.csv"))

winners = []
for _, row in summary.iterrows():
    family = row["family"]
    ov_file = os.path.join(OUT, f"{PREFIX}_overlay_points_{family}.csv")
    if not os.path.isfile(ov_file):
        print(f"Missing overlay file for {family}")
        continue

    ov = pd.read_csv(ov_file)
    target_cols_sorted = sorted(set(
        c.replace("_exp", "").replace("_phy", "").replace("_hrnn_oof", "")
        for c in ov.columns
    ))

    y_true_list = []
    y_pred_list = []
    r2_list = []
    rmse_list = []

    for t in target_cols_sorted:
        yt = ov[f"{t}_exp"].values.astype(float)
        yp = ov[f"{t}_hrnn_oof"].values.astype(float)
        y_true_list.append(yt)
        y_pred_list.append(yp)
        r2_list.append(_r2(yt, yp))
        rmse_list.append(_rmse(yt, yp))

    best_std_test = float(row["best_std_test"])
    best_score = float(row["best_score_mean_test"])
    best_mean_train = float(row["best_mean_train"]) if pd.notna(row.get("best_mean_train")) else None
    best_gap = float(row.get("gap_train_minus_test_score_space", 0)) if pd.notna(row.get("gap_train_minus_test_score_space")) else None
    complexity = float(row["complexity"]) if pd.notna(row.get("complexity")) else 0.0

    w = PseudoFamilyWinner(
        family=family,
        best_score=best_score,
        best_std_test=best_std_test,
        best_mean_train=best_mean_train,
        best_gap=best_gap,
        complexity=complexity,
        rmse_per_output=rmse_list,
        r2_per_output=r2_list,
        oof_y_true=np.column_stack(y_true_list),
        oof_y_pred=np.column_stack(y_pred_list),
    )
    winners.append(w)
    print(f"Loaded {family}: R2={r2_list}, RMSE={rmse_list}")

print(f"Loaded {len(winners)} winners")

# Build y_model (0D physics) from the first overlay file
# All families share the same y_model (same experiment data)
y_model_list = []
for t in target_cols_sorted:
    y_model_list.append(first_overlay[f"{t}_phy"].values.astype(float))
Y_model = np.column_stack(y_model_list)

# --- Regenerate plots ---

from plots import (
    plot_oof_pred_vs_true_per_target,
    plot_winners_rmse_by_target,
    plot_oof_vs_phy_overlay_per_target,
    plot_oof_vs_phy_overlay_panel,
)

for w in winners:
    print(f"Plotting {w.family}...")

    # OOF pred vs true (uses the winner object directly for oof_y_true/pred)
    plot_oof_pred_vs_true_per_target(w, target_cols=target_cols_sorted, out_dir=OUT, prefix=PREFIX)

    # Overlay per target
    plot_oof_vs_phy_overlay_per_target(
        winner=w, target_cols=target_cols_sorted, y_model=Y_model,
        out_dir=OUT, prefix=PREFIX,
    )

    # Overlay panel
    plot_oof_vs_phy_overlay_panel(
        winner=w, target_cols=target_cols_sorted, y_model=Y_model,
        out_path=os.path.join(OUT, f"{PREFIX}_overlay_panel_{w.family}.png"),
        figure_title="Compara\u00e7\u00e3o entre modelo 0D e modelo HRNN vencedor",
    )

# Winners RMSE by target
plot_winners_rmse_by_target(
    winners=winners, target_cols=target_cols_sorted,
    out_path=os.path.join(OUT, f"{PREFIX}_winners_rmse_by_target.png"),
)

print("Done regenerating plots.")

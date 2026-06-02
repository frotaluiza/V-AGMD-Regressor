#!/usr/bin/env python3
"""Check gap filter impact across all stages."""
import os, sys, pandas as pd
base = r"C:\Users\frota\OneDrive\Documentos\TCC\Codigos\Codigo-revisado"

files = [
    (r"stage2_only_dados_att_com_var_com_phy_20260601_151146\stage2_restricted_hybrid_comparison\agmd_stage2_restricted_hybrids_dados_att_com_var_com_phy_summary.csv", "Stage 2"),
    (r"stage1_only_dados_att_com_var_com_phy_20260601_012129\stage1_baseline_search\agmd_stage1_baseline_dados_att_com_var_com_phy_summary.csv", "Stage 1"),
    (r"stage0_classical_models\agmd_stage0_classical_dados_att_com_var_com_phy_summary.csv", "Stage 0",),
]
# Try stage0 from the stage1 directory
import glob
stage0_files = list(glob.glob(os.path.join(base, "stage1_only*", "stage0_classical_models", "*.csv")))
if stage0_files:
    files.append((os.path.relpath(stage0_files[0], base), "Stage 0"))

for f, label in files:
    path = os.path.join(base, f)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        gap = r.get("gap_rmse_Flux", None)
        tol = r.get("gap_filter_tol", None)
        if gap is not None and tol is not None:
            passed = gap < tol
            print(f"[{label}] {r['family']}: gap={gap:.4f}  tol={tol:.2f}  passed={passed}")
print("Done")

import pandas as pd, numpy as np, os
from sklearn.metrics import r2_score, mean_squared_error

# Latest Stage 2 run (includes HRNN and dropout)
RUN = r"C:\Users\frota\OneDrive\Documentos\TCC\Codigos\Codigo-revisado\stage2_only_dados_att_com_var_com_phy_20260603_001451\stage2_restricted_hybrid_comparison"
SUMMARY = os.path.join(RUN, "agmd_stage2_hybrids_dados_att_com_var_com_phy_summary.csv")
OVERLAY = os.path.join(RUN, "agmd_stage2_hybrids_dados_att_com_var_com_phy_overlay_points_KerasMLP_ZohanResidual_Restricted.csv")

summary = pd.read_csv(SUMMARY)
ov = pd.read_csv(OVERLAY)

# Compute 0D model metrics
targets = ["Alim_T_out", "Ref_T_out", "Flux"]
phy_rmse, phy_r2 = {}, {}
for t in targets:
    yt = ov[f"{t}_exp"].values.astype(float)
    yp = ov[f"{t}_phy"].values.astype(float)
    phy_rmse[t] = float(np.sqrt(mean_squared_error(yt, yp)))
    phy_r2[t] = float(r2_score(yt, yp))

# Build table data sorted by Flux RMSE
rows = []
# Add 0D model
rows.append({
    "family": "Modelo físico 0D",
    "r2_a": phy_r2["Alim_T_out"],
    "rmse_a": phy_rmse["Alim_T_out"],
    "r2_r": phy_r2["Ref_T_out"],
    "rmse_r": phy_rmse["Ref_T_out"],
    "r2_f": phy_r2["Flux"],
    "rmse_f": phy_rmse["Flux"],
    "cv_rmse": None,
})
# Add models from summary
for _, r in summary.iterrows():
    fam = r["family"].replace("KerasMLP_", "")
    rows.append({
        "family": fam,
        "r2_a": r["r2_Alim_T_out"],
        "rmse_a": r["rmse_Alim_T_out"],
        "r2_r": r["r2_Ref_T_out"],
        "rmse_r": r["rmse_Ref_T_out"],
        "r2_f": r["r2_Flux"],
        "rmse_f": r["rmse_Flux"],
        "cv_rmse": abs(r["best_score_mean_test"]) if pd.notna(r.get("best_score_mean_test")) else None,
    })

# Sort by Flux RMSE
rows.sort(key=lambda x: x["rmse_f"])

# Generate LaTeX
out_dir = r"C:\Users\frota\OneDrive\Documentos\TCC\Revisão\TCC_editado\Textual\tables"
os.makedirs(out_dir, exist_ok=True)

lines = []
def emit(s):
    lines.append(s)

emit(r"\begin{table}[H]")
emit(r"\centering")
emit(r"\caption[Desempenho consolidado dos modelos]{Desempenho consolidado dos modelos avaliados, ordenados pelo RMSE do fluxo de permeado (OOF). As métricas OOF são calculadas sobre predições \textit{out-of-fold} conforme descrito na Seção~\ref{sec:avaliacao_oof}. O RMSE CV refere-se à média da validação cruzada durante a seleção de hiperparâmetros. Fonte: Autor.}")
emit(r"\label{tab:resultados_consolidados}")
emit(r"\begin{tabular}{l c c c c c c c}")
emit(r"\hline")
emit(r"\textbf{Modelo} &")
emit(r"\multicolumn{2}{c}{\textbf{$Alim\_T_{out}$}} &")
emit(r"\multicolumn{2}{c}{\textbf{$Ref\_T_{out}$}} &")
emit(r"\multicolumn{2}{c}{\textbf{$Flux$}} &")
emit(r"\textbf{RMSE CV} \\")
emit(r"\cline{2-3} \cline{4-5} \cline{6-7}")
emit(r" & R$^2$ & RMSE & R$^2$ & RMSE & R$^2$ & RMSE & $Flux$ \\")
emit(r"\hline")

for r in rows:
    fam = r["family"]
    cv_str = f"{r['cv_rmse']:.3f}" if r["cv_rmse"] is not None else "---"
    if fam == "ZohanResidual_Restricted":
        emit(r"\rowcolor{orange!15}")
    emit(f"{fam} & {r['r2_a']:.3f} & {r['rmse_a']:.3f} & {r['r2_r']:.3f} & {r['rmse_r']:.3f} & {r['r2_f']:.3f} & {r['rmse_f']:.3f} & {cv_str} \\\\")

emit(r"\hline")
emit(r"\end{tabular}")
emit(r"\end{table}")

output = "\n".join(lines)
print(output)

out_path = os.path.join(out_dir, "consolidated_results.tex")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(output)
print(f"\nSaved to {out_path}")

# Also print CV RMSE comparison
print("\n=== CV vs OOF Comparison ===")
for r in rows:
    if r["cv_rmse"] is not None:
        diff = abs(r["rmse_f"] - r["cv_rmse"])
        print(f"{r['family']:35s} | CV={r['cv_rmse']:.3f} | OOF={r['rmse_f']:.3f} | diff={diff:.3f}")

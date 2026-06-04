import pandas as pd, numpy as np, os
from sklearn.metrics import r2_score, mean_squared_error

# --- Paths ---
CLASSICAL_DIR = r"C:\Users\frota\OneDrive\Documentos\TCC\Codigos\Codigo-revisado\results_dados_att_com_var_com_phy_20260603_153025\stage0_classical_models"
HYBRID_DIR = r"C:\Users\frota\OneDrive\Documentos\TCC\Codigos\Codigo-revisado\stage2_only_dados_att_com_var_com_phy_20260603_185001\stage2_restricted_hybrid_comparison"
OVERLAY_PREV = r"C:\Users\frota\OneDrive\Documentos\TCC\Codigos\Codigo-revisado\stage2_only_dados_att_com_var_com_phy_20260602_012037\stage2_restricted_hybrid_comparison\agmd_stage2_hybrids_dados_att_com_var_com_phy_overlay_points_KerasMLP_ZohanResidual_Restricted.csv"

OUT_DIR = r"C:\Users\frota\OneDrive\Documentos\TCC\Revisão\TCC_editado\Textual\tables"
os.makedirs(OUT_DIR, exist_ok=True)

targets = ["Alim_T_out", "Ref_T_out", "Flux"]

# --- 0D model metrics ---
ov = pd.read_csv(OVERLAY_PREV)
phy_rmse, phy_r2 = {}, {}
for t in targets:
    yt = ov[f"{t}_exp"].values.astype(float)
    yp = ov[f"{t}_phy"].values.astype(float)
    phy_rmse[t] = np.sqrt(mean_squared_error(yt, yp))
    phy_r2[t] = r2_score(yt, yp)

# --- Table 1: CV RMSE (ALL models) ---
rows_cv = []
cs = pd.read_csv(os.path.join(CLASSICAL_DIR, "agmd_stage0_classical_dados_att_com_var_com_phy_summary.csv"))
for _, r in cs.iterrows():
    rows_cv.append({"family": r["family"], "type": "Clássico", "cv_rmse": abs(r["best_score_mean_test"])})

hs = pd.read_csv(os.path.join(HYBRID_DIR, "agmd_stage2_hybrids_dados_att_com_var_com_phy_summary.csv"))
for _, r in hs.iterrows():
    rows_cv.append({"family": r["family"].replace("KerasMLP_", ""), "type": "Híbrido", "cv_rmse": abs(r["best_score_mean_test"])})

rows_cv.append({"family": "Modelo físico 0D", "type": "Físico", "cv_rmse": None})
rows_cv.sort(key=lambda x: x["cv_rmse"] if x["cv_rmse"] is not None else 999)

# --- Table 2: OOF RMSE (ALL models) ---
rows_oof = []
for _, r in cs.iterrows():
    r2f = r.get("r2_Flux")
    rmf = r.get("rmse_Flux")
    if pd.notna(r2f) and str(r2f).strip():
        rows_oof.append({"family": r["family"], "type": "Clássico", "r2_f": float(r2f), "rmse_f": float(rmf),
                         "r2_a": float(r.get("r2_Alim_T_out", 0)), "rmse_a": float(r.get("rmse_Alim_T_out", 0)),
                         "r2_r": float(r.get("r2_Ref_T_out", 0)), "rmse_r": float(r.get("rmse_Ref_T_out", 0))})

for _, r in hs.iterrows():
    rows_oof.append({"family": r["family"].replace("KerasMLP_", ""), "type": "Híbrido",
                     "r2_f": float(r["r2_Flux"]), "rmse_f": float(r["rmse_Flux"]),
                     "r2_a": float(r["r2_Alim_T_out"]), "rmse_a": float(r["rmse_Alim_T_out"]),
                     "r2_r": float(r["r2_Ref_T_out"]), "rmse_r": float(r["rmse_Ref_T_out"])})

rows_oof.append({"family": "Modelo físico 0D", "type": "Físico",
                 "r2_f": float(phy_r2["Flux"]), "rmse_f": float(phy_rmse["Flux"]),
                 "r2_a": float(phy_r2["Alim_T_out"]), "rmse_a": float(phy_rmse["Alim_T_out"]),
                 "r2_r": float(phy_r2["Ref_T_out"]), "rmse_r": float(phy_rmse["Ref_T_out"])})
rows_oof.sort(key=lambda x: x["rmse_f"])


def emit_table(title, label, rows, has_type=True):
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{%s}" % title)
    lines.append(r"\label{%s}" % label)
    ncols = 8 if has_type else 7
    lines.append(r"\begin{tabular}{l l c c c c c c}" if has_type else r"\begin{tabular}{l c c c c c c}")
    lines.append(r"\hline")
    if has_type:
        lines.append(r"\textbf{Modelo} & \textbf{Tipo} & \multicolumn{2}{c}{\textbf{$Alim\_T_{out}$}} & \multicolumn{2}{c}{\textbf{$Ref\_T_{out}$}} & \multicolumn{2}{c}{\textbf{$Flux$}} \\")
        lines.append(r"\cline{3-4} \cline{5-6} \cline{7-8}")
        lines.append(r" &  & R$^2$ & RMSE & R$^2$ & RMSE & R$^2$ & RMSE \\")
    else:
        lines.append(r"\textbf{Modelo} & \multicolumn{2}{c}{\textbf{$Alim\_T_{out}$}} & \multicolumn{2}{c}{\textbf{$Ref\_T_{out}$}} & \multicolumn{2}{c}{\textbf{$Flux$}} \\")
        lines.append(r"\cline{2-3} \cline{4-5} \cline{6-7}")
        lines.append(r" & R$^2$ & RMSE & R$^2$ & RMSE & R$^2$ & RMSE \\")
    lines.append(r"\hline")

    for r in rows:
        fam = r["family"]
        if fam == "ZohanResidual_Restricted":
            lines.append(r"\rowcolor{orange!15}")
        if has_type:
            t = r["type"]
            lines.append(f"{fam} & {t} & {r['r2_a']:.3f} & {r['rmse_a']:.3f} & {r['r2_r']:.3f} & {r['rmse_r']:.3f} & {r['r2_f']:.3f} & {r['rmse_f']:.3f} \\\\")
        else:
            lines.append(f"{fam} & {r['r2_a']:.3f} & {r['rmse_a']:.3f} & {r['r2_r']:.3f} & {r['rmse_r']:.3f} & {r['r2_f']:.3f} & {r['rmse_f']:.3f} \\\\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    if not has_type:
        lines.append(r"\vspace{0.2cm}")
        lines.append(r"\footnotesize{Nota: os valores de CV RMSE são apresentados como referência da etapa de seleção; as métricas OOF constituem a avaliação final.}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# Generate CV RMSE table (simplified - just RMSE)
lines_cv = []
lines_cv.append(r"\begin{table}[H]")
lines_cv.append(r"\centering")
lines_cv.append(r"\caption[Desempenho em validação cruzada (CV RMSE)]{Desempenho em validação cruzada (CV RMSE) para o alvo $Flux$ em todos os modelos avaliados, ordenados pelo menor erro. O RMSE CV corresponde à média dos erros obtidos nas partições da validação cruzada durante a busca de hiperparâmetros. Fonte: Autor.}")
lines_cv.append(r"\label{tab:cv_rmse_all}")
lines_cv.append(r"\begin{tabular}{l c c}")
lines_cv.append(r"\hline")
lines_cv.append(r"\textbf{Modelo} & \textbf{Tipo} & \textbf{RMSE CV ($Flux$)} \\")
lines_cv.append(r"\hline")
for r in rows_cv:
    fam = r["family"]
    cv_str = f"{r['cv_rmse']:.4f}" if r['cv_rmse'] is not None else "---"
    if fam == "ZohanResidual_Restricted":
        lines_cv.append(r"\rowcolor{orange!15}")
    lines_cv.append(f"{fam} & {r['type']} & {cv_str} \\\\")
lines_cv.append(r"\hline")
lines_cv.append(r"\end{tabular}")
lines_cv.append(r"\end{table}")
lines_cv.append("")

lines_cv.append(emit_table(
    r"Desempenho \textit{out-of-fold} (OOF) para todas as vari\'aveis de sa\'{\i}da em todos os modelos avaliados, ordenados pelo RMSE do fluxo de permeado. As m\'etricas OOF s\~ao obtidas a partir de predi\c{c}\~oes de modelos retreinados independentemente em cada parti\c{c}\~ao da valida\c{c}\~ao cruzada, conforme descrito na Se\c{c}\~ao~\ref{sec:avaliacao_oof}. Fonte: Autor.",
    "tab:oof_all", rows_oof, has_type=True
))

output = "\n".join(lines_cv)
out_path = os.path.join(OUT_DIR, "tables_final.tex")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(output)
print(output)
print(f"\nSaved to {out_path}")

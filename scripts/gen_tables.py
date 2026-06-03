import pandas as pd, numpy as np, os
from sklearn.metrics import r2_score, mean_squared_error

# Sources
CLASSICAL = r"C:\Users\frota\OneDrive\Documentos\TCC\Codigos\Codigo-revisado\results_dados_att_com_var_com_phy_20260603_153025\stage0_classical_models\agmd_stage0_classical_dados_att_com_var_com_phy_summary.csv"
HYBRID = r"C:\Users\frota\OneDrive\Documentos\TCC\Codigos\Codigo-revisado\stage2_only_dados_att_com_var_com_phy_20260603_185001\stage2_restricted_hybrid_comparison\agmd_stage2_hybrids_dados_att_com_var_com_phy_summary.csv"
OVERLAY_0D = r"C:\Users\frota\OneDrive\Documentos\TCC\Codigos\Codigo-revisado\stage2_only_dados_att_com_var_com_phy_20260602_012037\stage2_restricted_hybrid_comparison\agmd_stage2_hybrids_dados_att_com_var_com_phy_overlay_points_KerasMLP_ZohanResidual_Restricted.csv"

classical = pd.read_csv(CLASSICAL)
hybrid = pd.read_csv(HYBRID)

# Compute 0D model metrics
ov = pd.read_csv(OVERLAY_0D)
targets = ["Alim_T_out", "Ref_T_out", "Flux"]
phy_rmse, phy_r2 = {}, {}
for t in targets:
    yt = ov[f"{t}_exp"].values.astype(float)
    yp = ov[f"{t}_phy"].values.astype(float)
    phy_rmse[t] = float(np.sqrt(mean_squared_error(yt, yp)))
    phy_r2[t] = float(r2_score(yt, yp))

# ---- Build combined list ----
classical_map = {r["family"]: r for _, r in classical.iterrows()}
hybrid_map = {r["family"]: r for _, r in hybrid.iterrows()}

# Order: classical first (by CV), then hybrids (by OOF), then 0D
classical_order = ["MLP_sklearn", "OLS", "Lasso_MultiTask", "Lasso_Indep(MultiOutputRegressor)",
                    "Ridge", "ElasticNet_Indep(MultiOutputRegressor)", "ElasticNet_MultiTask",
                    "GB", "RF", "DT"]
hybrid_order = ["KerasMLP_ZohanResidual_Restricted", "KerasMLP_ZohanHRNN_Restricted",
                "KerasMLP_Luc_Restricted", "KerasMLP_FrozenBaseline",
                "KerasMLP_ZohanResidual_Dropout", "KerasMLP_ZohanHPD_Restricted",
                "KerasMLP_DropoutBaseline"]

def get_classical(name):
    return classical_map.get(name)
def get_hybrid(name):
    return hybrid_map.get(name)

def short_name(name):
    return name.replace("KerasMLP_", "").replace("_Restricted", "").replace("(MultiOutputRegressor)", "").replace("_Indep", "").replace("_Dep", "")

# ============================================================================
# TABLE 1: CV RMSE (ALL MODELS)
# ============================================================================
lines = []
def emit(s):
    lines.append(s)

emit(r"\begin{table}[H]")
emit(r"\centering")
emit(r"\caption[RMSE de validação cruzada para todos os modelos]{RMSE de validação cruzada (CV) para todos os modelos avaliados, ordenados pelo desempenho no alvo $Flux$. Fonte: Autor.}")
emit(r"\label{tab:cv_rmse_all}")
emit(r"\begin{tabular}{l c c c}")
emit(r"\hline")
emit(r"\textbf{Modelo} & \textbf{$Flux$} & \textbf{$Alim\_T_{out}$} & \textbf{$Ref\_T_{out}$} \\")
emit(r"\hline")

# Classical
for name in classical_order:
    r = get_classical(name)
    if r is None: continue
    emit(f"{short_name(name)} & {float(r['rmse_cv_Flux']):.3f} & --- & --- \\\\")

emit(r"\hline")
# Hybrids
for name in hybrid_order:
    r = get_hybrid(name)
    if r is None: continue
    emit(f"{short_name(name)} & {abs(float(r['best_score_mean_test'])):.3f} & --- & --- \\\\")

emit(r"\hline")
emit(f"Modelo f\'{{\\i}}sico 0D & --- & --- & --- \\\\")
emit(r"\hline")
emit(r"\end{tabular}")
emit(r"\end{table}")
emit("")

# ============================================================================
# TABLE 2: OOF RMSE (ALL MODELS)
# ============================================================================
emit(r"\begin{table}[H]")
emit(r"\centering")
emit(r"\caption[Desempenho OOF consolidado]{Desempenho consolidado dos modelos avaliados com base nas predições \textit{out-of-fold} (OOF), ordenados pelo RMSE do fluxo de permeado. As métricas OOF são calculadas sobre predições de modelos retreinados independentemente em cada partição da validação cruzada (\cite{Cawley2010Overfitting}). Fonte: Autor.}")
emit(r"\label{tab:oof_all}")
emit(r"\begin{tabular}{l c c c c c c}")
emit(r"\hline")
emit(r"\textbf{Modelo} &")
emit(r"\multicolumn{2}{c}{\textbf{$Alim\_T_{out}$}} &")
emit(r"\multicolumn{2}{c}{\textbf{$Ref\_T_{out}$}} &")
emit(r"\multicolumn{2}{c}{\textbf{$Flux$}} \\")
emit(r"\cline{2-3} \cline{4-5} \cline{6-7}")
emit(r" & R$^2$ & RMSE & R$^2$ & RMSE & R$^2$ & RMSE \\")
emit(r"\hline")

# Classical models
for name in classical_order:
    r = get_classical(name)
    if r is None: continue
    r2_a = r.get("r2_Alim_T_out")
    rm_a = r.get("rmse_Alim_T_out")
    r2_r = r.get("r2_Ref_T_out")
    rm_r = r.get("rmse_Ref_T_out")
    r2_f = r.get("r2_Flux")
    rm_f = r.get("rmse_Flux")
    if pd.isna(r2_f) or pd.isna(rm_f):
        emit(f"{short_name(name)} & --- & --- & --- & --- & --- & --- \\\\")
    else:
        emit(f"{short_name(name)} & {float(r2_a):.3f} & {float(rm_a):.3f} & {float(r2_r):.3f} & {float(rm_r):.3f} & {float(r2_f):.3f} & {float(rm_f):.3f} \\\\")

emit(r"\hline")
# Zero-row space
emit(r"\multicolumn{7}{c}{} \\")
emit(r"\multicolumn{7}{c}{\textbf{Modelos Híbridos}} \\")
emit(r"\multicolumn{7}{c}{} \\")
emit(r"\hline")

# Hybrids sorted by OOF Flux RMSE
hybrid_sorted = sorted([get_hybrid(n) for n in hybrid_order if get_hybrid(n) is not None], key=lambda r: float(r["rmse_Flux"]))
for r in hybrid_sorted:
    if r is None: continue
    nm = short_name(r["family"])
    r2_a = r.get("r2_Alim_T_out")
    rm_a = r.get("rmse_Alim_T_out")
    r2_r = r.get("r2_Ref_T_out")
    rm_r = r.get("rmse_Ref_T_out")
    r2_f = r.get("r2_Flux")
    rm_f = r.get("rmse_Flux")
    tag = r" \rowcolor{orange!15}" if r["family"] == "KerasMLP_ZohanResidual_Restricted" else ""
    if tag:
        emit(tag)
    emit(f"{nm} & {float(r2_a):.3f} & {float(rm_a):.3f} & {float(r2_r):.3f} & {float(rm_r):.3f} & {float(r2_f):.3f} & {float(rm_f):.3f} \\\\")

emit(r"\hline")
emit(r"\multicolumn{7}{c}{} \\")
emit(r"\multicolumn{7}{c}{\textbf{Modelo Físico de Referência}} \\")
emit(r"\multicolumn{7}{c}{} \\")
emit(r"\hline")
emit(f"Modelo f\'{{\\i}}sico 0D & {phy_r2['Alim_T_out']:.3f} & {phy_rmse['Alim_T_out']:.3f} & {phy_r2['Ref_T_out']:.3f} & {phy_rmse['Ref_T_out']:.3f} & {phy_r2['Flux']:.3f} & {phy_rmse['Flux']:.3f} \\\\")
emit(r"\hline")
emit(r"\end{tabular}")
emit(r"\end{table}")

out_path = r"C:\Users\frota\OneDrive\Documentos\TCC\Revisão\TCC_editado\Textual\tables\consolidated_tables.tex"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"Saved to {out_path}")
print("\n".join(lines))

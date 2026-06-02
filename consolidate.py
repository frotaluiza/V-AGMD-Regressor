#!/usr/bin/env python3
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def consolidate_results(base_dir: Path, out_dir: Path, dataset_stem: str = "dados_att_com_var_com_phy"):
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_files = {
        "stage0_classical_models": f"agmd_stage0_classical_{dataset_stem}_summary.csv",
        "stage1_baseline_search": f"agmd_stage1_baseline_{dataset_stem}_summary.csv",
        "stage2_restricted_hybrid_comparison": f"agmd_stage2_restricted_hybrids_{dataset_stem}_summary.csv",
    }

    stage_pretty = {
        "stage0_classical_models": "Stage 0 - Modelos cl\u00e1ssicos",
        "stage1_baseline_search": "Stage 1 - Busca baseline KerasMLP",
        "stage2_restricted_hybrid_comparison": "Stage 2 - H\u00edbridos restritos",
    }

    stage_order = {
        "stage0_classical_models": 0,
        "stage1_baseline_search": 1,
        "stage2_restricted_hybrid_comparison": 2,
    }

    all_frames = []
    for stage_folder, filename in summary_files.items():
        path = base_dir / stage_folder / filename
        if not path.exists():
            print(f"[AVISO] Arquivo n\u00e3o encontrado: {path}")
            continue

        df = pd.read_csv(path)
        df["stage_folder"] = stage_folder
        df["stage_pretty"] = stage_pretty[stage_folder]
        df["stage_order"] = stage_order[stage_folder]
        all_frames.append(df)

    if not all_frames:
        raise FileNotFoundError("Nenhum summary.csv foi encontrado.")

    df_all = pd.concat(all_frames, ignore_index=True)

    for c in [
        "best_overall_mean_test_score", "threshold_1se_score_space",
        "selected_score_minus_threshold", "best_score_mean_test",
        "best_std_test", "best_mean_train", "gap_train_minus_test_score_space",
        "rmse_cv_Flux", "rmse_train_Flux", "gap_rmse_Flux", "complexity",
        "rank_by_score_space", "rank_by_flux_rmse",
    ]:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    for c in ["rmse_Flux", "rmse_Alim_T_out", "rmse_Ref_T_out",
              "r2_Flux", "r2_Alim_T_out", "r2_Ref_T_out"]:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    rmse_col = "rmse_Flux" if "rmse_Flux" in df_all.columns else "rmse_cv_Flux"
    alim_col = "rmse_Alim_T_out" if "rmse_Alim_T_out" in df_all.columns else "rmse_cv_Flux"
    ref_col = "rmse_Ref_T_out" if "rmse_Ref_T_out" in df_all.columns else "rmse_cv_Flux"

    sort_keys = [rmse_col, "gap_rmse_Flux", "complexity"]
    df_ranked = df_all.sort_values(
        by=sort_keys,
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    df_ranked.insert(0, "global_rank_flux", np.arange(1, len(df_ranked) + 1))

    df_best_by_stage = (
        df_all.sort_values(
            by=["stage_order", rmse_col, "gap_rmse_Flux", "complexity"],
            ascending=[True, True, True, True],
            na_position="last",
        )
        .groupby("stage_folder", as_index=False)
        .head(1)
        .sort_values("stage_order")
        .reset_index(drop=True)
    )

    df_top10 = df_ranked.head(10).copy()

    df_all.to_csv(out_dir / "all_models_from_summary.csv", index=False, encoding="utf-8-sig")
    df_ranked.to_csv(out_dir / "ranking_global_by_flux_from_summary.csv", index=False, encoding="utf-8-sig")
    df_best_by_stage.to_csv(out_dir / "best_model_by_stage_from_summary.csv", index=False, encoding="utf-8-sig")
    df_top10.to_csv(out_dir / "top10_global_by_flux_from_summary.csv", index=False, encoding="utf-8-sig")

    latex_cols_raw = [
        "global_rank_flux", "stage_pretty", "family",
        rmse_col, alim_col, ref_col,
        "gap_rmse_Flux", "complexity",
    ]
    # deduplicate keeping order
    seen_col = set()
    latex_cols = []
    for c in latex_cols_raw:
        if c in df_ranked.columns and c not in seen_col:
            latex_cols.append(c)
            seen_col.add(c)

    temp_latex = df_ranked[latex_cols].copy()
    for c in temp_latex.columns:
        if c not in ("global_rank_flux", "stage_pretty", "family"):
            temp_latex[c] = temp_latex[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    latex = temp_latex.to_latex(
        index=False, escape=True,
        caption="Compara\u00e7\u00e3o consolidada dos modelos avaliados.",
        label="tab:comparacao_global_modelos",
    )
    (out_dir / "ranking_global_by_flux_from_summary.tex").write_text(latex, encoding="utf-8")

    # Gr\u00e1ficos
    temp = df_ranked.dropna(subset=[rmse_col]).copy()
    plt.figure(figsize=(12, max(5, 0.45 * len(temp))))
    labels = [s[:38] + "..." if len(str(s)) > 38 else str(s) for s in temp["family"]]
    plt.barh(labels, temp[rmse_col])
    plt.gca().invert_yaxis()
    plt.xlabel("RMSE do Fluxo")
    plt.ylabel("Modelo")
    plt.title("Ranking global dos modelos por RMSE do Fluxo")
    plt.tight_layout()
    plt.savefig(out_dir / "ranking_rmse_flux.png", dpi=300, bbox_inches="tight")
    plt.close()

    temp2 = df_ranked.dropna(subset=[alim_col, ref_col]).copy()
    if len(temp2) > 0:
        x = np.arange(len(temp2))
        width = 0.25
        plt.figure(figsize=(max(12, 0.7 * len(temp2)), 6))
        plt.bar(x - width, temp2[rmse_col], width=width, label="Flux")
        plt.bar(x, temp2[alim_col], width=width, label="Alim_T_out")
        plt.bar(x + width, temp2[ref_col], width=width, label="Ref_T_out")
        labels = [str(s)[:24] + "..." if len(str(s)) > 24 else str(s) for s in temp2["family"]]
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("RMSE")
        plt.title("Compara\u00e7\u00e3o de RMSE por alvo")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "rmse_by_target_grouped.png", dpi=300, bbox_inches="tight")
        plt.close()

    temp3 = df_ranked.dropna(subset=[rmse_col, "complexity"]).copy()
    plt.figure(figsize=(9, 6))
    plt.scatter(temp3["complexity"], temp3[rmse_col])
    for _, row in temp3.iterrows():
        label = str(row["family"])[:20]
        plt.annotate(label, (row["complexity"], row[rmse_col]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Complexidade")
    plt.ylabel("RMSE do Fluxo")
    plt.title("Trade-off entre RMSE do Fluxo e complexidade")
    plt.tight_layout()
    plt.savefig(out_dir / "error_vs_complexity.png", dpi=300, bbox_inches="tight")
    plt.close()

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Resumo da consolida\u00e7\u00e3o\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Base analisada: {base_dir}\n\n")
        f.write("Top 10 modelos por RMSE do Fluxo:\n")
        for _, row in df_top10.iterrows():
            r_flx = row.get(rmse_col, "N/A")
            r_ali = row.get(alim_col, "N/A")
            r_ref = row.get(ref_col, "N/A")
            gap = row.get("gap_rmse_Flux", "N/A")
            r_flx_str = f"{r_flx:.6f}" if isinstance(r_flx, (int, float)) else str(r_flx)
            r_ali_str = f"{r_ali:.6f}" if isinstance(r_ali, (int, float)) else str(r_ali)
            r_ref_str = f"{r_ref:.6f}" if isinstance(r_ref, (int, float)) else str(r_ref)
            gap_str = f"{gap:.6f}" if isinstance(gap, (int, float)) else str(gap)
            f.write(
                f"- Rank {int(row['global_rank_flux'])}: {row['family']} "
                f"[{row['stage_pretty']}] | "
                f"RMSE_Flux={r_flx_str} | "
                f"RMSE_Alim={r_ali_str} | "
                f"RMSE_Ref={r_ref_str} | "
                f"Gap_Flux={gap_str} | "
                f"Complexidade={row['complexity']:.3f}\n"
            )

        f.write("\nMelhor modelo por etapa:\n")
        for _, row in df_best_by_stage.iterrows():
            val = row.get(rmse_col, "N/A")
            val_str = f"{val:.6f}" if isinstance(val, (int, float)) else str(val)
            f.write(f"- {row['stage_pretty']}: {row['family']} (RMSE_Flux={val_str})\n")

    print("=" * 80)
    print("[OK] Consolida\u00e7\u00e3o conclu\u00edda.")
    print(f"Arquivos salvos em: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Consolidate AGMD results.")
    parser.add_argument("--base-dir", type=str, required=True, help="Base results directory.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for consolidated analysis.")
    args = parser.parse_args()

    base = Path(args.base_dir)
    out = Path(args.out_dir) if args.out_dir else base / "consolidated_analysis"
    consolidate_results(base, out)

#!/usr/bin/env python3
"""
Orquestrador que executa o pipeline completo (run_pipeline.py) em paralelo
para dois datasets (normal e aumentado) e gera uma comparação cruzada.

Uso:
  python scripts/run_organizer.py \
      --pipeline-script scripts/run_pipeline.py \
      --csv-normal <caminho_csv_normal> \
      --csv-augmented <caminho_csv_aumentado>

Equivalente a: script_organizador.py
"""

import argparse
import json
import os
import re
import sys
import time
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Run two AGMD pipelines in parallel and compare results.")

    parser.add_argument("--pipeline-script", type=str, required=True,
                        help="Path to the main pipeline script, e.g. run_pipeline.py")
    parser.add_argument("--csv-normal", type=str, required=True,
                        help="Path to the normal dataset CSV.")
    parser.add_argument("--csv-augmented", type=str, required=True,
                        help="Path to the augmented dataset CSV.")
    parser.add_argument("--python-exe", type=str, default=sys.executable,
                        help="Python executable to use. Default: current interpreter.")
    parser.add_argument("--poll-seconds", type=int, default=5,
                        help="Polling interval while waiting for both runs.")
    parser.add_argument("--comparison-outdir", type=str, default="comparison_across_datasets",
                        help="Directory where the cross-dataset comparison will be saved.")

    return parser.parse_args()


@dataclass
class RunSpec:
    label: str
    csv_path: str
    is_augmented: bool


@dataclass
class RunResult:
    label: str
    csv_path: str
    is_augmented: bool
    dataset_stem: str
    returncode: int
    stdout_path: str
    stderr_path: str
    output_dir: Optional[str]
    summary_csv_path: Optional[str]
    best_row: Optional[Dict]
    status: str


def resolve_existing_path(path_str: str, base_dir: str) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = Path(base_dir) / p
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    return str(p)


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def safe_dataset_stem(csv_path: str) -> str:
    return Path(csv_path).stem.replace(" ", "_")


def newest_matching_results_dir(script_dir: str, dataset_stem: str, created_after: float) -> Optional[str]:
    root = Path(script_dir)
    pattern = re.compile(rf"^results_{re.escape(dataset_stem)}_\d{{8}}_\d{{6}}$")

    candidates: List[Tuple[float, str]] = []
    for item in root.iterdir():
        if not item.is_dir():
            continue
        if not pattern.match(item.name):
            continue
        try:
            mtime = item.stat().st_mtime
        except OSError:
            continue
        if mtime >= created_after - 2:
            candidates.append((mtime, str(item)))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_final_summary_csv(results_dir: str, dataset_stem: str) -> Optional[str]:
    root = Path(results_dir)
    preferred = root / "final_comparison" / f"agmd_final_full_comparison_{dataset_stem}_summary.csv"
    if preferred.exists():
        return str(preferred)
    generic = sorted(root.rglob("*summary.csv"))
    if generic:
        return str(generic[0])
    return None


def extract_best_row_from_summary(summary_csv_path: str) -> Dict:
    df = pd.read_csv(summary_csv_path)

    if df.empty:
        raise ValueError(f"Summary CSV is empty: {summary_csv_path}")

    if "rmse_cv_Flux" not in df.columns:
        raise KeyError(f"'rmse_cv_Flux' not found in {summary_csv_path}")

    work = df.copy()
    sort_cols = ["rmse_cv_Flux"]
    ascending = [True]

    if "gap_rmse_Flux" in work.columns:
        sort_cols.append("gap_rmse_Flux")
        ascending.append(True)

    if "complexity" in work.columns:
        sort_cols.append("complexity")
        ascending.append(True)

    if "r2_Flux" in work.columns:
        work["_neg_r2_Flux"] = -pd.to_numeric(work["r2_Flux"], errors="coerce")
        sort_cols.append("_neg_r2_Flux")
        ascending.append(True)

    best = work.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict()

    if "_neg_r2_Flux" in best:
        del best["_neg_r2_Flux"]

    return best


def launch_pipeline(run_spec: RunSpec, *, pipeline_script: str, python_exe: str, workdir: str, logs_dir: str):
    dataset_stem = safe_dataset_stem(run_spec.csv_path)

    stdout_path = os.path.join(logs_dir, f"{run_spec.label}_{dataset_stem}_stdout.log")
    stderr_path = os.path.join(logs_dir, f"{run_spec.label}_{dataset_stem}_stderr.log")

    stdout_f = open(stdout_path, "w", encoding="utf-8")
    stderr_f = open(stderr_path, "w", encoding="utf-8")

    cmd = [python_exe, pipeline_script, "--csv", run_spec.csv_path]

    print(f"[LAUNCH] {run_spec.label}: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(cmd, cwd=workdir, stdout=stdout_f, stderr=stderr_f)

    start_time = time.time()
    return proc, start_time, stdout_path, stderr_path


def wait_for_processes(process_map: Dict[str, Dict], poll_seconds: int = 5):
    pending = set(process_map.keys())

    while pending:
        finished_now = []
        for key in list(pending):
            proc = process_map[key]["proc"]
            ret = proc.poll()
            if ret is not None:
                finished_now.append((key, ret))

        for key, ret in finished_now:
            elapsed = time.time() - process_map[key]["start_time"]
            print(f"[DONE] {key} finished with code={ret} in {elapsed:.1f}s", flush=True)
            pending.remove(key)

        if pending:
            print(f"[WAIT] still running: {', '.join(sorted(pending))}", flush=True)
            time.sleep(poll_seconds)


def collect_run_result(run_spec: RunSpec, *, proc, start_time, stdout_path, stderr_path, pipeline_script: str):
    dataset_stem = safe_dataset_stem(run_spec.csv_path)
    returncode = proc.returncode if proc.returncode is not None else -999

    script_dir = str(Path(pipeline_script).resolve().parent.parent)
    output_dir = newest_matching_results_dir(script_dir=script_dir, dataset_stem=dataset_stem, created_after=start_time)

    summary_csv_path = None
    best_row = None
    status = "ok" if returncode == 0 else "failed"

    if output_dir is not None:
        summary_csv_path = find_final_summary_csv(output_dir, dataset_stem)

    if returncode == 0 and summary_csv_path is not None:
        try:
            best_row = extract_best_row_from_summary(summary_csv_path)
        except Exception as e:
            status = f"summary_parse_failed: {repr(e)}"
    elif returncode == 0 and summary_csv_path is None:
        status = "summary_not_found"

    return RunResult(
        label=run_spec.label,
        csv_path=run_spec.csv_path,
        is_augmented=run_spec.is_augmented,
        dataset_stem=dataset_stem,
        returncode=returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        output_dir=output_dir,
        summary_csv_path=summary_csv_path,
        best_row=best_row,
        status=status,
    )


def build_cross_dataset_comparison(results: List[RunResult]) -> pd.DataFrame:
    rows = []

    for r in results:
        if r.best_row is None:
            rows.append({
                "dataset_label": r.label,
                "csv_path": r.csv_path,
                "is_augmented": r.is_augmented,
                "status": r.status,
                "dataset_stem": r.dataset_stem,
                "family": None,
                "rmse_cv_Flux": None,
                "rmse_train_Flux": None,
                "gap_rmse_Flux": None,
                "complexity": None,
                "r2_Flux": None,
                "summary_csv_path": r.summary_csv_path,
                "output_dir": r.output_dir,
            })
            continue

        b = r.best_row
        rows.append({
            "dataset_label": r.label,
            "csv_path": r.csv_path,
            "is_augmented": r.is_augmented,
            "status": r.status,
            "dataset_stem": r.dataset_stem,
            "family": b.get("family"),
            "rmse_cv_Flux": b.get("rmse_cv_Flux"),
            "rmse_train_Flux": b.get("rmse_train_Flux"),
            "gap_rmse_Flux": b.get("gap_rmse_Flux"),
            "complexity": b.get("complexity"),
            "r2_Flux": b.get("r2_Flux"),
            "summary_csv_path": r.summary_csv_path,
            "output_dir": r.output_dir,
        })

    df = pd.DataFrame(rows)

    valid = df["rmse_cv_Flux"].notna()
    if valid.any():
        rank_df = df.loc[valid].copy()
        sort_cols = ["rmse_cv_Flux"]
        ascending = [True]

        if "gap_rmse_Flux" in rank_df.columns:
            sort_cols.append("gap_rmse_Flux")
            ascending.append(True)
        if "complexity" in rank_df.columns:
            sort_cols.append("complexity")
            ascending.append(True)
        if "r2_Flux" in rank_df.columns:
            rank_df["_neg_r2_Flux"] = -pd.to_numeric(rank_df["r2_Flux"], errors="coerce")
            sort_cols.append("_neg_r2_Flux")
            ascending.append(True)

        rank_df = rank_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
        rank_df["global_rank"] = range(1, len(rank_df) + 1)

        df = df.merge(rank_df[["dataset_label", "global_rank"]], on="dataset_label", how="left")
    else:
        df["global_rank"] = None

    return df


def choose_best_global_experiment(comparison_df: pd.DataFrame) -> Optional[Dict]:
    valid = comparison_df[comparison_df["rmse_cv_Flux"].notna()].copy()
    if valid.empty:
        return None

    sort_cols = ["rmse_cv_Flux"]
    ascending = [True]

    if "gap_rmse_Flux" in valid.columns:
        sort_cols.append("gap_rmse_Flux")
        ascending.append(True)
    if "complexity" in valid.columns:
        sort_cols.append("complexity")
        ascending.append(True)
    if "r2_Flux" in valid.columns:
        valid["_neg_r2_Flux"] = -pd.to_numeric(valid["r2_Flux"], errors="coerce")
        sort_cols.append("_neg_r2_Flux")
        ascending.append(True)

    best = valid.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict()
    if "_neg_r2_Flux" in best:
        del best["_neg_r2_Flux"]

    return best


def main():
    args = parse_args()

    pipeline_script = str(Path(args.pipeline_script).resolve())
    if not Path(pipeline_script).exists():
        raise FileNotFoundError(f"Pipeline script not found: {pipeline_script}")

    workdir = str(Path(pipeline_script).resolve().parent.parent)
    csv_normal = resolve_existing_path(args.csv_normal, workdir)
    csv_augmented = resolve_existing_path(args.csv_augmented, workdir)

    comparison_outdir = ensure_dir(str(Path(args.comparison_outdir).resolve()))
    logs_dir = ensure_dir(os.path.join(comparison_outdir, "logs"))

    runs = [
        RunSpec(label="normal", csv_path=csv_normal, is_augmented=False),
        RunSpec(label="augmented", csv_path=csv_augmented, is_augmented=True),
    ]

    process_map = {}
    for run in runs:
        proc, start_time, stdout_path, stderr_path = launch_pipeline(
            run, pipeline_script=pipeline_script,
            python_exe=args.python_exe, workdir=workdir, logs_dir=logs_dir,
        )
        process_map[run.label] = {
            "proc": proc,
            "start_time": start_time,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "run_spec": run,
        }

    wait_for_processes(process_map, poll_seconds=args.poll_seconds)

    collected = []
    for label, info in process_map.items():
        result = collect_run_result(
            info["run_spec"], proc=info["proc"], start_time=info["start_time"],
            stdout_path=info["stdout_path"], stderr_path=info["stderr_path"],
            pipeline_script=pipeline_script,
        )
        collected.append(result)

    comparison_df = build_cross_dataset_comparison(collected)
    best_global = choose_best_global_experiment(comparison_df)

    comparison_csv = os.path.join(comparison_outdir, "cross_dataset_comparison_summary.csv")
    comparison_df.to_csv(comparison_csv, index=False)

    metadata_json = os.path.join(comparison_outdir, "cross_dataset_run_metadata.json")
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in collected], f, ensure_ascii=False, indent=2)

    best_json = os.path.join(comparison_outdir, "cross_dataset_best_pipeline.json")
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(best_global, f, ensure_ascii=False, indent=2)

    print("\n=== CROSS-DATASET COMPARISON ===", flush=True)
    show_cols = ["dataset_label", "is_augmented", "family", "rmse_cv_Flux",
                  "rmse_train_Flux", "gap_rmse_Flux", "complexity", "r2_Flux",
                  "global_rank", "status"]
    existing = [c for c in show_cols if c in comparison_df.columns]
    print(comparison_df[existing].to_string(index=False), flush=True)

    print("\n=== BEST GLOBAL PIPELINE ===", flush=True)
    if best_global is None:
        print("No valid best pipeline could be determined.", flush=True)
    else:
        for k, v in best_global.items():
            print(f"{k}: {v}", flush=True)

    print(f"\nComparison files saved in: {comparison_outdir}", flush=True)


if __name__ == "__main__":
    main()

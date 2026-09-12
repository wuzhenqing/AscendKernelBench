#!/usr/bin/env python
"""Aggregate runs/{run_name}/eval_results.json into fast_p / pass@k reports.

Example:
    python scripts/analyze.py --run-name dev_run
"""

from __future__ import annotations

import argparse
import json
import sys

import _bootstrap  # noqa: F401
from rich.console import Console
from rich.table import Table

from ascend_kernel_bench import rundir
from ascend_kernel_bench.io_util import read_json_object
from ascend_kernel_bench.score import (
    compute_pass_at_k,
    sample_speedup,
    summarize_eval_results,
)

console = Console()


def _summary_table(run_name: str, summary: dict, pass_at_k: dict) -> Table:
    """Build the headline metrics table for a run."""
    table = Table(title=f"AscendKernelBench report: {run_name}")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("problems", str(summary["total_problems"]))
    table.add_row("samples", str(summary["total_samples"]))
    table.add_row(
        "compiled",
        f"{summary['compiled']} "
        f"({summary['compiled'] / max(summary['total_samples'], 1):.1%})",
    )
    table.add_row("correct (fast_0 denominator)", str(summary["correct"]))
    table.add_row("npu-reference samples", str(summary.get("npu_reference", 0)))
    table.add_row("cpu-reference samples", str(summary.get("cpu_reference", 0)))
    table.add_row(
        "flagged excessive speedup",
        str(summary.get("excessive_speedup", 0)),
    )
    for key, value in summary["fast_p"].items():
        table.add_row(key, f"{value:.3f}")
    table.add_row(
        "geomean speedup (correct only)",
        f"{summary['geometric_mean_speedup_correct_only']:.3f}",
    )
    sol = summary.get("mean_sol_score")
    table.add_row(
        "mean SOL score (roofline)",
        f"{sol:.3f}" if isinstance(sol, (int, float)) else "-",
    )
    for key, value in pass_at_k["average"].items():
        table.add_row(key, f"{value:.3f}")
    return table


def _per_problem_table(eval_results: dict) -> Table:
    """Build the per-problem compiled/correct/speedup table."""
    table = Table(title="per-problem detail")
    table.add_column("problem", style="bold")
    table.add_column("samples", justify="right")
    table.add_column("compiled", justify="right")
    table.add_column("correct", justify="right")
    table.add_column("best speedup", justify="right")
    for problem_id, samples in sorted(eval_results.items()):
        compiled = sum(1 for sample in samples if sample.get("compiled"))
        correct = sum(1 for sample in samples if sample.get("correctness"))
        speedups = [s for s in (sample_speedup(x) for x in samples) if s]
        best = f"{max(speedups):.2f}x" if speedups else "-"
        style = "green" if correct else ("yellow" if compiled else "red")
        table.add_row(
            f"[{style}]{problem_id}[/{style}]",
            str(len(samples)),
            str(compiled),
            str(correct),
            best,
        )
    return table


def main() -> None:
    """Print fast_p, pass@k, and SOL summary for a saved run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    run_dir = rundir.RUNS_DIR / args.run_name
    results_path = run_dir / "eval_results.json"
    if not results_path.is_file():
        sys.exit(f"not found: {results_path} (run scripts/evaluate.py first)")
    try:
        eval_results = read_json_object(results_path)
    except (json.JSONDecodeError, ValueError) as exc:
        sys.exit(f"invalid eval_results.json: {exc}")

    summary = summarize_eval_results(eval_results)
    pass_at_k = compute_pass_at_k(eval_results)
    console.print(_summary_table(args.run_name, summary, pass_at_k))
    console.print(_per_problem_table(eval_results))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Aggregate runs/{run_name}/eval_results.json into fast_p / pass@k reports.

Example:
    python scripts/analyze.py --run-name dev_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.table import Table

from ascend_kernel_bench import rundir
from ascend_kernel_bench.score import (
    compute_pass_at_k,
    sample_speedup,
    summarize_eval_results,
)

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    run_dir = rundir.RUNS_DIR / args.run_name
    results_path = run_dir / "eval_results.json"
    if not results_path.is_file():
        sys.exit(f"not found: {results_path} (run scripts/evaluate.py first)")
    eval_results = json.loads(results_path.read_text(encoding="utf-8"))

    summary = summarize_eval_results(eval_results)
    pass_at_k = compute_pass_at_k(eval_results)

    table = Table(title=f"AscendKernelBench report: {args.run_name}")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("problems", str(summary["total_problems"]))
    table.add_row("samples", str(summary["total_samples"]))
    table.add_row("compiled", f"{summary['compiled']} "
                              f"({summary['compiled'] / max(summary['total_samples'], 1):.1%})")
    table.add_row("correct (fast_0 denominator)", str(summary["correct"]))
    for key, value in summary["fast_p"].items():
        table.add_row(key, f"{value:.3f}")
    table.add_row("geomean speedup (correct only)",
                  f"{summary['geometric_mean_speedup_correct_only']:.3f}")
    for key, value in pass_at_k["average"].items():
        table.add_row(key, f"{value:.3f}")
    console.print(table)

    per_problem = Table(title="per-problem detail")
    per_problem.add_column("problem", style="bold")
    per_problem.add_column("samples", justify="right")
    per_problem.add_column("compiled", justify="right")
    per_problem.add_column("correct", justify="right")
    per_problem.add_column("best speedup", justify="right")
    for problem_id, samples in sorted(eval_results.items()):
        n = len(samples)
        compiled = sum(1 for s in samples if s.get("compiled"))
        correct = sum(1 for s in samples if s.get("correctness"))
        speedups = [s for s in (sample_speedup(x) for x in samples) if s]
        best = f"{max(speedups):.2f}x" if speedups else "-"
        style = "green" if correct else ("yellow" if compiled else "red")
        per_problem.add_row(f"[{style}]{problem_id}[/{style}]",
                            str(n), str(compiled), str(correct), best)
    console.print(per_problem)


if __name__ == "__main__":
    main()

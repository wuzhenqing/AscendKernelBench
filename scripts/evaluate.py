#!/usr/bin/env python
"""Batch evaluation of generated samples in runs/{run_name}/.

Each sample is evaluated in an isolated worker subprocess (build -> 5 seeded
correctness trials -> NPU-Event timing vs the torch_npu eager reference).
Writes per-sample eval_result.json, then aggregates eval_results.json and
pass_at_k_results.json.

Example:
    python scripts/evaluate.py --run-name dev_run --device npu:0
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from rich.console import Console

from ascend_kernel_bench import rundir
from ascend_kernel_bench.cli_util import (
    cli_progress,
    eval_result_lines,
    load_eval_runtime,
    sample_status_label,
)
from ascend_kernel_bench.eval import evaluate_run
from ascend_kernel_bench.score import summarize_eval_results

console = Console()


def main() -> None:
    """Evaluate every generated sample in a run directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--hardware", default=None)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--no-perf", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    runtime = load_eval_runtime(
        config_path=args.config,
        hardware=args.hardware,
    )
    config, hardware = runtime.config, runtime.hardware
    run_dir = rundir.RUNS_DIR / args.run_name
    if not run_dir.is_dir():
        sys.exit(f"run dir not found: {run_dir}")

    samples = list(rundir.iter_sample_dirs(run_dir))
    if not samples:
        sys.exit(f"no samples in {run_dir}")
    console.print(
        f"evaluating {len(samples)} samples from {run_dir} on {args.device}"
    )

    with cli_progress(console) as progress:
        bar = progress.add_task("evaluating", total=len(samples))

        def on_sample(
            task_id: str, sample_id: int, result: dict[str, object]
        ) -> None:
            progress.update(bar, description=f"{task_id} s{sample_id}")
            style, label = sample_status_label(result)
            progress.console.print(
                f"  [{style}]{task_id} s{sample_id}: {label}[/{style}]"
            )
            _, lines = eval_result_lines(result)
            for line in lines:
                if line.startswith("error:"):
                    progress.console.print(f"    {line}")
            progress.advance(bar)

        results = evaluate_run(
            run_dir,
            hardware=hardware,
            config=config,
            device=args.device,
            measure_performance=not args.no_perf,
            on_sample=on_sample,
        )

    console.print(f"wrote {run_dir / 'eval_results.json'}")
    summary = summarize_eval_results(results)
    sol = summary.get("mean_sol_score")
    sol_text = f"{sol:.3f}" if isinstance(sol, int | float) else "-"
    console.print(
        f"compiled {summary['compiled']}/{summary['total_samples']}, "
        f"correct {summary['correct']}/{summary['total_samples']}, "
        f"fast_p {summary['fast_p']}, mean SOL {sol_text}"
    )


if __name__ == "__main__":
    main()

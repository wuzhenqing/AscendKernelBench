#!/usr/bin/env python
"""Evaluate generated samples in a run.

Each sample is evaluated in an isolated worker subprocess (build -> 5 seeded
correctness trials -> NPU-Event timing vs the torch_npu eager reference).
Writes per-sample eval_result.json, then aggregates eval_results.json and
pass_at_k_results.json.

Examples:
    python scripts/evaluate.py relu_demo
    python scripts/evaluate.py relu_demo 1
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
    print_eval_report,
    sample_status_label,
)
from ascend_kernel_bench.eval import evaluate_run

console = Console()


def main() -> None:
    """Evaluate complete samples in a run directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="run name under runs/, or an existing path")
    parser.add_argument(
        "level",
        nargs="?",
        type=int,
        help="evaluate only this level (omit to evaluate every level)",
    )
    args = parser.parse_args()

    try:
        run_dir = rundir.resolve_run(args.run)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    samples = list(rundir.iter_sample_dirs(run_dir, level=args.level))
    if not samples:
        if args.level is not None:
            sys.exit(f"no samples in {run_dir} for level {args.level}")
        sys.exit(f"no samples in {run_dir}")

    if args.level is not None:
        scope = f"level {args.level} of {run_dir}"
    else:
        scope = str(run_dir)
    console.print(f"evaluating {len(samples)} samples from {scope}")

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

        results = evaluate_run(run_dir, args.level, on_sample=on_sample)

    console.print(f"wrote {run_dir / 'eval_results.json'}")
    print_eval_report(console, run_dir.name, results)


if __name__ == "__main__":
    main()

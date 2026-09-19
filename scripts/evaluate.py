#!/usr/bin/env python
"""Evaluate generated samples in a run directory.

Each sample runs in an isolated worker subprocess (build, correctness trials,
NPU-event timing) and lands in eval_results.json.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from loguru import logger
from rich.console import Console

from ascend_kernel_bench import rundir
from ascend_kernel_bench.cli_util import (
    cli_progress,
    eval_result_lines,
    print_eval_report,
    sample_status_label,
)
from ascend_kernel_bench.eval import evaluate_run
from ascend_kernel_bench.log import die, setup_logging

console = Console()


def main() -> None:
    """Evaluate complete samples in a run directory."""
    setup_logging()
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
        die(str(exc))

    samples = list(rundir.iter_sample_dirs(run_dir, level=args.level))
    if not samples:
        if args.level is not None:
            die(f"no samples in {run_dir} for level {args.level}")
        die(f"no samples in {run_dir}")

    if args.level is not None:
        scope = f"level {args.level} of {run_dir}"
    else:
        scope = str(run_dir)
    logger.info("evaluating {} samples from {}", len(samples), scope)

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

    logger.info("wrote {}", run_dir / "eval_results.json")
    print_eval_report(console, run_dir.name, results)


if __name__ == "__main__":
    main()

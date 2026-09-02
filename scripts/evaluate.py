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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ascend_kernel_bench import rundir
from ascend_kernel_bench.config import load_eval_config, load_hardware_profile
from ascend_kernel_bench.dataset import load_task
from ascend_kernel_bench.eval import eval_sample
from ascend_kernel_bench.score import compute_pass_at_k

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--hardware", default=None)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--no-perf", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_eval_config(args.config)
    hardware = load_hardware_profile(args.hardware or config.hardware)
    run_dir = rundir.RUNS_DIR / args.run_name
    if not run_dir.is_dir():
        sys.exit(f"run dir not found: {run_dir}")

    samples = list(rundir.iter_sample_dirs(run_dir))
    if not samples:
        sys.exit(f"no samples in {run_dir}")
    console.print(f"evaluating {len(samples)} samples from {run_dir} on {args.device}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task("evaluating", total=len(samples))
        for task_id, sample_id, sdir in samples:
            progress.update(bar, description=f"{task_id} s{sample_id}")
            task = load_task(task_id)
            result = eval_sample(
                task, sdir,
                hardware=hardware, config=config,
                device=args.device, measure_performance=not args.no_perf,
            )
            mark = "green OK" if result["correctness"] else (
                "yellow COMPILE-FAIL" if not result["compiled"] else "red WRONG")
            progress.console.print(f"  [{mark.split()[0]}]{task_id} s{sample_id}: "
                                   f"{mark.split()[1]}[/{mark.split()[0]}]")
            progress.advance(bar)

    results = rundir.collect_eval_results(run_dir)
    rundir.write_eval_results(run_dir, results)
    rundir.write_pass_at_k(run_dir, compute_pass_at_k(results))
    console.print(f"wrote {run_dir / 'eval_results.json'}")
    from ascend_kernel_bench.score import summarize_eval_results
    summary = summarize_eval_results(results)
    console.print(f"compiled {summary['compiled']}/{summary['total_samples']}, "
                  f"correct {summary['correct']}/{summary['total_samples']}, "
                  f"fast_p {summary['fast_p']}")


if __name__ == "__main__":
    main()

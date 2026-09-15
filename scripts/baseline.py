#!/usr/bin/env python
"""Measure torch_npu eager baselines and archive them per hardware.

Baselines are measured on this machine (docs/guide/evaluation.md) and
archived to results/baseline/{hardware}/{task_slug}.json so runs stay
comparable across sessions. Evaluation itself always re-measures the
reference live in the same worker; the archive is the cross-run record.

Example:
    python scripts/baseline.py --level 1 --hardware ascend910b2 --device npu:0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from loguru import logger
from rich.console import Console

from ascend_kernel_bench._paths import BASELINE_DIR
from ascend_kernel_bench.cli_util import (
    cli_progress,
    load_eval_runtime,
    select_tasks,
)
from ascend_kernel_bench.config import EvalConfig
from ascend_kernel_bench.dataset import Task
from ascend_kernel_bench.io_util import write_json_atomic
from ascend_kernel_bench.log import die, setup_logging
from ascend_kernel_bench.process import IsolatedJsonWorker
from ascend_kernel_bench.timing import l2_clear_bytes

console = Console()


def measure_baseline(
    task: Task,
    *,
    config: EvalConfig,
    device: str,
    out_path: Path,
    l2_clear_size: int,
) -> dict:
    """Measure one task's reference runtime in an isolated subprocess.

    Args:
        task: Loaded KernelBench task.
        config: Evaluation timeouts, seed, precision, and trial counts.
        device: Runtime device string, for example ``npu:0``.
        out_path: Archive path under ``results/baseline/``.
        l2_clear_size: L2 flush buffer size forwarded to the worker.

    Returns:
        Timing statistics written to ``out_path``.

    Raises:
        RuntimeError: If the worker process exits non-zero.
        subprocess.TimeoutExpired: If the worker exceeds ``eval_timeout``.
    """
    outcome = IsolatedJsonWorker.for_baseline(config.eval_timeout).run(
        {
            "task_py": task.task_py,
            "device": device,
            "precision": config.precision,
            "seed": config.seed,
            "num_warmup": config.num_warmup,
            "num_perf_trials": config.num_perf_trials,
            "l2_clear_size": l2_clear_size,
        }
    )
    if outcome.returncode != 0:
        raise RuntimeError(f"baseline worker failed: {outcome.stderr[-1000:]}")
    if outcome.payload is None:
        raise RuntimeError(
            f"invalid baseline worker JSON: {outcome.parse_error}"
        )
    stats = outcome.payload
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_path, stats)
    return stats


def main() -> None:
    """Archive eager ``torch_npu`` baselines for the selected tasks."""
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--hardware", default=None)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    runtime = load_eval_runtime(config_path=args.config, hardware=args.hardware)
    config, hardware = runtime.config, runtime.hardware
    tasks = select_tasks(level=args.level, task_ids=args.task)
    if not tasks:
        die("no tasks found")

    out_dir = BASELINE_DIR / hardware.name
    flush = l2_clear_bytes(hardware.l2_cache_mb)
    logger.info(
        "measuring {} baselines on {} -> {}",
        len(tasks),
        hardware.name,
        out_dir,
    )
    with cli_progress(console) as progress:
        bar = progress.add_task("baseline", total=len(tasks))
        for task in tasks:
            progress.update(bar, description=task.task_id)
            out_path = out_dir / f"{task.name}.json"
            try:
                stats = measure_baseline(
                    task,
                    config=config,
                    device=args.device,
                    out_path=out_path,
                    l2_clear_size=flush,
                )
                if stats.get("supported_on_npu", True):
                    progress.console.print(
                        f"  {task.task_id}: {stats['mean']:.4f} ms"
                    )
                else:
                    progress.console.print(
                        f"  [yellow]{task.task_id}: not supported on NPU"
                        f" (CPU-reference task)[/yellow]"
                    )
            except Exception as exc:
                logger.error("{}: {}", task.task_id, exc)
                progress.console.print(f"  [red]{task.task_id}: {exc}[/red]")
            progress.advance(bar)


if __name__ == "__main__":
    main()

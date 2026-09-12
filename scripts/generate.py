#!/usr/bin/env python
"""Batch generation: n samples per task, saved under runs/{run_name}/.

Generation is decoupled from evaluation (docs/guide/workflows.md):
samples land on disk and can be evaluated later, including on another
machine.

Example:
    python scripts/generate.py --level 1 --n-samples 2 --run-name dev_run
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

import _bootstrap  # noqa: F401
from rich.console import Console

from ascend_kernel_bench import rundir
from ascend_kernel_bench.cli_util import (
    add_operator_mode_argument,
    cli_progress,
    generation_run_config,
    load_eval_runtime,
    resolve_generation_settings,
    select_tasks,
)
from ascend_kernel_bench.llm import LLMClient
from ascend_kernel_bench.modes import OperatorModeError
from ascend_kernel_bench.prompt import SYSTEM_PROMPT, build_prompt

console = Console()


def main() -> None:
    """Generate one or more samples per selected task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="task id(s); overrides --level",
    )
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--hardware", default=None)
    parser.add_argument(
        "--prompt-mode",
        default=None,
        choices=["zero_shot", "one_shot", "few_shot"],
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--config", default=None)
    add_operator_mode_argument(parser)
    args = parser.parse_args()

    try:
        runtime = load_eval_runtime(
            config_path=args.config,
            hardware=args.hardware,
            operator_mode=args.operator_mode,
        )
    except OperatorModeError as exc:
        sys.exit(str(exc))
    config, hardware = runtime.config, runtime.hardware
    settings = resolve_generation_settings(
        config,
        model=args.model,
        prompt_mode=args.prompt_mode,
        temperature=args.temperature,
        num_samples=args.n_samples,
    )
    operator_mode = config.operator_mode

    tasks = select_tasks(level=args.level, task_ids=args.task)
    if not tasks:
        sys.exit("no tasks found")

    run_name = args.run_name or f"gen_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = rundir.create_run(
        run_name,
        generation_run_config(
            settings,
            hardware_name=hardware.name,
            operator_mode=operator_mode,
            task_ids=[task.task_id for task in tasks],
        ),
    )
    console.print(
        f"run dir: {run_dir}  "
        f"({len(tasks)} tasks x {settings.num_samples} samples)"
    )

    client = LLMClient(
        settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
    total = len(tasks) * settings.num_samples
    failures = 0
    with cli_progress(console) as progress:
        bar = progress.add_task("generating", total=total)
        for task in tasks:
            prompt = build_prompt(
                task,
                hardware,
                mode=settings.prompt_mode,
                operator_mode=operator_mode,
            )
            for sample_id in range(settings.num_samples):
                progress.update(bar, description=f"{task.task_id} s{sample_id}")
                try:
                    result = client.generate(prompt, system=SYSTEM_PROMPT)
                    rundir.save_sample(
                        run_dir,
                        task.task_id,
                        sample_id,
                        prompt=prompt,
                        generation=result.generation,
                        raw_response=result.raw_text,
                    )
                except Exception as exc:
                    failures += 1
                    console.print(
                        f"[red]generate failed {task.task_id} "
                        f"sample {sample_id}: {exc}[/red]"
                    )
                progress.advance(bar)
    console.print(
        f"done: {total - failures}/{total} samples saved to {run_dir}"
    )
    sys.exit(1 if failures == total else 0)


if __name__ == "__main__":
    main()

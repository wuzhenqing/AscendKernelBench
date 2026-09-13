#!/usr/bin/env python
"""Single-task end-to-end run: prompt -> generate -> build -> evaluate.

Example:
    python scripts/run_single.py --task level1/19_ReLU --model deepseek-v4-flash
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from rich.console import Console
from rich.panel import Panel

from ascend_kernel_bench import rundir
from ascend_kernel_bench.cli_util import (
    GenerationSettings,
    eval_result_lines,
    generation_run_config,
    load_eval_runtime,
    resolve_generation_settings,
)
from ascend_kernel_bench.config import EvalConfig, HardwareProfile
from ascend_kernel_bench.dataset import Task, load_task
from ascend_kernel_bench.eval import eval_sample
from ascend_kernel_bench.llm import LLMClient
from ascend_kernel_bench.prompt import SYSTEM_PROMPT, build_prompt

console = Console()


def _build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for a single-task run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", required=True, help="task id, e.g. level1/19_ReLU"
    )
    parser.add_argument(
        "--model", default=None, help="override generation model"
    )
    parser.add_argument(
        "--hardware", default=None, help="hardware profile name"
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--run-name", default=None, help="default: single_{task}"
    )
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument(
        "--prompt-mode",
        default=None,
        choices=["zero_shot", "one_shot", "few_shot"],
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--no-perf", action="store_true", help="skip timing")
    parser.add_argument("--config", default=None, help="eval config yaml")
    return parser


def _generate_sample(
    task: Task,
    hardware: HardwareProfile,
    settings: GenerationSettings,
    run_dir: Path,
    sample_id: int,
) -> Path:
    """Generate one sample and persist it under ``run_dir``."""
    prompt = build_prompt(
        task,
        hardware,
        mode=settings.prompt_mode,
    )
    console.print(
        f"prompt: {len(prompt)} chars, mode={settings.prompt_mode}, "
        f"model={settings.model}"
    )
    client = LLMClient(
        settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
    with console.status("[bold green]Generating with LLM..."):
        result = client.generate(prompt, system=SYSTEM_PROMPT)
    console.print(
        f"generation: asc={len(result.generation.custom_op_asc)} chars, "
        f"model_new={len(result.generation.model_new_py)} chars"
    )
    sample_path = rundir.save_sample(
        run_dir,
        task.task_id,
        sample_id,
        prompt=prompt,
        generation=result.generation,
        raw_response=result.raw_text,
    )
    console.print(f"sample saved to {sample_path}")
    return sample_path


def _evaluate_sample(
    task: Task,
    sample_path: Path,
    *,
    hardware: HardwareProfile,
    config: EvalConfig,
    device: str,
    measure_performance: bool,
) -> dict[str, object]:
    """Build and evaluate one generated sample on the NPU."""
    with console.status("[bold cyan]Building + evaluating on NPU..."):
        return eval_sample(
            task,
            sample_path,
            hardware=hardware,
            config=config,
            device=device,
            measure_performance=measure_performance,
        )


def _print_eval_panel(eval_result: dict[str, object]) -> None:
    """Print the per-sample evaluation panel."""
    style, lines = eval_result_lines(eval_result)
    console.print(
        Panel("\n".join(lines), title="eval_result", border_style=style)
    )


def main() -> None:
    """Generate and evaluate a single task end to end."""
    args = _build_parser().parse_args()

    runtime = load_eval_runtime(
        config_path=args.config,
        hardware=args.hardware,
    )
    config, hardware = runtime.config, runtime.hardware
    settings = resolve_generation_settings(
        config,
        model=args.model,
        prompt_mode=args.prompt_mode,
        temperature=args.temperature,
    )

    task = load_task(args.task)
    run_name = args.run_name or f"single_{task.name}"
    run_dir = rundir.create_run(
        run_name,
        generation_run_config(
            settings,
            hardware_name=hardware.name,
            task_ids=[task.task_id],
            device=args.device,
        ),
    )

    console.rule(f"[bold]{task.task_id}[/bold] on {hardware.name}")
    sample_path = _generate_sample(
        task,
        hardware,
        settings,
        run_dir,
        args.sample_id,
    )
    eval_result = _evaluate_sample(
        task,
        sample_path,
        hardware=hardware,
        config=config,
        device=args.device,
        measure_performance=not args.no_perf,
    )
    _print_eval_panel(eval_result)

    results = rundir.collect_eval_results(run_dir)
    rundir.write_eval_results(run_dir, results)
    console.print(f"results: {run_dir / 'eval_results.json'}")
    sys.exit(0 if eval_result.get("correctness") else 1)


if __name__ == "__main__":
    main()

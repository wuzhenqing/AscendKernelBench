#!/usr/bin/env python
"""Batch generation: LLM generates n samples per task, saved to runs/{run_name}/.

Generation is decoupled from evaluation (README 3.6): samples land on disk
and can be evaluated repeatedly, on this or another machine.

Example:
    python scripts/generate.py --level 1 --n-samples 2 --run-name dev_run
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ascend_kernel_bench import rundir
from ascend_kernel_bench.config import load_eval_config, load_hardware_profile
from ascend_kernel_bench.dataset import discover_tasks, load_task
from ascend_kernel_bench.llm import LLMClient
from ascend_kernel_bench.prompt import SYSTEM_PROMPT, build_prompt

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--task", action="append", default=None,
                        help="task id(s); overrides --level")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--hardware", default=None)
    parser.add_argument("--prompt-mode", default=None,
                        choices=["zero_shot", "one_shot", "few_shot"])
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_eval_config(args.config)
    hardware = load_hardware_profile(args.hardware or config.hardware)
    gen_cfg = dict(config.generation)
    model = args.model or gen_cfg.get("model", "deepseek-v4-flash")
    n_samples = args.n_samples or int(gen_cfg.get("num_samples", 1))
    prompt_mode = args.prompt_mode or gen_cfg.get("prompt_mode", "one_shot")
    temperature = args.temperature
    if temperature is None:
        temperature = float(gen_cfg.get("temperature", 0.0))
    max_tokens = int(gen_cfg.get("max_tokens", 16384))

    if args.task:
        tasks = [load_task(t) for t in args.task]
    elif args.level is not None:
        tasks = discover_tasks(level=args.level)
    else:
        tasks = discover_tasks()
    if not tasks:
        sys.exit("no tasks found")

    run_name = args.run_name or f"gen_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = rundir.create_run(run_name, {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_mode": prompt_mode,
        "num_samples": n_samples,
        "hardware": hardware.name,
        "tasks": [t.task_id for t in tasks],
    })
    console.print(f"run dir: {run_dir}  ({len(tasks)} tasks x {n_samples} samples)")

    client = LLMClient(model, temperature=temperature, max_tokens=max_tokens)
    total = len(tasks) * n_samples
    failures = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task("generating", total=total)
        for task in tasks:
            prompt = build_prompt(task, hardware, mode=prompt_mode)
            for sample_id in range(n_samples):
                progress.update(bar, description=f"{task.task_id} s{sample_id}")
                try:
                    result = client.generate(prompt, system=SYSTEM_PROMPT)
                    rundir.save_sample(
                        run_dir, task.task_id, sample_id,
                        prompt=prompt, generation=result.generation,
                        raw_response=result.raw_text,
                    )
                except Exception as exc:
                    failures += 1
                    console.print(f"[red]generate failed {task.task_id} "
                                  f"sample {sample_id}: {exc}[/red]")
                progress.advance(bar)
    console.print(f"done: {total - failures}/{total} samples saved to {run_dir}")
    sys.exit(1 if failures == total else 0)


if __name__ == "__main__":
    main()

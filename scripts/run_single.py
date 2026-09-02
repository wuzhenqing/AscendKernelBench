#!/usr/bin/env python
"""Single-task end-to-end run: prompt -> generate -> build -> evaluate.

Example:
    python scripts/run_single.py --task level1/19_ReLU --model deepseek-v4-flash
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel

from ascend_kernel_bench import rundir
from ascend_kernel_bench.config import load_eval_config, load_hardware_profile
from ascend_kernel_bench.dataset import load_task
from ascend_kernel_bench.eval import eval_sample
from ascend_kernel_bench.llm import LLMClient
from ascend_kernel_bench.prompt import SYSTEM_PROMPT, build_prompt

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="task id, e.g. level1/19_ReLU")
    parser.add_argument("--model", default=None, help="override generation model")
    parser.add_argument("--hardware", default=None, help="hardware profile name")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--run-name", default=None, help="default: single_{task}")
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--prompt-mode", default=None,
                        choices=["zero_shot", "one_shot", "few_shot"])
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--no-perf", action="store_true", help="skip timing")
    parser.add_argument("--config", default=None, help="eval config yaml")
    args = parser.parse_args()

    config = load_eval_config(args.config)
    hardware = load_hardware_profile(args.hardware or config.hardware)
    gen_cfg = dict(config.generation)
    model = args.model or gen_cfg.get("model", "deepseek-v4-flash")
    prompt_mode = args.prompt_mode or gen_cfg.get("prompt_mode", "one_shot")
    temperature = args.temperature
    if temperature is None:
        temperature = float(gen_cfg.get("temperature", 0.0))
    max_tokens = int(gen_cfg.get("max_tokens", 16384))

    task = load_task(args.task)
    run_name = args.run_name or f"single_{task.name}"
    run_dir = rundir.create_run(run_name, {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_mode": prompt_mode,
        "hardware": hardware.name,
        "device": args.device,
        "tasks": [task.task_id],
    })

    console.rule(f"[bold]{task.task_id}[/bold] on {hardware.name}")
    prompt = build_prompt(task, hardware, mode=prompt_mode)
    console.print(f"prompt: {len(prompt)} chars, mode={prompt_mode}, model={model}")

    client = LLMClient(model, temperature=temperature, max_tokens=max_tokens)
    with console.status("[bold green]Generating with LLM..."):
        result = client.generate(prompt, system=SYSTEM_PROMPT)
    console.print(f"generation: asc={len(result.generation.custom_op_asc)} chars, "
                  f"model_new={len(result.generation.model_new_py)} chars")

    sdir = rundir.save_sample(
        run_dir, task.task_id, args.sample_id,
        prompt=prompt, generation=result.generation, raw_response=result.raw_text,
    )
    console.print(f"sample saved to {sdir}")

    with console.status("[bold cyan]Building + evaluating on NPU..."):
        eval_result = eval_sample(
            task, sdir,
            hardware=hardware, config=config,
            device=args.device, measure_performance=not args.no_perf,
        )

    compiled = eval_result["compiled"]
    correct = eval_result["correctness"]
    style = "green" if correct else ("yellow" if compiled else "red")
    lines = [f"compiled: {compiled}", f"correctness: {correct}"]
    if eval_result.get("runtime") and eval_result.get("ref_runtime"):
        speedup = eval_result["ref_runtime"] / eval_result["runtime"]
        lines.append(f"runtime: {eval_result['runtime']:.4f} ms "
                     f"(ref {eval_result['ref_runtime']:.4f} ms, speedup {speedup:.2f}x)")
    err = (eval_result.get("metadata") or {}).get("compilation_error") or \
        (eval_result.get("metadata") or {}).get("runtime_error")
    if err:
        lines.append(f"error: {str(err)[:800]}")
    console.print(Panel("\n".join(lines), title="eval_result", border_style=style))

    results = rundir.collect_eval_results(run_dir)
    rundir.write_eval_results(run_dir, results)
    console.print(f"results: {run_dir / 'eval_results.json'}")
    sys.exit(0 if correct else 1)


if __name__ == "__main__":
    main()

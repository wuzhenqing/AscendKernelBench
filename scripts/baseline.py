#!/usr/bin/env python
"""Measure torch_npu eager baselines and archive them per hardware profile.

Baselines are measured on this machine (README 3.6) and archived to
results/baseline/{hardware}/{task_slug}.json so runs stay comparable across
sessions. Evaluation itself always re-measures the reference live in the same
worker; the archive is the cross-run record.

Example:
    python scripts/baseline.py --level 1 --hardware ascend910b2 --device npu:0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ascend_kernel_bench._paths import BASELINE_DIR
from ascend_kernel_bench.config import load_eval_config, load_hardware_profile
from ascend_kernel_bench.dataset import discover_tasks, load_task

console = Console()

WORKER_SNIPPET = """
import json, sys
from pathlib import Path
sys.path.insert(0, {src!r})
import torch, torch_npu
from ascend_kernel_bench.timing import get_timing_stats, time_execution_with_npu_event

cfg = json.loads(Path(sys.argv[1]).read_text())
ref_globals = {{}}
exec(compile(cfg["task_py"], "<task.py>", "exec"), ref_globals)
Model, get_init_inputs, get_inputs = (
    ref_globals["Model"], ref_globals["get_init_inputs"], ref_globals["get_inputs"])
device = torch.device(cfg["device"])
torch.npu.set_device(device.index or 0)
dtype = getattr(torch, {{"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}}[cfg["precision"]])
torch.manual_seed(cfg["seed"]); torch.npu.manual_seed(cfg["seed"])
try:
    model = Model(*get_init_inputs()).to(device=device, dtype=dtype)
    model.eval()
    def draw():
        return [
            (x.to(device=device, dtype=dtype) if torch.is_floating_point(x) else x.to(device=device))
            if isinstance(x, torch.Tensor) else x
            for x in get_inputs()
        ]
    # Reseed after construction so the input sequence matches eval.py's
    # perf-phase RNG state and archived baselines stay comparable with the
    # live reference timing. Same adaptive protocol as eval.py: input sets up
    # to 256 MB are redrawn every trial, larger ones stay fixed.
    torch.manual_seed(cfg["seed"]); torch.npu.manual_seed(cfg["seed"])
    box = [draw()]
    prev = [None]
    input_bytes = sum(x.numel() * x.element_size() for x in box[0] if isinstance(x, torch.Tensor))
    fresh_per_trial = input_bytes <= 256 * 1024 * 1024
    def refresh():
        # Keeping the previous set alive stops the caching allocator from
        # handing back identical data_ptrs for a data_ptr-keyed cache to hit.
        prev[0] = box[0]
        box[0] = draw()
    # Rewind the RNG so the per-trial refresh sequence starts from the same
    # first draw eval.py uses (its timed() reseeds before the loop).
    torch.manual_seed(cfg["seed"]); torch.npu.manual_seed(cfg["seed"])
    torch.npu.synchronize(device=device)
    with torch.no_grad():
        times = time_execution_with_npu_event(
            lambda: model(*box[0]), [],
            num_warmup=cfg["num_warmup"], num_trials=cfg["num_perf_trials"], device=device,
            setup=refresh if fresh_per_trial else None)
    stats = get_timing_stats(times)
    stats["supported_on_npu"] = True
    stats["timing_fresh_inputs"] = bool(fresh_per_trial)
except Exception as exc:
    # Tasks torch_npu cannot run are not benchmark failures: they are the
    # operator-extension opportunities the CPU-reference fallback in eval.py
    # exists to surface. Record them explicitly instead of dropping them.
    stats = {{"supported_on_npu": False, "error": repr(exc)}}
Path(cfg["out_path"]).write_text(json.dumps(stats))
"""


def measure_baseline(task, *, config, device: str, out_path: Path) -> dict:
    """Measure one task's reference runtime in an isolated subprocess."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="akb_baseline_") as tmpdir:
        cfg = {
            "task_py": task.task_py,
            "device": device,
            "precision": config.precision,
            "seed": config.seed,
            "num_warmup": config.num_warmup,
            "num_perf_trials": config.num_perf_trials,
            "out_path": str(Path(tmpdir) / "stats.json"),
        }
        cfg_path = Path(tmpdir) / "cfg.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-c", WORKER_SNIPPET.format(src=str(Path(__file__).resolve().parent.parent / "src")), str(cfg_path)],
            capture_output=True, text=True, timeout=config.eval_timeout,
            start_new_session=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"baseline worker failed: {proc.stderr[-1000:]}")
        stats = json.loads(Path(cfg["out_path"]).read_text())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    os.replace(tmp, out_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--hardware", default=None)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_eval_config(args.config)
    hardware = load_hardware_profile(args.hardware or config.hardware)
    if args.task:
        tasks = [load_task(t) for t in args.task]
    elif args.level is not None:
        tasks = discover_tasks(level=args.level)
    else:
        tasks = discover_tasks()
    if not tasks:
        sys.exit("no tasks found")

    out_dir = BASELINE_DIR / hardware.name
    console.print(f"measuring {len(tasks)} baselines on {hardware.name} -> {out_dir}")
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console,
    ) as progress:
        bar = progress.add_task("baseline", total=len(tasks))
        for task in tasks:
            progress.update(bar, description=task.task_id)
            out_path = out_dir / f"{task.name}.json"
            try:
                stats = measure_baseline(task, config=config, device=args.device,
                                         out_path=out_path)
                if stats.get("supported_on_npu", True):
                    progress.console.print(f"  {task.task_id}: {stats['mean']:.4f} ms")
                else:
                    progress.console.print(
                        f"  [yellow]{task.task_id}: not supported on NPU"
                        f" (CPU-reference task)[/yellow]"
                    )
            except Exception as exc:
                progress.console.print(f"  [red]{task.task_id}: {exc}[/red]")
            progress.advance(bar)


if __name__ == "__main__":
    main()

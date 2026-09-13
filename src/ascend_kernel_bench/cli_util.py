"""Shared argparse and CLI helpers for the benchmark scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console
    from rich.progress import Progress

from .config import (
    EvalConfig,
    HardwareProfile,
    load_eval_config,
    load_hardware_profile,
)
from .dataset import Task, discover_tasks, load_task


@dataclass(frozen=True)
class GenerationSettings:
    """Resolved generation settings after YAML defaults and CLI overrides."""

    model: str
    prompt_mode: str
    temperature: float
    max_tokens: int
    num_samples: int


def resolve_generation_settings(
    config: EvalConfig,
    *,
    model: str | None = None,
    prompt_mode: str | None = None,
    temperature: float | None = None,
    num_samples: int | None = None,
) -> GenerationSettings:
    """Resolve generation settings from YAML with optional CLI overrides.

    Args:
        config: Loaded evaluation configuration.
        model: Optional ``--model`` override.
        prompt_mode: Optional ``--prompt-mode`` override.
        temperature: Optional ``--temperature`` override.
        num_samples: Optional ``--n-samples`` override.

    Returns:
        Frozen settings used by ``generate.py`` and ``run_single.py``.
    """
    gen_cfg = dict(config.generation)
    return GenerationSettings(
        model=model or str(gen_cfg.get("model", "deepseek-v4-flash")),
        prompt_mode=prompt_mode or str(gen_cfg.get("prompt_mode", "one_shot")),
        temperature=float(
            gen_cfg.get("temperature", 0.0)
            if temperature is None
            else temperature
        ),
        max_tokens=int(gen_cfg.get("max_tokens", 16384)),
        num_samples=(
            int(gen_cfg.get("num_samples", 1))
            if num_samples is None
            else int(num_samples)
        ),
    )


@dataclass(frozen=True)
class EvalRuntime:
    """Loaded evaluation config plus the resolved hardware profile."""

    config: EvalConfig
    hardware: HardwareProfile


def load_eval_runtime(
    *,
    config_path: str | None = None,
    hardware: str | None = None,
) -> EvalRuntime:
    """Load YAML defaults and resolve the hardware profile.

    Args:
        config_path: Optional ``--config`` path.
        hardware: Optional ``--hardware`` name or YAML path.

    Returns:
        Frozen config and hardware used by the benchmark CLIs.

    Raises:
        FileNotFoundError: If the config or hardware YAML is missing.
    """
    config = load_eval_config(config_path)
    return EvalRuntime(
        config=config,
        hardware=load_hardware_profile(hardware or config.hardware),
    )


def select_tasks(
    *,
    level: int | None = None,
    task_ids: list[str] | None = None,
) -> list[Task]:
    """Load tasks from explicit IDs, one level, or the full corpus.

    Args:
        level: Optional ``--level`` filter.
        task_ids: Optional repeated ``--task`` identifiers.

    Returns:
        Loaded tasks. Explicit IDs win over ``level``.
    """
    if task_ids is not None:
        return [load_task(item) for item in task_ids]
    if level is not None:
        return discover_tasks(level=level)
    return discover_tasks()


def generation_run_config(
    settings: GenerationSettings,
    *,
    hardware_name: str,
    task_ids: list[str],
    **extra: object,
) -> dict[str, object]:
    """Return the mapping written to ``generation_config.yaml``.

    Args:
        settings: Resolved generation settings.
        hardware_name: Hardware profile name.
        task_ids: Task identifiers included in the run.
        **extra: Optional extra keys such as ``device``.

    Returns:
        YAML-serializable generation record.
    """
    payload: dict[str, object] = {
        "model": settings.model,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "prompt_mode": settings.prompt_mode,
        "num_samples": settings.num_samples,
        "hardware": hardware_name,
        "tasks": list(task_ids),
    }
    payload.update(extra)
    return payload


def eval_result_lines(result: dict[str, object]) -> tuple[str, list[str]]:
    """Return Rich style and summary lines for one evaluation result.

    Args:
        result: Per-sample evaluation dict.

    Returns:
        ``(style, lines)`` suitable for a status panel or log line.
    """
    style, label = sample_status_label(result)
    lines = [
        f"status: {label}",
        f"compiled: {result.get('compiled')}",
        f"correctness: {result.get('correctness')}",
    ]
    runtime = result.get("runtime")
    ref_runtime = result.get("ref_runtime")
    if (
        isinstance(runtime, int | float)
        and isinstance(ref_runtime, int | float)
        and runtime
    ):
        speedup = ref_runtime / runtime
        lines.append(
            f"runtime: {runtime:.4f} ms "
            f"(ref {ref_runtime:.4f} ms, speedup {speedup:.2f}x)"
        )
    metadata = result.get("metadata") or {}
    if isinstance(metadata, dict):
        err = metadata.get("compilation_error") or metadata.get("runtime_error")
        if err:
            lines.append(f"error: {str(err)[:800]}")
    return style, lines


def cli_progress(console: Console) -> Progress:
    """Return the shared Rich progress bar used by the batch CLIs.

    Args:
        console: Rich console that owns the progress output.

    Returns:
        A ``Progress`` context-manager instance.
    """
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def sample_status_label(result: dict[str, object]) -> tuple[str, str]:
    """Return Rich style and short label for one evaluation result.

    Args:
        result: Per-sample evaluation dict.

    Returns:
        ``(style, label)`` such as ``("green", "OK")``.
    """
    if result.get("correctness"):
        return "green", "OK"
    if not result.get("compiled"):
        return "yellow", "COMPILE-FAIL"
    return "red", "WRONG"

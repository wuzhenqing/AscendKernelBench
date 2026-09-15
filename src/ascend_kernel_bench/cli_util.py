"""Shared CLI helpers: settings resolution, task selection, and reports.

Reporting lives in :mod:`.report`. This module keeps the generation and
evaluation setting factories that the batch scripts share.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .config import (
    EvalConfig,
    HardwareProfile,
    load_eval_config,
    load_hardware_profile,
)
from .dataset import Task, discover_tasks, load_task
from .report import (
    cli_progress,
    eval_result_lines,
    print_eval_report,
    sample_status_label,
)

__all__ = [
    "EvalRuntime",
    "GenerationSettings",
    "cli_progress",
    "eval_result_lines",
    "generation_run_config",
    "load_eval_runtime",
    "print_eval_report",
    "resolve_generation_settings",
    "sample_status_label",
    "select_tasks",
]


class GenerationSettings(BaseModel):
    """Resolved generation settings after YAML defaults and CLI overrides."""

    model_config = ConfigDict(frozen=True)

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
        Frozen settings used by ``generate.py``.
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


class EvalRuntime(BaseModel):
    """Loaded evaluation config plus the resolved hardware profile."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

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

"""Run directory layout and result persistence (docs/guide/results.md).

``runs/{run_name}/`` holds generation_config.yaml, per-sample directories
``level{L}/{task}/sample_{i}/`` (prompt.txt, custom_op.asc, model_new.py,
build/, eval_result.json), and the aggregate eval_results.json aligned with
KernelBench's schema.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from ._paths import RUNS_DIR
from .io_util import read_json_object, write_json_atomic, write_yaml_atomic
from .llm import AscendCGeneration


def create_run(run_name: str, generation_config: dict) -> Path:
    """Create ``runs/{run_name}/`` and stamp the generation config.

    Args:
        run_name: Directory name under :data:`RUNS_DIR`.
        generation_config: Mapping written to ``generation_config.yaml``.
            An empty mapping skips the file.

    Returns:
        Path of the run directory.
    """
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "generation_config.yaml"
    if generation_config:
        write_yaml_atomic(cfg_path, generation_config)
    return run_dir


def resolve_run(run: str | Path) -> Path:
    """Resolve a run name or directory path to an existing run directory.

    An existing directory is used as-is. A bare name is resolved under
    :data:`RUNS_DIR`.

    Args:
        run: Run directory name or filesystem path.

    Returns:
        Absolute path of the run directory.

    Raises:
        FileNotFoundError: If the resolved path is not a directory.
    """
    path = Path(run)
    if path.is_dir():
        return path.resolve()
    if path.is_absolute() or len(path.parts) > 1:
        raise FileNotFoundError(f"run dir not found: {path}")
    named = RUNS_DIR / path.name
    if named.is_dir():
        return named
    raise FileNotFoundError(f"run dir not found: {named}")


def generation_hardware_name(run_dir: Path) -> str | None:
    """Return the hardware profile name recorded at generation time.

    Args:
        run_dir: Run directory that may contain ``generation_config.yaml``.

    Returns:
        Hardware profile name, or ``None`` when the file is missing or
        has no usable ``hardware`` field.
    """
    path = Path(run_dir) / "generation_config.yaml"
    if not path.is_file():
        return None
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(loaded, dict):
        return None
    name = loaded.get("hardware")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def sample_dir(run_dir: Path, task_id: str, sample_id: int) -> Path:
    """Return ``runs/{run}/level{L}/{task}/sample_{i}/``.

    Args:
        run_dir: Run directory.
        task_id: ``levelN/stem`` path segment.
        sample_id: Integer sample index.

    Returns:
        Path of the sample directory (not created).
    """
    return Path(run_dir) / task_id / f"sample_{sample_id}"


def save_sample(
    run_dir: Path,
    task_id: str,
    sample_id: int,
    *,
    prompt: str,
    generation: AscendCGeneration,
    raw_response: str = "",
) -> Path:
    """Persist one generated sample (prompt, both deliverables, raw text).

    Args:
        run_dir: Run directory.
        task_id: ``levelN/stem`` path segment.
        sample_id: Integer sample index.
        prompt: Full user prompt written to ``prompt.txt``.
        generation: Parsed deliverables.
        raw_response: Optional raw model text.

    Returns:
        Path of the created sample directory.
    """
    out_dir = sample_dir(run_dir, task_id, sample_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (out_dir / "custom_op.asc").write_text(
        generation.custom_op_asc, encoding="utf-8"
    )
    (out_dir / "model_new.py").write_text(
        generation.model_new_py, encoding="utf-8"
    )
    if raw_response:
        (out_dir / "response_raw.txt").write_text(
            raw_response, encoding="utf-8"
        )
    return out_dir


def iter_sample_dirs(
    run_dir: Path, level: int | None = None
) -> Iterator[tuple[str, int, Path]]:
    """Yield ``(task_id, sample_id, dir)`` for every complete sample.

    A sample is complete when both ``custom_op.asc`` and ``model_new.py``
    exist.

    Args:
        run_dir: Run directory under ``runs/``.
        level: When set, only visit ``level{level}/`` task directories.

    Yields:
        Task id, sample index, and directory path.
    """
    pattern = f"level{level}/*" if level is not None else "level*/*"
    for task_dir in sorted(Path(run_dir).glob(pattern)):
        if not task_dir.is_dir():
            continue
        task_id = f"{task_dir.parent.name}/{task_dir.name}"
        for sdir in sorted(task_dir.glob("sample_*")):
            if (sdir / "custom_op.asc").is_file() and (
                sdir / "model_new.py"
            ).is_file():
                yield task_id, int(sdir.name.split("_", 1)[1]), sdir


def load_eval_result(sample_dir_path: Path) -> dict[str, Any] | None:
    """Load ``eval_result.json`` from a sample directory, if present.

    Args:
        sample_dir_path: Per-sample directory.

    Returns:
        Parsed result dict, or ``None`` when the file is missing.
    """
    path = Path(sample_dir_path) / "eval_result.json"
    if not path.is_file():
        return None
    try:
        return read_json_object(path)
    except (json.JSONDecodeError, ValueError):
        return None


def collect_eval_results(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Assemble the KernelBench-compatible eval_results mapping for a run.

    Args:
        run_dir: Run directory under ``runs/``.

    Returns:
        Mapping of task id to sample result dicts that include
        ``sample_id``. Samples without ``eval_result.json`` are omitted.
    """
    results: dict[str, list[dict]] = {}
    for task_id, sample_id, sdir in iter_sample_dirs(run_dir):
        result = load_eval_result(sdir)
        if result is None:
            continue
        results.setdefault(task_id, []).append(
            {"sample_id": sample_id, **result}
        )
    return results


def write_eval_results(
    run_dir: Path, results: dict[str, list[dict[str, Any]]]
) -> Path:
    """Write ``eval_results.json`` atomically.

    Args:
        run_dir: Run directory under ``runs/``.
        results: KernelBench-compatible problem-to-samples mapping.

    Returns:
        Path of the written file.
    """
    path = Path(run_dir) / "eval_results.json"
    write_json_atomic(path, results)
    return path


def write_pass_at_k(run_dir: Path, pass_at_k: dict) -> Path:
    """Write ``pass_at_k_results.json`` atomically.

    Args:
        run_dir: Run directory under ``runs/``.
        pass_at_k: Mapping produced by :func:`score.compute_pass_at_k`.

    Returns:
        Path of the written file.
    """
    path = Path(run_dir) / "pass_at_k_results.json"
    write_json_atomic(path, pass_at_k)
    return path

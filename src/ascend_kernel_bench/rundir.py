"""Run directory layout and result persistence (README 4.6).

``runs/{run_name}/`` holds generation_config.yaml, per-sample directories
``level{L}/{task}/sample_{i}/`` (prompt.txt, custom_op.asc, model_new.py,
build/, eval_result.json), and the aggregate eval_results.json aligned with
KernelBench's schema.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from ._paths import RUNS_DIR
from .llm import AscendCGeneration


def create_run(run_name: str, generation_config: dict) -> Path:
    """Create runs/{run_name}/ and stamp it with the generation config."""
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "generation_config.yaml"
    if generation_config:
        _write_yaml_atomic(cfg_path, generation_config)
    return run_dir


def sample_dir(run_dir: Path, task_id: str, sample_id: int) -> Path:
    """runs/{run}/level{L}/{task}/sample_{i}/"""
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
    """Persist one generated sample (prompt, both deliverables, raw text)."""
    out_dir = sample_dir(run_dir, task_id, sample_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (out_dir / "custom_op.asc").write_text(generation.custom_op_asc, encoding="utf-8")
    (out_dir / "model_new.py").write_text(generation.model_new_py, encoding="utf-8")
    if raw_response:
        (out_dir / "response_raw.txt").write_text(raw_response, encoding="utf-8")
    return out_dir


def iter_sample_dirs(run_dir: Path):
    """Yield (task_id, sample_id, dir) for every generated sample in a run."""
    for task_dir in sorted(Path(run_dir).glob("level*/*")):
        if not task_dir.is_dir():
            continue
        task_id = f"{task_dir.parent.name}/{task_dir.name}"
        for sdir in sorted(task_dir.glob("sample_*")):
            if (sdir / "custom_op.asc").is_file() and (sdir / "model_new.py").is_file():
                yield task_id, int(sdir.name.split("_", 1)[1]), sdir


def load_eval_result(sample_dir_path: Path) -> dict | None:
    path = Path(sample_dir_path) / "eval_result.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_eval_results(run_dir: Path) -> dict[str, list[dict]]:
    """Assemble the KernelBench-compatible eval_results mapping for a run."""
    results: dict[str, list[dict]] = {}
    for task_id, sample_id, sdir in iter_sample_dirs(run_dir):
        result = load_eval_result(sdir)
        if result is None:
            continue
        results.setdefault(task_id, []).append({"sample_id": sample_id, **result})
    return results


def write_eval_results(run_dir: Path, results: dict[str, list[dict]]) -> Path:
    path = Path(run_dir) / "eval_results.json"
    _write_json_atomic(path, results)
    return path


def write_pass_at_k(run_dir: Path, pass_at_k: dict) -> Path:
    path = Path(run_dir) / "pass_at_k_results.json"
    _write_json_atomic(path, pass_at_k)
    return path


def _write_json_atomic(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _write_yaml_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    os.replace(tmp, path)

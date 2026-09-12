"""Isolated evaluation of one generated Ascend C sample.

Static check, then build, correctness, and timing run in a worker
subprocess. See docs/guide/evaluation.md for the evaluation protocol.

The host entry (:func:`eval_sample`) writes a config JSON, spawns
``python -m ascend_kernel_bench.worker`` as a process-group leader via
Popen, kills the whole group on timeout, and reads the result from a JSON
file — CANN log spam on stdout never pollutes the result channel.

Worker-side logic lives in :mod:`ascend_kernel_bench.eval_device`. Result
payload helpers live in :mod:`ascend_kernel_bench.eval_result`.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .checker import check_custom_op_asc, check_model_new
from .config import EvalConfig, HardwareProfile
from .dataset import Task
from .eval_device import eval_sample_on_device
from .eval_result import eval_protocol_metadata, fail_result
from .io_util import read_json_object, write_json_atomic
from .modes import require_implemented_mode
from .timing import l2_clear_bytes

__all__ = [
    "eval_protocol_metadata",
    "eval_sample",
    "eval_sample_on_device",
    "fail_result",
]

# Backward-compatible alias used by the worker and older call sites.
_fail = fail_result


def _persist_eval_result(
    sample_dir: Path, result: dict[str, Any]
) -> dict[str, Any]:
    """Write ``eval_result.json`` when ``sample_dir`` exists and return it."""
    if Path(sample_dir).is_dir():
        write_json_atomic(Path(sample_dir) / "eval_result.json", result)
    return result


def eval_sample(
    task: Task,
    sample_dir: Path,
    *,
    hardware: HardwareProfile,
    config: EvalConfig,
    device: str = "npu:0",
    measure_performance: bool = True,
) -> dict[str, Any]:
    """Host entry: static check, then isolated worker subprocess.

    ``sample_dir`` must already contain ``custom_op.asc`` and
    ``model_new.py``. The result dict is also written to
    ``sample_dir/eval_result.json``.

    Args:
        task: Loaded KernelBench task (reference ``Model`` source).
        sample_dir: Directory containing the two generated deliverables.
        hardware: Profile used for CMake arch, L2 flush size, and SOL.
        config: Evaluation timeouts, seeds, trials, and operator mode.
        device: Runtime device string, for example ``npu:0``.
        measure_performance: When False, skip timing and the post-timing
            re-check.

    Returns:
        Per-sample result dict (also persisted as ``eval_result.json``).
    """
    sample_dir = Path(sample_dir)
    model_new_path = sample_dir / "model_new.py"
    asc_path = sample_dir / "custom_op.asc"
    if not asc_path.is_file() or not model_new_path.is_file():
        return _persist_eval_result(
            sample_dir,
            _fail(
                compilation_error=(
                    "sample dir missing custom_op.asc or model_new.py"
                )
            ),
        )

    violations = check_custom_op_asc(asc_path.read_text(encoding="utf-8"))
    violations += check_model_new(model_new_path.read_text(encoding="utf-8"))
    if violations:
        return _persist_eval_result(
            sample_dir,
            _fail(
                compilation_error="; ".join(violations),
                static_check_error=violations,
            ),
        )

    # Configure and build each get build_timeout; the host budget covers both
    # plus the evaluation itself.
    timeout_s = config.eval_timeout + 2 * config.build_timeout
    result = _run_eval_worker(
        {
            "task_py": task.task_py,
            "sample_dir": str(sample_dir),
            "cmake_arch": hardware.cmake_arch,
            "hardware_name": hardware.name,
            "device": device,
            "measure_performance": measure_performance,
            "seed": config.seed,
            "num_correct_trials": config.num_correct_trials,
            "num_perf_trials": config.num_perf_trials,
            "num_warmup": config.num_warmup,
            "precision": config.precision,
            "tolerances": config.tolerances,
            "excessive_speedup": config.excessive_speedup,
            "build_timeout": config.build_timeout,
            "operator_mode": require_implemented_mode(config.operator_mode),
            "memory_bandwidth_gbps": hardware.memory_bandwidth_gbps,
            "peak_tflops": hardware.peak_tflops_for(config.precision),
            "l2_clear_size": l2_clear_bytes(hardware.l2_cache_mb),
        },
        timeout_s=timeout_s,
    )
    return _persist_eval_result(sample_dir, result)


def _run_eval_worker(cfg: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    """Spawn ``python -m ascend_kernel_bench.worker`` and return its payload.

    Args:
        cfg: Worker config without ``result_path`` (added here).
        timeout_s: Host budget covering configure, build, and evaluation.

    Returns:
        Parsed worker result, or a failed payload on timeout / I/O error.
    """
    with tempfile.TemporaryDirectory(prefix="akb_eval_") as tmpdir:
        result_path = Path(tmpdir) / "result.json"
        cfg_path = Path(tmpdir) / "cfg.json"
        write_json_atomic(cfg_path, {**cfg, "result_path": str(result_path)})
        env = dict(os.environ)
        env.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "0")
        proc = subprocess.Popen(
            [sys.executable, "-m", "ascend_kernel_bench.worker", str(cfg_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=env,
        )
        try:
            _, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # start_new_session makes the worker a process-group leader, so
            # killpg reaps the worker and anything it forked (a hung NPU job
            # included) instead of leaking the group on the device.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return _fail(runtime_error=f"eval timed out after {timeout_s}s")

        if proc.returncode != 0:
            err = (stderr or "").strip()
            return _fail(
                runtime_error=err[-2000:]
                or f"worker exited with code {proc.returncode}"
            )

        return _read_worker_payload(result_path)


def _read_worker_payload(result_path: Path) -> dict[str, Any]:
    """Load the worker JSON object, or return a failed payload.

    Args:
        result_path: Path written by ``worker.main``.

    Returns:
        Parsed result dict, or a failed payload when the file is missing,
        invalid JSON, or not an object.
    """
    if not result_path.is_file():
        return _fail(runtime_error="worker produced no result.json")
    try:
        return read_json_object(result_path)
    except json.JSONDecodeError as exc:
        return _fail(runtime_error=f"invalid worker JSON: {exc}")
    except ValueError:
        return _fail(runtime_error="worker JSON is not an object")

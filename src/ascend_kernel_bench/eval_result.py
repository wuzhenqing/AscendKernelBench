"""KernelBench-compatible evaluation result payloads."""

from __future__ import annotations

from typing import Any

__all__ = [
    "compiled_result",
    "eval_protocol_metadata",
    "fail_result",
]

_TOP_LEVEL_RESULT_KEYS = {
    "compiled",
    "correctness",
    "runtime",
    "runtime_stats",
    "ref_runtime",
    "ref_runtime_stats",
    "metadata",
}


def fail_result(
    *,
    compiled: bool = False,
    compilation_error: str | None = None,
    runtime_error: str | None = None,
    **extra_metadata: object,
) -> dict[str, Any]:
    """Return a failed evaluation payload with empty timing fields.

    Args:
        compiled: Whether the shared library built successfully.
        compilation_error: Build or static-check diagnostic.
        runtime_error: Worker or candidate runtime diagnostic.
        **extra_metadata: Extra keys merged into ``metadata``. Top-level
            result keys such as ``correctness`` are ignored so callers
            cannot overwrite the payload schema.

    Returns:
        KernelBench-compatible result dict with ``correctness`` False.
    """
    extras = {
        key: value
        for key, value in extra_metadata.items()
        if key not in _TOP_LEVEL_RESULT_KEYS
    }
    return {
        "compiled": compiled,
        "correctness": False,
        "runtime": None,
        "runtime_stats": None,
        "ref_runtime": None,
        "ref_runtime_stats": None,
        "metadata": {
            "compilation_error": compilation_error,
            "runtime_error": runtime_error,
            **extras,
        },
    }


def compiled_result(
    *,
    correctness: bool,
    metadata: dict[str, Any],
    runtime: float | None = None,
    runtime_stats: dict[str, Any] | None = None,
    ref_runtime: float | None = None,
    ref_runtime_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a post-build result payload.

    Args:
        correctness: Final correctness decision.
        metadata: Stage-specific settings and diagnostics.
        runtime: Candidate mean latency, or None.
        runtime_stats: Candidate timing statistics, or None.
        ref_runtime: NPU reference mean latency, or None.
        ref_runtime_stats: NPU reference timing statistics, or None.

    Returns:
        KernelBench-compatible result dict with ``compiled`` True.
    """
    return {
        "compiled": True,
        "correctness": correctness,
        "runtime": runtime,
        "runtime_stats": runtime_stats,
        "ref_runtime": ref_runtime,
        "ref_runtime_stats": ref_runtime_stats,
        "metadata": metadata,
    }


def eval_protocol_metadata(
    *,
    hardware_name: str,
    precision: str,
    seed: int,
    num_correct_trials: int,
    num_warmup: int,
    num_perf_trials: int,
    operator_mode: str,
    l2_clear_size: int,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Return the evaluation-protocol snapshot stored on scored samples.

    Args:
        hardware_name: Profile name recorded in results.
        precision: Floating dtype key (``fp32``, ``fp16``, or ``bf16``).
        seed: Base RNG seed for init, correctness, and timing.
        num_correct_trials: Number of seeded correctness trials.
        num_warmup: Warmup calls before measured trials.
        num_perf_trials: Retained NPU-event measurements per model.
        operator_mode: Compilation mode (currently ``aclnn``).
        l2_clear_size: Bytes allocated to flush L2 before each timed call.
        atol: Absolute comparison tolerance used for this sample.
        rtol: Relative comparison tolerance used for this sample.

    Returns:
        Metadata keys that identify how the sample was evaluated.
    """
    return {
        "hardware": hardware_name,
        "precision": precision,
        "seed": seed,
        "correctness_trials": num_correct_trials,
        "num_warmup": num_warmup,
        "num_perf_trials": num_perf_trials,
        "operator_mode": operator_mode,
        "l2_clear_size": int(l2_clear_size),
        "atol": atol,
        "rtol": rtol,
    }

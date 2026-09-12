"""Hardware Speed-of-Light (SOL) scoring after NVIDIA SOL-ExecBench.

SOL-ExecBench ranks kernels by how much of the gap between a software
baseline and an analytic hardware lower bound they close:

    S = (T_b - T_sol) / ((T_k - T_sol) + (T_b - T_sol))

``S = 0.5`` matches the baseline latency; ``S -> 1`` as the kernel
approaches the bound. This module uses a documented roofline estimate
(memory traffic / profile bandwidth, optionally FLOPs / peak TFLOPS),
not NVIDIA SOLAR. Bounds are therefore a reporting aid, not a claim that
the machine was characterized with SOLAR.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence

_MS_PER_S = 1_000.0
_BYTES_PER_GB = 1e9
_FLOPS_PER_TFLOP = 1e12


def roofline_bound_ms(
    *,
    bytes_moved: int,
    bandwidth_gbps: float,
    flops: float | None = None,
    peak_tflops: float | None = None,
) -> float | None:
    """Return a roofline lower bound in milliseconds, or None if undefined.

    The memory term is ``bytes_moved / bandwidth``. When both ``flops`` and
    ``peak_tflops`` are positive, the compute term is included and the
    bound is the max of the two (classic roofline).

    Args:
        bytes_moved: Estimated bytes transferred (inputs + outputs).
        bandwidth_gbps: Profile peak memory bandwidth in GB/s.
        flops: Optional floating-point operation count.
        peak_tflops: Optional peak TFLOPS for the active precision.

    Returns:
        Bound in milliseconds, or None when the memory term is undefined.
    """
    if bytes_moved <= 0 or bandwidth_gbps <= 0:
        return None
    t_mem_ms = (bytes_moved / (bandwidth_gbps * _BYTES_PER_GB)) * _MS_PER_S
    if flops is None or peak_tflops is None or flops <= 0 or peak_tflops <= 0:
        return t_mem_ms
    t_compute_ms = (flops / (peak_tflops * _FLOPS_PER_TFLOP)) * _MS_PER_S
    return max(t_mem_ms, t_compute_ms)


def sol_score(
    kernel_ms: float,
    baseline_ms: float,
    sol_ms: float,
) -> float | None:
    """Return the SOL-ExecBench-style score in ``[0, 1]``, or None.

    ``T_sol`` is clamped below both measured times so a noisy bound cannot
    invert the score. A non-positive baseline-to-SOL gap is undefined.

    Args:
        kernel_ms: Candidate mean latency in milliseconds.
        baseline_ms: Reference mean latency in milliseconds.
        sol_ms: Analytic lower bound in milliseconds.

    Returns:
        Score in ``[0, 1]``, or None when any input is non-positive.
    """
    if kernel_ms <= 0 or baseline_ms <= 0 or sol_ms <= 0:
        return None
    sol_ms = min(sol_ms, kernel_ms * 0.999, baseline_ms * 0.999)
    gap = baseline_ms - sol_ms
    if gap <= 0:
        return None
    score = gap / ((kernel_ms - sol_ms) + gap)
    return max(0.0, min(1.0, score))


def attach_sol_metadata(
    metadata: MutableMapping[str, object],
    *,
    kernel_ms: float | None,
    baseline_ms: float | None,
    bytes_moved: int,
    bandwidth_gbps: float,
    flops: float | None = None,
    peak_tflops: float | None = None,
) -> None:
    """Record the roofline bound and SOL score on ``metadata`` when defined.

    Args:
        metadata: Mutable result metadata mapping.
        kernel_ms: Candidate mean latency, or None if untimed.
        baseline_ms: NPU reference mean latency, or None.
        bytes_moved: Estimated input-plus-output tensor bytes.
        bandwidth_gbps: Profile memory bandwidth in GB/s.
        flops: Optional FLOP count. Unused unless ``peak_tflops`` is set.
        peak_tflops: Optional peak TFLOPS for the active precision.
    """
    bound = roofline_bound_ms(
        bytes_moved=bytes_moved,
        bandwidth_gbps=bandwidth_gbps,
        flops=flops,
        peak_tflops=peak_tflops,
    )
    if bound is None:
        return
    metadata["bytes_moved"] = int(bytes_moved)
    metadata["sol_bound_ms"] = float(f"{bound:.6g}")
    metadata["sol_bound_kind"] = (
        "roofline" if flops and peak_tflops else "roofline_memory"
    )
    if not isinstance(kernel_ms, (int, float)):
        return
    if not isinstance(baseline_ms, (int, float)):
        return
    score = sol_score(float(kernel_ms), float(baseline_ms), bound)
    if score is not None:
        metadata["sol_score"] = float(f"{score:.6g}")


def task_declared_flops(namespace: Mapping[str, object]) -> float | None:
    """Return a positive FLOP count declared by the task, if any.

    A task may set a top-level ``FLOPS`` or ``NUM_FLOPS`` number. The
    vendored KernelBench corpus does not, so the default SOL bound stays
    memory-only unless an author adds that constant.

    Args:
        namespace: Executed task globals.

    Returns:
        Positive FLOP count, or None.
    """
    for key in ("FLOPS", "NUM_FLOPS"):
        value = namespace.get(key)
        if isinstance(value, (int, float)) and float(value) > 0:
            return float(value)
    return None


def mean_sol_score(samples: Sequence[Mapping[str, object]]) -> float | None:
    """Return the mean of per-sample ``metadata.sol_score`` values.

    Args:
        samples: Flat list of evaluation result dicts.

    Returns:
        Arithmetic mean, or None when no sample recorded a SOL score.
    """
    scores: list[float] = []
    for sample in samples:
        metadata = sample.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            continue
        value = metadata.get("sol_score")
        if isinstance(value, (int, float)):
            scores.append(float(value))
    if not scores:
        return None
    return sum(scores) / len(scores)

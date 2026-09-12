"""NPU Event timing with per-trial L2 clearing (docs/guide/evaluation.md).

Semantics follow KernelBench's ``time_execution_with_cuda_event`` ported to
``torch.npu``: warmup with synchronize, empty_cache, then per trial
synchronize -> event pair around the callable -> L2 thrash -> synchronize,
discarding the first trial. torch imports stay inside functions so the host
process can import this module without initializing an NPU runtime.

The L2 flush buffer is at least 256 MiB and at least twice the profile L2
size (SOL-ExecBench-style isolation). Warmup defaults to 10 calls.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

DEFAULT_L2_CLEAR_BYTES = 256 * 1024 * 1024
DEFAULT_NUM_WARMUP = 10
DEFAULT_NUM_TRIALS = 100
# Input sets larger than this reuse one allocation during timing (KernelBench
# L1 already ships multi-gigabyte cases). Smaller sets are redrawn per trial.
REFRESH_INPUT_BYTES_LIMIT = 256 * 1024 * 1024


def l2_clear_bytes(l2_cache_mb: int) -> int:
    """Return a flush-buffer size at least 256 MiB and at least 2x L2.

    Args:
        l2_cache_mb: Profile L2 capacity in mebibytes. ``0`` uses the
            256 MiB default only.

    Returns:
        Buffer size in bytes.
    """
    if l2_cache_mb <= 0:
        return DEFAULT_L2_CLEAR_BYTES
    twice = 2 * int(l2_cache_mb) * 1024 * 1024
    return max(DEFAULT_L2_CLEAR_BYTES, twice)


def clear_l2_cache(
    device: Any,
    size_bytes: int = DEFAULT_L2_CLEAR_BYTES,
) -> None:
    """Thrash device memory so the next kernel misses in L2.

    Args:
        device: ``torch.device`` or device index accepted by ``torch.empty``.
        size_bytes: Allocation size. Rounded down to a whole number of
            int64 elements; a non-positive size is a no-op.
    """
    import torch

    n_elements = max(int(size_bytes), 0) // 8
    if n_elements <= 0:
        return
    dummy = torch.empty((n_elements,), dtype=torch.int64, device=device)
    dummy.fill_(42)
    del dummy


def time_execution_with_npu_event(
    kernel_fn: Callable[..., Any],
    args: Sequence[Any],
    num_warmup: int = DEFAULT_NUM_WARMUP,
    num_trials: int = DEFAULT_NUM_TRIALS,
    discard_first: int = 1,
    device: Any = None,
    setup: Callable[[], None] | None = None,
    l2_clear_size: int = DEFAULT_L2_CLEAR_BYTES,
) -> list[float]:
    """Time ``kernel_fn(*args)`` over trials with ``torch.npu.Event`` (ms).

    ``setup`` (optional) runs before every warmup and timed call, outside the
    event window — use it to hand the callable fresh inputs per trial so a
    result cache cannot replay one computation across all trials.

    Args:
        kernel_fn: Callable invoked as ``kernel_fn(*args)``.
        args: Positional arguments forwarded to ``kernel_fn``.
        num_warmup: Synchronized warmup calls before measured trials.
        num_trials: Number of retained measurements.
        discard_first: Measured trials dropped before retention.
        device: NPU device. Defaults to the current device.
        setup: Optional per-trial setup (input refresh).
        l2_clear_size: Bytes allocated to flush L2 before each timed call.

    Returns:
        Elapsed times in milliseconds for the retained trials.
    """
    import torch

    if device is None:
        device = torch.npu.current_device()
    previous_device = torch.npu.current_device()
    torch.npu.set_device(device)
    try:
        for _ in range(num_warmup):
            if setup is not None:
                setup()
            kernel_fn(*args)
            torch.npu.synchronize(device=device)

        # Releases PyTorch's caching allocator, not the device L2 cache.
        torch.npu.empty_cache()

        elapsed_times: list[float] = []
        for trial in range(num_trials + discard_first):
            if setup is not None:
                setup()
            torch.npu.synchronize(device=device)

            start_event = torch.npu.Event(enable_timing=True)
            end_event = torch.npu.Event(enable_timing=True)

            clear_l2_cache(device=device, size_bytes=l2_clear_size)

            start_event.record()
            kernel_fn(*args)
            end_event.record()

            torch.npu.synchronize(device=device)

            elapsed_time_ms = start_event.elapsed_time(end_event)
            if trial >= discard_first:
                elapsed_times.append(elapsed_time_ms)
    finally:
        torch.npu.set_device(previous_device)

    return elapsed_times


def get_timing_stats(elapsed_times: Sequence[float]) -> dict[str, float | int]:
    """Return mean/std/min/max/num_trials rounded to 3 significant digits.

    Args:
        elapsed_times: Per-trial latencies in milliseconds.

    Returns:
        Statistics mapping used in ``eval_result.json``.

    Raises:
        ValueError: If ``elapsed_times`` is empty.
    """
    if not elapsed_times:
        raise ValueError("elapsed_times must be a non-empty sequence")
    values = np.asarray(elapsed_times, dtype=np.float64)
    return {
        "mean": float(f"{float(np.mean(values)):.3g}"),
        "std": float(f"{float(np.std(values)):.3g}"),
        "min": float(f"{float(np.min(values)):.3g}"),
        "max": float(f"{float(np.max(values)):.3g}"),
        "num_trials": len(elapsed_times),
    }

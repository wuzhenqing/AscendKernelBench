"""NPU Event timing with per-trial L2 clearing (README 3.6).

Semantics follow KernelBench's ``time_execution_with_cuda_event`` ported to
``torch.npu``: warmup with synchronize, empty_cache, then per trial
synchronize -> event pair around the callable -> L2 thrash -> synchronize,
discarding the first trial. torch imports stay inside functions so the host
process can import this module without initializing an NPU runtime.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def clear_l2_cache(device: Any) -> None:
    """Clear L2 cache by thrashing with a 256MB dummy tensor."""
    import torch

    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device)
    dummy.fill_(42)
    del dummy


def time_execution_with_npu_event(
    kernel_fn,
    args: list,
    num_warmup: int = 3,
    num_trials: int = 100,
    discard_first: int = 1,
    device: Any = None,
    setup=None,
) -> list[float]:
    """Time ``kernel_fn(*args)`` over trials with torch.npu.Event (ms).

    ``setup`` (optional) runs before every warmup and timed call, outside the
    event window — use it to hand the callable fresh inputs per trial so a
    result cache cannot replay one computation across all trials.
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

            clear_l2_cache(device=device)

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


def get_timing_stats(elapsed_times: list[float]) -> dict[str, float | int]:
    """Timing statistics (mean/std/min/max/num_trials, 3-significant digits)."""
    return {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

"""Isolated eager-reference timing worker for archived baselines.

Evaluation always re-measures the live NPU reference in the same process as
the candidate. This worker only archives ``torch_npu`` eager timings so
runs stay comparable across sessions. See docs/guide/evaluation.md.
"""

from __future__ import annotations

import sys
from typing import Any

from .compare import move_value_to_device, tensor_nbytes
from .eval_device import exec_python_source
from .io_util import load_cfg_argv, pop_required_path, write_json_atomic
from .runtime import npu_device_index, seed_torch, torch_dtype_for
from .timing import (
    DEFAULT_L2_CLEAR_BYTES,
    REFRESH_INPUT_BYTES_LIMIT,
    get_timing_stats,
    time_execution_with_npu_event,
)


def measure_reference(cfg: dict[str, Any]) -> dict[str, Any]:
    """Time the task ``Model`` on NPU using the evaluation timing protocol.

    Args:
        cfg: Worker config with ``task_py``, ``device``, ``precision``,
            ``seed``, ``num_warmup``, ``num_perf_trials``, and optional
            ``l2_clear_size``.

    Returns:
        Timing statistics plus ``supported_on_npu``. Unsupported eager
        ops return ``supported_on_npu: False`` and an error string.
    """
    import torch
    import torch_npu  # noqa: F401

    ref_globals = exec_python_source(cfg["task_py"], "<task.py>")
    model_cls = ref_globals["Model"]
    get_init_inputs = ref_globals["get_init_inputs"]
    get_inputs = ref_globals["get_inputs"]

    device = torch.device(cfg["device"])
    torch.npu.set_device(device.index or npu_device_index(cfg["device"]))
    dtype = torch_dtype_for(cfg["precision"])
    l2_clear_size = int(cfg.get("l2_clear_size", DEFAULT_L2_CLEAR_BYTES))

    def process_input(value: Any) -> Any:
        """Move a tensor onto the baseline device."""
        return move_value_to_device(value, device, dtype)

    def draw() -> list[Any]:
        """Draw one processed input set."""
        return [process_input(item) for item in get_inputs()]

    try:
        seed_torch(cfg["seed"])
        model = model_cls(*get_init_inputs()).to(device=device, dtype=dtype)
        model.eval()
        # Reseed after construction so the input sequence matches
        # eval_device timing.
        seed_torch(cfg["seed"])
        box: list[Any] = [draw()]
        prev: list[Any] = [None]
        input_bytes = tensor_nbytes(box[0])
        fresh_per_trial = input_bytes <= REFRESH_INPUT_BYTES_LIMIT

        def refresh() -> None:
            """Replace the live input set and keep the previous allocation."""
            prev[0] = box[0]
            box[0] = draw()

        seed_torch(cfg["seed"])
        torch.npu.synchronize(device=device)
        with torch.no_grad():
            times = time_execution_with_npu_event(
                lambda: model(*box[0]),
                [],
                num_warmup=cfg["num_warmup"],
                num_trials=cfg["num_perf_trials"],
                device=device,
                setup=refresh if fresh_per_trial else None,
                l2_clear_size=l2_clear_size,
            )
        stats = get_timing_stats(times)
        stats["supported_on_npu"] = True
        stats["timing_fresh_inputs"] = bool(fresh_per_trial)
        return stats
    except Exception as exc:
        return {"supported_on_npu": False, "error": repr(exc)}


def main(argv: list[str]) -> None:
    """Read ``cfg.json``, time the reference, and write ``out_path``.

    Args:
        argv: ``[prog, cfg.json]``.
    """
    cfg = load_cfg_argv(argv)
    out_path = pop_required_path(cfg, "out_path")
    stats = measure_reference(cfg)
    write_json_atomic(out_path, stats)


if __name__ == "__main__":
    main(sys.argv)

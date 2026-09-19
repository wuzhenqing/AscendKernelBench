"""Runtime environment facts recorded beside each evaluation result.

Captures the software stack a score was produced on: CANN, PyTorch,
torch-npu, and the live device name. No locked clock is claimed.
"""

from __future__ import annotations

import contextlib
import os
import re
from typing import Any

_CANN_VERSION_RE = re.compile(r"cann-(\d+(?:\.\d+)*)", re.IGNORECASE)
PRECISION_DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def npu_device_index(device: str) -> int:
    """Return the numeric index from a device string, or 0 without a colon."""
    if ":" not in device:
        return 0
    return int(device.split(":", 1)[1])


def seed_torch(value: int) -> None:
    """Seed the CPU and NPU RNGs with value."""
    import torch

    torch.manual_seed(value)
    torch.npu.manual_seed(value)


def torch_dtype_for(precision: str) -> Any:
    """Return the torch floating dtype for a precision key, default fp32."""
    import torch

    return getattr(torch, PRECISION_DTYPES.get(precision, "float32"))


def cann_runtime_facts() -> dict[str, str]:
    """Return CANN identity from the environment, or empty when unset."""
    facts: dict[str, str] = {}
    explicit = os.environ.get("ASCEND_VERSION") or os.environ.get(
        "CANN_VERSION"
    )
    if explicit:
        facts["cann_version"] = explicit.strip()
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get(
        "ASCEND_TOOLKIT_HOME"
    )
    if home:
        facts["ascend_home"] = home
        if "cann_version" not in facts:
            match = _CANN_VERSION_RE.search(home)
            if match:
                facts["cann_version"] = match.group(1)
    if "cann_version" not in facts:
        set_env = os.environ.get("CANN_SET_ENV", "")
        match = _CANN_VERSION_RE.search(set_env)
        if match:
            facts["cann_version"] = match.group(1)
    return facts


def npu_runtime_metadata(device: str) -> dict[str, Any]:
    """Return versions and device identity for the current NPU process."""
    metadata: dict[str, Any] = {"device": device, **cann_runtime_facts()}
    try:
        import torch
        import torch_npu
    except ImportError:
        return metadata
    metadata["torch_version"] = torch.__version__
    metadata["torch_npu_version"] = getattr(torch_npu, "__version__", "unknown")
    with contextlib.suppress(Exception):
        metadata["device_name"] = torch.npu.get_device_name(
            npu_device_index(device)
        )
    return metadata

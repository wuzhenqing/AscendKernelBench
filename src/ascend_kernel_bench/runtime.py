"""Runtime environment facts recorded beside each evaluation result.

SOL-ExecBench publishes the software stack used for a score. This module
captures the same class of facts for an Ascend worker: CANN, PyTorch,
torch-npu, and the live device name. It does not claim a locked clock or a
SOLAR characterization of the machine.
"""

from __future__ import annotations

import os
import re
from typing import Any

_CANN_VERSION_RE = re.compile(r"cann-(\d+(?:\.\d+)*)", re.IGNORECASE)
PRECISION_DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def npu_device_index(device: str) -> int:
    """Return the numeric index from a device string such as ``npu:0``.

    Args:
        device: Runtime device string.

    Returns:
        Device index; ``0`` when the string has no colon.
    """
    if ":" not in device:
        return 0
    return int(device.split(":", 1)[1])


def seed_torch(value: int) -> None:
    """Seed CPU and NPU RNGs to ``value``.

    Args:
        value: Seed forwarded to ``torch.manual_seed`` and
            ``torch.npu.manual_seed``.
    """
    import torch

    torch.manual_seed(value)
    torch.npu.manual_seed(value)


def torch_dtype_for(precision: str) -> Any:
    """Return the torch floating dtype for an evaluation precision key.

    Args:
        precision: ``fp32``, ``fp16``, or ``bf16``. Unknown keys use fp32.

    Returns:
        A ``torch.dtype``.
    """
    import torch

    return getattr(torch, PRECISION_DTYPES.get(precision, "float32"))


def cann_runtime_facts() -> dict[str, str]:
    """Return CANN identity parsed from the process environment.

    Returns:
        Zero or more of ``cann_version`` and ``ascend_home``. Missing
        environment variables produce an empty mapping rather than an error.
    """
    facts: dict[str, str] = {}
    explicit = os.environ.get("ASCEND_VERSION") or os.environ.get("CANN_VERSION")
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
    """Return versions and device identity for the current NPU process.

    Args:
        device: Runtime device string, for example ``npu:0``.

    Returns:
        Metadata keys ``device``, optional CANN facts, and when the
        backends import, ``torch_version``, ``torch_npu_version``, and
        ``device_name``.
    """
    metadata: dict[str, Any] = {"device": device, **cann_runtime_facts()}
    try:
        import torch
        import torch_npu
    except ImportError:
        return metadata
    metadata["torch_version"] = torch.__version__
    metadata["torch_npu_version"] = getattr(torch_npu, "__version__", "unknown")
    try:
        metadata["device_name"] = torch.npu.get_device_name(
            npu_device_index(device)
        )
    except Exception:
        pass
    return metadata

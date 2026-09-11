"""Operator compilation modes.

AscendKernelBench will support two ways to turn generated Ascend C into a
callable PyTorch operator:

- ``aclnn`` (implemented): a modern CMake operator project that builds a
  process-local shared library (``libcustom_op.so``) and loads it inside the
  evaluating PyTorch process. The library is never installed into site-packages
  or the global CANN OPP vendors path, so parallel jobs and multi-machine
  workers do not lock a shared Python / CANN install.
- ``jit`` (not implemented yet): KernelBench-style just-in-time compilation
  of sources inside the PyTorch process. Reserved; requesting it fails clearly.

This module is the single source of mode names and validation.
"""

from __future__ import annotations

ACLNN_MODE = "aclnn"
JIT_MODE = "jit"
OPERATOR_MODES = (ACLNN_MODE, JIT_MODE)
DEFAULT_OPERATOR_MODE = ACLNN_MODE

# Artifact produced by the ACLNN CMake project and loaded in-process.
ACLNN_SHARED_LIBRARY_NAME = "libcustom_op.so"


class OperatorModeError(ValueError):
    """Unknown or not-yet-implemented operator mode."""


def normalize_operator_mode(mode: str | None) -> str:
    """Return a canonical mode name or raise :class:`OperatorModeError`."""
    resolved = DEFAULT_OPERATOR_MODE if mode is None else str(mode).strip().lower()
    if resolved not in OPERATOR_MODES:
        raise OperatorModeError(
            f"Unknown operator_mode {mode!r}; expected one of {OPERATOR_MODES}"
        )
    return resolved


def require_implemented_mode(mode: str | None) -> str:
    """Validate ``mode`` and reject the reserved JIT path."""
    resolved = normalize_operator_mode(mode)
    if resolved == JIT_MODE:
        raise OperatorModeError(
            "JIT operator mode is not implemented yet. "
            "Use operator_mode='aclnn' to build a process-local shared library "
            "and load it in the evaluating PyTorch process."
        )
    return resolved

"""Load a process-local ACLNN shared library into the current PyTorch process.

The evaluator builds ``libcustom_op.so`` next to the sample sources and then
calls :func:`load_process_local_op`, which uses ``torch.ops.load_library``.
That is the only supported deployment: no pybind import, no ``cmake --install``,
no ``pip install``, and no CANN OPP / ``custom_opp`` package. Parallel workers
each load their own sample-local ``.so``.
"""

from __future__ import annotations

from pathlib import Path

from .modes import ACLNN_SHARED_LIBRARY_NAME


class LoadError(RuntimeError):
    """Raised when the sample-local shared library cannot be loaded."""


def has_torch_library(source: str) -> bool:
    return "TORCH_LIBRARY" in source


def load_process_local_op(so_path: Path, source: str = "") -> None:
    """``torch.ops.load_library`` ``so_path`` so ``torch.ops.custom_op.*`` works.

    ``source`` is optional and used only to fail fast when the generated
    file is not a ``TORCH_LIBRARY`` operator.
    """
    so_path = Path(so_path).resolve()
    if not so_path.is_file():
        raise LoadError(f"shared library not found: {so_path}")
    if source and not has_torch_library(source):
        raise LoadError(
            f"{so_path.name} is missing TORCH_LIBRARY; the evaluator loads "
            "operators with torch.ops.load_library only"
        )
    try:
        import torch
    except ImportError as exc:
        raise LoadError("torch is required to load a TORCH_LIBRARY operator") from exc
    try:
        torch.ops.load_library(str(so_path))
    except Exception as exc:
        raise LoadError(f"torch.ops.load_library({so_path}) failed: {exc!r}") from exc


def expected_library_name() -> str:
    return ACLNN_SHARED_LIBRARY_NAME

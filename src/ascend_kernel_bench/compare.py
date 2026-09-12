"""Dtype-aware comparison of reference and candidate outputs.

KernelBench-style matching: floating/complex tensors use ``allclose``;
integer and Boolean tensors require exact equality. Nested tuples/lists
are compared recursively. CPU-reference mode moves values to host memory
before comparison so an NPU candidate can be scored against a CPU model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

CompareFn = Callable[[Any, Any], bool]


def resolve_tolerances(
    precision: str,
    tolerances: Mapping[str, Mapping[str, float]],
    task_tolerance: Mapping[str, Any] | None,
) -> tuple[float, float]:
    """Resolve ``atol`` / ``rtol`` from a task override or the eval config.

    Args:
        precision: Floating dtype key (``fp32``, ``fp16``, or ``bf16``).
        tolerances: Per-precision mapping from the evaluation config.
        task_tolerance: Optional nonempty task ``TOLERANCE`` dict.

    Returns:
        ``(atol, rtol)``. Missing task keys fall back to ``1e-4``.
    """
    if task_tolerance:
        return (
            float(task_tolerance.get("atol", 1e-4)),
            float(task_tolerance.get("rtol", 1e-4)),
        )
    tol = tolerances.get(precision, {"atol": 1e-4, "rtol": 1e-4})
    return float(tol["atol"]), float(tol["rtol"])


def is_discrete_tensor(value: Any) -> bool:
    """Return True if ``value`` is a non-floating, non-complex tensor.

    Args:
        value: Object that may be a ``torch.Tensor``.

    Returns:
        True when ``value`` is an integer or Boolean tensor.
    """
    import torch

    return isinstance(value, torch.Tensor) and not (
        torch.is_floating_point(value) or torch.is_complex(value)
    )


def tensor_match(ref: Any, new: Any, *, atol: float, rtol: float) -> bool:
    """Return True if two tensors match in shape and values.

    Integer/Boolean pairs use exact equality. Other tensors use
    ``torch.allclose`` with ``atol`` / ``rtol``. A dtype that
    ``allclose`` rejects is treated as a mismatch.

    Args:
        ref: Reference tensor.
        new: Candidate tensor.
        atol: Absolute tolerance for floating/complex comparison.
        rtol: Relative tolerance for floating/complex comparison.

    Returns:
        True when shapes and values are considered equal.
    """
    import torch

    if not isinstance(ref, torch.Tensor) or not isinstance(new, torch.Tensor):
        return False
    if ref.shape != new.shape:
        return False
    if is_discrete_tensor(ref) and is_discrete_tensor(new):
        return bool(torch.equal(ref, new))
    try:
        return bool(torch.allclose(ref, new, atol=atol, rtol=rtol))
    except RuntimeError:
        return False


def outputs_match(ref: Any, new: Any, *, atol: float, rtol: float) -> bool:
    """Recursively compare a reference output tree to a candidate tree.

    Args:
        ref: Reference output (tensor, sequence, or scalar).
        new: Candidate output of the same structure.
        atol: Absolute tolerance forwarded to :func:`tensor_match`.
        rtol: Relative tolerance forwarded to :func:`tensor_match`.

    Returns:
        True when the trees match under KernelBench comparison rules.
    """
    import torch

    if isinstance(ref, torch.Tensor):
        return tensor_match(ref, new, atol=atol, rtol=rtol)
    if isinstance(ref, (tuple, list)):
        return (
            isinstance(new, (tuple, list))
            and len(ref) == len(new)
            and all(
                outputs_match(left, right, atol=atol, rtol=rtol)
                for left, right in zip(ref, new, strict=True)
            )
        )
    return bool(ref == new)


def to_cpu_compare_value(value: Any) -> Any:
    """Move a tensor to CPU; cast floating values to float32 for comparison.

    Args:
        value: Tensor or non-tensor passthrough.

    Returns:
        A CPU tensor suitable for comparison, or ``value`` unchanged.
    """
    import torch

    if not isinstance(value, torch.Tensor):
        return value
    if torch.is_floating_point(value):
        return value.float().cpu()
    return value.cpu()


def to_cpu_compare_tree(value: Any) -> Any:
    """Apply :func:`to_cpu_compare_value` to a tensor or a flat sequence.

    Args:
        value: Tensor or one-level sequence of values.

    Returns:
        The same structure with tensors moved to CPU.
    """
    if isinstance(value, (tuple, list)):
        return [to_cpu_compare_value(item) for item in value]
    return to_cpu_compare_value(value)


def compare_candidate_outputs(
    ref: Any,
    new: Any,
    *,
    ref_mode: str,
    atol: float,
    rtol: float,
    custom_check: CompareFn | None,
) -> bool:
    """Compare candidate outputs, optionally after a CPU transfer.

    Args:
        ref: Reference output tree.
        new: Candidate output tree.
        ref_mode: ``"npu"`` or ``"cpu"``. CPU mode moves both trees to host
            memory and casts floating tensors to float32.
        atol: Absolute tolerance for the default matcher.
        rtol: Relative tolerance for the default matcher.
        custom_check: Optional task-provided ``custom_check(ref, out)``.

    Returns:
        True when the candidate is accepted.
    """
    if ref_mode == "cpu":
        ref = to_cpu_compare_tree(ref)
        new = to_cpu_compare_tree(new)
    if custom_check is not None:
        return bool(custom_check(ref, new))
    return outputs_match(ref, new, atol=atol, rtol=rtol)


def max_abs_diff(ref: Any, new: Any) -> float:
    """Return a finite max-abs difference for equal-shaped top-level tensors.

    Args:
        ref: Reference value.
        new: Candidate value.

    Returns:
        The maximum absolute difference, or ``0.0`` when it cannot be
        computed (non-tensors, shape mismatch, or a non-numeric dtype).
    """
    import torch

    if not (
        isinstance(ref, torch.Tensor)
        and isinstance(new, torch.Tensor)
        and ref.shape == new.shape
    ):
        return 0.0
    if ref.device != new.device:
        ref = ref.cpu()
        new = new.cpu()
    if torch.is_floating_point(ref) or torch.is_floating_point(new):
        ref = ref.float()
        new = new.float()
    try:
        value = float(torch.abs(ref - new).max().item())
    except (RuntimeError, TypeError):
        return 0.0
    if value != value:  # NaN
        return 0.0
    return value


def snapshot_inputs(inputs: Sequence[Any]) -> list[Any]:
    """Clone top-level tensors so later mutations can be detected.

    Args:
        inputs: Candidate forward arguments after device transfer.

    Returns:
        A list of cloned tensors and unchanged non-tensors.
    """
    import torch

    return [
        item.clone() if isinstance(item, torch.Tensor) else item
        for item in inputs
    ]


def inputs_were_mutated(inputs: Sequence[Any], snapshot: Sequence[Any]) -> bool:
    """Return True if any top-level tensor differs from its snapshot.

    Args:
        inputs: Arguments after the candidate forward.
        snapshot: Clones taken before that forward.

    Returns:
        True when any paired tensors are not exactly equal.
    """
    import torch

    return any(
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and not torch.equal(left, right)
        for left, right in zip(inputs, snapshot, strict=True)
    )


def move_value_to_device(value: Any, device: Any, dtype: Any) -> Any:
    """Move a tensor to ``device``, casting only floating dtypes.

    Integer and Boolean tensors keep their dtype. Non-tensors pass through.

    Args:
        value: Tensor or passthrough object.
        device: ``torch.device`` accepted by ``Tensor.to``.
        dtype: Floating dtype applied to floating tensors.

    Returns:
        A device-resident tensor, or ``value`` unchanged.
    """
    import torch

    if not isinstance(value, torch.Tensor):
        return value
    if torch.is_floating_point(value):
        return value.to(device=device, dtype=dtype)
    return value.to(device=device)


def tensor_nbytes(value: Any) -> int:
    """Return the number of bytes in tensors in ``value`` (one nesting level).

    Args:
        value: A tensor, a sequence of tensors, or an unrelated object.

    Returns:
        Sum of ``numel * element_size`` over tensors; ``0`` otherwise.
    """
    import torch

    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return sum(tensor_nbytes(item) for item in value)
    return 0

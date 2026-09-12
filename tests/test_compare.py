"""Dtype-aware output comparison helpers."""

import pytest

from ascend_kernel_bench.compare import (
    compare_candidate_outputs,
    inputs_were_mutated,
    is_discrete_tensor,
    max_abs_diff,
    move_value_to_device,
    outputs_match,
    resolve_tolerances,
    snapshot_inputs,
    tensor_nbytes,
)

torch = pytest.importorskip("torch")


def test_discrete_and_float_match() -> None:
    ints = torch.tensor([1, 2, 3])
    floats = torch.tensor([1.0, 2.0, 3.0])
    assert is_discrete_tensor(ints)
    assert not is_discrete_tensor(floats)
    assert outputs_match(ints, ints.clone(), atol=1e-4, rtol=1e-4)
    assert not outputs_match(
        ints, torch.tensor([1, 2, 4]), atol=1e-4, rtol=1e-4
    )
    close = torch.tensor([1.0, 2.0, 3.00001])
    assert outputs_match(floats, close, atol=1e-4, rtol=1e-4)


def test_nested_and_custom_check() -> None:
    ref = (torch.tensor([1.0]), torch.tensor([2.0]))
    new = [torch.tensor([1.0]), torch.tensor([2.0])]
    assert outputs_match(ref, new, atol=1e-5, rtol=1e-5)
    assert compare_candidate_outputs(
        ref,
        new,
        ref_mode="npu",
        atol=1e-5,
        rtol=1e-5,
        custom_check=None,
    )
    assert compare_candidate_outputs(
        ref,
        new,
        ref_mode="npu",
        atol=0.0,
        rtol=0.0,
        custom_check=lambda _r, _n: True,
    )


def test_tensor_nbytes_and_max_diff() -> None:
    tensor = torch.ones(4, 4, dtype=torch.float32)
    assert tensor_nbytes(tensor) == 64
    assert tensor_nbytes([tensor, tensor]) == 128
    assert tensor_nbytes("x") == 0
    other = tensor + 0.5
    assert abs(max_abs_diff(tensor, other) - 0.5) < 1e-6
    assert max_abs_diff(tensor, torch.ones(2, 2)) == 0.0


def test_move_value_to_device_casts_only_floats() -> None:
    device = torch.device("cpu")
    floats = torch.ones(2, dtype=torch.float64)
    ints = torch.tensor([1, 2], dtype=torch.int32)
    moved = move_value_to_device(floats, device, torch.float32)
    assert moved.dtype == torch.float32
    assert move_value_to_device(ints, device, torch.float32).dtype == torch.int32
    assert move_value_to_device(3, device, torch.float32) == 3


def test_snapshot_detects_input_mutation() -> None:
    tensor = torch.ones(2)
    inputs = [tensor]
    snapshot = snapshot_inputs(inputs)
    assert not inputs_were_mutated(inputs, snapshot)
    tensor.add_(1)
    assert inputs_were_mutated(inputs, snapshot)


def test_resolve_tolerances_prefers_task_override() -> None:
    atol, rtol = resolve_tolerances(
        "fp32",
        {"fp32": {"atol": 1e-4, "rtol": 1e-4}},
        {"atol": 1e-2},
    )
    assert atol == 1e-2
    assert rtol == 1e-4

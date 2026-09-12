"""Host-side evaluation helpers that do not require an NPU."""

import pytest

from ascend_kernel_bench.eval import (
    _read_worker_payload,
    eval_protocol_metadata,
    fail_result,
)
from ascend_kernel_bench.eval_device import (
    cpu_reference_inputs,
    exec_python_source,
)
from ascend_kernel_bench.eval_result import compiled_result
from ascend_kernel_bench.runtime import npu_device_index


def test_fail_result_drops_reserved_keys() -> None:
    result = fail_result(
        compiled=True,
        correctness=False,
        runtime_error="boom",
        static_check_error=["x"],
    )
    assert result["correctness"] is False
    assert result["compiled"] is True
    assert result["runtime"] is None
    assert "correctness" not in result["metadata"]
    assert result["metadata"]["runtime_error"] == "boom"
    assert result["metadata"]["static_check_error"] == ["x"]


def test_eval_protocol_metadata_snapshot() -> None:
    metadata = eval_protocol_metadata(
        hardware_name="ascend910b2",
        precision="fp32",
        seed=42,
        num_correct_trials=5,
        num_warmup=10,
        num_perf_trials=100,
        operator_mode="aclnn",
        l2_clear_size=384 * 1024 * 1024,
        atol=1e-4,
        rtol=1e-4,
    )
    assert metadata["hardware"] == "ascend910b2"
    assert metadata["seed"] == 42
    assert metadata["num_warmup"] == 10
    assert metadata["num_perf_trials"] == 100
    assert metadata["l2_clear_size"] == 384 * 1024 * 1024
    assert metadata["operator_mode"] == "aclnn"


def test_compiled_result_schema() -> None:
    payload = compiled_result(
        correctness=True,
        metadata={"hardware": "ascend910b2"},
        runtime=1.5,
    )
    assert payload["compiled"] is True
    assert payload["correctness"] is True
    assert payload["runtime"] == 1.5
    assert payload["ref_runtime"] is None


def test_npu_device_index() -> None:
    assert npu_device_index("npu:0") == 0
    assert npu_device_index("npu:3") == 3
    assert npu_device_index("npu") == 0


def test_read_worker_payload_rejects_non_object(tmp_path) -> None:
    missing = _read_worker_payload(tmp_path / "absent.json")
    assert missing["correctness"] is False
    assert "no result.json" in missing["metadata"]["runtime_error"]

    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    result = _read_worker_payload(bad)
    assert result["correctness"] is False
    assert "not an object" in result["metadata"]["runtime_error"]

    ok = tmp_path / "ok.json"
    ok.write_text('{"compiled": true, "correctness": true}', encoding="utf-8")
    assert _read_worker_payload(ok)["compiled"] is True


def test_exec_python_source_populates_namespace() -> None:
    namespace = exec_python_source("VALUE = 7\n", "<mem>")
    assert namespace["VALUE"] == 7


def test_cpu_reference_inputs_casts_floats() -> None:
    pytest.importorskip("torch")
    import torch

    raw = [torch.ones(2, dtype=torch.float64), torch.tensor([1, 2])]
    out = cpu_reference_inputs(raw, torch)
    assert out[0].dtype == torch.float32
    assert out[1].dtype == torch.int64

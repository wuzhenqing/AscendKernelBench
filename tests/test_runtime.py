"""Software-stack metadata helpers."""

from ascend_kernel_bench.runtime import cann_runtime_facts, npu_runtime_metadata


def test_cann_runtime_facts_from_home(monkeypatch) -> None:
    monkeypatch.setenv("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.1.0")
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_VERSION", raising=False)
    monkeypatch.delenv("CANN_VERSION", raising=False)
    monkeypatch.delenv("CANN_SET_ENV", raising=False)
    facts = cann_runtime_facts()
    assert facts["cann_version"] == "9.1.0"
    assert facts["ascend_home"] == "/usr/local/Ascend/cann-9.1.0"


def test_cann_runtime_facts_prefers_explicit_version(monkeypatch) -> None:
    monkeypatch.setenv("ASCEND_VERSION", "9.1.0")
    monkeypatch.setenv("ASCEND_HOME_PATH", "/opt/cann-8.0.0")
    facts = cann_runtime_facts()
    assert facts["cann_version"] == "9.1.0"


def test_npu_runtime_metadata_always_records_device() -> None:
    metadata = npu_runtime_metadata("npu:0")
    assert metadata["device"] == "npu:0"

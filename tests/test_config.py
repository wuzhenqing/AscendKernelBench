"""Hardware profile and eval config loading."""

from ascend_kernel_bench.config import load_eval_config, load_hardware_profile
from ascend_kernel_bench.timing import DEFAULT_L2_CLEAR_BYTES, l2_clear_bytes


def test_default_eval_config_warmup() -> None:
    config = load_eval_config()
    assert config.num_warmup == 10
    assert config.num_perf_trials == 100
    assert config.operator_mode == "aclnn"


def test_910b2_profile_loads_sol_fields() -> None:
    hardware = load_hardware_profile("ascend910b2")
    assert hardware.memory_bandwidth_gbps == 1600
    assert hardware.l2_cache_mb == 192
    assert hardware.cube_core_num == 24
    assert hardware.vector_core_num == 48
    assert hardware.peak_tflops_for("fp16") == 376.0
    assert hardware.peak_tflops_for("int8") == 0.0
    assert l2_clear_bytes(hardware.l2_cache_mb) == 384 * 1024 * 1024
    assert l2_clear_bytes(0) == DEFAULT_L2_CLEAR_BYTES

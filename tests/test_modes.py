import pytest

from ascend_kernel_bench.modes import (
    ACLNN_MODE,
    DEFAULT_OPERATOR_MODE,
    JIT_MODE,
    OPERATOR_MODES,
    OperatorModeError,
    normalize_operator_mode,
    require_implemented_mode,
)
from ascend_kernel_bench.prompt import build_prompt
from ascend_kernel_bench.config import load_eval_config, load_hardware_profile
from ascend_kernel_bench.dataset import load_task


def test_default_mode_is_aclnn() -> None:
    assert DEFAULT_OPERATOR_MODE == ACLNN_MODE == "aclnn"
    assert normalize_operator_mode(None) == ACLNN_MODE
    assert require_implemented_mode("ACLNN") == ACLNN_MODE


def test_unknown_mode_rejected() -> None:
    with pytest.raises(OperatorModeError, match="Unknown"):
        normalize_operator_mode("opp")


def test_jit_reserved() -> None:
    assert JIT_MODE in OPERATOR_MODES
    with pytest.raises(OperatorModeError, match="not implemented"):
        require_implemented_mode("jit")


def test_prompt_rejects_jit() -> None:
    task = load_task("level1/19_ReLU")
    hardware = load_hardware_profile("ascend910b2")
    with pytest.raises(OperatorModeError, match="not implemented"):
        build_prompt(task, hardware, mode="zero_shot", operator_mode="jit")


def test_eval_config_default_mode() -> None:
    config = load_eval_config()
    assert config.operator_mode == ACLNN_MODE

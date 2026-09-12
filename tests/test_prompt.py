"""Prompt construction stays English and describes local ACLNN loading."""

from ascend_kernel_bench.config import load_hardware_profile
from ascend_kernel_bench.dataset import load_task
from ascend_kernel_bench.prompt import (
    SYSTEM_PROMPT,
    build_prompt,
    load_examples,
)


def test_system_prompt_is_english_process_local() -> None:
    assert "libcustom_op.so" in SYSTEM_PROMPT
    assert "torch.ops.load_library" in SYSTEM_PROMPT
    assert "TORCH_LIBRARY" in SYSTEM_PROMPT
    assert "OPP" in SYSTEM_PROMPT
    assert not any("\u4e00" <= ch <= "\u9fff" for ch in SYSTEM_PROMPT)


def test_aclnn_prompt_forbids_opp_install() -> None:
    prompt = build_prompt(
        load_task("level1/19_ReLU"),
        load_hardware_profile("ascend910b2"),
        mode="one_shot",
        operator_mode="aclnn",
    )
    assert "libcustom_op.so" in prompt
    assert "custom_opp" in prompt
    assert "torch.ops.load_library" in prompt
    assert "torch.ops.custom_op" in prompt
    assert "TORCH_LIBRARY" in prompt
    assert "TORCH_LIBRARY_IMPL" in prompt
    assert "Do NOT use" in prompt and "PYBIND11_MODULE" in prompt
    assert "Do not emit" in prompt or "must NOT emit" in prompt
    assert not any("\u4e00" <= ch <= "\u9fff" for ch in prompt)


def test_few_shot_includes_both_examples() -> None:
    examples = load_examples()
    assert [ex.name for ex in examples] == [
        "001_elementwise_add",
        "002_leaky_relu",
    ]
    prompt = build_prompt(
        load_task("level1/19_ReLU"),
        load_hardware_profile("ascend910b2"),
        mode="few_shot",
    )
    assert "001_elementwise_add" in prompt
    assert "002_leaky_relu" in prompt

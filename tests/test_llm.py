import json

from ascend_kernel_bench.llm import (
    _RETRY_REMINDER,
    AscendCGeneration,
    _append_retry_turn,
    extract_generation,
    validate_generation,
)


def test_extract_generation_from_json_body() -> None:
    text = json.dumps(
        {
            "custom_op_asc": (
                "__global__ __vector__ void k() {}\n"
                "TORCH_LIBRARY(custom_op, m) {}\n"
                "TORCH_LIBRARY_IMPL(custom_op, PrivateUse1, m) {}"
            ),
            "model_new_py": (
                "class ModelNew:\n"
                "    def forward(self, x):\n"
                "        return torch.ops.custom_op.run(x)"
            ),
        }
    )
    generation = extract_generation(text)
    assert "__global__" in generation.custom_op_asc
    assert "class ModelNew" in generation.model_new_py


def test_extract_generation_from_json_in_fence() -> None:
    body = json.dumps(
        {
            "custom_op_asc": "__global__ __vector__ void k() {}",
            "model_new_py": "class ModelNew:\n    pass",
        }
    )
    generation = extract_generation(f"```json\n{body}\n```")
    assert "__global__" in generation.custom_op_asc
    assert "class ModelNew" in generation.model_new_py


def test_extract_generation_rejects_incomplete_json() -> None:
    text = json.dumps({"custom_op_asc": "__global__ __vector__ void k() {}"})
    try:
        extract_generation(text)
    except ValueError as exc:
        assert "no JSON fields and no fenced code blocks" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected ValueError for a one-field payload")


def test_validate_rejects_pybind() -> None:
    gen = AscendCGeneration(
        custom_op_asc=(
            "__global__ __vector__ void k() {}\n"
            "PYBIND11_MODULE(custom_op, m) {}"
        ),
        model_new_py=(
            "class ModelNew:\n"
            "    def forward(self, x):\n"
            "        return torch.ops.custom_op.run(x)"
        ),
    )
    problems = validate_generation(gen)
    assert any("PYBIND11_MODULE" in item for item in problems)


def test_validate_accepts_torch_library() -> None:
    gen = AscendCGeneration(
        custom_op_asc=(
            "__global__ __vector__ void k() {}\n"
            "// ==================== ASCEND_HOST_SECTION ====================\n"
            "TORCH_LIBRARY(custom_op, m) {}\n"
            "TORCH_LIBRARY_IMPL(custom_op, PrivateUse1, m) {}"
        ),
        model_new_py=(
            "class ModelNew:\n"
            "    def forward(self, x):\n"
            "        return torch.ops.custom_op.run(x)"
        ),
    )
    assert validate_generation(gen) == []


def test_validate_rejects_missing_host_section_marker() -> None:
    gen = AscendCGeneration(
        custom_op_asc=(
            "__global__ __vector__ void k() {}\n"
            "TORCH_LIBRARY(custom_op, m) {}\n"
            "TORCH_LIBRARY_IMPL(custom_op, PrivateUse1, m) {}"
        ),
        model_new_py=(
            "class ModelNew:\n"
            "    def forward(self, x):\n"
            "        return torch.ops.custom_op.run(x)"
        ),
    )
    problems = validate_generation(gen)
    assert any("ASCEND_HOST_SECTION" in item for item in problems)


def test_validate_rejects_host_only() -> None:
    gen = AscendCGeneration(
        custom_op_asc="at::Tensor run() { return at::empty({}); }",
        model_new_py="class ModelNew: ...",
    )
    problems = validate_generation(gen)
    assert any("__global__" in item for item in problems)
    assert any("TORCH_LIBRARY" in item for item in problems)
    assert any("torch.ops.custom_op" in item for item in problems)


def test_extract_generation_from_filename_fences() -> None:
    text = (
        "```custom_op.asc\n"
        "__global__ __vector__ void k() {}\n"
        "TORCH_LIBRARY(custom_op, m) {}\n"
        "TORCH_LIBRARY_IMPL(custom_op, PrivateUse1, m) {}\n"
        "```\n"
        "```model_new.py\n"
        "class ModelNew:\n"
        "    def forward(self, x):\n"
        "        return torch.ops.custom_op.run(x)\n"
        "```\n"
    )
    generation = extract_generation(text)
    assert "__global__" in generation.custom_op_asc
    assert "class ModelNew" in generation.model_new_py


def test_append_retry_turn_includes_previous_answer() -> None:
    messages: list[dict] = [{"role": "user", "content": "prompt"}]
    _append_retry_turn(messages, "incomplete")
    assert messages[-2] == {"role": "assistant", "content": "incomplete"}
    assert messages[-1] == {"role": "user", "content": _RETRY_REMINDER}

    empty: list[dict] = []
    _append_retry_turn(empty, "")
    assert empty == [{"role": "user", "content": _RETRY_REMINDER}]


def test_extract_generation_from_language_tags() -> None:
    text = (
        "```cpp\n"
        "__global__ __vector__ void k() {}\n"
        "TORCH_LIBRARY(custom_op, m) {}\n"
        "```\n"
        "```python\n"
        "class ModelNew:\n"
        "    pass\n"
        "```\n"
    )
    generation = extract_generation(text)
    assert "__global__" in generation.custom_op_asc
    assert "class ModelNew" in generation.model_new_py

from ascend_kernel_bench.llm import AscendCGeneration, validate_generation


def test_validate_rejects_pybind() -> None:
    gen = AscendCGeneration(
        custom_op_asc="__global__ __vector__ void k() {}\nPYBIND11_MODULE(custom_op, m) {}",
        model_new_py="class ModelNew:\n    def forward(self, x):\n        return torch.ops.custom_op.run(x)",
    )
    problems = validate_generation(gen)
    assert any("PYBIND11_MODULE" in item for item in problems)


def test_validate_accepts_torch_library() -> None:
    gen = AscendCGeneration(
        custom_op_asc=(
            "__global__ __vector__ void k() {}\n"
            "TORCH_LIBRARY(custom_op, m) {}\n"
            "TORCH_LIBRARY_IMPL(custom_op, PrivateUse1, m) {}"
        ),
        model_new_py="class ModelNew:\n    def forward(self, x):\n        return torch.ops.custom_op.run(x)",
    )
    assert validate_generation(gen) == []


def test_validate_rejects_host_only() -> None:
    gen = AscendCGeneration(
        custom_op_asc="at::Tensor run() { return at::empty({}); }",
        model_new_py="class ModelNew: ...",
    )
    problems = validate_generation(gen)
    assert any("__global__" in item for item in problems)
    assert any("TORCH_LIBRARY" in item for item in problems)
    assert any("torch.ops.custom_op" in item for item in problems)

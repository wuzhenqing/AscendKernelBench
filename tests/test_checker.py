from ascend_kernel_bench.checker import check_custom_op_asc, check_model_new
from ascend_kernel_bench.prompt import load_examples

_TORCH_LIB_BINDING = """
TORCH_LIBRARY(custom_op, m) { m.def("run(Tensor x) -> Tensor"); }
TORCH_LIBRARY_IMPL(custom_op, PrivateUse1, m) { m.impl("run", TORCH_FN(run)); }
"""


def _minimal_kernel(binding: str) -> str:
    return f"""
__global__ __vector__ void k(__gm__ uint8_t* x) {{
    AscendC::InitSocState();
}}
{binding}
"""


def test_shipped_examples_pass_static_checks() -> None:
    for example in load_examples():
        assert check_custom_op_asc(example.custom_op_asc) == []
        assert check_model_new(example.model_new_py) == []


def test_torch_library_binding_is_accepted() -> None:
    assert check_custom_op_asc(_minimal_kernel(_TORCH_LIB_BINDING)) == []


def test_missing_binding_is_rejected() -> None:
    violations = check_custom_op_asc(_minimal_kernel("// no binding"))
    assert any("TORCH_LIBRARY" in item for item in violations)


def test_pybind_binding_is_rejected() -> None:
    source = _minimal_kernel("PYBIND11_MODULE(custom_op, m) {}")
    violations = check_custom_op_asc(source)
    assert any("pybind11" in item or "PYBIND11" in item for item in violations)
    assert any("TORCH_LIBRARY" in item for item in violations)


def test_vendor_aclnn_still_banned() -> None:
    source = _minimal_kernel(_TORCH_LIB_BINDING + "\nvoid h() { aclnnAdd(nullptr); }\n")
    violations = check_custom_op_asc(source)
    assert any("aclnn" in item for item in violations)


def test_model_new_allows_torch_ops_custom_op() -> None:
    source = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.ops.custom_op.run(x)
"""
    assert check_model_new(source) == []


def test_model_new_rejects_import_custom_op() -> None:
    source = """
import torch
import torch.nn as nn
import custom_op

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return custom_op.run(x)
"""
    violations = check_model_new(source)
    assert any("import custom_op" in item for item in violations)


def test_model_new_bans_load_library_and_vendor_ops() -> None:
    load_lib = """
import torch
import torch.nn as nn
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        torch.ops.load_library("libcustom_op.so")
        return torch.ops.custom_op.run(x)
"""
    assert any("load_library" in item for item in check_model_new(load_lib))

    vendor = """
import torch
import torch.nn as nn
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.ops.aten.add(x, x)
"""
    assert any("vendor" in item or "aten" in item for item in check_model_new(vendor))

"""Static checks for generated custom_op.asc source."""

from __future__ import annotations

import re

from .rules import PatternRule, run_rules
from .text import dedupe, strip_cpp_comments

_ASC_KERNEL_MARKERS = ("__global__", "__vector__")

_ASC_AT_ALLOWED_CALLS = {
    "Tensor",
    "empty",
    "empty_like",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "empty_strided",
    "from_blob",
    "scalar_tensor",
    "tensor",
}

_ASC_COMPUTE_METHODS = (
    "matmul",
    "mm",
    "bmm",
    "addmm",
    "baddbmm",
    "mv",
    "ger",
    "outer",
    "relu",
    "relu_",
    "sigmoid",
    "sigmoid_",
    "tanh",
    "tanh_",
    "gelu",
    "silu",
    "softmax",
    "log_softmax",
    "leaky_relu",
    "elu",
    "selu",
    "add",
    "add_",
    "sub",
    "sub_",
    "mul",
    "mul_",
    "div",
    "div_",
    "pow",
    "sum",
    "mean",
    "amax",
    "amin",
    "argmax",
    "argmin",
    "prod",
    "norm",
    "var",
    "std",
    "cumsum",
    "topk",
    "sort",
    "clamp",
    "gather",
    "scatter",
    "index_select",
    "conv1d",
    "conv2d",
    "conv3d",
)

########################### BANNED PATTERN CATALOG ###########################
ASC_BANNED_RULES: tuple[PatternRule, ...] = (
    PatternRule(
        (r"\baclnn[A-Z]\w*",),
        "vendor prebuilt operator (aclnn*) — implement the kernel yourself",
        include_match=True,
    ),
    PatternRule(
        (r"\baclop\w*",),
        "legacy vendor operator API (aclop*)",
        include_match=True,
    ),
    PatternRule(
        (r"\bstd::system\s*\(|\bsystem\s*\(",),
        "host process execution",
        include_match=True,
    ),
    PatternRule(
        (r"\bpopen\s*\(|\bexecl\w*\s*\(|\bexecv\w*\s*\(|\bfork\s*\(",),
        "host process execution",
        include_match=True,
    ),
    PatternRule(
        (r"\bsocket\s*\(|\bconnect\s*\(",),
        "network access",
        include_match=True,
    ),
    PatternRule(
        (r"\bdlopen\s*\(|\bdlsym\s*\(",),
        "dynamic loading",
        include_match=True,
    ),
    PatternRule(
        (r"#\s*include\s*<ATen/ops/",),
        "ATen operator headers",
        include_match=True,
    ),
    PatternRule(
        (r"\bstd::thread\b|\bpthread_create\b|\bstd::async\b",),
        "host threads (timing manipulation)",
        include_match=True,
    ),
)
########################### BANNED PATTERN CATALOG ###########################


def _check_asc_binding(code: str) -> list[str]:
    """Require kernel markers and a process-local torch.library binding."""
    violations: list[str] = []
    for marker in _ASC_KERNEL_MARKERS:
        if marker not in code:
            violations.append(
                f"custom_op.asc missing {marker!r}: not an Ascend C kernel "
                "(host-only ATen implementations are not allowed)"
            )
    if not re.search(r"\bTORCH_LIBRARY\s*\(", code):
        violations.append(
            "custom_op.asc missing TORCH_LIBRARY(...): register the operator "
            "schema so the evaluator can torch.ops.load_library the "
            "process-local .so"
        )
    if not re.search(r"\bTORCH_LIBRARY_IMPL\s*\(", code):
        violations.append(
            "custom_op.asc missing TORCH_LIBRARY_IMPL(...): bind the NPU "
            "implementation (PrivateUse1) for torch.ops.custom_op"
        )
    if "PYBIND11_MODULE" in code or "#include <pybind11/" in code:
        violations.append(
            "pybind11 is not used; register with TORCH_LIBRARY / "
            "TORCH_LIBRARY_IMPL and let the evaluator call "
            "torch.ops.load_library"
        )
    return violations


def _check_asc_aten(code: str) -> list[str]:
    """Flag host-side ATen/libtorch compute calls."""
    violations: list[str] = []
    bad_aten = sorted(
        {
            match.group(1)
            for match in re.finditer(r"\bat::(\w+)\s*\(", code)
            if match.group(1) not in _ASC_AT_ALLOWED_CALLS
        }
        | {
            match.group(1)
            for match in re.finditer(r"\btorch::(\w+)\s*\(", code)
            if match.group(1) not in _ASC_AT_ALLOWED_CALLS
        }
    )
    for name in bad_aten:
        violations.append(
            f"host-side ATen call at::{name}(...) — compute must live in the "
            "Ascend C kernel, not in libtorch"
        )
    if re.search(r"\bat::native::", code):
        violations.append(
            "at::native:: call — direct ATen kernel reuse is not allowed"
        )
    return violations


def _check_asc_methods(code: str) -> list[str]:
    """Flag host-side tensor compute methods (.relu(), ->matmul())."""
    method_re = r"(?:\.|->)(" + "|".join(_ASC_COMPUTE_METHODS) + r")\s*\("
    bad_methods = sorted(
        {match.group(1) for match in re.finditer(method_re, code)}
    )
    return [
        f"host-side tensor method .{name}(...) — compute must live in the "
        "Ascend C kernel"
        for name in bad_methods
    ]


def check_custom_op_asc(source: str) -> list[str]:
    """Return static-check violations for generated custom_op.asc source.

    Args:
        source: Raw custom_op.asc text.

    Returns:
        Deduplicated human-readable violations; empty means pass.
    """
    code = strip_cpp_comments(source)
    violations: list[str] = []
    violations.extend(_check_asc_binding(code))
    violations.extend(_check_asc_aten(code))
    violations.extend(_check_asc_methods(code))
    violations.extend(run_rules(code, ASC_BANNED_RULES))
    return dedupe(violations)

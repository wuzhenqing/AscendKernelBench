"""ACLNN build contract: process-local .so, no global install."""

from pathlib import Path

import pytest

from ascend_kernel_bench.build import BuildError, find_built_library, build_custom_op
from ascend_kernel_bench.loader import LoadError, load_process_local_op
from ascend_kernel_bench.modes import ACLNN_SHARED_LIBRARY_NAME
from ascend_kernel_bench._paths import BUILD_TEMPLATE_DIR


def test_cmake_template_is_process_local() -> None:
    text = (BUILD_TEMPLATE_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "CMAKE_SKIP_INSTALL_RULES" in text
    assert "add_library(custom_op SHARED" in text
    assert "npu_op_package(" not in text
    assert "custom_opp_*" in text  # mentioned only as something we do not produce
    assert "BUILD_RPATH" in text
    assert "libcustom_op" in text or 'OUTPUT_NAME "custom_op"' in text
    assert "CMAKE_ASC_ARCHITECTURES" in text
    assert "find_package(pybind11" not in text
    assert "pybind11::module" not in text
    assert "torch.ops.load_library" in text


def test_find_built_library_prefers_libcustom_op(tmp_path: Path) -> None:
    preferred = tmp_path / ACLNN_SHARED_LIBRARY_NAME
    preferred.write_bytes(b"so")
    (tmp_path / "custom_op.cpython-310-x86_64-linux-gnu.so").write_bytes(b"old")
    assert find_built_library(tmp_path) == preferred


def test_find_built_library_missing(tmp_path: Path) -> None:
    with pytest.raises(BuildError, match="not found"):
        find_built_library(tmp_path)


def test_build_rejects_jit(tmp_path: Path) -> None:
    with pytest.raises(BuildError, match="not implemented"):
        build_custom_op("source", tmp_path, cmake_arch="dav-2201", operator_mode="jit")


def test_loader_requires_binding(tmp_path: Path) -> None:
    so_path = tmp_path / ACLNN_SHARED_LIBRARY_NAME
    so_path.write_bytes(b"not-a-real-so")
    with pytest.raises(LoadError, match="TORCH_LIBRARY"):
        load_process_local_op(so_path, "// empty kernel")


def test_loader_missing_file(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="not found"):
        load_process_local_op(
            tmp_path / "missing.so",
            "TORCH_LIBRARY(custom_op, m) {}",
        )

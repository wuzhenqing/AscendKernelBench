"""ACLNN build contract: process-local .so, no global install."""

import os
import sys
import time
from pathlib import Path

import pytest

from ascend_kernel_bench._paths import BUILD_TEMPLATE_DIR
from ascend_kernel_bench.build import (
    SHARED_LIBRARY_NAME,
    BuildError,
    LoadError,
    _cmake_cache_matches,
    _write_if_changed,
    build_custom_op,
    find_built_library,
    load_custom_op,
)


def test_cmake_template_is_process_local() -> None:
    text = (BUILD_TEMPLATE_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "CMAKE_SKIP_INSTALL_RULES" in text
    assert "add_library(custom_op SHARED" in text
    assert "npu_op_package(" not in text
    assert "custom_opp_*" in text
    assert "BUILD_RPATH" in text
    assert "libcustom_op" in text or 'OUTPUT_NAME "custom_op"' in text
    assert "CMAKE_ASC_ARCHITECTURES" in text
    assert "find_package(pybind11" not in text
    assert "pybind11::module" not in text
    assert "torch.ops.load_library" in text


def test_cmake_template_ccache_covers_asc() -> None:
    # The .asc translation unit dominates build time; the ccache option must
    # cover the ASC compiler, not only the C/CXX launchers.
    text = (BUILD_TEMPLATE_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "CMAKE_ASC_COMPILER_LAUNCHER" in text


def test_write_if_changed_preserves_mtime(tmp_path: Path) -> None:
    target = tmp_path / "custom_op.asc"
    _write_if_changed(target, "kernel v1")
    old_mtime = os.stat(target).st_mtime_ns
    time.sleep(0.01)
    _write_if_changed(target, "kernel v1")
    assert os.stat(target).st_mtime_ns == old_mtime


def test_write_if_changed_updates_on_change(tmp_path: Path) -> None:
    target = tmp_path / "custom_op.asc"
    _write_if_changed(target, "kernel v1")
    _write_if_changed(target, "kernel v2")
    assert target.read_text(encoding="utf-8") == "kernel v2"


def test_cmake_cache_matches(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    assert not _cmake_cache_matches(build_dir, "dav-2201", False)
    (build_dir / "CMakeCache.txt").write_text(
        "CMAKE_ASC_ARCHITECTURES:STRING=dav-2201\n"
        f"Python3_EXECUTABLE:UNINITIALIZED={sys.executable}\n"
        "ENABLE_CCACHE:BOOL=OFF\n",
        encoding="utf-8",
    )
    assert _cmake_cache_matches(build_dir, "dav-2201", False)
    assert not _cmake_cache_matches(build_dir, "dav-3510", False)
    assert not _cmake_cache_matches(build_dir, "dav-2201", True)


def test_find_built_library_prefers_sample_dir(tmp_path: Path) -> None:
    preferred = tmp_path / SHARED_LIBRARY_NAME
    preferred.write_bytes(b"so")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / SHARED_LIBRARY_NAME).write_bytes(b"nested")
    assert find_built_library(tmp_path) == preferred


def test_find_built_library_falls_back_to_build_dir(tmp_path: Path) -> None:
    nested = tmp_path / "build" / SHARED_LIBRARY_NAME
    nested.parent.mkdir()
    nested.write_bytes(b"so")
    assert find_built_library(tmp_path) == nested


def test_find_built_library_missing(tmp_path: Path) -> None:
    with pytest.raises(BuildError, match="not found"):
        find_built_library(tmp_path)


def test_build_custom_op_has_no_mode_switch(tmp_path: Path) -> None:
    # Compilation is always the CMake path; operator_mode is not a build input.
    assert "operator_mode" not in build_custom_op.__code__.co_varnames


def test_loader_requires_binding(tmp_path: Path) -> None:
    so_path = tmp_path / SHARED_LIBRARY_NAME
    so_path.write_bytes(b"not-a-real-so")
    with pytest.raises(LoadError, match="TORCH_LIBRARY"):
        load_custom_op(so_path, "// empty kernel")


def test_loader_missing_file(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="not found"):
        load_custom_op(
            tmp_path / "missing.so",
            "TORCH_LIBRARY(custom_op, m) {}",
        )

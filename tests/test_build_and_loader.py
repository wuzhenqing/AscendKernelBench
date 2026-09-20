"""ACLNN build contract: process-local .so, no global install."""

import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from ascend_kernel_bench._paths import BUILD_TEMPLATE_DIR
from ascend_kernel_bench.build import (
    SHARED_LIBRARY_NAME,
    BuildError,
    LoadError,
    _cmake_cache_matches,
    _pch_key,
    _TorchFacts,
    _write_if_changed,
    build_custom_op,
    ensure_host_pch,
    extract_kernel_signatures,
    find_built_library,
    generate_launcher_decls,
    generate_launcher_source,
    load_custom_op,
    split_asc_source,
)
from ascend_kernel_bench.checks.text import HOST_SECTION_MARKER


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


def test_read_cmake_cache(tmp_path: Path) -> None:
    from ascend_kernel_bench.build import _read_cmake_cache

    assert _read_cmake_cache(tmp_path) is None
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text(
        "CMAKE_ASC_ARCHITECTURES:STRING=dav-2201\nENABLE_CCACHE:BOOL=OFF\n",
        encoding="utf-8",
    )
    assert _read_cmake_cache(build_dir) == {
        "CMAKE_ASC_ARCHITECTURES": "dav-2201",
        "ENABLE_CCACHE": "OFF",
    }


def test_cmake_cache_matches() -> None:
    expected = {
        "CMAKE_ASC_ARCHITECTURES": "dav-2201",
        "Python3_EXECUTABLE": sys.executable,
        "ENABLE_CCACHE": "OFF",
    }
    assert not _cmake_cache_matches(None, expected)
    cached = dict(expected)
    assert _cmake_cache_matches(cached, expected)
    assert not _cmake_cache_matches(
        {**cached, "CMAKE_ASC_ARCHITECTURES": "dav-3510"}, expected
    )
    assert not _cmake_cache_matches({**cached, "ENABLE_CCACHE": "ON"}, expected)
    # Extra cached variables beyond the expected set are fine.
    extra = {**cached, "CMAKE_BUILD_TYPE": "Release"}
    assert _cmake_cache_matches(extra, expected)


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


_SPLIT_SOURCE = f"""\
#include "kernel_operator.h"
__global__ __vector__ void add_custom(__gm__ uint8_t* x, __gm__ uint8_t* z,
                                      uint32_t totalLength)
{{
    AscendC::InitSocState();
}}
// ==================== {HOST_SECTION_MARKER} ====================
#include <torch/library.h>
TORCH_LIBRARY(custom_op, m) {{ m.def("run(Tensor x) -> Tensor"); }}
"""


def test_split_asc_source_legacy_returns_none() -> None:
    legacy = "#include <torch/extension.h>\n__global__ __vector__ void k() {}\n"
    assert split_asc_source(legacy) is None


def test_split_asc_source_splits_at_marker() -> None:
    device, host = split_asc_source(_SPLIT_SOURCE)  # type: ignore[misc]
    assert "kernel_operator.h" in device
    assert HOST_SECTION_MARKER not in device
    assert "torch/library.h" in host
    assert HOST_SECTION_MARKER not in host


def test_split_asc_source_rejects_multiple_markers() -> None:
    second = f"// ==== {HOST_SECTION_MARKER} ====\n"
    with pytest.raises(BuildError, match="exactly one"):
        split_asc_source(_SPLIT_SOURCE + second)


def test_split_asc_source_ignores_prose_mentions() -> None:
    source = f"# not a marker: {HOST_SECTION_MARKER} in prose\n" + _SPLIT_SOURCE
    assert split_asc_source(source) is not None


def test_extract_kernel_signatures_basic() -> None:
    device, _ = split_asc_source(_SPLIT_SOURCE)  # type: ignore[misc]
    sigs = extract_kernel_signatures(device)
    assert [sig.name for sig in sigs] == ["add_custom"]
    sig = sigs[0]
    assert "__gm__ uint8_t* x" in sig.decl_params
    assert "__gm__" not in sig.host_params
    assert "uint8_t* x" in sig.host_params
    assert sig.arg_names == "x, z, totalLength"


def test_extract_kernel_signatures_ignores_comments() -> None:
    source = (
        "// __global__ __vector__ void fake(__gm__ uint8_t* y)\n"
        "__global__ __vector__ void real(__gm__ uint8_t* x) { }\n"
    )
    sigs = extract_kernel_signatures(source)
    assert [sig.name for sig in sigs] == ["real"]


def test_extract_kernel_signatures_requires_kernel() -> None:
    with pytest.raises(BuildError, match="no __global__"):
        extract_kernel_signatures("class KernelAdd {};")
    with pytest.raises(BuildError, match="cannot parse parameter"):
        extract_kernel_signatures("__global__ __vector__ void k(uint32_t) { }")
    with pytest.raises(BuildError, match="conflicting parameter"):
        extract_kernel_signatures(
            "__global__ __vector__ void k(uint32_t a) { }\n"
            "__global__ __vector__ void k(uint32_t a, uint32_t b) { }\n"
        )


def test_generate_launcher_source_roundtrip() -> None:
    device, _ = split_asc_source(_SPLIT_SOURCE)  # type: ignore[misc]
    sigs = extract_kernel_signatures(device)
    launcher = generate_launcher_source(sigs)
    assert "__global__ __vector__ void add_custom(" in launcher
    assert 'extern "C" void add_custom_launch(' in launcher
    assert "uint32_t numBlocks, void* stream" in launcher
    launch_call = "add_custom<<<numBlocks, 0, stream>>>(x, z, totalLength);"
    assert launch_call in launcher
    decls = generate_launcher_decls(sigs)
    assert decls.count('extern "C" void add_custom_launch(') == 1
    assert decls.rstrip().endswith(";")


def test_pch_key_changes_with_facts() -> None:
    facts = _TorchFacts(
        bisheng="/opt/bisheng",
        bisheng_version="clang 15",
        torch_version="2.10.0",
        torch_npu_version="2.10.0",
        python_version="3.12.13",
        include_dirs=("/a/include",),
        defines=("USE_DISTRIBUTED",),
        gcc_toolchain="/usr",
    )
    key = _pch_key(facts)
    assert key == _pch_key(facts)
    assert key != _pch_key(replace(facts, torch_version="2.11.0"))


def test_ensure_host_pch_disabled_without_torch(monkeypatch) -> None:
    monkeypatch.setenv("ASCEND_KERNEL_BENCH_DISABLE_PCH", "1")
    assert ensure_host_pch() is None

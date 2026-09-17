"""Build and load a process-local ``libcustom_op.so``.

Writes ``custom_op.asc`` into the sample directory, compiles it with the
fixed CMake template, and loads the resulting shared library with
``torch.ops.load_library``. Nothing is installed globally.

The CMake backend follows the official asc-devkit torch.library sample
(CANN >= 9.1). If a future torch_npu ships an AscendC-aware
``cpp_extension`` (``load_inline``-style JIT for ``.asc`` sources), this
module's two entry points — ``build_custom_op`` / ``load_custom_op`` —
are the only surface that needs to change.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from ._paths import BUILD_TEMPLATE_DIR

SHARED_LIBRARY_NAME = "libcustom_op.so"
DEFAULT_CANN_SET_ENV = "/usr/local/Ascend/cann-9.1.0/set_env.sh"


class BuildError(RuntimeError):
    """Raised when configure or build fails; message carries compiler output."""


class LoadError(RuntimeError):
    """Raised when the sample-local shared library cannot be loaded."""


@lru_cache(maxsize=1)
def cann_env() -> dict[str, str]:
    """Capture the environment produced by sourcing CANN's set_env.sh.

    Cached per process. Set ``CANN_SET_ENV`` to override the script path.
    If that file is missing, the current environment is used as-is.
    """
    set_env = os.environ.get("CANN_SET_ENV", DEFAULT_CANN_SET_ENV)
    if not Path(set_env).is_file():
        return dict(os.environ)
    result = subprocess.run(
        ["bash", "-c", f'source "{set_env}" >/dev/null 2>&1 && env'],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise BuildError(
            f"Failed to source CANN env {set_env}: {result.stderr}"
        )
    env = dict(os.environ)
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            env[key] = value
    return env


def build_custom_op(
    asc_source: str,
    work_dir: Path,
    *,
    cmake_arch: str,
    timeout_s: int = 600,
) -> Path:
    """Write ``custom_op.asc`` into ``work_dir`` and build ``libcustom_op.so``.

    The sample directory doubles as the build cache: sources are written
    only when their content changes, and CMake configure is skipped when
    the existing cache already matches the request, so re-evaluating an
    unchanged sample is an incremental no-op build.

    Args:
        asc_source: Generated Ascend C source.
        work_dir: Sample directory that will hold sources and ``.so``.
        cmake_arch: Value passed to ``CMAKE_ASC_ARCHITECTURES``.
        timeout_s: Separate budget for configure and for build.

    Returns:
        Path to the process-local shared library.

    Raises:
        BuildError: If CMake fails or no ``.so`` is produced.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    # Write only on content change (torch's _maybe_write pattern): the
    # sample dir doubles as the build cache, so an unchanged source lets
    # make skip the Ascend C recompile on re-evaluation.
    _write_if_changed(work_dir / "custom_op.asc", asc_source)
    _write_if_changed(
        work_dir / "CMakeLists.txt",
        (BUILD_TEMPLATE_DIR / "CMakeLists.txt").read_text(encoding="utf-8"),
    )

    build_dir = work_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    env = cann_env()
    env.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "0")
    enable_ccache = os.environ.get("AKB_ENABLE_CCACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # Configure is idempotent; skip it when the existing cache already
    # matches this request. A changed CMakeLists.txt still triggers CMake's
    # own re-configure during the build step.
    if not _cmake_cache_matches(build_dir, cmake_arch, enable_ccache):
        configure_cmd = [
            "cmake",
            "-S",
            str(work_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_ASC_ARCHITECTURES={cmake_arch}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={work_dir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]
        if enable_ccache:
            configure_cmd.append("-DENABLE_CCACHE=ON")
        _run_cmake(
            "configure",
            configure_cmd,
            cwd=work_dir,
            env=env,
            timeout_s=timeout_s,
        )
    _run_cmake(
        "build",
        ["cmake", "--build", str(build_dir), "-j"],
        cwd=work_dir,
        env=env,
        timeout_s=timeout_s,
    )
    return find_built_library(work_dir)


def _cmake_cache_matches(
    build_dir: Path, cmake_arch: str, enable_ccache: bool
) -> bool:
    """Return True when the existing configure cache matches this request."""
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return False
    values: dict[str, str] = {}
    text = cache.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key.split(":", 1)[0]] = value
    return (
        values.get("CMAKE_ASC_ARCHITECTURES") == cmake_arch
        and values.get("Python3_EXECUTABLE") == sys.executable
        and values.get("ENABLE_CCACHE") == ("ON" if enable_ccache else "OFF")
    )


def _write_if_changed(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` only when it differs.

    Preserves mtime for unchanged content so the CMake build stays
    incremental across evaluations of the same sample.
    """
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def find_built_library(work_dir: Path) -> Path:
    """Return ``libcustom_op.so`` from ``work_dir`` or ``work_dir/build``."""
    work_dir = Path(work_dir)
    for candidate in (
        work_dir / SHARED_LIBRARY_NAME,
        work_dir / "build" / SHARED_LIBRARY_NAME,
    ):
        if candidate.is_file():
            return candidate
    raise BuildError(
        f"Built shared library not found in {work_dir} "
        f"(expected {SHARED_LIBRARY_NAME})"
    )


def load_custom_op(so_path: Path, source: str = "") -> None:
    """Load ``so_path`` so ``torch.ops.custom_op.*`` becomes callable.

    Args:
        so_path: Process-local ``libcustom_op.so``.
        source: Optional ``custom_op.asc`` text used for a fast marker check.

    Raises:
        LoadError: If the file is missing, lacks ``TORCH_LIBRARY``, or
            ``torch.ops.load_library`` fails.
    """
    so_path = Path(so_path).resolve()
    if not so_path.is_file():
        raise LoadError(f"shared library not found: {so_path}")
    if source and "TORCH_LIBRARY" not in source:
        raise LoadError(
            f"{so_path.name} is missing TORCH_LIBRARY; the evaluator loads "
            "operators with torch.ops.load_library only"
        )
    try:
        import torch
    except ImportError as exc:
        raise LoadError(
            "torch is required to load a TORCH_LIBRARY operator"
        ) from exc
    try:
        torch.ops.load_library(str(so_path))
    except Exception as exc:
        raise LoadError(
            f"torch.ops.load_library({so_path}) failed: {exc!r}"
        ) from exc


def _run_cmake(
    stage: str,
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_s: int,
) -> None:
    """Run one CMake stage, persist the log, and raise on failure."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise BuildError(f"{stage} timed out after {timeout_s}s") from exc
    log = (result.stdout or "") + "\n" + (result.stderr or "")
    log_path = cwd / "build" / f"{stage}.log"
    with contextlib.suppress(OSError):
        log_path.write_text(log, encoding="utf-8")
    if result.returncode != 0:
        tail = "\n".join(log.strip().splitlines()[-60:])
        raise BuildError(f"{stage} failed (see {log_path}):\n{tail}")

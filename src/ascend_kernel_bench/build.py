"""Build and load a process-local ``libcustom_op.so``.

Writes ``custom_op.asc`` into the sample directory, compiles it with the
fixed CMake template, and loads the resulting shared library with
``torch.ops.load_library``. Nothing is installed globally.
"""

from __future__ import annotations

import contextlib
import os
import shutil
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
    (work_dir / "custom_op.asc").write_text(asc_source, encoding="utf-8")
    shutil.copy(
        BUILD_TEMPLATE_DIR / "CMakeLists.txt", work_dir / "CMakeLists.txt"
    )

    build_dir = work_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    env = cann_env()
    env.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "0")

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
    if os.environ.get("AKB_ENABLE_CCACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        configure_cmd.append("-DENABLE_CCACHE=ON")

    _run_cmake(
        "configure", configure_cmd, cwd=work_dir, env=env, timeout_s=timeout_s
    )
    _run_cmake(
        "build",
        ["cmake", "--build", str(build_dir), "-j"],
        cwd=work_dir,
        env=env,
        timeout_s=timeout_s,
    )
    return find_built_library(work_dir)


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

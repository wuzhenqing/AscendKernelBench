"""Build LLM-generated Ascend C sources with the fixed CMake project.

The build contract (README 3.1/3.3): the LLM writes a single self-contained
``custom_op.asc``; this module copies ``build_template/CMakeLists.txt`` into
the sample work directory, injects ``CMAKE_ASC_ARCHITECTURES`` from the
hardware profile, and produces an importable ``custom_op`` Python extension.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from ._paths import BUILD_TEMPLATE_DIR

DEFAULT_CANN_SET_ENV = "/usr/local/Ascend/cann-9.1.0/set_env.sh"


class BuildError(RuntimeError):
    """Raised when configure or build fails; message carries compiler output."""


@lru_cache(maxsize=1)
def cann_env() -> dict[str, str]:
    """Capture the environment produced by sourcing CANN's set_env.sh.

    Cached per process; the CANN env is stable for a machine. Set
    ``CANN_SET_ENV`` to override the set_env.sh location.
    """
    set_env = os.environ.get("CANN_SET_ENV", DEFAULT_CANN_SET_ENV)
    if not Path(set_env).is_file():
        # Fall back to the current environment (already inside a sourced shell).
        return dict(os.environ)
    result = subprocess.run(
        ["bash", "-c", f'source "{set_env}" >/dev/null 2>&1 && env'],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise BuildError(f"Failed to source CANN env {set_env}: {result.stderr}")
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
    """Write ``custom_op.asc`` into ``work_dir`` and build ``custom_op``.

    Returns the path to the built extension module. Raises BuildError with
    the compiler log on failure.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "custom_op.asc").write_text(asc_source, encoding="utf-8")
    shutil.copy(BUILD_TEMPLATE_DIR / "CMakeLists.txt", work_dir / "CMakeLists.txt")

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
        # Torch/torch_npu/pybind11 are located via Python introspection, so the
        # interpreter must be the one from the active environment.
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]
    build_cmd = ["cmake", "--build", str(build_dir), "-j"]

    _run_checked("configure", configure_cmd, cwd=work_dir, env=env, timeout_s=timeout_s)
    _run_checked("build", build_cmd, cwd=work_dir, env=env, timeout_s=timeout_s)

    candidates = sorted(work_dir.glob("custom_op*.so"))
    if not candidates:
        raise BuildError(f"Built module not found in {work_dir}")
    return candidates[0]


def _run_checked(
    stage: str,
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_s: int,
) -> None:
    try:
        result = subprocess.run(
            cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired as exc:
        raise BuildError(f"{stage} timed out after {timeout_s}s") from exc
    log = (result.stdout or "") + "\n" + (result.stderr or "")
    log_path = cwd / "build" / f"{stage}.log"
    try:
        log_path.write_text(log, encoding="utf-8")
    except OSError:
        pass
    if result.returncode != 0:
        tail = "\n".join(log.strip().splitlines()[-60:])
        raise BuildError(f"{stage} failed (see {log_path}):\n{tail}")

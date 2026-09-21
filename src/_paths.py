"""Repository layout constants.

Data directories are anchored to the repository root. A pip install needs
ASCEND_KERNEL_BENCH_REPO_ROOT pointing at a checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
_repo_root_override = os.environ.get("ASCEND_KERNEL_BENCH_REPO_ROOT")
REPO_ROOT = (
    Path(_repo_root_override).expanduser().resolve()
    if _repo_root_override
    else PACKAGE_DIR.parent.parent
)
CONFIGS_DIR = REPO_ROOT / "configs"
HARDWARE_DIR = CONFIGS_DIR / "hardware"
EVAL_DEFAULT_CONFIG = CONFIGS_DIR / "eval_default.yaml"
BUILD_TEMPLATE_DIR = REPO_ROOT / "build_template"
KB_ROOT = REPO_ROOT / "KernelBench"
RESULTS_DIR = REPO_ROOT / "results"
BASELINE_DIR = RESULTS_DIR / "baseline"
RUNS_DIR = REPO_ROOT / "runs"
PROMPT_EXAMPLES_DIR = PACKAGE_DIR / "prompts" / "examples"

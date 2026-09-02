"""Repository layout constants.

Data directories (``configs/``, ``KernelBench/``, ``build_template/``,
``runs/``) are anchored to the repository root, so running from a checkout
needs no configuration. When the package is pip-installed, point the
``AKB_REPO_ROOT`` environment variable at a checkout containing those
directories.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
_repo_root_override = os.environ.get("AKB_REPO_ROOT")
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
SCRIPTS_DIR = REPO_ROOT / "scripts"
RESULTS_DIR = REPO_ROOT / "results"
BASELINE_DIR = RESULTS_DIR / "baseline"
RUNS_DIR = REPO_ROOT / "runs"
PROMPT_EXAMPLES_DIR = PACKAGE_DIR / "prompts" / "examples"

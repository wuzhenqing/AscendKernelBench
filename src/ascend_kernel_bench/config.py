"""Hardware profile and evaluation configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ._paths import EVAL_DEFAULT_CONFIG, HARDWARE_DIR


@dataclass(frozen=True)
class HardwareProfile:
    """Hardware profile (README 3.8): CMake arch for builds, specs for prompts."""

    name: str
    soc_version: str
    cmake_arch: str
    ai_core_num: int
    ub_size_kb: int
    l2_cache_mb: int = 0
    hbm_gb: int = 0
    memory_bandwidth_gbps: int = 0
    supported_dtypes: list[str] = field(default_factory=list)
    api_style: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "HardwareProfile":
        return cls(
            name=data["name"],
            soc_version=data["soc_version"],
            cmake_arch=data["cmake_arch"],
            ai_core_num=int(data["ai_core_num"]),
            ub_size_kb=int(data["ub_size_kb"]),
            l2_cache_mb=int(data.get("l2_cache_mb", 0)),
            hbm_gb=int(data.get("hbm_gb", 0)),
            memory_bandwidth_gbps=int(data.get("memory_bandwidth_gbps", 0)),
            supported_dtypes=list(data.get("supported_dtypes", [])),
            api_style=str(data.get("api_style", "")),
        )


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation defaults (README 3.5/3.6)."""

    hardware: str = "ascend910b2"
    num_correct_trials: int = 5
    seed: int = 42
    tolerances: dict = field(
        default_factory=lambda: {
            "fp32": {"atol": 1e-4, "rtol": 1e-4},
            "fp16": {"atol": 1e-2, "rtol": 1e-2},
            "bf16": {"atol": 1e-2, "rtol": 1e-2},
        }
    )
    precision: str = "fp32"
    num_perf_trials: int = 100
    num_warmup: int = 3
    excessive_speedup: float = 10.0
    build_timeout: int = 600
    eval_timeout: int = 300
    generation: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "EvalConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def load_hardware_profile(name_or_path: str) -> HardwareProfile:
    """Load a hardware profile by name (configs/hardware/<name>.yaml) or path."""
    candidate = Path(name_or_path)
    if not candidate.is_file():
        candidate = HARDWARE_DIR / f"{name_or_path}.yaml"
    if not candidate.is_file():
        available = sorted(p.stem for p in HARDWARE_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"Hardware profile not found: {name_or_path}. Available: {available}"
        )
    return HardwareProfile.from_dict(yaml.safe_load(candidate.read_text()))


def load_eval_config(path: str | Path | None = None) -> EvalConfig:
    """Load evaluation config; defaults to configs/eval_default.yaml."""
    cfg_path = Path(path) if path else EVAL_DEFAULT_CONFIG
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Eval config not found: {cfg_path}")
    return EvalConfig.from_dict(yaml.safe_load(cfg_path.read_text()) or {})

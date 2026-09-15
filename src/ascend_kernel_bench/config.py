"""Hardware profile and evaluation configuration loading.

YAML files are parsed with PyYAML, then validated by frozen Pydantic
models so required fields, defaults, and unknown-key ignoring live in
one place instead of hand-written ``from_dict`` mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ._paths import EVAL_DEFAULT_CONFIG, HARDWARE_DIR


def _default_tolerances() -> dict[str, dict[str, float]]:
    """Return the KernelBench-style per-precision comparison tolerances."""
    return {
        "fp32": {"atol": 1e-4, "rtol": 1e-4},
        "fp16": {"atol": 1e-2, "rtol": 1e-2},
        "bf16": {"atol": 1e-2, "rtol": 1e-2},
    }


class HardwareProfile(BaseModel):
    """Hardware profile: CMake arch for builds, specs for prompts and SOL.

    See docs/reference/configuration.md.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str
    soc_version: str
    cmake_arch: str
    ai_core_num: int
    ub_size_kb: int
    l2_cache_mb: int = 0
    hbm_gb: int = 0
    memory_bandwidth_gbps: float = 0.0
    supported_dtypes: list[str] = Field(default_factory=list)
    api_style: str = ""
    cube_core_num: int = 0
    vector_core_num: int = 0
    peak_tflops: dict[str, float] = Field(default_factory=dict)

    @field_validator("peak_tflops", mode="before")
    @classmethod
    def _coerce_peak_tflops(cls, value: object) -> dict[str, float]:
        """Accept missing, empty, or loosely typed peak-TFLOPS mappings."""
        if not value:
            return {}
        if not isinstance(value, Mapping):
            return {}
        return {str(key): float(item) for key, item in value.items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HardwareProfile:
        """Build a profile from a hardware YAML mapping.

        Args:
            data: Mapping loaded from ``configs/hardware/*.yaml``.

        Returns:
            An immutable :class:`HardwareProfile`.
        """
        return cls.model_validate(data)

    def peak_tflops_for(self, precision: str) -> float:
        """Return datasheet peak TFLOPS for ``precision``, or 0.0.

        Args:
            precision: One of ``fp32``, ``fp16``, ``bf16``.

        Returns:
            Peak TFLOPS from the profile, or 0.0 when unset.
        """
        return float(self.peak_tflops.get(precision, 0.0) or 0.0)


class EvalConfig(BaseModel):
    """Evaluation defaults (docs/reference/configuration.md)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    hardware: str = "ascend910b2"
    num_correct_trials: int = 5
    seed: int = 42
    tolerances: dict[str, dict[str, float]] = Field(
        default_factory=_default_tolerances
    )
    precision: str = "fp32"
    num_perf_trials: int = 100
    num_warmup: int = 10
    excessive_speedup: float = 10.0
    build_timeout: int = 600
    eval_timeout: int = 300
    generation: dict[str, Any] = Field(default_factory=dict)

    @field_validator("generation", mode="before")
    @classmethod
    def _generation_mapping(cls, value: object) -> dict[str, Any]:
        """Treat a missing or null ``generation`` block as an empty mapping."""
        return value if isinstance(value, dict) else {}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvalConfig:
        """Build an eval config, ignoring unknown top-level keys.

        Args:
            data: Mapping loaded from an eval YAML file.

        Returns:
            An immutable :class:`EvalConfig`.
        """
        return cls.model_validate(data)


def load_hardware_profile(name_or_path: str) -> HardwareProfile:
    """Load a hardware profile by name or filesystem path.

    Args:
        name_or_path: ``configs/hardware/<name>.yaml`` stem, or a file path.

    Returns:
        The loaded :class:`HardwareProfile`.

    Raises:
        FileNotFoundError: If neither a path nor a named profile exists.
    """
    candidate = Path(name_or_path)
    if not candidate.is_file():
        candidate = HARDWARE_DIR / f"{name_or_path}.yaml"
    if not candidate.is_file():
        available = sorted(path.stem for path in HARDWARE_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"Hardware profile not found: {name_or_path}. "
            f"Available: {available}"
        )
    loaded = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    return HardwareProfile.from_dict(loaded)


def load_eval_config(path: str | Path | None = None) -> EvalConfig:
    """Load evaluation config; defaults to ``configs/eval_default.yaml``.

    Args:
        path: Optional YAML path. ``None`` uses the repository default.

    Returns:
        The loaded :class:`EvalConfig`.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    cfg_path = Path(path) if path else EVAL_DEFAULT_CONFIG
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Eval config not found: {cfg_path}")
    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return EvalConfig.from_dict(loaded)

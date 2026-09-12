"""Hardware profile and evaluation configuration loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ._paths import EVAL_DEFAULT_CONFIG, HARDWARE_DIR
from .modes import DEFAULT_OPERATOR_MODE, require_implemented_mode


@dataclass(frozen=True)
class HardwareProfile:
    """Hardware profile: CMake arch for builds, specs for prompts and SOL.

    See docs/reference/configuration.md.
    """

    name: str
    soc_version: str
    cmake_arch: str
    ai_core_num: int
    ub_size_kb: int
    l2_cache_mb: int = 0
    hbm_gb: int = 0
    memory_bandwidth_gbps: float = 0.0
    supported_dtypes: list[str] = field(default_factory=list)
    api_style: str = ""
    cube_core_num: int = 0
    vector_core_num: int = 0
    peak_tflops: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HardwareProfile:
        """Build a profile from a hardware YAML mapping.

        Args:
            data: Mapping loaded from ``configs/hardware/*.yaml``.

        Returns:
            An immutable :class:`HardwareProfile`.

        Raises:
            KeyError: If a required field is missing.
        """
        raw_peaks = data.get("peak_tflops") or {}
        peaks = {str(key): float(value) for key, value in raw_peaks.items()}
        return cls(
            name=str(data["name"]),
            soc_version=str(data["soc_version"]),
            cmake_arch=str(data["cmake_arch"]),
            ai_core_num=int(data["ai_core_num"]),
            ub_size_kb=int(data["ub_size_kb"]),
            l2_cache_mb=int(data.get("l2_cache_mb", 0)),
            hbm_gb=int(data.get("hbm_gb", 0)),
            memory_bandwidth_gbps=float(data.get("memory_bandwidth_gbps", 0)),
            supported_dtypes=list(data.get("supported_dtypes", [])),
            api_style=str(data.get("api_style", "")),
            cube_core_num=int(data.get("cube_core_num", 0)),
            vector_core_num=int(data.get("vector_core_num", 0)),
            peak_tflops=peaks,
        )

    def peak_tflops_for(self, precision: str) -> float:
        """Return datasheet peak TFLOPS for ``precision``, or 0.0.

        Args:
            precision: One of ``fp32``, ``fp16``, ``bf16``.

        Returns:
            Peak TFLOPS from the profile, or 0.0 when unset.
        """
        return float(self.peak_tflops.get(precision, 0.0) or 0.0)


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation defaults (docs/reference/configuration.md)."""

    hardware: str = "ascend910b2"
    num_correct_trials: int = 5
    seed: int = 42
    tolerances: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "fp32": {"atol": 1e-4, "rtol": 1e-4},
            "fp16": {"atol": 1e-2, "rtol": 1e-2},
            "bf16": {"atol": 1e-2, "rtol": 1e-2},
        }
    )
    precision: str = "fp32"
    num_perf_trials: int = 100
    num_warmup: int = 10
    excessive_speedup: float = 10.0
    build_timeout: int = 600
    eval_timeout: int = 300
    operator_mode: str = DEFAULT_OPERATOR_MODE
    generation: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvalConfig:
        """Build an eval config, ignoring unknown top-level keys.

        Args:
            data: Mapping loaded from an eval YAML file.

        Returns:
            An immutable :class:`EvalConfig`.

        Raises:
            OperatorModeError: If ``operator_mode`` is unknown or JIT.
        """
        known = set(cls.__dataclass_fields__)
        filtered = {key: value for key, value in data.items() if key in known}
        if "operator_mode" in filtered:
            filtered["operator_mode"] = require_implemented_mode(
                filtered["operator_mode"]
            )
        return cls(**filtered)


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

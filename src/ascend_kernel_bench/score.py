"""Scoring: fast_p and pass@k, KernelBench-compatible schema.

See docs/guide/results.md for scoring rules and result interpretation.

``eval_results.json`` maps problem_id -> list of per-sample dicts with
``sample_id``, ``compiled``, ``correctness``, ``metadata``, ``runtime``,
``runtime_stats``. ``pass_at_k_results.json`` stores the unbiased pass@k
estimates. fast_p denominators count every sample, including compile
failures — matching KernelBench semantics. Format compatibility does NOT
imply score comparability across hardware/baselines.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import geometric_mean
from typing import Any

from .sol import mean_sol_score

FAST_P_THRESHOLDS = (0.0, 0.5, 0.8, 1.0, 1.5, 2.0)


def _threshold_name(t: float) -> str:
    """Return the KernelBench-style key, for example ``fast_0.5``."""
    return f"fast_{t:g}"


def sample_speedup(sample: Mapping[str, Any]) -> float | None:
    """Return speedup (ref mean / kernel mean), or None if unavailable.

    Args:
        sample: One evaluation result dict.

    Returns:
        Speedup ratio, or None when the sample is incorrect, flagged, or
        missing an NPU baseline.
    """
    if not sample.get("correctness"):
        return None
    if (sample.get("metadata") or {}).get("excessive_speedup"):
        return None
    runtime = sample.get("runtime")
    ref_runtime = sample.get("ref_runtime")
    if runtime and ref_runtime and runtime > 0:
        return ref_runtime / runtime
    return None


def fast_p(samples: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """fast_p over a flat sample list; denominator includes all samples.

    fast_0 is the correctness rate: every correct sample counts, including
    correct samples without an NPU baseline (CPU-reference mode, where no
    speedup exists) and samples flagged for excessive speedup — the flag only
    excludes them from the p > 0 speedup thresholds and the geometric mean
    (docs/guide/results.md: marked for manual review, not auto-failed).

    Args:
        samples: Flat list of evaluation result dicts.

    Returns:
        Mapping from ``fast_p`` keys to rates in ``[0, 1]``.
    """
    total = len(samples)
    if total == 0:
        return {_threshold_name(t): 0.0 for t in FAST_P_THRESHOLDS}
    result: dict[str, float] = {}
    for t in FAST_P_THRESHOLDS:
        count = 0
        for sample in samples:
            if not sample.get("correctness"):
                continue
            if t == 0.0:
                count += 1
                continue
            speedup = sample_speedup(sample)
            if speedup is not None and speedup > t:
                count += 1
        result[_threshold_name(t)] = count / total
    return result


def geometric_mean_speedup(samples: Sequence[Mapping[str, Any]]) -> float:
    """Geometric mean speedup over correct, non-flagged samples.

    Args:
        samples: Flat list of evaluation result dicts.

    Returns:
        Geometric mean, or ``0.0`` when no eligible speedup exists.
    """
    speedups = [
        s for s in (sample_speedup(x) for x in samples) if s is not None
    ]
    if not speedups:
        return 0.0
    return float(geometric_mean(speedups))


def pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Standard unbiased pass@k estimator (KernelBench / HumanEval).

    Args:
        num_samples: Collected samples for one problem.
        num_correct: Correct samples among them.
        k: Draw size.

    Returns:
        Unbiased pass@k estimate in ``[0, 1]``.
    """
    if num_samples < k:
        return float(num_correct > 0)
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(
        num_samples, k
    )


def summarize_eval_results(
    eval_results: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Aggregate an eval_results.json mapping into headline metrics.

    Args:
        eval_results: Mapping of problem id to per-sample result dicts.

    Returns:
        Headline counts, reference-mode / flag tallies, ``fast_p``,
        geometric-mean speedup, optional mean SOL score, and per-problem
        tallies.
    """
    all_samples = [s for samples in eval_results.values() for s in samples]
    compiled = sum(1 for s in all_samples if s.get("compiled"))
    correct = sum(1 for s in all_samples if s.get("correctness"))

    def _meta(sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return ``sample['metadata']`` when it is a mapping, else ``{}``."""
        metadata = sample.get("metadata") or {}
        return metadata if isinstance(metadata, Mapping) else {}

    cpu_reference = sum(
        1 for sample in all_samples if _meta(sample).get("reference") == "cpu"
    )
    npu_reference = sum(
        1 for sample in all_samples if _meta(sample).get("reference") == "npu"
    )
    flagged = sum(
        1 for sample in all_samples if _meta(sample).get("excessive_speedup")
    )
    per_problem: dict[str, dict[str, Any]] = {}
    for problem_id, samples in eval_results.items():
        n = len(samples)
        c = sum(1 for s in samples if s.get("correctness"))
        per_problem[str(problem_id)] = {
            "num_samples": n,
            "num_correct": c,
            "any_correct": c > 0,
        }
    summary: dict[str, Any] = {
        "total_samples": len(all_samples),
        "total_problems": len(eval_results),
        "compiled": compiled,
        "correct": correct,
        "cpu_reference": cpu_reference,
        "npu_reference": npu_reference,
        "excessive_speedup": flagged,
        "fast_p": fast_p(all_samples),
        "geometric_mean_speedup_correct_only": geometric_mean_speedup(
            all_samples
        ),
        "mean_sol_score": mean_sol_score(all_samples),
        "per_problem": per_problem,
    }
    return summary


def compute_pass_at_k(
    eval_results: dict[str, list[dict]], ks: Sequence[int] = (1, 5, 10)
) -> dict[str, dict[str, float]]:
    """Compute pass@k per problem and the unweighted average.

    Args:
        eval_results: Mapping of problem id to per-sample result dicts.
        ks: Requested ``k`` values. A problem contributes to ``pass@k``
            only when it has enough samples (except ``k == 1``).

    Returns:
        ``{"per_problem": ..., "average": ...}``.
    """
    per_problem: dict[str, dict[str, float]] = {}
    for problem_id, samples in eval_results.items():
        n = len(samples)
        c = sum(1 for s in samples if s.get("correctness"))
        per_problem[problem_id] = {
            f"pass@{k}": pass_at_k(n, c, k) for k in ks if k <= n or k == 1
        }
    if not per_problem:
        return {"per_problem": {}, "average": {}}
    average = {}
    for k in ks:
        key = f"pass@{k}"
        values = [v[key] for v in per_problem.values() if key in v]
        if values:
            average[key] = sum(values) / len(values)
    return {"per_problem": per_problem, "average": average}

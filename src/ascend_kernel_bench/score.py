"""Scoring: fast_p and pass@k (README 3.6), KernelBench-compatible schema.

``eval_results.json`` maps problem_id -> list of per-sample dicts with
``sample_id``, ``compiled``, ``correctness``, ``metadata``, ``runtime``,
``runtime_stats``. ``pass_at_k_results.json`` stores the unbiased pass@k
estimates. fast_p denominators count every sample, including compile
failures — matching KernelBench semantics. Format compatibility does NOT
imply score comparability across hardware/baselines.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

FAST_P_THRESHOLDS = (0.0, 0.5, 0.8, 1.0, 1.5, 2.0)


def _threshold_name(t: float) -> str:
    """KernelBench-style key: fast_0, fast_0.5, fast_1, ..."""
    return f"fast_{t:g}"


def sample_speedup(sample: dict) -> float | None:
    """Speedup (ref mean / kernel mean) of one sample, or None if unavailable."""
    if not sample.get("correctness"):
        return None
    if (sample.get("metadata") or {}).get("excessive_speedup"):
        return None
    runtime = sample.get("runtime")
    ref_runtime = sample.get("ref_runtime")
    if runtime and ref_runtime and runtime > 0:
        return ref_runtime / runtime
    return None


def fast_p(samples: Sequence[dict]) -> dict[str, float]:
    """fast_p over a flat sample list; denominator includes all samples.

    fast_0 is the correctness rate: every correct sample counts, including
    correct samples without an NPU baseline (CPU-reference mode, where no
    speedup exists) and samples flagged for excessive speedup — the flag only
    excludes them from the p > 0 speedup thresholds and the geometric mean
    (README section 5: marked for manual review, not auto-failed).
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


def geometric_mean_speedup(samples: Sequence[dict]) -> float:
    """Geometric mean speedup over correct, non-flagged samples."""
    speedups = [s for s in (sample_speedup(x) for x in samples) if s is not None]
    if not speedups:
        return 0.0
    return math.exp(sum(math.log(s) for s in speedups) / len(speedups))


def pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Standard unbiased pass@k estimator (KernelBench / HumanEval)."""
    if num_samples < k:
        return float(num_correct > 0)
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


def summarize_eval_results(eval_results: dict[str, list[dict]]) -> dict:
    """Aggregate an eval_results.json mapping into headline metrics."""
    all_samples = [s for samples in eval_results.values() for s in samples]
    compiled = sum(1 for s in all_samples if s.get("compiled"))
    correct = sum(1 for s in all_samples if s.get("correctness"))
    per_problem = {}
    for problem_id, samples in eval_results.items():
        n = len(samples)
        c = sum(1 for s in samples if s.get("correctness"))
        per_problem[problem_id] = {
            "num_samples": n,
            "num_correct": c,
            "any_correct": c > 0,
        }
    return {
        "total_samples": len(all_samples),
        "total_problems": len(eval_results),
        "compiled": compiled,
        "correct": correct,
        "fast_p": fast_p(all_samples),
        "geometric_mean_speedup_correct_only": geometric_mean_speedup(all_samples),
        "per_problem": per_problem,
    }


def compute_pass_at_k(
    eval_results: dict[str, list[dict]], ks: Sequence[int] = (1, 5, 10)
) -> dict[str, dict[str, float]]:
    """pass@k per problem and averaged, for multi-sample runs."""
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

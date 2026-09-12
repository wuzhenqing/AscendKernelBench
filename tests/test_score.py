"""fast_p, pass@k, and SOL aggregation."""

from ascend_kernel_bench.score import (
    compute_pass_at_k,
    fast_p,
    geometric_mean_speedup,
    pass_at_k,
    sample_speedup,
    summarize_eval_results,
)


def _sample(
    *,
    correct: bool,
    runtime: float | None = None,
    ref_runtime: float | None = None,
    flagged: bool = False,
    sol: float | None = None,
) -> dict:
    metadata: dict = {}
    if flagged:
        metadata["excessive_speedup"] = True
    if sol is not None:
        metadata["sol_score"] = sol
    return {
        "compiled": True,
        "correctness": correct,
        "runtime": runtime,
        "ref_runtime": ref_runtime,
        "metadata": metadata,
    }


def test_fast_p_denominator_includes_failures() -> None:
    samples = [
        _sample(correct=False),
        _sample(correct=True, runtime=1.0, ref_runtime=2.0),
        _sample(correct=True, runtime=1.0, ref_runtime=1.0),
        _sample(correct=True, runtime=0.05, ref_runtime=1.0, flagged=True),
    ]
    metrics = fast_p(samples)
    assert metrics["fast_0"] == 0.75
    assert metrics["fast_1"] == 0.25
    assert metrics["fast_2"] == 0.0


def test_sample_speedup_excludes_flagged_and_cpu() -> None:
    assert (
        sample_speedup(_sample(correct=True, runtime=1.0, ref_runtime=2.0))
        == 2.0
    )
    assert sample_speedup(_sample(correct=True, runtime=1.0)) is None
    assert (
        sample_speedup(
            _sample(correct=True, runtime=0.1, ref_runtime=2.0, flagged=True)
        )
        is None
    )


def test_summarize_includes_mean_sol() -> None:
    results = {
        "level1/a": [
            _sample(correct=True, runtime=1.0, ref_runtime=2.0, sol=0.6),
            _sample(correct=False),
        ]
    }
    summary = summarize_eval_results(results)
    assert summary["total_samples"] == 2
    assert summary["correct"] == 1
    assert summary["cpu_reference"] == 0
    assert summary["npu_reference"] == 0
    assert summary["excessive_speedup"] == 0
    assert summary["mean_sol_score"] == 0.6
    assert geometric_mean_speedup(results["level1/a"]) == 2.0


def test_summarize_counts_reference_modes() -> None:
    results = {
        "p": [
            {
                "compiled": True,
                "correctness": True,
                "metadata": {"reference": "cpu", "excessive_speedup": True},
            },
            {
                "compiled": True,
                "correctness": True,
                "metadata": {"reference": "npu"},
            },
        ]
    }
    summary = summarize_eval_results(results)
    assert summary["cpu_reference"] == 1
    assert summary["npu_reference"] == 1
    assert summary["excessive_speedup"] == 1


def test_pass_at_k_estimator() -> None:
    assert pass_at_k(1, 1, 1) == 1.0
    assert pass_at_k(2, 0, 1) == 0.0
    assert pass_at_k(2, 1, 1) == 0.5
    report = compute_pass_at_k(
        {"p": [_sample(correct=True), _sample(correct=False)]},
        ks=(1, 5),
    )
    assert report["average"]["pass@1"] == 0.5
    assert "pass@5" not in report["average"]

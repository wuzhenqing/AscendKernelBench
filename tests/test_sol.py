"""Roofline SOL bound and SOL-ExecBench-style score."""

from ascend_kernel_bench.sol import (
    attach_sol_metadata,
    mean_sol_score,
    roofline_bound_ms,
    sol_score,
    task_declared_flops,
)


def test_roofline_memory_bound() -> None:
    # 1.6e9 bytes at 1600 GB/s = 1 ms.
    bound = roofline_bound_ms(bytes_moved=1_600_000_000, bandwidth_gbps=1600)
    assert bound is not None
    assert abs(bound - 1.0) < 1e-9


def test_roofline_undefined_without_bandwidth() -> None:
    assert roofline_bound_ms(bytes_moved=1024, bandwidth_gbps=0) is None
    assert roofline_bound_ms(bytes_moved=0, bandwidth_gbps=1600) is None


def test_roofline_takes_max_of_memory_and_compute() -> None:
    # Memory 1 ms; compute 2 ms -> 2 ms.
    bound = roofline_bound_ms(
        bytes_moved=1_600_000_000,
        bandwidth_gbps=1600,
        flops=2e9,
        peak_tflops=1.0,
    )
    assert bound is not None
    assert abs(bound - 2.0) < 1e-9


def test_sol_score_baseline_is_half() -> None:
    score = sol_score(kernel_ms=2.0, baseline_ms=2.0, sol_ms=1.0)
    assert score is not None
    assert abs(score - 0.5) < 1e-9


def test_sol_score_closer_to_bound_is_higher() -> None:
    closer = sol_score(kernel_ms=1.1, baseline_ms=2.0, sol_ms=1.0)
    farther = sol_score(kernel_ms=1.8, baseline_ms=2.0, sol_ms=1.0)
    assert closer is not None and farther is not None
    assert closer > farther
    assert 0.0 <= farther < closer <= 1.0


def test_attach_and_mean_sol_score() -> None:
    metadata: dict[str, object] = {}
    attach_sol_metadata(
        metadata,
        kernel_ms=1.5,
        baseline_ms=2.0,
        bytes_moved=1_600_000_000,
        bandwidth_gbps=1600,
    )
    assert metadata["sol_bound_kind"] == "roofline_memory"
    assert "sol_score" in metadata
    compute_meta: dict[str, object] = {}
    attach_sol_metadata(
        compute_meta,
        kernel_ms=2.0,
        baseline_ms=3.0,
        bytes_moved=1_600_000_000,
        bandwidth_gbps=1600,
        flops=2e9,
        peak_tflops=1.0,
    )
    assert compute_meta["sol_bound_kind"] == "roofline"
    samples = [
        {"metadata": metadata},
        {"metadata": {}},
        {"correctness": True},
    ]
    mean = mean_sol_score(samples)
    assert mean == metadata["sol_score"]


def test_task_declared_flops() -> None:
    assert task_declared_flops({}) is None
    assert task_declared_flops({"FLOPS": 0}) is None
    assert task_declared_flops({"FLOPS": 1.5e9}) == 1.5e9
    assert task_declared_flops({"NUM_FLOPS": 100}) == 100.0

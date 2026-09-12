"""Timing helpers that do not require an NPU."""

import pytest

from ascend_kernel_bench.timing import (
    DEFAULT_L2_CLEAR_BYTES,
    get_timing_stats,
    l2_clear_bytes,
)


def test_l2_clear_bytes_is_at_least_256_mib() -> None:
    assert l2_clear_bytes(0) == DEFAULT_L2_CLEAR_BYTES
    assert l2_clear_bytes(64) == DEFAULT_L2_CLEAR_BYTES
    assert l2_clear_bytes(192) == 384 * 1024 * 1024


def test_get_timing_stats() -> None:
    stats = get_timing_stats([1.0, 2.0, 3.0])
    assert stats["num_trials"] == 3
    assert stats["mean"] == 2.0
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0


def test_get_timing_stats_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        get_timing_stats([])

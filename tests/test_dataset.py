"""Vendored KernelBench corpus: 270 tasks, original contract."""

from collections import Counter

import pytest

from ascend_kernel_bench.dataset import (
    KERNELBENCH_LEVEL_COUNTS,
    KERNELBENCH_TASK_COUNT,
    discover_tasks,
    load_task,
)


def test_kernelbench_task_count() -> None:
    tasks = discover_tasks()
    assert len(tasks) == KERNELBENCH_TASK_COUNT == 270
    counts = Counter(task.level for task in tasks)
    assert dict(counts) == KERNELBENCH_LEVEL_COUNTS


def test_task_ids_are_level_stem() -> None:
    tasks = discover_tasks()
    ids = {task.task_id for task in tasks}
    assert "level1/19_ReLU" in ids
    assert "level1/20_LeakyReLU" in ids
    assert all(
        "/" in task.task_id and not task.task_id.endswith(".py")
        for task in tasks
    )


def test_invalid_task_id_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid task id"):
        load_task("not-a-task")


def test_load_task_keeps_reference_source() -> None:
    task = load_task("level1/19_ReLU")
    assert "class Model" in task.task_py
    assert "def get_inputs" in task.task_py
    assert "def get_init_inputs" in task.task_py
    assert task.path.name == "19_ReLU.py"

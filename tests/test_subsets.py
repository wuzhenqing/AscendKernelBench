"""The fixed level 1 subset used to compare models and harness revisions."""

from pathlib import Path

from ascend_kernel_bench._paths import REPO_ROOT
from ascend_kernel_bench.cli_util import read_task_ids, select_tasks
from ascend_kernel_bench.dataset import discover_tasks

SUBSET_PATH = REPO_ROOT / "configs" / "subsets" / "level1_20.txt"


def test_subset_manifest_is_every_fifth_level1_task() -> None:
    ids = read_task_ids(SUBSET_PATH)
    expected = [task.task_id for task in discover_tasks(level=1)[4::5]]
    assert ids == expected
    assert len(ids) == 20


def test_subset_tasks_all_load() -> None:
    tasks = select_tasks(tasks_file=SUBSET_PATH)
    assert len(tasks) == 20
    assert {task.level for task in tasks} == {1}
    for task in tasks:
        assert task.path.is_file()
        assert "class Model" in task.task_py


def test_read_task_ids_skips_comments_and_blanks(tmp_path: Path) -> None:
    manifest = tmp_path / "subset.txt"
    manifest.write_text(
        "# header\n\nlevel1/19_ReLU\n  level1/20_LeakyReLU  \n",
        encoding="utf-8",
    )
    assert read_task_ids(manifest) == ["level1/19_ReLU", "level1/20_LeakyReLU"]


def test_explicit_task_ids_win_over_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "subset.txt"
    manifest.write_text("level1/19_ReLU\n", encoding="utf-8")
    tasks = select_tasks(
        task_ids=["level1/20_LeakyReLU"],
        tasks_file=manifest,
    )
    assert [task.task_id for task in tasks] == ["level1/20_LeakyReLU"]

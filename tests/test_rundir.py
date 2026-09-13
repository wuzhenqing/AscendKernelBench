"""Run-directory persistence helpers."""

from pathlib import Path

import pytest

from ascend_kernel_bench.io_util import write_json_atomic
from ascend_kernel_bench.llm import AscendCGeneration
from ascend_kernel_bench.rundir import (
    collect_eval_results,
    create_run,
    generation_hardware_name,
    iter_sample_dirs,
    load_eval_result,
    resolve_run,
    save_sample,
    write_eval_results,
)


def test_save_and_collect_roundtrip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("ascend_kernel_bench.rundir.RUNS_DIR", tmp_path)
    run_dir = create_run("demo", {"model": "x"})
    assert (run_dir / "generation_config.yaml").is_file()
    generation = AscendCGeneration(
        custom_op_asc=(
            "__global__ __vector__ void k() {}\n"
            "TORCH_LIBRARY\nTORCH_LIBRARY_IMPL"
        ),
        model_new_py="class ModelNew:\n    pass\ntorch.ops.custom_op",
    )
    sample = save_sample(
        run_dir,
        "level1/19_ReLU",
        0,
        prompt="hello",
        generation=generation,
        raw_response="raw",
    )
    assert (sample / "prompt.txt").read_text(encoding="utf-8") == "hello"
    assert (sample / "response_raw.txt").is_file()
    found = list(iter_sample_dirs(run_dir))
    assert found == [("level1/19_ReLU", 0, sample)]
    write_json_atomic(
        sample / "eval_result.json",
        {"compiled": True, "correctness": True},
    )
    collected = collect_eval_results(run_dir)
    assert collected["level1/19_ReLU"][0]["sample_id"] == 0
    write_eval_results(run_dir, collected)
    assert (run_dir / "eval_results.json").is_file()
    (sample / "eval_result.json").write_text("{not-json", encoding="utf-8")
    assert load_eval_result(sample) is None


def test_resolve_run_name_and_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("ascend_kernel_bench.rundir.RUNS_DIR", tmp_path)
    named = tmp_path / "demo"
    named.mkdir()
    assert resolve_run("demo") == named.resolve()
    assert resolve_run(named) == named.resolve()
    with pytest.raises(FileNotFoundError, match="run dir not found"):
        resolve_run("missing")
    with pytest.raises(FileNotFoundError, match="run dir not found"):
        resolve_run(tmp_path / "missing")


def test_iter_sample_dirs_level_filter(tmp_path: Path) -> None:
    first = tmp_path / "level1" / "19_ReLU" / "sample_0"
    second = tmp_path / "level2" / "1_MLP" / "sample_0"
    for sample in (first, second):
        sample.mkdir(parents=True)
        (sample / "custom_op.asc").write_text("x", encoding="utf-8")
        (sample / "model_new.py").write_text("y", encoding="utf-8")
    all_samples = list(iter_sample_dirs(tmp_path))
    assert [item[0] for item in all_samples] == [
        "level1/19_ReLU",
        "level2/1_MLP",
    ]
    level1 = list(iter_sample_dirs(tmp_path, level=1))
    assert level1 == [("level1/19_ReLU", 0, first)]


def test_generation_hardware_name(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert generation_hardware_name(run_dir) is None
    (run_dir / "generation_config.yaml").write_text(
        "hardware: ascend910b2\n", encoding="utf-8"
    )
    assert generation_hardware_name(run_dir) == "ascend910b2"

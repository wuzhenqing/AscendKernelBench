"""Run-directory persistence helpers."""

from pathlib import Path

from ascend_kernel_bench.io_util import write_json_atomic
from ascend_kernel_bench.llm import AscendCGeneration
from ascend_kernel_bench.rundir import (
    collect_eval_results,
    create_run,
    iter_sample_dirs,
    load_eval_result,
    save_sample,
    write_eval_results,
)


def test_save_and_collect_roundtrip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("ascend_kernel_bench.rundir.RUNS_DIR", tmp_path)
    run_dir = create_run("demo", {"model": "x"})
    assert (run_dir / "generation_config.yaml").is_file()
    generation = AscendCGeneration(
        custom_op_asc="__global__ __vector__ void k() {}\nTORCH_LIBRARY\nTORCH_LIBRARY_IMPL",
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

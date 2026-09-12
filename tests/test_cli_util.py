"""Shared CLI helpers."""

from argparse import ArgumentParser

import pytest

from ascend_kernel_bench.cli_util import (
    add_operator_mode_argument,
    apply_operator_mode,
    cli_progress,
    eval_result_lines,
    generation_run_config,
    load_eval_runtime,
    resolve_generation_settings,
    sample_status_label,
    select_tasks,
)
from ascend_kernel_bench.config import EvalConfig
from ascend_kernel_bench.modes import OperatorModeError


def test_apply_operator_mode_default() -> None:
    config = apply_operator_mode(EvalConfig(), None)
    assert config.operator_mode == "aclnn"


def test_apply_operator_mode_rejects_jit() -> None:
    with pytest.raises(OperatorModeError, match="not implemented"):
        apply_operator_mode(EvalConfig(), "jit")


def test_operator_mode_argument_choices() -> None:
    parser = ArgumentParser()
    add_operator_mode_argument(parser)
    args = parser.parse_args(["--operator-mode", "aclnn"])
    assert args.operator_mode == "aclnn"


def test_resolve_generation_settings_overrides() -> None:
    config = EvalConfig(
        generation={
            "model": "yaml-model",
            "prompt_mode": "few_shot",
            "temperature": 0.7,
            "max_tokens": 4096,
            "num_samples": 3,
        }
    )
    defaults = resolve_generation_settings(config)
    assert defaults.model == "yaml-model"
    assert defaults.prompt_mode == "few_shot"
    assert defaults.temperature == 0.7
    assert defaults.max_tokens == 4096
    assert defaults.num_samples == 3
    override = resolve_generation_settings(
        config, model="cli-model", num_samples=2, temperature=0.0
    )
    assert override.model == "cli-model"
    assert override.num_samples == 2
    assert override.temperature == 0.0
    assert override.prompt_mode == "few_shot"


def test_load_eval_runtime_defaults() -> None:
    runtime = load_eval_runtime()
    assert runtime.config.operator_mode == "aclnn"
    assert runtime.hardware.name == "ascend910b2"


def test_select_tasks_explicit_id() -> None:
    tasks = select_tasks(task_ids=["level1/19_ReLU"])
    assert [task.task_id for task in tasks] == ["level1/19_ReLU"]
    assert select_tasks(task_ids=[]) == []


def test_cli_progress_builds_bar() -> None:
    from rich.console import Console
    from rich.progress import Progress

    progress = cli_progress(Console(record=True))
    assert isinstance(progress, Progress)


def test_generation_run_config_includes_extra_keys() -> None:
    settings = resolve_generation_settings(EvalConfig())
    record = generation_run_config(
        settings,
        hardware_name="ascend910b2",
        operator_mode="aclnn",
        task_ids=["level1/19_ReLU"],
        device="npu:0",
    )
    assert record["hardware"] == "ascend910b2"
    assert record["tasks"] == ["level1/19_ReLU"]
    assert record["device"] == "npu:0"
    assert record["num_samples"] == settings.num_samples


def test_eval_result_lines_includes_runtime_and_error() -> None:
    style, lines = eval_result_lines(
        {
            "compiled": True,
            "correctness": True,
            "runtime": 2.0,
            "ref_runtime": 4.0,
        }
    )
    assert style == "green"
    assert lines[0] == "status: OK"
    assert any("speedup 2.00x" in line for line in lines)

    _, failed = eval_result_lines(
        {
            "compiled": False,
            "correctness": False,
            "metadata": {"compilation_error": "missing kernel"},
        }
    )
    assert failed[0] == "status: COMPILE-FAIL"
    assert any(line.startswith("error: missing kernel") for line in failed)


def test_sample_status_label() -> None:
    assert sample_status_label({"correctness": True, "compiled": True}) == (
        "green",
        "OK",
    )
    assert sample_status_label({"correctness": False, "compiled": False}) == (
        "yellow",
        "COMPILE-FAIL",
    )
    assert sample_status_label({"correctness": False, "compiled": True}) == (
        "red",
        "WRONG",
    )

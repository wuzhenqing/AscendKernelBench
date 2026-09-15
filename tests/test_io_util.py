"""Atomic writers."""

import json
from pathlib import Path

import pytest

from ascend_kernel_bench.io_util import (
    load_cfg_argv,
    pop_required_path,
    read_json_object,
    write_json_atomic,
    write_yaml_atomic,
)


def test_json_and_yaml_atomic_roundtrip(tmp_path: Path) -> None:
    json_path = tmp_path / "nested" / "out.json"
    yaml_path = tmp_path / "nested" / "out.yaml"
    write_json_atomic(json_path, {"a": 1})
    write_yaml_atomic(yaml_path, {"b": 2})
    assert json_path.read_text(encoding="utf-8").startswith("{")
    assert "b:" in yaml_path.read_text(encoding="utf-8")
    assert not json_path.with_suffix(".json.tmp").exists()
    assert not yaml_path.with_suffix(".yaml.tmp").exists()


def test_read_json_object_requires_mapping(tmp_path: Path) -> None:
    path = tmp_path / "obj.json"
    write_json_atomic(path, {"ok": True})
    assert read_json_object(path)["ok"] is True
    path.write_text("[1]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        read_json_object(path)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_json_object(path)


def test_load_cfg_argv(tmp_path: Path) -> None:
    path = tmp_path / "cfg.json"
    write_json_atomic(path, {"device": "npu:0"})
    assert load_cfg_argv(["prog", str(path)])["device"] == "npu:0"
    with pytest.raises(SystemExit):
        load_cfg_argv(["prog"])
    with pytest.raises(SystemExit):
        load_cfg_argv(["prog", str(tmp_path / "missing.json")])


def test_pop_required_path() -> None:
    path = pop_required_path({"result_path": "/tmp/out.json"}, "result_path")
    assert path.name == "out.json"
    with pytest.raises(SystemExit) as missing:
        pop_required_path({}, "result_path")
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as blank:
        pop_required_path({"result_path": "  "}, "result_path")
    assert blank.value.code == 2
    with pytest.raises(SystemExit) as wrong_type:
        pop_required_path({"result_path": ["not", "a", "path"]}, "result_path")
    assert wrong_type.value.code == 2

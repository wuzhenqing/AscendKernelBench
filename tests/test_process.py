"""Isolated JSON worker helper."""

import sys
from pathlib import Path

from ascend_kernel_bench.process import IsolatedJsonWorker


def test_isolated_json_worker_reads_payload() -> None:
    script = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "cfg = json.loads(Path(sys.argv[1]).read_text())\n"
        "Path(cfg['result_path']).write_text(json.dumps({'ok': True}))\n"
    )
    worker = IsolatedJsonWorker.for_eval(timeout_s=10)
    worker.argv_for = lambda cfg_path: [  # type: ignore[method-assign]
        sys.executable,
        "-c",
        script,
        str(cfg_path),
    ]
    outcome = worker.run({"x": 1})
    assert outcome.timed_out is False
    assert outcome.returncode == 0
    assert outcome.payload == {"ok": True}


def test_isolated_json_worker_nonzero_keeps_stderr() -> None:
    script = "import sys; print('boom', file=sys.stderr); sys.exit(3)"
    worker = IsolatedJsonWorker.for_eval(timeout_s=10)
    worker.argv_for = lambda cfg_path: [  # type: ignore[method-assign]
        sys.executable,
        "-c",
        script,
    ]
    outcome = worker.run({})
    assert outcome.returncode == 3
    assert "boom" in outcome.stderr
    assert outcome.payload is None
    assert outcome.parse_error == "worker produced no result.json"


def test_eval_worker_argv_points_at_repo_script() -> None:
    worker = IsolatedJsonWorker.for_eval(timeout_s=1)
    argv = worker.argv_for(Path("/tmp/cfg.json"))
    assert argv[0] == sys.executable
    assert argv[1].endswith("_eval_worker.py")
    assert argv[2] == "/tmp/cfg.json"

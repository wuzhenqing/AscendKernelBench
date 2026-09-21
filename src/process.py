"""Isolated JSON-in / JSON-out child processes.

Generation-adjacent tools (evaluation and baseline measurement) share one
pattern: write a config object, spawn a worker, read a JSON result.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from ._paths import REPO_ROOT
from .io_util import read_json_object, write_json_atomic


@dataclass(frozen=True)
class IsolatedJobResult:
    """Outcome of one isolated worker process."""

    returncode: int
    stderr: str
    payload: dict[str, Any] | None
    timed_out: bool
    parse_error: str | None = None


class IsolatedJsonWorker:
    """Run a child that reads cfg.json and writes one JSON object.

    Use for_eval or for_baseline rather than constructing argv by hand.
    """

    def __init__(
        self,
        *,
        timeout_s: int,
        result_key: str,
        prefix: str,
        raise_on_timeout: bool = False,
    ) -> None:
        """Record the timeout and the cfg key that names the result file.

        result_key is injected into cfg and is result_path or out_path. With
        raise_on_timeout set, a timeout re-raises TimeoutExpired after the
        process group is killed (baseline); evaluation reports a failed
        payload instead.
        """
        self.timeout_s = timeout_s
        self.result_key = result_key
        self.prefix = prefix
        self.raise_on_timeout = raise_on_timeout

    @classmethod
    def for_eval(cls, timeout_s: int) -> IsolatedJsonWorker:
        """Return the worker used by host-side sample evaluation."""
        return cls(
            timeout_s=timeout_s,
            result_key="result_path",
            prefix="ascend_kernel_bench_eval_",
        )

    @classmethod
    def for_baseline(cls, timeout_s: int) -> IsolatedJsonWorker:
        """Return the worker that archives eager-reference timings."""
        return cls(
            timeout_s=timeout_s,
            result_key="out_path",
            prefix="ascend_kernel_bench_baseline_",
            raise_on_timeout=True,
        )

    def argv_for(self, cfg_path: Path) -> list[str]:
        """Return the child argv for cfg_path.

        Returns:
            [python, worker, cfg.json].

        Raises:
            FileNotFoundError: If the eval worker script is missing.
        """
        if self.result_key == "result_path":
            worker = REPO_ROOT / "scripts" / "_eval_worker.py"
            if not worker.is_file():
                raise FileNotFoundError(f"eval worker script not found: {worker}")
            return [sys.executable, str(worker), str(cfg_path)]
        return [
            sys.executable,
            "-m",
            "ascend_kernel_bench.baseline_worker",
            str(cfg_path),
        ]

    def run(self, cfg: dict[str, Any]) -> IsolatedJobResult:
        """Write cfg, spawn the worker, and parse its JSON result.

        Args:
            cfg: Worker configuration; the result key is overwritten with
                a path inside the temporary directory.

        Returns:
            Process status plus a parsed object when the child succeeded.

        Raises:
            subprocess.TimeoutExpired: When raise_on_timeout is set and the
                child exceeds timeout_s.
        """
        with tempfile.TemporaryDirectory(prefix=self.prefix) as tmpdir:
            result_path = Path(tmpdir) / "result.json"
            cfg_path = Path(tmpdir) / "cfg.json"
            write_json_atomic(cfg_path, {**cfg, self.result_key: str(result_path)})
            env = dict(os.environ)
            env.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "0")
            argv = self.argv_for(cfg_path)
            logger.debug("isolated worker: {}", " ".join(argv))
            return self._communicate(argv, env, result_path)

    def _communicate(
        self,
        argv: list[str],
        env: dict[str, str],
        result_path: Path,
    ) -> IsolatedJobResult:
        """Spawn argv and collect stderr plus an optional JSON payload."""
        #################### WORKER SPAWN ####################
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=env,
        )
        try:
            _, stderr = proc.communicate(timeout=self.timeout_s)
        except subprocess.TimeoutExpired as exc:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            if self.raise_on_timeout:
                raise exc
            code = proc.returncode if proc.returncode is not None else -1
            return IsolatedJobResult(
                returncode=code,
                stderr="",
                payload=None,
                timed_out=True,
            )
        #################### WORKER SPAWN ####################
        payload, parse_error = _load_payload(result_path)
        return IsolatedJobResult(
            returncode=proc.returncode or 0,
            stderr=stderr or "",
            payload=payload,
            timed_out=False,
            parse_error=parse_error,
        )


def _load_payload(
    result_path: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return (object, None) or (None, diagnostic)."""
    if not result_path.is_file():
        return None, "worker produced no result.json"
    try:
        return read_json_object(result_path), None
    except json.JSONDecodeError as exc:
        return None, f"invalid worker JSON: {exc}"
    except ValueError:
        return None, "worker JSON is not an object"

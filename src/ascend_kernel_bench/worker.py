"""Isolated NPU eval worker: read cfg.json, write result.json, exit."""

from __future__ import annotations

import sys

from .io_util import load_cfg_argv, pop_required_path, write_json_atomic


def main(argv: list[str]) -> None:
    """Read ``cfg.json``, evaluate one sample, and write ``result_path``.

    Args:
        argv: ``[prog, cfg.json]``.
    """
    cfg = load_cfg_argv(argv)
    result_path = pop_required_path(cfg, "result_path")
    from .eval_device import eval_sample_on_device
    from .eval_result import fail_result

    try:
        result = eval_sample_on_device(**cfg)
    except Exception as exc:
        # Build failures return earlier with compiled=False; anything escaping
        # to here happened after the build, so it is a runtime failure.
        result = fail_result(compiled=True, runtime_error=repr(exc))
    write_json_atomic(result_path, result)


if __name__ == "__main__":
    main(sys.argv)

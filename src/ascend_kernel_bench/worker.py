"""Isolated NPU eval worker: read cfg.json, write result.json, exit."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} <cfg.json>", file=sys.stderr)
        sys.exit(2)

    cfg = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    result_path = Path(cfg.pop("result_path"))
    from .eval import _fail, eval_sample_on_device

    try:
        result = eval_sample_on_device(**cfg)
    except Exception as exc:
        # Build failures return earlier with compiled=False; anything escaping
        # to here happened after the build, so it is a runtime failure.
        result = _fail(compiled=True, runtime_error=repr(exc))
    result_path.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv)

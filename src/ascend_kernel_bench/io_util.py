"""Atomic JSON and YAML writers used by evaluation and run persistence."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def pop_required_path(cfg: dict[str, Any], key: str) -> Path:
    """Pop a required filesystem path from a worker config.

    Args:
        cfg: Mutable worker configuration.
        key: Key whose value must be a nonempty string path.

    Returns:
        The path value.

    Raises:
        SystemExit: If ``key`` is missing or not a nonempty string.
    """
    raw = cfg.pop(key, None)
    if not isinstance(raw, str) or not raw.strip():
        print(f"cfg.json missing {key}", file=sys.stderr)
        sys.exit(2)
    return Path(raw)


def load_cfg_argv(argv: list[str]) -> dict[str, Any]:
    """Read a worker ``cfg.json`` path from ``argv``.

    Args:
        argv: ``[prog, cfg.json]``.

    Returns:
        Parsed configuration mapping.

    Raises:
        SystemExit: If the argument list or file is unusable.
    """
    if len(argv) != 2:
        print(f"Usage: {argv[0]} <cfg.json>", file=sys.stderr)
        sys.exit(2)
    try:
        return read_json_object(Path(argv[1]))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"invalid cfg.json: {exc}", file=sys.stderr)
        sys.exit(2)


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object from ``path``.

    Args:
        path: Existing JSON file.

    Returns:
        The parsed object.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the top-level value is not an object.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 ``text`` via a sibling temp file, fsync, and ``os.replace``.

    Args:
        path: Destination path. Missing parents are created.
        text: File contents.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write ``payload`` as UTF-8 JSON via a temp file and ``os.replace``.

    Args:
        path: Destination path. Missing parents are created.
        payload: Any JSON-serializable object.
    """
    _write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False))


def write_yaml_atomic(path: Path, payload: Any) -> None:
    """Write ``payload`` as UTF-8 YAML via a temp file and ``os.replace``.

    Args:
        path: Destination path. Missing parents are created.
        payload: Any object accepted by ``yaml.safe_dump``.
    """
    _write_text_atomic(
        path, yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    )

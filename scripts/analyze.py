#!/usr/bin/env python
"""Print fast_p / pass@k tables from a saved run."""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from rich.console import Console

from src.cli_util import print_eval_report, resolve_run_dir
from src.io_util import read_json_object
from src.log import die, setup_logging

console = Console()


def main() -> None:
    """Print the saved evaluation report for a run."""
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="run name under runs/, or an existing path")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)

    results_path = run_dir / "eval_results.json"
    if not results_path.is_file():
        die(f"not found: {results_path} (run scripts/evaluate.py first)")
    try:
        eval_results = read_json_object(results_path)
    except (json.JSONDecodeError, ValueError) as exc:
        die(f"invalid eval_results.json: {exc}")

    print_eval_report(console, run_dir.name, eval_results)


if __name__ == "__main__":
    main()

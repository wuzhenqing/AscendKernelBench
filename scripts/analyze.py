#!/usr/bin/env python
"""Print fast_p / pass@k tables from a saved run.

Example:
    python scripts/analyze.py relu_demo
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from rich.console import Console

from ascend_kernel_bench import rundir
from ascend_kernel_bench.cli_util import print_eval_report
from ascend_kernel_bench.io_util import read_json_object
from ascend_kernel_bench.log import die, setup_logging

console = Console()


def main() -> None:
    """Print the saved evaluation report for a run."""
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="run name under runs/, or an existing path")
    args = parser.parse_args()

    try:
        run_dir = rundir.resolve_run(args.run)
    except FileNotFoundError as exc:
        die(str(exc))

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

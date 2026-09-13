"""AscendKernelBench evaluation engine.

A library with separable modules for dataset, prompt, generation, static
checks, isolated evaluation, timing, SOL scoring, and reporting. Host
orchestration lives in ``eval.py``; the NPU worker body lives in
``eval_device.py``. See docs/reference/architecture.md.
"""

from .eval import evaluate_run
from .score import compute_pass_at_k, fast_p, summarize_eval_results
from .sol import mean_sol_score, sol_score

__version__ = "0.1.0"
__all__ = [
    "compute_pass_at_k",
    "evaluate_run",
    "fast_p",
    "mean_sol_score",
    "sol_score",
    "summarize_eval_results",
]

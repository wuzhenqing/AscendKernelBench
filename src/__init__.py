"""AscendKernelBench evaluation engine.

Host orchestration is eval.py, the NPU worker body is eval_device.py; the
rest covers dataset, prompt, generation, checks, timing, and scoring.
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

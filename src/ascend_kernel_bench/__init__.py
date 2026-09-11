"""AscendKernelBench evaluation engine.

A pure evaluation library with no LLM orchestration dependency: dataset,
build, eval, timing, score, prompt, llm and checker are separable modules
that share stable entry points (docs/reference/architecture.md).
"""

from .modes import ACLNN_MODE, DEFAULT_OPERATOR_MODE, JIT_MODE, OPERATOR_MODES

__version__ = "0.1.0"
__all__ = [
    "ACLNN_MODE",
    "DEFAULT_OPERATOR_MODE",
    "JIT_MODE",
    "OPERATOR_MODES",
]

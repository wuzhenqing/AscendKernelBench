"""Static anti-cheat checks (implementation in :mod:`.checks`).

See :mod:`ascend_kernel_bench.checks` for the policy and the split
between regex catalogs, the Python AST visitor, and Ascend C rules.
"""

from .checks import check_custom_op_asc, check_model_new

__all__ = ["check_custom_op_asc", "check_model_new"]

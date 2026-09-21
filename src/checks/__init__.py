"""Static anti-cheat checks for generated samples.

check_model_new checks the Python wrapper, check_custom_op_asc the Ascend C
source; both return violations. nn layers may hold parameters, never compute.
"""

from .ascend_c import check_custom_op_asc
from .python_source import check_model_new

__all__ = ["check_custom_op_asc", "check_model_new"]

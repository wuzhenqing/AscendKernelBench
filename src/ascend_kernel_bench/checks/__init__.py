"""Static anti-cheat checks for generated samples.

Two entry points:

- :func:`check_model_new` inspects the Python wrapper. Regex checks run on
  comment-stripped, string-masked source; semantic checks run on the AST, so
  import aliases (``import torch as t``), direct imports (``from
  torch.nn.functional import sigmoid``), tensor-method calls
  (``x.softmax(-1)``) and operators (``A @ B``) cannot slip through spelling
  variations. ``nn`` layers MAY be constructed as parameter containers (the
  evaluator seeds candidate and reference construction identically, so
  identical construction reproduces the reference weights) but must never be
  called — compute belongs to the custom op. The wrapper must call
  ``torch.ops.custom_op``, not ``import custom_op``.
- :func:`check_custom_op_asc` inspects the Ascend C source: it must contain a
  real ``__global__ __vector__`` kernel and a ``TORCH_LIBRARY`` /
  ``TORCH_LIBRARY_IMPL`` binding (not pybind11), and must not call ATen
  compute ops, vendor prebuilt ops (aclnn/aclop), or host side effects
  (process execution, networking, dynamic loading, threads).

Both return lists of human-readable violations; empty means pass. Static
checks are advisory: they close accidental and low-effort bypasses, not
determined obfuscation — the fresh-input timing protocol and the
excessive-speedup flag are the runtime backstops.

Implementation is split by strategy: :mod:`.rules` for regex catalogs,
:mod:`.python_ast` for the wrapper visitor, and :mod:`.ascend_c` for
host-side C++ checks. Import :mod:`ascend_kernel_bench.checker` for the
stable public names.
"""

from .ascend_c import check_custom_op_asc
from .python_source import check_model_new

__all__ = ["check_custom_op_asc", "check_model_new"]

"""Static anti-cheat checks for generated samples (README section 5).

Two entry points:

- :func:`check_model_new` inspects the Python wrapper. Regex checks run on
  comment-stripped, string-masked source; semantic checks run on the AST, so
  import aliases (``import torch as t``), direct imports (``from
  torch.nn.functional import sigmoid``), tensor-method calls
  (``x.softmax(-1)``) and operators (``A @ B``) cannot slip through spelling
  variations. ``nn`` layers MAY be constructed as parameter containers (the
  evaluator seeds candidate and reference construction identically, so
  identical construction reproduces the reference weights) but must never be
  called — compute belongs to the custom op.
- :func:`check_custom_op_asc` inspects the Ascend C source: it must contain a
  real ``__global__ __vector__`` kernel and the pybind module, and must not
  call ATen compute ops, vendor prebuilt ops (aclnn/aclop), or host side
  effects (process execution, networking, dynamic loading, threads).

Both return lists of human-readable violations; empty means pass. Static
checks are advisory: they close accidental and low-effort bypasses, not
determined obfuscation — the fresh-input timing protocol and the
excessive-speedup flag are the runtime backstops.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize


def _mask_string_constants(source: str) -> str:
    """Blank string constants so embedded text cannot trip the regex rules."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and getattr(node, "end_lineno", None) is not None
            and getattr(node, "end_col_offset", None) is not None
        ):
            continue
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        ranges.append((start, end))

    if not ranges:
        return source
    parts: list[str] = []
    cursor = 0
    for start, end in sorted(ranges):
        parts.append(source[cursor:start])
        parts.append(" " * (end - start))
        cursor = end
    parts.append(source[cursor:])
    return "".join(parts)


def _strip_comments(code: str) -> str:
    """Blank Python ``#`` comments in place, preserving all other bytes.

    Reconstruction from token strings would drop inter-token whitespace and
    break space-sensitive patterns (``import numpy``), so comments are
    replaced by spaces at their exact offsets instead.
    """
    lines = code.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    chars = list(code)
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        for tok in tokens:
            if tok.type != tokenize.COMMENT:
                continue
            (srow, scol), (erow, ecol) = tok.start, tok.end
            start = offsets[srow - 1] + scol
            end = offsets[erow - 1] + ecol
            for i in range(start, end):
                chars[i] = " "
    except (tokenize.TokenError, IndentationError):
        return "\n".join(
            line.split("#", 1)[0].rstrip() for line in code.splitlines()
        )
    return "".join(chars)


# --- Bypass checks (strictly prohibited) ---

TRY_EXCEPT_PATTERNS = [r"\btry\s*:", r"\bexcept\s*:", r"\bexcept\s+\w+"]
PASS_PATTERN = r"\bpass\b"

CPU_FALLBACK_PATTERNS = [
    r"\.cpu\s*\(",
    r"\.numpy\s*\(",
    r"\bimport\s+numpy\b",
    r"\bfrom\s+numpy\b",
    r"\bnumpy\s*\.\s*\w+",
]

# --- torch_npu / aclnn escape hatches ---
NPU_NATIVE_PATTERNS = [
    r"torch_npu\.npu_\w+",
    r"\baclnn\w*",
    r"torch\.ops\.op_plugin",
    r"torch\.ops\.atlas",
]

# --- Timing manipulation checks ---

STREAM_PATTERNS = [
    r"torch\.cuda\.Stream\s*\(",
    r"cuda\.Stream\s*\(",
    r"torch\.npu\.Stream\s*\(",
    r"npu\.Stream\s*\(",
    r"with\s+torch\.(cuda|npu)\.stream",
    r"\.wait_stream\s*\(",
    r"\.record_stream\s*\(",
]

THREAD_PATTERNS = [
    r"threading\.Thread\s*\(",
    r"\bimport\s+threading\b",
    r"\bfrom\s+threading\s+import\b",
    r"multiprocessing\.(Process|Pool|Manager|Queue|Pipe)",
    r"\bimport\s+multiprocessing\b",
    r"concurrent\.futures",
    r"ThreadPoolExecutor",
    r"ProcessPoolExecutor",
]

LAZY_TENSOR_PATTERNS = [
    r"_make_subclass",
    r"class\s+\w+.*\(torch\.Tensor\)",
    r"class\s+\w+.*\(Tensor\)",
    r"torch\.Tensor\.__new__",
]

TIMING_EVENT_PATCH_PATTERNS = [
    r"torch\.(cuda|npu)\.Event\.record\s*=",
    r"torch\.(cuda|npu)\.Event\.elapsed_time\s*=",
    r"torch\.(cuda|npu)\.synchronize\s*=",
    r"torch\.(cuda|npu)\.Event\s*=",
    r"time\.perf_counter\s*=",
    r"time\.time\s*=",
]

RESULT_CACHE_PATTERNS = [
    r"_cached_output",
    r"_result_cache",
    r"functools\.(lru_cache|cache)",
]


def _check_bypass(code: str) -> list[str]:
    violations: list[str] = []
    for pattern in TRY_EXCEPT_PATTERNS:
        if re.search(pattern, code):
            violations.append("Contains try-except block (potential fallback bypass)")
            break
    if re.search(PASS_PATTERN, code):
        violations.append("Contains 'pass' statement (inheritance bypass)")
    for pattern in CPU_FALLBACK_PATTERNS:
        if re.search(pattern, code):
            violations.append("Contains CPU/NumPy fallback pattern")
            break
    return violations


def _check_npu_native(code: str) -> list[str]:
    for pattern in NPU_NATIVE_PATTERNS:
        match = re.search(pattern, code)
        if match:
            return [f"Uses vendor native op shortcut: {match.group(0)}"]
    return []


def _check_stream_injection(code: str) -> list[str]:
    for pattern in STREAM_PATTERNS:
        if re.search(pattern, code):
            return ["Uses stream primitives (potential timing manipulation)"]
    return []


def _check_thread_injection(code: str) -> list[str]:
    for pattern in THREAD_PATTERNS:
        if re.search(pattern, code):
            return ["Uses threading/multiprocessing (potential timing manipulation)"]
    return []


def _check_lazy_eval(code: str) -> list[str]:
    for pattern in LAZY_TENSOR_PATTERNS:
        if re.search(pattern, code):
            return ["Uses lazy tensor pattern (potential correctness hack)"]
    return []


def _check_timing_event_patch(code: str) -> list[str]:
    for pattern in TIMING_EVENT_PATCH_PATTERNS:
        if re.search(pattern, code):
            return ["Reassigns timing function (monkey patch detected)"]
    return []


def _check_result_cache(code: str) -> list[str]:
    for pattern in RESULT_CACHE_PATTERNS:
        if re.search(pattern, code):
            return ["Caches results across calls (outputs must depend on current inputs)"]
    return []


# ---------------------------------------------------------------------------
# AST-level semantic checks for model_new.py
# ---------------------------------------------------------------------------

# nn names that are structure, not compute: constructing them is always fine.
_STRUCTURAL_NN_NAMES = {
    "Module",
    "Parameter",
    "ParameterList",
    "ParameterDict",
    "ModuleList",
    "ModuleDict",
    "init",
    "parameter",
}

# torch.<name>() calls allowed in the wrapper: allocation, metadata, and data
# movement/layout glue. Anything else under torch.* is treated as compute.
_TORCH_ALLOWED_CALLS = {
    # allocation / construction
    "empty", "zeros", "ones", "full", "empty_like", "zeros_like", "ones_like",
    "full_like", "empty_strided", "tensor", "as_tensor", "scalar_tensor",
    "arange", "rand", "randn", "randint", "rand_like", "randn_like",
    "Tensor", "device", "no_grad", "inference_mode",
    # metadata
    "is_tensor", "is_floating_point", "numel", "manual_seed",
    # data movement / layout glue (no arithmetic on tensor values)
    "cat", "concat", "concatenate", "stack", "vstack", "hstack", "dstack",
    "split", "chunk", "unbind", "reshape", "transpose", "permute",
    "squeeze", "unsqueeze", "flatten", "unflatten", "clone", "detach",
    "narrow", "select", "expand", "repeat", "tile", "broadcast_to",
    "view_as", "movedim", "moveaxis", "swapaxes", "swapdims", "roll", "flip",
}

# Method names that only exist on tensor-like objects and compute values.
# Flagged regardless of receiver spelling (x.softmax(-1), a.matmul(b), ...).
_TENSOR_COMPUTE_METHODS = {
    "matmul", "mm", "bmm", "addmm", "baddbmm", "dot", "cross", "mv", "ger",
    "outer", "einsum",
    "relu", "relu_", "sigmoid", "sigmoid_", "tanh", "tanh_", "gelu", "silu",
    "softmax", "log_softmax", "leaky_relu", "elu", "selu", "hardsigmoid",
    "hardswish", "softplus", "softsign", "mish",
    "add", "add_", "sub", "sub_", "mul", "mul_", "div", "div_", "pow", "pow_",
    "sum", "mean", "max", "min", "amax", "amin", "argmax", "argmin",
    "prod", "norm", "var", "std", "cumsum", "cumprod", "topk", "sort",
    "logsumexp", "clamp", "clamp_",
    "conv1d", "conv2d", "conv3d",
    "gather", "scatter", "scatter_", "index_select", "index_add",
    "masked_fill", "masked_fill_", "masked_select",
}

_BANNED_IMPORT_ROOTS = {"ctypes", "subprocess", "socket", "importlib"}

_DYNAMIC_CALLS = {"exec", "eval", "compile", "__import__", "globals", "locals", "vars"}

# Calls whose result is a plain Python scalar (shape arithmetic is legal).
_SCALAR_BUILTINS = {"len", "int", "float", "round", "abs", "min", "max", "sum", "str", "bool", "repr"}
_SCALAR_METHODS = {"numel", "nelement", "dim", "size", "item", "__len__"}


class _WrapperSemantics(ast.NodeVisitor):
    """AST checks for ``model_new.py``; see module docstring for the policy."""

    def __init__(self, source: str) -> None:
        self.violations: list[str] = []
        self.aliases: dict[str, str] = {}
        self.co_refs: set[str] = set()  # refs aliasing the custom_op module
        self.holders: set[str] = set()  # refs assigned nn-layer constructions
        self.scalar_refs: set[str] = set()  # refs holding plain Python scalars
        self.calls_custom_op = False
        tree = ast.parse(source)
        self._collect_aliases(tree)
        for _ in range(3):  # fixed-point pass: assignment chains
            self._collect_bindings(tree)
        self.visit(tree)
        if not self.calls_custom_op:
            self.violations.append(
                "never calls the compiled custom_op module — the wrapper must "
                "call the Ascend C operator"
            )

    # -- name resolution ---------------------------------------------------

    def _resolve(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = self._resolve(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    def _ref_key(self, node: ast.AST) -> str | None:
        """Dotted key for assignment targets: name or attribute path."""
        if isinstance(node, ast.Name):
            return node.id
        return self._resolve(node) if isinstance(node, ast.Attribute) else None

    def _is_custom_op_path(self, path: str) -> bool:
        if path == "custom_op" or path.startswith("custom_op."):
            return True
        return any(path == ref or path.startswith(ref + ".") for ref in self.co_refs)

    # -- pre-passes ----------------------------------------------------------

    def _collect_aliases(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in _BANNED_IMPORT_ROOTS:
                        self._flag(f"banned import: {alias.name}")
                    # `import a.b` binds `a`; `import a.b as c` binds c -> a.b
                    self.aliases[alias.asname or root] = (
                        alias.name if alias.asname else root
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.split(".")[0] in _BANNED_IMPORT_ROOTS:
                    self._flag(f"banned import: from {module}")
                for alias in node.names:
                    if alias.name == "*":
                        if module.split(".")[0] in {"torch", "torch_npu"}:
                            self._flag(f"star import from {module} (unauditable)")
                        continue
                    self.aliases[alias.asname or alias.name] = (
                        f"{module}.{alias.name}" if module else alias.name
                    )

    def _collect_bindings(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            elif isinstance(node, (ast.For, ast.comprehension)):
                if (
                    isinstance(node.iter, ast.Call)
                    and isinstance(node.iter.func, ast.Name)
                    and node.iter.func.id == "range"
                ):
                    for name_node in ast.walk(node.target):
                        if isinstance(name_node, ast.Name):
                            self.scalar_refs.add(name_node.id)
                continue
            else:
                continue
            # Tuple unpacking: B, C, H, W = x.shape  /  a, b = x.size(0), n
            if len(targets) == 1 and isinstance(targets[0], (ast.Tuple, ast.List)):
                elts = targets[0].elts
                if self._is_scalar(value):
                    for elt in elts:
                        if isinstance(elt, ast.Name):
                            self.scalar_refs.add(elt.id)
                elif isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(elts):
                    for tgt, val in zip(elts, value.elts):
                        if isinstance(tgt, ast.Name) and self._is_scalar(val):
                            self.scalar_refs.add(tgt.id)
                continue
            for target in targets:
                key = self._ref_key(target)
                if key is None:
                    continue
                if self._is_scalar(value):
                    self.scalar_refs.add(key)
                path = self._resolve(value)
                if path and self._is_custom_op_path(path):
                    self.co_refs.add(key)
                if self._is_nn_layer_call(value):
                    self.holders.add(key)

    # -- scalar (non-tensor) expression analysis -----------------------------

    def _is_scalar(self, node: ast.AST) -> bool:
        """True for expressions that cannot carry tensor data: constants,
        shape arithmetic, len()/int()/math.*, and names bound to such."""
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Name):
            return node.id in self.scalar_refs
        if isinstance(node, ast.Attribute):
            if node.attr in {"shape", "sizes"}:
                return True  # x.shape is a Size tuple of ints
            key = self._resolve(node)
            return key in self.scalar_refs if key else False
        if isinstance(node, ast.Subscript):
            base = node.value
            if isinstance(base, ast.Attribute) and base.attr in {"shape", "sizes"}:
                return True  # x.shape[i]
            if (
                isinstance(base, ast.Call)
                and isinstance(base.func, ast.Attribute)
                and base.func.attr == "size"
            ):
                return True  # x.size(i)[j] is int; x.size(i) itself is int
            return isinstance(base, ast.Name) and base.id in self.scalar_refs
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                return func.id in _SCALAR_BUILTINS
            if isinstance(func, ast.Attribute):
                if func.attr in _SCALAR_METHODS:
                    return True
                path = self._resolve(func)
                return bool(path and path.startswith("math."))
            return False
        if isinstance(node, ast.BinOp):
            return (
                isinstance(
                    node.op,
                    (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow),
                )
                and self._is_scalar(node.left)
                and self._is_scalar(node.right)
            )
        if isinstance(node, ast.UnaryOp):
            return isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)) and self._is_scalar(
                node.operand
            )
        if isinstance(node, (ast.Tuple, ast.List)):
            return all(self._is_scalar(elt) for elt in node.elts)
        return False

    # -- call checks ---------------------------------------------------------

    def _is_nn_layer_call(self, node: ast.AST) -> bool:
        """True for constructions ``nn.<Layer>(...)`` of non-structural layers."""
        if not isinstance(node, ast.Call):
            return False
        path = self._resolve(node.func)
        if not path or not path.startswith("torch.nn."):
            return False
        rest = path[len("torch.nn."):]
        if rest.startswith(("functional", "init", "parameter")):
            return False
        return rest.split(".")[0] not in _STRUCTURAL_NN_NAMES

    def _is_holder_call(self, node: ast.Call) -> bool:
        """True when the call invokes an nn layer as a function (the compute)."""
        func = node.func
        if isinstance(func, (ast.Name, ast.Attribute)):
            key = self._ref_key(func)
            if key in self.holders:
                return True
            if isinstance(func, ast.Attribute):
                # self.layers[0](x) — subscripted holder
                if isinstance(func.value, ast.Subscript):
                    base_key = self._ref_key(func.value.value)
                    if base_key in self.holders:
                        return True
                # holder.forward(x)
                if func.attr == "forward":
                    base_key = self._ref_key(func.value)
                    if base_key in self.holders:
                        return True
        # nn.Conv2d(...)(x) — construct-and-call in one expression
        return isinstance(func, ast.Call) and self._is_nn_layer_call(func)

    def _flag(self, message: str) -> None:
        self.violations.append(message)

    def visit_Call(self, node: ast.Call) -> None:
        func_path = self._resolve(node.func)
        if func_path and self._is_custom_op_path(func_path):
            self.calls_custom_op = True
            self.generic_visit(node)
            return

        flagged = False
        if self._is_holder_call(node):
            self._flag(
                "calls an nn layer as a function — nn modules may only hold "
                "parameters; the compute must go through custom_op"
            )
            flagged = True
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in _DYNAMIC_CALLS:
                self._flag(f"dynamic code execution ({name}())")
                flagged = True
            elif name == "getattr" and node.args:
                base = self._resolve(node.args[0])
                if base and (base == "torch" or base.startswith(("torch.", "torch_npu"))):
                    self._flag("getattr() on torch modules (dynamic op access)")
                    flagged = True

        if func_path is not None:
            if func_path.startswith("torch_npu."):
                self._flag(f"vendor native op shortcut: {func_path}()")
                flagged = True
            elif func_path.startswith("torch.ops."):
                self._flag(f"vendor op-plugin call: {func_path}()")
                flagged = True
            elif func_path.startswith("torch.nn.functional."):
                self._flag(f"torch.nn.functional compute: {func_path}()")
                flagged = True
            elif func_path.startswith("torch.nn."):
                pass  # layer construction: allowed as parameter container
            elif func_path == "torch" or func_path.startswith("torch."):
                parts = func_path.split(".")
                if len(parts) == 2:
                    if parts[1] not in _TORCH_ALLOWED_CALLS:
                        self._flag(
                            f"torch.{parts[1]}() is not allowed in model_new.py "
                            "(only allocation and data-movement glue are)"
                        )
                        flagged = True
                else:
                    self._flag(f"{func_path}() is not allowed in model_new.py")
                    flagged = True
            elif func_path.startswith(("os.system", "os.popen", "os.exec", "os.spawn")):
                self._flag(f"host process execution: {func_path}()")
                flagged = True

        if (
            not flagged
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _TENSOR_COMPUTE_METHODS
        ):
            self._flag(
                f"tensor-method compute (.{node.func.attr}(...)) — compute must "
                "live in the Ascend C kernel"
            )
        self.generic_visit(node)

    # -- operator / comparison checks ----------------------------------------

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.MatMult):
            self._flag("@ (matmul) operator — compute must live in the Ascend C kernel")
        elif isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
        ):
            if not (self._is_scalar(node.left) and self._is_scalar(node.right)):
                self._flag(
                    "arithmetic on non-scalar values — tensor compute must live "
                    "in the Ascend C kernel (integer shape arithmetic is fine)"
                )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if not isinstance(node.op, ast.MatMult):
            if not (self._is_scalar(node.target) and self._is_scalar(node.value)):
                self._flag(
                    "in-place arithmetic on non-scalar values — tensor compute "
                    "must live in the Ascend C kernel"
                )
        else:
            self._flag("@ (matmul) operator — compute must live in the Ascend C kernel")
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)) and not self._is_scalar(
            node.operand
        ):
            self._flag(
                "unary arithmetic on a non-scalar value — tensor compute must "
                "live in the Ascend C kernel"
            )
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
            self.generic_visit(node)
            return
        operands = [node.left, *node.comparators]
        none_check = any(
            isinstance(o, ast.Constant) and o.value is None for o in operands
        )
        if not none_check and not all(self._is_scalar(o) for o in operands):
            self._flag(
                "comparison on non-scalar values — tensor comparisons are "
                "compute and belong in the Ascend C kernel"
            )
        self.generic_visit(node)


def _dedupe(violations: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for violation in violations:
        if violation not in seen:
            seen.add(violation)
            out.append(violation)
    return out


def check_model_new(source: str) -> list[str]:
    """Return static-check violations for generated ``model_new.py`` source."""
    code = _strip_comments(_mask_string_constants(source))
    violations: list[str] = []
    violations.extend(_check_bypass(code))
    violations.extend(_check_npu_native(code))
    violations.extend(_check_stream_injection(code))
    violations.extend(_check_thread_injection(code))
    violations.extend(_check_lazy_eval(code))
    violations.extend(_check_timing_event_patch(code))
    violations.extend(_check_result_cache(code))
    try:
        violations.extend(_WrapperSemantics(source).violations)
    except SyntaxError as exc:
        violations.append(f"model_new.py does not parse as Python: {exc}")
    return _dedupe(violations)


# ---------------------------------------------------------------------------
# Ascend C source checks (custom_op.asc)
# ---------------------------------------------------------------------------

_ASC_REQUIRED_MARKERS = ("__global__", "__vector__", "PYBIND11_MODULE")

# at:: host-side calls that are pure allocation/construction, never compute.
_ASC_AT_ALLOWED_CALLS = {
    "Tensor", "empty", "empty_like", "zeros", "zeros_like", "ones",
    "ones_like", "full", "full_like", "empty_strided", "from_blob",
    "scalar_tensor", "tensor",
}

_ASC_COMPUTE_METHODS = (
    "matmul", "mm", "bmm", "addmm", "baddbmm", "mv", "ger", "outer",
    "relu", "relu_", "sigmoid", "sigmoid_", "tanh", "tanh_", "gelu", "silu",
    "softmax", "log_softmax", "leaky_relu", "elu", "selu",
    "add", "add_", "sub", "sub_", "mul", "mul_", "div", "div_", "pow",
    "sum", "mean", "amax", "amin", "argmax", "argmin", "prod", "norm",
    "var", "std", "cumsum", "topk", "sort", "clamp", "gather", "scatter",
    "index_select", "conv1d", "conv2d", "conv3d",
)

_ASC_BANNED_PATTERNS = [
    (r"\baclnn[A-Z]\w*", "vendor prebuilt operator (aclnn*) — implement the kernel yourself"),
    (r"\baclop\w*", "legacy vendor operator API (aclop*)"),
    (r"\bstd::system\s*\(|\bsystem\s*\(", "host process execution"),
    (r"\bpopen\s*\(|\bexecl\w*\s*\(|\bexecv\w*\s*\(|\bfork\s*\(", "host process execution"),
    (r"\bsocket\s*\(|\bconnect\s*\(", "network access"),
    (r"\bdlopen\s*\(|\bdlsym\s*\(", "dynamic loading"),
    (r"#\s*include\s*<ATen/ops/", "ATen operator headers"),
    (r"\bstd::thread\b|\bpthread_create\b|\bstd::async\b", "host threads (timing manipulation)"),
]


def _strip_cpp_comments(code: str) -> str:
    """Blank // and /* */ comments in place so markers in comments don't count."""
    code = re.sub(
        r"/\*.*?\*/", lambda m: " " * (m.end() - m.start()), code, flags=re.DOTALL
    )
    return re.sub(r"//[^\n]*", lambda m: " " * (m.end() - m.start()), code)


def check_custom_op_asc(source: str) -> list[str]:
    """Return static-check violations for generated ``custom_op.asc`` source.

    The host wrapper may allocate outputs and launch kernels; all compute must
    be inside the ``__global__ __vector__`` Ascend C kernel. ATen compute
    calls, vendor prebuilt ops (aclnn/aclop) and host side effects are banned.
    """
    code = _strip_cpp_comments(source)
    violations: list[str] = []

    for marker in _ASC_REQUIRED_MARKERS:
        if marker not in code:
            violations.append(
                f"custom_op.asc missing {marker!r}: not an Ascend C kernel "
                "(host-only ATen implementations are not allowed)"
            )

    bad_aten = sorted(
        {
            m.group(1)
            for m in re.finditer(r"\bat::(\w+)\s*\(", code)
            if m.group(1) not in _ASC_AT_ALLOWED_CALLS
        }
        | {
            m.group(1)
            for m in re.finditer(r"\btorch::(\w+)\s*\(", code)
            if m.group(1) not in _ASC_AT_ALLOWED_CALLS
        }
    )
    for name in bad_aten:
        violations.append(
            f"host-side ATen call at::{name}(...) — compute must live in the "
            "Ascend C kernel, not in libtorch"
        )
    if re.search(r"\bat::native::", code):
        violations.append("at::native:: call — direct ATen kernel reuse is not allowed")

    method_re = r"(?:\.|->)(" + "|".join(_ASC_COMPUTE_METHODS) + r")\s*\("
    bad_methods = sorted({m.group(1) for m in re.finditer(method_re, code)})
    for name in bad_methods:
        violations.append(
            f"host-side tensor method .{name}(...) — compute must live in the "
            "Ascend C kernel"
        )

    for pattern, what in _ASC_BANNED_PATTERNS:
        match = re.search(pattern, code)
        if match:
            violations.append(f"{what}: {match.group(0)}")
    return _dedupe(violations)

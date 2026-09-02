"""Component-based prompt construction (README 4.4).

Pure-Python assembly, in order: problem_statement (the reference Model source
code, exactly as KernelBench presents it), hardware_block (from the hardware
profile), examples_block (verified one-shot example pair), output_contract
(dual code-block markers, module name ``custom_op``), instruction. Modes:
zero_shot / one_shot (default) / few_shot.

The example block teaches the Ascend C language itself — kernel class
structure, ``__global__ __vector__``, UB budgeting and tiling, host launch,
pybind binding — because Ascend C is scarce in LLM corpora. Task semantics
are NOT spelled out beyond the Model source: mapping a PyTorch reference to
an Ascend C operator is precisely the capability under test.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._paths import PROMPT_EXAMPLES_DIR
from .config import HardwareProfile
from .dataset import Task

SYSTEM_PROMPT = (
    "You are an expert Ascend C kernel engineer. You write correct, "
    "high-performance Ascend C operators for Huawei Ascend NPUs, compiled "
    "through a fixed CMake + pybind11 pipeline."
)

OUTPUT_CONTRACT = """\
## Output Contract

You must output exactly two fenced code blocks, tagged with their filenames:

1. ```custom_op.asc — one self-contained Ascend C source file with four parts:
   a. kernel class: `Init` (data partition across cores, GM buffers) and
      `Process` (UB allocation, copy-in, compute, copy-out);
   b. the kernel function annotated `__global__ __vector__`, calling
      `AscendC::InitSocState()`, `Init`, `Process`, `AscendC::PipeBarrier<PIPE_ALL>()`;
   c. host wrapper taking `const at::Tensor&` arguments, fetching the current
      NPU stream via `c10_npu::getCurrentNPUStream().stream(false)`,
      allocating outputs, launching with `<<<numBlocks, 0, stream>>>`;
   d. `PYBIND11_MODULE(custom_op, m)` exporting the host function(s).
   The module name MUST be `custom_op`. Exported function names are free
   (`run` is the convention); multi-kernel tasks may export several entries.
   The host wrapper may only allocate memory and launch kernels — all compute
   must happen inside the Ascend C kernel, never in host-side ATen calls
   (`at::matmul`, `tensor.relu()`, ...) or vendor prebuilt ops (`aclnn*`).
2. ```model_new.py — class `ModelNew` with the SAME `__init__` and `forward`
   signatures as the reference `Model`. It is a thin wrapper: import
   `custom_op` and call the compiled operator. Keep all optimisation work in
   the `.asc` file.
   - Parameters: if the reference `Model` has parameters (e.g. `nn.Conv2d`,
     `nn.Linear`, norm layers), you MAY instantiate the same `nn` modules in
     `ModelNew.__init__` as parameter containers — the evaluator seeds the
     RNG identically before constructing `Model` and `ModelNew`, so identical
     construction yields identical weights — but you must NEVER call them;
     pass their `.weight`/`.bias` tensors into your custom op.

Do not output any test code, `if __name__ == "__main__"` blocks, or prose
between the two code blocks. In model_new.py, ALL tensor compute must go
through `custom_op`: no torch native operators in any form — free functions
(`torch.matmul`), tensor methods (`x.softmax(...)`), operators (`A @ B`,
`A + B` on tensors), or comparisons on tensor data — and no nn.functional,
torch_npu/aclnn shortcuts, CPU/NumPy fallbacks, try/except, dynamic imports
(`importlib`, `__import__`, `getattr` on torch), or result caching. Integer
shape arithmetic (shapes, strides, counts) is of course allowed.
"""

INSTRUCTION = """\
## Instruction

Implement the operator defined by the reference model above in Ascend C.
Generate real, compilable code: every API you use must exist in the new-style
Ascend C API shown in the example. Output only the two code blocks defined by
the Output Contract.
"""


@dataclass(frozen=True)
class PromptExample:
    """A verified example pair (task input -> expected answer)."""

    name: str
    task_py: str
    custom_op_asc: str
    model_new_py: str


def load_examples() -> list[PromptExample]:
    """Load verified few-shot example assets shipped with the engine."""
    examples: list[PromptExample] = []
    for example_dir in sorted(PROMPT_EXAMPLES_DIR.iterdir()):
        if not example_dir.is_dir():
            continue
        examples.append(
            PromptExample(
                name=example_dir.name,
                task_py=(example_dir / "task.py").read_text(encoding="utf-8"),
                custom_op_asc=(example_dir / "custom_op.asc").read_text(
                    encoding="utf-8"
                ),
                model_new_py=(example_dir / "model_new.py").read_text(
                    encoding="utf-8"
                ),
            )
        )
    return examples


def _problem_statement(task: Task) -> str:
    return f"""\
## Problem Statement

Implement the operator defined by the reference PyTorch model below as an
Ascend C kernel on the target NPU.

```python
{task.task_py}
```
"""


def _hardware_block(hw: HardwareProfile) -> str:
    dtypes = ", ".join(hw.supported_dtypes)
    return f"""\
## Target Hardware Contract

- SoC: {hw.soc_version} (CMake arch `{hw.cmake_arch}`)
- AI cores: {hw.ai_core_num}; UB budget per core: {hw.ub_size_kb} KB
- HBM: {hw.hbm_gb} GB, bandwidth ~{hw.memory_bandwidth_gbps} GB/s
- Supported dtypes: {dtypes}

### Ascend C API style (mandatory)

{hw.api_style}
"""


def _examples_block(examples: list[PromptExample]) -> str:
    parts = ["## Example\n"]
    for example in examples:
        parts.append(
            f"### Example task: {example.name}\n\n"
            f"Reference Model:\n\n```python\n{example.task_py}\n```\n\n"
            f"Expected answer:\n\n"
            f"```custom_op.asc\n{example.custom_op_asc}\n```\n\n"
            f"```model_new.py\n{example.model_new_py}\n```\n"
        )
    return "\n".join(parts)


def build_prompt(
    task: Task,
    hardware: HardwareProfile,
    *,
    mode: str = "one_shot",
    examples: list[PromptExample] | None = None,
) -> str:
    """Assemble the full generation prompt for one task."""
    if mode not in {"zero_shot", "one_shot", "few_shot"}:
        raise ValueError(f"Unknown prompt mode: {mode}")
    components = [_problem_statement(task), _hardware_block(hardware)]
    if mode != "zero_shot":
        pool = examples if examples is not None else load_examples()
        if not pool:
            raise ValueError("no prompt examples available")
        chosen = pool[:1] if mode == "one_shot" else pool
        components.append(_examples_block(chosen))
    components.extend([OUTPUT_CONTRACT, INSTRUCTION])
    return "\n".join(components)

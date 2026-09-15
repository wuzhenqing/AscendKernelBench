"""Component-based prompt construction (docs/reference/configuration.md).

Pure-Python assembly, in order: problem_statement (the reference Model source
code, exactly as KernelBench presents it), hardware_block (from the hardware
profile), examples_block (verified one-shot example pair), output_contract
(dual code-block markers, module name ``custom_op``), instruction. Modes:
zero_shot / one_shot (default) / few_shot.

The example block teaches the Ascend C language itself — kernel class
structure, ``__global__ __vector__``, UB budgeting and tiling, host launch,
and the process-local ``TORCH_LIBRARY`` binding — because Ascend C is scarce
in LLM corpora. Task semantics are NOT spelled out beyond the Model source:
mapping a PyTorch reference to an Ascend C operator is precisely the
capability under test.

Prompts describe the implemented ACLNN operator-project path: a fixed modern
CMake project builds ``libcustom_op.so`` and the evaluator calls
``torch.ops.load_library`` on that file. Do not ask the model for pybind11,
an OPP / custom_opp install project, or in-process JIT compilation.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from ._paths import PROMPT_EXAMPLES_DIR
from .config import HardwareProfile
from .dataset import Task

SYSTEM_PROMPT = (
    "You are an expert Ascend C kernel engineer. You write correct, "
    "high-performance Ascend C operators for Huawei Ascend NPUs. The "
    "benchmark compiles your operator with a fixed modern CMake project into "
    "a process-local shared library (libcustom_op.so) and loads it with "
    "torch.ops.load_library inside the evaluating PyTorch process. Register "
    "the operator with TORCH_LIBRARY / TORCH_LIBRARY_IMPL. Do not use "
    "pybind11. The library is never installed into site-packages or the "
    "global CANN OPP path."
)

OUTPUT_CONTRACT = """\
## Output Contract

The benchmark already owns the CMake operator project. You must NOT emit
CMakeLists.txt, build.sh, op_host / op_kernel trees, framework plugins,
custom_opp packages, or any install / pip / OPP deployment step. Output
exactly two fenced code blocks, tagged with their filenames:

1. ```custom_op.asc — one self-contained Ascend C source file with four parts:
   a. kernel class: `Init` (data partition across cores, GM buffers) and
      `Process` (UB allocation, copy-in, compute, copy-out);
   b. the kernel function annotated `__global__ __vector__`, calling
      `AscendC::InitSocState()`, `Init`, `Process`,
      `AscendC::PipeBarrier<PIPE_ALL>()`;
   c. host wrapper taking `const at::Tensor&` arguments, fetching the current
      NPU stream via `c10_npu::getCurrentNPUStream().stream(false)`,
      allocating outputs, launching with `<<<numBlocks, 0, stream>>>`;
   d. a process-local torch.library binding. Register the host function with
      `TORCH_LIBRARY(custom_op, m)` and bind the NPU implementation with
      `TORCH_LIBRARY_IMPL(custom_op, PrivateUse1, m)`. Do NOT use
      `PYBIND11_MODULE` or any pybind11 header. The evaluator will
      `torch.ops.load_library` the resulting `libcustom_op.so`.
   The library namespace MUST be `custom_op`. Exported function names are
   free (`run` is the convention); multi-kernel tasks may export several
   entries. Schema strings must match the host function (for example
   `run(Tensor x, Tensor y) -> Tensor`). The host wrapper may only allocate
   memory and launch kernels — all compute must happen inside the Ascend C
   kernel, never in host-side ATen calls (`at::matmul`, `tensor.relu()`,
   ...) or vendor prebuilt ops (`aclnn*`). Do not call `cmake --install` or
   write files outside this source.
2. ```model_new.py — class `ModelNew` with the SAME `__init__` and `forward`
   signatures as the reference `Model`. It is a thin wrapper: call
   `torch.ops.custom_op.run(...)` (or the names you exported). The evaluator
   already loaded `libcustom_op.so` with `torch.ops.load_library`. Do not
   `import custom_op`, do not call `torch.ops.load_library`, do not change
   `sys.path`, and do not install anything. Keep all optimisation work in
   the `.asc` file.
   - Parameters: if the reference `Model` has parameters (e.g. `nn.Conv2d`,
     `nn.Linear`, norm layers), you MAY instantiate the same `nn` modules in
     `ModelNew.__init__` as parameter containers — the evaluator seeds the
     RNG identically before constructing `Model` and `ModelNew`, so identical
     construction yields identical weights — but you must NEVER call them;
     pass their `.weight`/`.bias` tensors into your custom op.

Do not output any test code, `if __name__ == "__main__"` blocks, or prose
between the two code blocks. In model_new.py, ALL tensor compute must go
through `torch.ops.custom_op`: no torch native operators in any form —
free functions
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
Ascend C API shown in the example. The evaluator will compile this file with
its fixed CMake project into libcustom_op.so and load that library with
torch.ops.load_library; you only write the two code blocks defined by the
Output Contract.
"""


class PromptMode(str, Enum):
    """How many bundled examples the user prompt includes."""

    ZERO_SHOT = "zero_shot"
    ONE_SHOT = "one_shot"
    FEW_SHOT = "few_shot"

    def chosen_examples(self, pool: list[PromptExample]) -> list[PromptExample]:
        """Return the example slice this mode should embed.

        Args:
            pool: Bundled or caller-supplied examples.

        Returns:
            ``[]`` for zero-shot, the first example for one-shot, or
            the full pool for few-shot.
        """
        if self is PromptMode.ZERO_SHOT:
            return []
        if self is PromptMode.ONE_SHOT:
            return pool[:1]
        return list(pool)


class PromptExample(BaseModel):
    """A verified example pair (task input -> expected answer)."""

    model_config = ConfigDict(frozen=True)

    name: str
    task_py: str
    custom_op_asc: str
    model_new_py: str


def load_examples() -> list[PromptExample]:
    """Load verified few-shot example assets shipped with the engine.

    Returns:
        Examples sorted by directory name.

    Raises:
        OSError: If an example directory is missing a required file.
    """
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


class PromptBuilder:
    """Assemble the generation prompt as an ordered list of sections."""

    def __init__(
        self,
        task: Task,
        hardware: HardwareProfile,
        *,
        examples: list[PromptExample] | None = None,
    ) -> None:
        """Bind the task, hardware profile, and optional example override.

        Args:
            task: Reference KernelBench task.
            hardware: Profile injected into the hardware-contract block.
            examples: Optional override of bundled examples.
        """
        self.task = task
        self.hardware = hardware
        self._examples = examples

    def build(self, mode: str | PromptMode = PromptMode.ONE_SHOT) -> str:
        """Return the complete English user prompt (no system message).

        Args:
            mode: ``zero_shot``, ``one_shot``, or ``few_shot``.

        Returns:
            Assembled markdown prompt.

        Raises:
            ValueError: If ``mode`` is unknown or examples are required
                but missing.
        """
        try:
            if isinstance(mode, PromptMode):
                resolved = mode
            else:
                resolved = PromptMode(mode)
        except ValueError as exc:
            raise ValueError(f"Unknown prompt mode: {mode}") from exc
        sections = [self._problem_statement(), self._hardware_block()]
        chosen = resolved.chosen_examples(self._example_pool(resolved))
        if chosen:
            sections.append(self._examples_block(chosen))
        sections.extend([OUTPUT_CONTRACT, INSTRUCTION])
        return "\n".join(sections)

    def _example_pool(self, mode: PromptMode) -> list[PromptExample]:
        """Load bundled examples when the mode needs them."""
        if mode is PromptMode.ZERO_SHOT:
            return []
        pool = self._examples if self._examples is not None else load_examples()
        if not pool:
            raise ValueError("no prompt examples available")
        return pool

    def _problem_statement(self) -> str:
        """Return the English problem-statement block."""
        return f"""\
## Problem Statement

Implement the operator defined by the reference PyTorch model below as an
Ascend C kernel on the target NPU.

```python
{self.task.task_py}
```
"""

    def _hardware_block(self) -> str:
        """Return the English hardware-contract block."""
        hw = self.hardware
        dtypes = ", ".join(hw.supported_dtypes)
        cores = f"{hw.ai_core_num}"
        if hw.cube_core_num or hw.vector_core_num:
            cores = (
                f"{hw.ai_core_num} (cube={hw.cube_core_num}, "
                f"vector={hw.vector_core_num})"
            )
        return f"""\
## Target Hardware Contract

- SoC: {hw.soc_version} (CMake arch `{hw.cmake_arch}`)
- AI cores: {cores}; UB budget per core: {hw.ub_size_kb} KB
- HBM: {hw.hbm_gb} GB, bandwidth ~{hw.memory_bandwidth_gbps} GB/s
- Supported dtypes: {dtypes}

### Ascend C API style (mandatory)

{hw.api_style}
"""

    def _examples_block(self, examples: list[PromptExample]) -> str:
        """Render verified example pairs as prompt markdown."""
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
    """Assemble the full generation prompt for one task.

    Args:
        task: Reference KernelBench task.
        hardware: Profile injected into the hardware-contract block.
        mode: ``zero_shot``, ``one_shot``, or ``few_shot``.
        examples: Optional override of bundled examples.

    Returns:
        The complete English user prompt (no system message).

    Raises:
        ValueError: If ``mode`` is unknown or examples are required
            but missing.
    """
    return PromptBuilder(task, hardware, examples=examples).build(mode)

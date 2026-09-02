# Task Authoring Guide

The benchmark task set IS the vendored KernelBench copy under
`KernelBench/level{1..4}/*.py` — 270 problems, committed to this repository,
used in place. A task is a single self-contained Python file in the original
KernelBench format.

## Contract

Each task file must define:

| Symbol | Signature | Notes |
|---|---|---|
| `Model` | `torch.nn.Module` | `forward()` implements the reference operator. |
| `get_init_inputs` | `() -> list` | Constructor args for `Model`; deterministic, no RNG. |
| `get_inputs` | `() -> list` | Forward args; RNG consumed only here. |
| `TOLERANCE` | `dict` (optional) | e.g. `{"atol": 1e-3, "rtol": 1e-3}` overrides the eval-config tolerance. |
| `custom_check` | `(ref_out, out) -> bool` (optional) | Replaces tolerance comparison (e.g. argmax ties). |

Rules (enforced by the evaluator and static checker):

- `get_inputs()` must return CPU tensors; the evaluator moves them to the NPU.
- `Model.forward()` must be deterministic given fixed inputs.
- No I/O, no network, no `torch.compile` in task files.

Evaluation semantics a task author should know:

- Both the reference `Model` and the candidate `ModelNew` run in `eval()`
  mode under `torch.no_grad()`. The evaluator re-seeds the RNG identically
  before constructing `Model` and `ModelNew`, so a candidate that constructs
  the same modules in the same order sees identical weights — this is what
  makes parameterized tasks (conv, linear, norm) winnable.
- Floating-point outputs are compared with the tolerance above; integer and
  bool outputs (indices, masks) are compared with exact `torch.equal`, since
  `torch.allclose` does not accept non-floating dtypes. `TOLERANCE` only
  affects floating-point comparison.
- Timing redraws inputs every trial when the input set is small (≤ 256 MB,
  same seed sequence for reference and candidate); larger input sets stay
  fixed during timing, followed by one post-timing correctness re-check on a
  fresh input set. Either way, `get_inputs()` must not depend on global
  state carried between calls.

## Task identity

The task id is `level{L}/{file_stem}`, e.g. `level1/19_ReLU` for
`KernelBench/level1/19_ReLU.py`. The id is used as the run-directory path and
as the KernelBench-compatible `problem_id` in `eval_results.json`.

## No specification documents

Unlike the original design draft, tasks carry **no** `spec.md`. The prompt
presents the `Model` source code as the entire problem statement — the same
trust KernelBench places in the model for CUDA. Mapping a PyTorch reference to
a correct, fast Ascend C kernel from the model source alone is precisely the
capability under test; a model that cannot do it should score accordingly.

## Correctness reference and the CPU fallback

The primary correctness reference is `torch_npu` eager on the NPU (same
device, same inputs, same precision). If the reference `Model` cannot run on
NPU at all — `torch_npu` lacks the op — the evaluator falls back to a CPU
reference (fp32) and compares the candidate's NPU output against it. A task
that passes this way is an operator the LLM effectively added to `torch_npu`;
results are marked `"reference": "cpu"` in the sample metadata, have no NPU
baseline speedup, and still count toward `fast_0` (the correctness rate).

## Adding a task

Drop a KernelBench-format `.py` file into `KernelBench/level{L}/`. The
dataset loader validates the contract statically (`Model`, `get_inputs`,
`get_init_inputs` present, parseable) at discovery time.

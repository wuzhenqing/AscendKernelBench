# AscendKernelBench

[![Quality](https://github.com/wuzhenqing/AscendKernelBench/actions/workflows/quality.yml/badge.svg)](https://github.com/wuzhenqing/AscendKernelBench/actions/workflows/quality.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A reproducible benchmark for **LLM-generated Ascend C kernels** on Huawei
Ascend NPUs. A language model reads a PyTorch reference `Model` and writes
two files. The harness builds them into a process-local `libcustom_op.so`,
loads that library with `torch.ops.load_library`, checks outputs against
the reference, and times eligible kernels against live `torch_npu` eager
execution.

[Documentation](docs/index.md) ·
[Getting started](docs/guide/getting-started.md) ·
[Evaluation protocol](docs/guide/evaluation.md) ·
[Results and metrics](docs/guide/results.md) ·
[Contributing](CONTRIBUTING.md)

## Why this benchmark

AscendKernelBench is a KernelBench-style *generate, then evaluate* harness
for Ascend C, not a CUDA-to-NPU score translator.

- **270 vendored KernelBench tasks** in four levels: 100 operators, 100
  fused operators, 50 networks or subgraphs, and 20 model tasks.
- **Process-local operators.** ACLNN mode compiles `custom_op.asc` into
  `libcustom_op.so` next to the sample. The evaluator loads it inside the
  worker PyTorch process. Nothing is installed into site-packages or the
  global CANN OPP path, so parallel jobs do not lock a shared environment.
- **Kernels are actually checked.** Static anti-cheat on both files, five
  seeded correctness trials, input-mutation rejection, isolated workers,
  NPU-event timing with L2 flush, and a post-timing fresh-input re-check.
- **Honest scores.** `fast_p` and `pass@k` follow KernelBench semantics.
  An optional roofline **SOL score** follows NVIDIA SOL-ExecBench's
  formula, using the hardware-profile bandwidth (not NVIDIA SOLAR). Each
  scored sample records the protocol snapshot and software stack used to
  produce it.
- **English generation prompts** and a fixed CMake template. The model
  writes `custom_op.asc` and `model_new.py` only.

The current workflow supports generation and evaluation scripts. It does
not implement an automatic compile-error repair agent.

## Choose your environment

| Work | Machine without an NPU | Linux Ascend host |
| --- | --- | --- |
| Read tasks and build prompts | Yes | Yes |
| Generate candidates with a remote LLM | Yes | Yes |
| Lint and unit-test | Yes | Yes |
| Analyze saved `eval_results.json` | Yes | Yes |
| Compile, check, or time a kernel | No | Required |

**Neither `--no-perf` nor CPU-reference fallback enables evaluation on a
laptop.** Candidates always run on an Ascend NPU. CPU fallback changes
only the reference used for correctness; those samples have no NPU
speedup and no SOL score.

## Quick start

Use Python 3.10 or later. On an Ascend host the recommended experiment
environment is conda env `akb` with **PyTorch 2.10.0** and
**torch-npu 2.10.0.post6** (CANN 9.1.0 pairing):

```bash
git clone https://github.com/wuzhenqing/AscendKernelBench.git
cd AscendKernelBench

# Host without an NPU: orchestration, lint, and unit tests
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On a Linux Ascend host with CANN 9.1.0, create the experiment env:

```bash
conda env create -f environment.yml
conda activate akb
source /usr/local/Ascend/cann-9.1.0/set_env.sh
python -m pip install -e ".[dev]"
```

The base extra does **not** install CANN, the NPU driver, or a hardware
runtime. See [getting started](docs/guide/getting-started.md).

Generate (any machine with network access to your LLM):

```bash
export OPENAI_BASE_URL="https://your-provider.example/v1"
export OPENAI_API_KEY="your-api-key"

python scripts/generate.py \
  --task level1/19_ReLU \
  --model your-served-model \
  --hardware ascend910b2 \
  --run-name relu_demo
```

Evaluate and report (Ascend host, exclusive device):

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --run-name relu_demo \
  --hardware ascend910b2 \
  --device npu:0

python scripts/analyze.py --run-name relu_demo
```

## Command-line tools

| Script | Purpose |
| --- | --- |
| `scripts/generate.py` | Sample an OpenAI-compatible model; write sources under `runs/` |
| `scripts/evaluate.py` | Build, check, and time every sample in a run |
| `scripts/run_single.py` | One-task generate + evaluate loop |
| `scripts/analyze.py` | Print `fast_p`, `pass@k`, geomean speedup, mean SOL |
| `scripts/baseline.py` | Archive live `torch_npu` eager timings (not the eval denominator) |

## How kernels are scored

| Metric | Meaning |
| --- | --- |
| `fast_0` | Correctness rate over **collected** sample results |
| `fast_p` (`p > 0`) | Fraction of collected samples that are correct and strictly faster than `p`× the live NPU reference |
| `pass@k` | Unbiased chance of at least one correct sample in `k` draws |
| Geometric-mean speedup | Over correct, unflagged NPU-reference samples only |
| SOL score | `(T_b − T_sol) / ((T_k − T_sol) + (T_b − T_sol))`. `0.5` matches the baseline; `1.0` would match the roofline bound. Bound = estimated traffic / profile bandwidth |

Suspicious speedups (`> 10×` by default) stay in `fast_0` and `pass@k`
but are excluded from positive `fast_p` and the geometric mean. Generation
failures that never wrote a sample are **absent** from the denominator.
Compare runs only with matching hardware, CANN, precision, and timing
settings. These scores are **not** comparable to CUDA KernelBench or to
NVIDIA SOL-ExecBench numbers.

## Evaluation environment

The default protocol is:

- 5 seeded correctness trials; all must pass
- 10 warmups (SOL-ExecBench-style isolation) and 100 retained NPU-event
  trials (KernelBench-style statistics)
- L2 flush of `max(256 MiB, 2 × profile L2)` before each timed call
- Isolated worker process group; host kills the group on timeout
- `torch.library` only: `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL`, loaded
  with `torch.ops.load_library`

The default hardware profile is `ascend910b2`. `ascend950pr` is reserved
and must be validated before use.

## Repository layout

```text
KernelBench/                 Vendored reference tasks, level1–level4
src/ascend_kernel_bench/     Generation, build, evaluation, and scoring
scripts/                     Five benchmark CLIs
configs/                     Eval defaults and hardware profiles
build_template/              CMake project that writes libcustom_op.so
docs/                        English Markdown guides and references
.github/workflows/           Lint and unit-test quality gate
environment.yml              Conda recipe for the akb experiment env
```

## Development

```bash
python -m pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
pytest
```

Style is enforced by pre-commit: Ruff (PEP 8 + Google pydocstyle),
ruff-format (80 columns), and the standard hook set. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgments and license

The task set is vendored from
[KernelBench](https://github.com/ScalingIntelligence/KernelBench).
Scoring ideas follow KernelBench (`fast_p`, `pass@k`) and NVIDIA
[SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench) (isolated
timing and the SOL formula). The roofline bound used here is a
documented bandwidth estimate, not a SOLAR characterization.

Released under the [MIT License](LICENSE).

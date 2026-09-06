# AscendKernelBench

[![Documentation](https://github.com/wuzhenqing/AscendKernelBench/actions/workflows/docs.yml/badge.svg)](https://github.com/wuzhenqing/AscendKernelBench/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

AscendKernelBench evaluates the correctness and performance of **LLM-generated
Ascend C kernels on Huawei Ascend NPUs**. Given a PyTorch reference model, a
language model produces an Ascend C implementation and a thin Python wrapper.
The benchmark builds that implementation, checks its outputs, and measures it
against `torch_npu` eager execution.

**[Read the documentation](https://wuzhenqing.github.io/AscendKernelBench/)** ·
[Getting started](https://wuzhenqing.github.io/AscendKernelBench/guide/getting-started.html) ·
[CLI reference](https://wuzhenqing.github.io/AscendKernelBench/reference/cli.html) ·
[Evaluation protocol](https://wuzhenqing.github.io/AscendKernelBench/guide/evaluation.html)

## What is included

- **270 vendored KernelBench tasks** across four levels: 100 individual
  operators, 100 fused operators, 50 networks or subgraphs, and 20 model tasks.
- **Generation through an OpenAI-compatible endpoint**, with saved source files
  that can be evaluated later or transferred to another machine.
- **A shared CMake and pybind11 build path** for `custom_op.asc` and a
  `ModelNew` wrapper in `model_new.py`.
- **Isolated evaluation workers** with static checks, seeded correctness trials,
  input mutation checks, NPU event timing, and post-timing correctness checks.
- **Machine-readable results**, `fast_p` metrics, and `pass@k` estimates.

The current workflow supports generation and evaluation scripts. It does not
implement an automatic compile-error repair agent. The `ascend910b2` profile is
the default; `ascend950pr` is reserved and requires validation before use.

## Choose your environment

| Work | macOS / machine without an NPU | Linux Ascend host |
| --- | --- | --- |
| Read tasks and build prompts | Yes | Yes |
| Generate candidates with a remote LLM service | Yes | Yes |
| Analyze saved evaluation results | Yes | Yes |
| Build and preview documentation | Yes | Yes |
| Compile Ascend C, check candidate outputs, or measure performance | No | Required |

**Neither `--no-perf` nor CPU reference fallback enables evaluation on macOS.**
Candidates always require an Ascend NPU. CPU fallback changes only the reference
used for correctness; those samples have no NPU reference speedup.

## Quick start

Use Python 3.10 or later and run the commands below from a repository checkout:

```bash
git clone https://github.com/wuzhenqing/AscendKernelBench.git
cd AscendKernelBench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

This installs the orchestration dependencies. It does **not** install PyTorch,
`torch_npu`, CANN, or a hardware-specific runtime. See the
[installation guide](https://wuzhenqing.github.io/AscendKernelBench/guide/getting-started.html)
for the Linux Ascend prerequisites and repository layout requirements.

For generation, set the endpoint and credentials provided by your LLM service:

```bash
export OPENAI_BASE_URL="https://your-provider.example/v1"
export OPENAI_API_KEY="your-api-key"

python scripts/generate.py \
  --task level1/19_ReLU \
  --model your-served-model \
  --hardware ascend910b2 \
  --run-name relu_demo
```

Replace the endpoint, API key, and model placeholders. The configured default
model is not a bundled service. Generation makes remote API requests and stores
the prompt, raw response, and two source files under
`runs/relu_demo/level1/19_ReLU/sample_0/`. Use a new run name for each experiment.

On a configured **Linux Ascend host**, with the same checkout and generated run:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --run-name relu_demo \
  --hardware ascend910b2 \
  --device npu:0

python scripts/analyze.py --run-name relu_demo
```

The selected device must be available exclusively for evaluation; use separate
physical devices for a local LLM service. Evaluation takes its hardware settings
from the current CLI/configuration, not from the saved generation configuration.
See [workflows](https://wuzhenqing.github.io/AscendKernelBench/guide/workflows.html)
for batch sampling, single-task runs, archived baselines, and repeated evaluation.

## Interpret results

`fast_0` is correctness over collected sample results. For positive thresholds,
`fast_p` is the fraction of collected samples that are correct and exceed speedup
`p`; speedup is reference mean runtime divided by candidate mean runtime.
Suspicious speedups and CPU-reference samples are excluded from positive
thresholds. `pass@k` estimates the chance of finding a correct sample in `k`
draws and requires enough samples per problem.

Generation failures that produce no saved sample are absent from the evaluation
denominator. Check generation completeness before reporting a score, and compare
runs only with matching hardware, software, tasks, precision, and timing settings.
See [results and metrics](https://wuzhenqing.github.io/AscendKernelBench/guide/results.html)
for formulas, files, exclusions, and reproducibility limits. These scores are not
directly comparable with CUDA KernelBench results.

## Build the documentation

The VitePress site needs only Node.js and npm. Node.js 24 is used in CI and recorded
in `.nvmrc`.

```bash
npm ci
npm run docs:dev
```

For a production build with internal link and anchor checks:

```bash
npm run docs:check
npm run docs:preview
```

Open the `/AscendKernelBench/` path printed by the server. Pull requests build and
check the site; pushes to `main` publish it to GitHub Pages. See
[maintaining the docs](https://wuzhenqing.github.io/AscendKernelBench/guide/documentation.html)
for the publishing workflow.

## Repository layout

```text
KernelBench/                 Vendored reference tasks, level1 through level4
src/ascend_kernel_bench/      Generation, build, evaluation, and scoring modules
scripts/                     Five benchmark CLIs and documentation link checker
configs/                     Evaluation defaults and hardware profiles
build_template/              Shared Ascend C CMake project
docs/                        English user documentation and VitePress configuration
.github/workflows/docs.yml   Documentation checks and Pages deployment
runs/                        Local generated candidates and results (ignored by Git)
```

To contribute, start with the
[task authoring guide](https://wuzhenqing.github.io/AscendKernelBench/task_authoring.html),
[architecture reference](https://wuzhenqing.github.io/AscendKernelBench/reference/architecture.html),
or [troubleshooting guide](https://wuzhenqing.github.io/AscendKernelBench/guide/troubleshooting.html).
Documentation and tooling checks can run on macOS; kernel compilation, numerical
correctness, and performance claims require validation on the target Ascend host.

## Acknowledgments and license

The task set is vendored from
[KernelBench](https://github.com/ScalingIntelligence/KernelBench). AscendKernelBench
adapts the reference-model contract and evaluation ideas to Ascend C and
`torch_npu`. The project is released under the [MIT License](LICENSE).

---
layout: home
title: User documentation
hero:
  name: AscendKernelBench
  text: From PyTorch to Ascend C.
  tagline: A reproducible workflow for generating kernels, checking correctness, and measuring performance on Huawei Ascend NPUs.
  actions:
    - theme: brand
      text: Get started
      link: /guide/getting-started
    - theme: alt
      text: Understand the evaluation
      link: /guide/evaluation
    - theme: alt
      text: View on GitHub
      link: https://github.com/wuzhenqing/AscendKernelBench
features:
  - title: 270 reference tasks
    details: Four levels of vendored KernelBench tasks, from individual operators to complete models. Each task uses the same Python contract.
    link: /task_authoring
    linkText: Explore the task contract
  - title: Generate once, evaluate later
    details: Connect an OpenAI-compatible service and save Ascend C source with a Python wrapper. Move those artifacts to an NPU host when ready.
    link: /guide/workflows
    linkText: Follow the workflow
  - title: Correctness before speed
    details: Seeded correctness trials, input mutation checks, NPU event timing, and explicit handling of CPU references and suspicious speedups.
    link: /guide/results
    linkText: Read the metrics
---

## Choose your starting point

| Your environment | What you can do | Start here |
| --- | --- | --- |
| macOS or a machine without an Ascend NPU | Read tasks, construct prompts, generate candidates through a remote service, analyze existing JSON results, and build these docs | [Installation and first steps](/guide/getting-started) |
| Linux with a supported Ascend software stack | Compile candidates, check their outputs, collect timings, and record eager baselines | [Generation and evaluation](/guide/workflows) |
| Adding tasks or integrating an agent | Use the task contract and call individual engine modules | [Task authoring](/task_authoring) · [Architecture](/reference/architecture) |

## What the benchmark measures

The input is a PyTorch reference `Model`. A language model produces two files:
`custom_op.asc`, containing the Ascend C kernel and binding, and `model_new.py`,
containing a compatible `ModelNew` wrapper. The evaluator checks that wrapper,
builds a process-local `libcustom_op.so`, loads it in the PyTorch process, compares outputs, and measures eligible candidates against
`torch_npu` eager execution. [Results and metrics](/guide/results) explains how
`fast_p` and `pass@k` are calculated and which samples enter each denominator.

The project implements generation, isolated sample evaluation, and analysis.
It does not provide an automatic compile-error repair loop. A saved hardware
profile is a configuration input, not proof that a software or device combination
has passed validation. In particular, `ascend950pr` is reserved and must be checked
before use. See [configuration and hardware](/reference/configuration).

::: info Working on macOS
The documentation site builds without Python, CANN, or `torch_npu`. Actual kernel
compilation and evaluation require the Linux Ascend environment; CPU fallback
changes the reference only and does not make candidate evaluation run on a Mac.
:::

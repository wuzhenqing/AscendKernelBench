# Documentation

```text
/======================================================================\
|                                                                      |
|    +---------+        ___    __ __    ____                           |
|    |# # # # #|       /   |  / //_/   / __ )                          |
|    |#  NPU  #|      / /| | / ,<     / __  |                          |
|    |# CORE  #|     / ___ |/ /| |   / /_/ /                           |
|    +---------+    /_/  |_/_/ |_|  /_____/                            |
|                                                                      |
|              A S C E N D     K E R N E L     B E N C H               |
|                                                                      |
|        [ LLM ]=======>{ Ascend C }=======>{ libcustom_op.so }        |
|                 fast_p   |   pass@k   |   SOL score                  |
|                                                                      |
\======================================================================/
```

Guides and references for running AscendKernelBench. These pages live in
the repository and are meant to be read on GitHub. There is no published
documentation site.

## Start here

- [Installation and first steps](guide/getting-started.md)
- [Connect an LLM service](deploy_llm_service.md)

## Run the benchmark

- [Generation and evaluation](guide/workflows.md)
- [Evaluation protocol](guide/evaluation.md)
- [Results and metrics](guide/results.md)
- [Troubleshooting](guide/troubleshooting.md)

## Reference

- [Command line](reference/cli.md)
- [Configuration and hardware](reference/configuration.md)
- [Architecture and Python entry points](reference/architecture.md)
- [Write a benchmark task](task_authoring.md)

## Choose your starting point

| Your environment | What you can do | Start here |
| --- | --- | --- |
| macOS or a machine without an Ascend NPU | Read tasks, construct prompts, generate candidates through a remote service, and analyze existing JSON results | [Installation and first steps](guide/getting-started.md) |
| Linux with a supported Ascend software stack | Compile candidates, check their outputs, collect timings, and record eager baselines | [Generation and evaluation](guide/workflows.md) |
| Adding tasks or integrating an agent | Use the task contract and call individual engine modules | [Task authoring](task_authoring.md) · [Architecture](reference/architecture.md) |

## What the benchmark measures

The input is a PyTorch reference `Model`. A language model produces two files:
`custom_op.asc`, containing the Ascend C kernel and binding, and `model_new.py`,
containing a compatible `ModelNew` wrapper. The evaluator checks that wrapper,
builds a process-local `libcustom_op.so`, loads it in the PyTorch process,
compares outputs, and measures eligible candidates against `torch_npu` eager
execution. [Results and metrics](guide/results.md) explains how `fast_p`,
`pass@k`, and the optional roofline SOL score are calculated and which samples
enter each denominator.

The project implements generation, isolated sample evaluation, and analysis.
It does not provide an automatic compile-error repair loop. A saved hardware
profile is a configuration input, not proof that a software or device
combination has passed validation. In particular, `ascend950pr` is reserved
and must be checked before use. See
[configuration and hardware](reference/configuration.md).

> **Working on a machine without an NPU.** Kernel compilation and evaluation
> require the Linux Ascend environment. CPU fallback changes the reference
> only; it does not make candidate evaluation run on a laptop or Mac.

# CLI reference

Run these scripts from the checkout root after [installing the package](../guide/getting-started.md). Each supports `-h` / `--help`. There is no installed `akb` or `ascend-kernel-bench` command-line entry point.

## Common conventions

| Concept | Meaning |
| --- | --- |
| Task ID | `level{number}/{file_stem}`, for example `level1/19_ReLU`. Omit `KernelBench/` and `.py`. |
| `--task` and `--level` | In batch generation and baseline measurement, repeat `--task` to select tasks. It overrides `--level`. Omitting both selects all discovered tasks. |
| Run name | A directory name under `<repository>/runs/`. Use a simple, unique name such as `relu-demo`. There is no separate `--run-dir` option. |
| `--hardware` | A profile name such as `ascend910b2`, or the path to an existing YAML file. |
| `--config` | An evaluation YAML file. If omitted, loads `<repository>/configs/eval_default.yaml`. A relative path is resolved from the current working directory. |
| `--device` | Device string, default `npu:0`. Device evaluation requires an Ascend NPU. |

All batch commands process items sequentially. They have no CLI options for parallel workers, resume, or skipping previously completed samples.

## `scripts/generate.py`

Generate source files through the configured LLM endpoint. This command does not require an NPU.

```bash
python scripts/generate.py \
  --task level1/19_ReLU \
  --n-samples 10 \
  --model "$AKB_MODEL" \
  --run-name relu-10
```

| Option | Default | Purpose |
| --- | --- | --- |
| `--level INTEGER` | All levels | Discover tasks from one level. |
| `--task TASK_ID` | Unset | Select a task; repeat for multiple tasks. Overrides `--level`. |
| `--n-samples INTEGER` | `generation.num_samples`, otherwise `1` | Number of samples per task. Use a positive integer. |
| `--model NAME` | `generation.model`, otherwise `deepseek-v4-flash` | Endpoint model name. |
| `--hardware NAME_OR_PATH` | `hardware` from configuration | Select the hardware description injected into the prompt. |
| `--prompt-mode MODE` | `generation.prompt_mode`, otherwise `one_shot` | `zero_shot`, `one_shot`, or `few_shot`. |
| `--temperature FLOAT` | `generation.temperature`, otherwise `0.0` | Generation sampling temperature. |
| `--run-name NAME` | `gen_YYYYMMDD_HHMMSS` | Output run name, using local time when generated automatically. |
| `--config PATH` | `configs/eval_default.yaml` | Load generation and hardware settings from an evaluation YAML file. |
| `--operator-mode MODE` | `operator_mode` from configuration, otherwise `aclnn` | `aclnn` (process-local shared library) or `jit` (reserved, not implemented). |

`max_tokens` is configured through `generation.max_tokens`; there is no `--max-tokens` flag. Successful samples include `prompt.txt`, `custom_op.asc`, `model_new.py`, and, when nonempty, `response_raw.txt`.

Generation errors are printed and the batch continues. The script exits with status 1 if **all** generation attempts fail; a partially successful batch returns status 0. Check the final saved/expected count before proceeding. Configuration, task loading, and prompt-construction errors can also stop execution.

## `scripts/evaluate.py`

Build and evaluate every complete source pair found in an existing run. Requires the Ascend evaluation environment.

```bash
python scripts/evaluate.py \
  --run-name relu-10 \
  --hardware ascend910b2 \
  --device npu:0
```

| Option | Default | Purpose |
| --- | --- | --- |
| `--run-name NAME` | Required | Existing run under `runs/`. |
| `--hardware NAME_OR_PATH` | `hardware` from configuration | Compilation target and result hardware label. |
| `--device DEVICE` | `npu:0` | NPU device for each worker. |
| `--no-perf` | Disabled | Skip performance timing; compilation and correctness checks still run. |
| `--config PATH` | `configs/eval_default.yaml` | Evaluation protocol settings. |
| `--operator-mode MODE` | `operator_mode` from configuration, otherwise `aclnn` | `aclnn` (process-local shared library) or `jit` (reserved, not implemented). |

This command has no `--task`, `--level`, `--sample-id`, or generation options. It discovers sample directories under `runs/<name>/level*/*/sample_*` that contain both `custom_op.asc` and `model_new.py`.

It writes per-sample `eval_result.json`, followed by the run's `eval_results.json` and `pass_at_k_results.json`. It does not infer hardware from the saved generation configuration. Existing evaluation results are replaced.

Missing runs or runs with no samples cause a nonzero exit. Individual compilation and correctness failures are recorded as results; a completed batch does not return a failing exit code merely because candidates failed. Inspect the result files when using this command in automation.

## `scripts/run_single.py`

Generate one sample and immediately build and evaluate it. Requires both an LLM endpoint and the Ascend environment.

```bash
python scripts/run_single.py \
  --task level1/19_ReLU \
  --model "$AKB_MODEL" \
  --run-name relu-single
```

| Option | Default | Purpose |
| --- | --- | --- |
| `--task TASK_ID` | Required | Exactly one task. |
| `--model NAME` | Generation configuration | Endpoint model name. |
| `--hardware NAME_OR_PATH` | `hardware` from configuration | Prompt hardware description and build target. |
| `--device DEVICE` | `npu:0` | Evaluation device. |
| `--run-name NAME` | `single_<task_file_stem>` | For the example task, `single_19_ReLU`. |
| `--sample-id INTEGER` | `0` | Sample directory index. Use a nonnegative integer. |
| `--prompt-mode MODE` | Generation configuration | `zero_shot`, `one_shot`, or `few_shot`. |
| `--temperature FLOAT` | Generation configuration | Override sampling temperature. |
| `--no-perf` | Disabled | Skip timing while retaining NPU correctness checks. |
| `--config PATH` | `configs/eval_default.yaml` | Generation and evaluation settings. |
| `--operator-mode MODE` | `operator_mode` from configuration, otherwise `aclnn` | `aclnn` (process-local shared library) or `jit` (reserved, not implemented). |

Generation fallbacks match `generate.py`: model `deepseek-v4-flash`, prompt mode `one_shot`, temperature `0.0`, and maximum tokens `16384`. This script always generates one sample; it does not use `generation.num_samples`.

The command writes source files, a per-sample result, and aggregate `eval_results.json`. It does not write `pass_at_k_results.json`. The exit status is 0 for a correct candidate and 1 for an incorrect or failed candidate; an unhandled generation/setup error also exits unsuccessfully.

## `scripts/baseline.py`

Measure and archive PyTorch eager reference timings on the NPU, without an LLM call or generated candidate.

```bash
python scripts/baseline.py \
  --task level1/19_ReLU \
  --hardware ascend910b2 \
  --device npu:0
```

| Option | Default | Purpose |
| --- | --- | --- |
| `--level INTEGER` | All levels | Discover tasks from one level. |
| `--task TASK_ID` | Unset | Select a task; repeat to select more. Overrides `--level`. |
| `--hardware NAME_OR_PATH` | `hardware` from configuration | Select the hardware name used for the archive directory. |
| `--device DEVICE` | `npu:0` | Device on which to time the reference. |
| `--config PATH` | `configs/eval_default.yaml` | Precision, seed, warmup, trial count, and timeout settings. |

Results go to `results/baseline/<hardware.name>/<task_file_stem>.json`. The command can record `supported_on_npu: false` for a reference that fails within its measurement phase. Worker startup failures may leave no result file. Per-task exceptions are printed and the batch continues without a failing exit code solely for those exceptions.

The hardware profile labels the archive; it does not change the physical device selected by `--device`. Evaluation independently measures its own live reference and does not consume these archived files.

## `scripts/analyze.py`

Print aggregate and per-problem reports from existing results. Does not need an NPU or endpoint credentials.

```bash
python scripts/analyze.py --run-name relu-10
```

| Option | Default | Purpose |
| --- | --- | --- |
| `--run-name NAME` | Required | Read `runs/<name>/eval_results.json`. |

The script recalculates metrics in memory and prints them, including `fast_p`, geometric-mean speedup, `pass@k`, and mean roofline SOL score when present. It writes no report files, does not collect per-sample results, and does not update `pass_at_k_results.json`. A missing aggregate file causes a nonzero exit.

See [configuration](configuration.md) for defaults and environment variables, and [results](../guide/results.md) for the result schema and scoring rules.

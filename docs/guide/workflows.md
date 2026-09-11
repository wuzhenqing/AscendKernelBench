# Generation and evaluation workflows

Use separate generation and evaluation steps when developing on macOS, comparing models, or repeating measurements on an Ascend host. Each run keeps source files and results in the repository's `runs/` directory.

## Generate a small, explicit task set

After [configuring your environment](/guide/getting-started), generate candidates for selected tasks:

```bash
python scripts/generate.py \
  --task level1/19_ReLU \
  --task level1/20_LeakyReLU \
  --model "$AKB_MODEL" \
  --hardware ascend910b2 \
  --n-samples 10 \
  --temperature 0.7 \
  --run-name activations-10
```

Repeat `--task` to select multiple tasks. Alternatively, use `--level 1` for all level 1 tasks. Omitting both selectors generates candidates for **all discovered tasks**, so start with explicit task IDs when checking an endpoint.

The command processes tasks and samples sequentially. Sample IDs start at zero for each task. There is no CLI option for parallel generation or automatic resume.

Generation requests structured output first and falls back to fenced code blocks if that request fails. The client checks for basic source markers and can retry an invalid generation once. These retries address response generation or formatting; they do not use compiler or correctness feedback to repair the operator.

`one_shot` includes the first bundled example; `few_shot` includes every bundled example. The current checkout ships an elementwise-add example and a LeakyReLU example. `zero_shot` omits examples.

::: warning Preserve separate experiments
Use a new run name for each experiment. Reusing a name overwrites `generation_config.yaml` and matching sample source files, while old samples, build artifacts, and evaluation results can remain. Reusing a directory is not a clean restart.
:::

## Inspect generated files locally

A static check is available without a compiler, endpoint, or NPU:

```bash
python - <<'PY'
from pathlib import Path
from ascend_kernel_bench.checker import check_custom_op_asc, check_model_new

sample = Path("runs/activations-10/level1/19_ReLU/sample_0")
violations = check_custom_op_asc((sample / "custom_op.asc").read_text())
violations += check_model_new((sample / "model_new.py").read_text())
print("\n".join(violations) if violations else "No static violations found")
PY
```

Passing this check does not establish compiler compatibility or numerical correctness. Read the [evaluation protocol](/guide/evaluation) for the checks applied on the device.

## Move a run from macOS to Linux

Prepare the [Ascend environment](/guide/getting-started#prepare-an-ascend-evaluation-machine) on the destination, using the same repository revision and task sources. Copy the run directory into that checkout's `runs/` directory. For example, once the destination `runs/` directory exists:

```bash
scp -r runs/activations-10 \
  user@ascend-host:/path/to/AscendKernelBench/runs/
```

Replace the host and path with your own. Copy any custom evaluation or hardware YAML files separately and preserve their contents. Record the repository revision and environment versions alongside the run: `generation_config.yaml` captures generation settings and task IDs, but not a complete environment or evaluation configuration snapshot.

A newly generated run consists of portable source and text files. Native `custom_op*.so` files and CMake build directories from a previous evaluation are machine-specific; use a source-only copy when moving an evaluated run to another environment so the destination can build it afresh.

## Evaluate on the target NPU

On the Linux Ascend machine:

```bash
python scripts/evaluate.py \
  --run-name activations-10 \
  --hardware ascend910b2 \
  --device npu:0
```

The command evaluates every sample directory containing both required source files. Each sample passes through static checking, compilation, seeded correctness trials, and timing in an isolated worker process. The batch runs sequentially on the selected device.

To check correctness without timing:

```bash
python scripts/evaluate.py \
  --run-name activations-10 \
  --hardware ascend910b2 \
  --device npu:0 \
  --no-perf
```

This still needs the Ascend compiler and NPU. It produces no runtime or speedup measurement.

::: warning Select the hardware again
Evaluation does not load `generation_config.yaml` to select hardware. Pass the intended `--hardware`, or use an evaluation configuration containing the same hardware profile. Otherwise, the evaluator uses the current configuration's default, `ascend910b2`.
:::

The evaluator rebuilds and reevaluates samples on each invocation, overwrites their `eval_result.json` files, and rewrites the run's aggregate files. Save a separate copy of a run before comparing repeated evaluations. Archived baselines are not read by this command: it measures the reference in the same worker as the candidate.

## Read the report anywhere

After evaluation completes, run:

```bash
python scripts/analyze.py --run-name activations-10
```

The report shows compile and correctness counts, `fast_p`, geometric mean speedup, `pass@k`, and per-problem detail. Analysis reads the aggregate `eval_results.json`; it does not evaluate source files or refresh that aggregate from per-sample results.

You can copy the run back to macOS for analysis. For reporting alone, `runs/activations-10/eval_results.json` is sufficient. Keep the full run if you also want to inspect generated code or compilation diagnostics.

Ten successfully saved and evaluated samples per task allow `pass@1`, `pass@5`, and `pass@10` to be reported. Missing generations are not represented as failed evaluation samples. Check the generation command's final saved/expected count before interpreting scores: partially failed generation batches can exit successfully. See [results and metrics](/guide/results) for metric denominators and eligibility rules.

## Run one task end to end

With the LLM endpoint and Ascend environment available on the same machine:

```bash
python scripts/run_single.py \
  --task level1/19_ReLU \
  --model "$AKB_MODEL" \
  --hardware ascend910b2 \
  --device npu:0 \
  --run-name relu-end-to-end
```

This generates one sample, evaluates it, and writes `eval_results.json`. It exits with status 0 only when the sample is correct. Unlike `evaluate.py`, it does not write `pass_at_k_results.json`; `analyze.py` can still calculate and display the available metrics from the aggregate.

## Archive reference baselines

On the Ascend host, measure a reference without generating candidates:

```bash
python scripts/baseline.py \
  --task level1/19_ReLU \
  --hardware ascend910b2 \
  --device npu:0
```

This writes `results/baseline/ascend910b2/19_ReLU.json`. Unsupported NPU references are recorded with `supported_on_npu: false` when the worker can report the failure. The archive is a record of a particular machine and configuration, not an input to evaluation.

Archive paths contain the hardware profile name and task file stem, but not precision, configuration, timestamp, or level. Repeated measurements overwrite the same path; preserve copies when comparing configurations.

All flags are listed in the [CLI reference](/reference/cli), with defaults and precedence in [configuration](/reference/configuration).

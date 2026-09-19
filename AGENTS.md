# AGENTS.md

Working agreement for AI coding agents in this repository. `CONTRIBUTING.md`
is the human style and review contract; this file is the fast path to being
productive here without re-deriving the architecture.

**Maintenance rule:** if your change alters a command, directory, protocol
constant, or invariant described here, update this file in the same change.
A stale `AGENTS.md` is a bug.

## What this project is

AscendKernelBench is a *generate, then evaluate* benchmark for LLM-written
Ascend C kernels on Huawei Ascend NPUs. A model reads a PyTorch reference
`Model` from the vendored KernelBench corpus and must emit exactly two
artifacts: `custom_op.asc` and `model_new.py`. The harness compiles the
`.asc` into a **process-local** `libcustom_op.so`, loads it with
`torch.ops.load_library` inside a worker PyTorch process, checks outputs
against the reference, and times eligible kernels against live `torch_npu`
eager execution. Scoring is KernelBench `fast_p` / `pass@k` plus an optional
roofline SOL score.

```text
KernelBench task + hardware profile
  -> prompt.py -> llm.py -> runs/{run}/level{L}/{task}/sample_{i}/
       (prompt.txt, custom_op.asc, model_new.py, response_raw.txt)
  -> eval.py static checks (checker.py / checks/)
  -> isolated worker (scripts/_eval_worker.py)
  -> build.py + build_template/ -> libcustom_op.so -> torch.ops.load_library
  -> eval_device.py: seeded correctness -> NPU-event timing -> fresh-input re-check
  -> per-sample eval_result.json
  -> eval_results.json + pass_at_k_results.json -> score.py / scripts/analyze.py
```

Current scope: generation and evaluation CLIs only. There is **no**
automatic compile-error repair agent and no scheduler / multi-device queue.

## Commands

```bash
# Setup (any machine; NPU not required)
python -m pip install -e ".[dev]"
pre-commit install

# Quality gate — both must pass before you call a change done
pytest -q                              # 95 passed, 2 skipped on a torch-less host
pre-commit run --all-files             # ruff lint + format, whitespace, codespell

# Offline fallback when pre-commit cannot fetch hooks / write its cache
ruff check . && ruff format --check .

# Ascend host experiment environment (CANN 9.1.0 pairing)
conda env create -f environment.yml && conda activate akb
source /usr/local/Ascend/cann-9.1.0/set_env.sh
python -m pip install -e ".[dev]"

# Workflow CLIs (run from the checkout root)
export OPENAI_BASE_URL="https://provider.example/v1" OPENAI_API_KEY="..."
python scripts/generate.py --task level1/19_ReLU --model <model> \
    --hardware ascend910b2 --run-name relu_demo [--level N] [--n-samples N] \
    [--prompt-mode zero_shot|one_shot|few_shot] [--temperature T] [--config PATH]
# NPU commands need device access: without it aclInit fails with 507899 (see
# Local host notes). Have the user run them, or escalate the sandbox.
ASCEND_RT_VISIBLE_DEVICES=0 python scripts/evaluate.py relu_demo [level]
python scripts/analyze.py relu_demo
python scripts/baseline.py --level 1 --hardware ascend910b2 --device npu:0
```

CLI facts worth remembering:

* `scripts/evaluate.py` takes only `run` and an optional positional `level`.
  It has **no** `--config`/`--hardware` override: it always loads
  `configs/eval_default.yaml` and resolves hardware from the run's
  `generation_config.yaml` (falling back to `config.hardware`).
* `scripts/generate.py` and `scripts/baseline.py` accept `--config` /
  `--hardware`.
* `scripts/_eval_worker.py` is internal. Never call it by hand; it is spawned
  by `eval.eval_sample`.
* Offline analysis (`analyze.py`, `score.py`) needs neither torch nor an NPU.

## Layout

| Path | What lives there |
| --- | --- |
| `KernelBench/level{1..4}/` | Vendored corpus: 270 tasks (100/100/50/20). Read-only in spirit. |
| `src/ascend_kernel_bench/` | Engine: dataset, prompt, LLM, checks, eval, build, timing, scoring, reporting. |
| `src/ascend_kernel_bench/checks/` | Static anti-cheat: `rules.py` (regex catalogs), `python_ast.py` (wrapper AST), `ascend_c.py` (Ascend C), `python_source.py`/`text.py` (entry points and masking). |
| `src/ascend_kernel_bench/prompts/examples/` | Few-shot example pairs shipped as package data. |
| `scripts/` | The four user CLIs plus `_bootstrap.py` / `_eval_worker.py`. |
| `configs/eval_default.yaml` | Default protocol: 5 correctness trials, 10 warmup + 100 perf trials, tolerances, timeouts, generation defaults. |
| `configs/hardware/*.yaml` | Hardware profiles validated by `config.HardwareProfile`. Default `ascend910b2`; `ascend950pr` is reserved and unvalidated. |
| `build_template/CMakeLists.txt` | The single build path that produces `libcustom_op.so`. |
| `tests/` | NPU-free unit tests (CPU torch optional). |
| `docs/` | English guides (`guide/`) and references (`reference/`). |
| `runs/`, `results/` | Generated artifacts; gitignored. Never commit them. |

Module boundaries are strict; `docs/reference/architecture.md` has the full
API-layer table. In short: host code goes through `eval.evaluate_run`;
`eval_device` and `worker_main` are worker-side only; `score.py` never loads
kernels or tasks; `_paths.py` anchors all data directories to the repo root
(override with `AKB_REPO_ROOT` for an installed package).

## Invariants — do not break these

1. **Process-local operators.** Build in the sample directory and load with
   `torch.ops.load_library`. Never install into site-packages, `$ASCEND_OPP_PATH`,
   or a global vendor directory; never run `cmake --install`; never emit a
   `custom_opp_*.run` package. Parallel jobs must not share mutable install state.
2. **`torch.library` registration only.** `TORCH_LIBRARY` /
   `TORCH_LIBRARY_IMPL` under namespace `custom_op`. pybind11 is banned and the
   checker enforces it.
3. **Compute belongs to the device.** The host wrapper may allocate and launch
   only; no ATen compute, no `aclnn*`/`aclop` vendor ops, no host side effects
   (process, network, dlopen, threads). `model_new.py` may call only
   `torch.ops.custom_op` (plus integer shape arithmetic); `nn` modules may be
   parameter containers but must never be called.
4. **Static checks are advisory; the runtime protocol is the backstop.** The
   fresh-input re-check and `excessive_speedup` flag exist because regex/AST
   checks cannot stop determined obfuscation. Do not weaken either layer alone.
5. **Scoring semantics are KernelBench-compatible.** `fast_0` is the
   correctness rate over *collected* sample results. Generation failures that
   never wrote a sample are absent from the denominator, not failures.
   Suspicious speedups stay in `fast_0`/`pass@k` but are excluded from positive
   `fast_p` thresholds and the geometric mean. Do not "fix" that silently.
6. **The reference is measured live**, in the same worker, on the same device,
   in the same run. `results/baseline/` exists only as a cross-run archive and
   must never become the evaluation denominator.
7. **Keep the host NPU-free.** NPU imports (`torch`, `torch_npu`) stay inside
   worker and timing functions so prompt building, dataset discovery, and
   offline analysis work on machines without a device.
8. **English everywhere** in code, prompts, checker messages, comments, and docs.
9. **Never reformat or edit vendored `KernelBench/` tasks** to satisfy lint.
10. **CPU-reference fallback is not a free pass.** It only changes the
    correctness reference; such samples have no NPU speedup and no SOL score.

## Conventions

* Ruff (`pyproject.toml`): 80 columns, target py310, `E,F,W,I,UP,B,SIM,N,D,C4,PIE,RUF`,
  Google pydocstyle. `KernelBench/`, `runs/`, `results/` are excluded;
  `tests/**` skip docstring rules; prompt examples skip `D`/`N` so they keep
  the KernelBench `Model`/`A`/`B` contract.
* `from __future__ import annotations` plus type hints on public signatures;
  Google-style docstrings on public functions (summary, then `Args`/`Returns`/`Raises`
  when they add information). One responsibility per function.
* Frozen Pydantic models for YAML config and worker request payloads
  (`extra="ignore"`); dataclasses for internal value types.
* All file writes go through `io_util` atomic helpers. Result JSON must stay
  KernelBench-compatible and carry the protocol snapshot.
* Logging is loguru (`logger`); Rich is for CLIs and `report.py` only. Fatal CLI
  errors go through `log.die()`.
* Public API is re-exported from `__init__.py` and `__all__`; keep that list
  accurate when adding a public function.

## Tests

* Unit tests must never require an NPU. CPU `torch` is optional — gate it with
  the existing `importorskip` pattern (that is why 2 tests skip on a torch-less host).
* `pyproject.toml` sets `pythonpath = ["src"]`, so tests run against the
  checkout without an install.
* Kernel compile and NPU timing are host-side validation, not CI gates.
* Any change to scoring, comparison, SOL, dataset discovery, or the static
  checker needs a regression test that would have failed before the change.

## Common change recipes

| Change | Touch |
| --- | --- |
| New static check | `checks/rules.py` (regex catalog) or `checks/python_ast.py` / `checks/ascend_c.py`, plus `tests/test_checker.py`. |
| New hardware profile | `configs/hardware/<name>.yaml`; fields must satisfy `config.HardwareProfile`; mirror the comments in `ascend910b2.yaml`. |
| Protocol/timeout/trial change | `configs/eval_default.yaml` + `timing.py`/`eval.py` + `docs/guide/evaluation.md`. |
| Scoring change | `score.py` / `sol.py` + `tests/test_score.py` + `docs/guide/results.md` + README metrics table. |
| Prompt or output contract | `prompt.py` (`SYSTEM_PROMPT`, `OUTPUT_CONTRACT`) + `prompts/examples/` + `tests/test_prompt.py` + docs. |
| New CLI flag | Thin `scripts/*.py` argparse + `cli_util.py` factory + `docs/reference/cli.md`. |
| Build backend | `build.py` (`build_custom_op` / `load_custom_op` are the only surface) + `build_template/CMakeLists.txt`. |
| New task | Add to `KernelBench/levelN/` keeping `Model`, `get_inputs`, `get_init_inputs`; see `docs/task_authoring.md`. Task source is validated statically and never executed at load time. |

## Local host notes (this checkout, not upstream truths)

* Two Python environments exist here: the default base conda env (3.14, has
  `pytest`/`ruff`/`pre-commit` and an editable install, **no torch**) for tests
  and lint, and the `akb` conda env (`/root/miniconda3/envs/akb`, Python 3.12,
  torch 2.10.0 + torch-npu 2.10.0.post6, CANN 9.1.0) for anything NPU-related.
* **The NPU works here — but only outside the agent's file sandbox.** The host
  is an Ascend 910B2 (`npu-smi` 25.2.1, Health OK, 64 GB HBM) and the `akb`
  env reaches it: `torch.npu.is_available()` is `True`, `device_count()` is 1,
  and a matmul on `npu:0` succeeds. Under the default `workspace-write`
  sandbox, opening `/dev/davinci*` (mode 0666) fails with `EACCES`, which
  surfaces as `aclInit ... error code is 507899` from torch_npu and
  `dcmi module initialize failed. ret is -8005` from `npu-smi`. That is a
  sandbox artifact, **not** a driver fault — do not "diagnose" it as a broken
  install. Run build/evaluate/baseline with the sandbox escalated
  (`danger-full-access`) or from the user's own terminal, and expect a benign
  "CANN and HDK(driver) versions require processing for 32 padding size"
  warning when the NPU allocator initializes. `/dev/devmm_svm` reports
  `EACCES` even unsandboxed; torch_npu initializes fine regardless.
* The NPU path is smoke-verified on this host: `scripts/baseline.py --task
  level1/19_ReLU --hardware ascend910b2 --device npu:0`, run with the sandbox
  escalated, completed the 10-warmup/100-trial protocol and archived
  `results/baseline/ascend910b2/19_ReLU.json` (mean 10.5 ms). CANN toolchain
  variables (`ASCEND_TOOLKIT_HOME`, `ASCEND_OPP_PATH`, driver `lib64` in
  `LD_LIBRARY_PATH`, `bisheng` on `PATH`) are already present in
  non-interactive shells, so no `.bashrc` sourcing is required.
* `pre-commit run --all-files` cannot run in this container as-is: hook
  environments are fetched from github.com, which times out here, and the
  default `~/.cache/pre-commit` is outside the sandbox. Use
  `ruff check . && ruff format --check .` as the offline gate and say that
  pre-commit itself was not run.
* Re-evaluation of an unchanged sample reuses the sample-local `build/`
  directory, so builds are incremental. Set `AKB_ENABLE_CCACHE=1` to opt into
  ccache; set `CANN_SET_ENV` to point at a different `set_env.sh`.

## Definition of done

1. `pytest -q` and `pre-commit run --all-files` (or the ruff fallback) pass.
2. New behavior has a test that would have failed before the change.
3. CLI, result-field, or protocol changes update `README.md` and the matching
   `docs/` page — and this file if any of the above changed.
4. No global installs, no reformatted vendored tasks, no committed
   `runs/`, `results/`, `*.so`, or API keys.

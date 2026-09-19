# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working agreement

[`AGENTS.md`](AGENTS.md) is the canonical, actively maintained agent contract for
this repository — architecture, module boundaries, invariants, comment style, change
recipes, and notes about this specific checkout. It is imported below so it loads with
this file:

@AGENTS.md

**Maintenance rule.** If a change alters a command, directory, protocol constant, or
invariant that `AGENTS.md` documents, update `AGENTS.md` in the same change. Keep this
file to Claude-Code-specific additions only; do not copy the contract here.

Call the project AscendKernelBench in everything you write, including commit messages.
Do not abbreviate it as akb.

## Commands

```bash
# Setup — any machine, no torch and no NPU required; requirements.txt is the
# only dependency file and installs the pinned torch pair on Linux only.
python -m pip install -r requirements.txt

# Quality gate; both must pass before a change is done
pytest -q                                   # 107 passed, 2 skipped (2 skip without torch)
pre-commit run --all-files                  # works once hook envs are cached
ruff check . && ruff format --check .       # fallback when the hook fetch fails

# Narrower test runs (from the checkout root; pyproject sets pythonpath = ["src"])
pytest -q tests/test_score.py
pytest -q tests/test_score.py::test_summarize_includes_mean_sol
pytest -q -k pass_at_k

# Workflow CLIs (from the checkout root)
export OPENAI_BASE_URL=https://api.deepseek.com/v1 OPENAI_API_KEY=<key>
python scripts/generate.py --tasks-file configs/subsets/level1_20.txt \
    --model deepseek-flash --reasoning-effort high --run-name level1_20
ASCEND_RT_VISIBLE_DEVICES=0 python scripts/evaluate.py level1_20 [level]
python scripts/analyze.py level1_20         # offline: reads existing JSON, no torch or NPU
```

Compiling and timing kernels needs the AscendKernelBench conda env and an Ascend NPU.
Under this container's default sandbox, device access fails with `aclInit ... 507899` —
a sandbox artifact, not a broken driver. See "Local host notes" in `AGENTS.md` before
diagnosing any NPU error. Orchestration, prompt building, lint, unit tests, and offline
analysis all work with no device.

## Architecture in brief

A model reads a vendored KernelBench task and must emit exactly two files,
`custom_op.asc` (kernel + host wrapper + `TORCH_LIBRARY` binding) and `model_new.py`
(the `ModelNew` wrapper). The harness builds them into a **process-local**
`libcustom_op.so`, loads it with `torch.ops.load_library` in an isolated worker
process, checks outputs against a live `torch_npu` eager reference, and times
eligible kernels. Scoring is KernelBench `fast_p` / `pass@k` plus an optional
roofline SOL score.

```text
KernelBench task --> prompt.py --> llm.py --> runs/{run}/level{L}/{task}/sample_{i}/
  --> eval.py static checks (checks/) --> scripts/_eval_worker.py (one process per sample)
  --> build_template/ CMake --> libcustom_op.so --> eval_device.py
      (5 seeded correctness trials -> NPU-event timing -> fresh-input re-check)
  --> per-sample eval_result.json --> eval_results.json --> score.py / analyze.py
```

The layering is strict and `docs/reference/architecture.md` holds the full table:
host code calls `eval.evaluate_run`; `eval_device` and `worker_main` are worker-side
only; `score.py` never loads kernels, tasks, or NPU libraries. `_paths.py` anchors all
data directories (configs, tasks, build template, `runs/`, `results/`) to the
repository root, so a wheel install alone is not a working checkout — override with
`ASCEND_KERNEL_BENCH_REPO_ROOT`.

Generation talks to an OpenAI-compatible endpoint. `deepseek-flash` accepts
`reasoning_effort` and rejects `response_format`, so its answers arrive as a JSON
object with the two deliverable fields; `llm.py` accepts that layout and fenced code
blocks, and the note appended to every prompt asks for the JSON form.

The non-obvious rules that break things when violated (global `custom_opp_*.run`
installs, a second build path, host-side compute in the wrapper, using
`results/baseline/` as the evaluation denominator, importing torch at module scope,
reformatting vendored `KernelBench/`, changing the `fast_0`/`fast_p` denominator
semantics) are enumerated in the "Invariants" section of `AGENTS.md`. Read it before
touching `build.py`, `eval_device.py`, `checks/`, or `score.py`.

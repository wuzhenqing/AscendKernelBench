# Contributing to AscendKernelBench

Thank you for helping improve the harness. This document is the style contract
for the Python engine, scripts, and docs.

## Development environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pre-commit install   # optional
```

On an Ascend host, use the AscendKernelBench conda env (Python 3.12, CANN 9.1.0
torch pairing) and source `set_env.sh` before evaluate or baseline.

## Style gate

Optional before you submit Python changes:

```bash
pre-commit run --all-files
```

Pre-commit runs Ruff on `src/` and `scripts/` only (not vendored `KernelBench/`).
Few-shot prompt examples keep the KernelBench `Model` / `A` / `B` contract; do
not reformat vendored tasks to satisfy Ruff.

### Function quality

For code under `src/ascend_kernel_bench/`:

- One responsibility per function; extract a helper rather than nesting
  a second protocol inside a 200-line body.
- Google-style docstrings on public functions: a one-line summary,
  then `Args` / `Returns` / `Raises` when they add information.
- Type hints on public signatures (`from __future__ import annotations`).
- No global CANN / OPP install; operators stay process-local
  `libcustom_op.so` loaded with `torch.ops.load_library`.
- Prompts and checker messages stay in English.

Do not reformat vendored KernelBench tasks to “pass Ruff.”

## Documentation

User-facing docs are English Markdown pages under `docs/`. After
changing CLI flags, result fields, or the evaluation protocol:

1. Update the matching page (`docs/guide/evaluation.md`,
   `docs/guide/results.md`, `docs/reference/configuration.md`,
   `docs/reference/cli.md`).
2. Update `README.md` if the quick start, metrics, or layout changed.
3. Update [`AGENTS.md`](AGENTS.md) if a command, directory, protocol
   constant, or invariant it documents changed.
4. Keep links relative so the pages stay readable on GitHub.

## Pull requests

- Keep the change reviewable: one concern per PR when you can.
- Describe *why*, not only *what*.
- Do not commit `runs/`, `results/`, `.so` files, or API keys.

Expected review passes for engine changes:

1. **Protocol review** — KernelBench scoring semantics and process-local
   `torch.library` contract preserved.
2. **Style review** — pre-commit green when you use it; docstrings and types
   on new public functions.
3. **Docs review** — README and the relevant guide page match the tree.

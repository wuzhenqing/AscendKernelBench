# Contributing to AscendKernelBench

Thank you for helping improve the harness. This document is the style and
review contract for the Python engine, scripts, tests, and docs.

## Development environment

```bash
# Orchestration / lint / tests (any machine)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pre-commit install

# Optional: Ascend host experiment env
conda activate akb
source /usr/local/Ascend/cann-9.1.0/set_env.sh
```

`akb` pins Python 3.12, PyTorch 2.10.0, and torch-npu 2.10.0.post6 for
CANN 9.1.0. Recreate it with `conda env create -f environment.yml`.

## Style gate

Every change that touches Python must pass:

```bash
pre-commit run --all-files
pytest
```

Hooks enforce:

| Tool | What it checks |
| --- | --- |
| Ruff lint | PEP 8 (`E`/`W`), pyflakes, isort, pyupgrade, bugbear, plus Google-convention pydocstyle (`D`) |
| Ruff format | 80-column wrap, consistent quotes |
| pre-commit-hooks | Trailing whitespace, EOF, YAML/TOML/JSON, debug leftovers |
| codespell | Common misspellings |

Vendored `KernelBench/` tasks are excluded.
Few-shot prompt examples keep the KernelBench `Model` / `A` / `B`
contract and are excluded from docstring/naming rules.

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

## Tests

Unit tests live in `tests/` and must not require an NPU. They may import
CPU `torch`. Kernel compile / NPU timing is a host-side validation step,
not a CI gate.

If you change scoring, comparison, SOL, dataset discovery, or the static
checker, add or update a unit test that would have failed before the
change.

## Documentation

User-facing docs are English Markdown pages under `docs/`. After
changing CLI flags, result fields, or the evaluation protocol:

1. Update the matching page (`docs/guide/evaluation.md`,
   `docs/guide/results.md`, `docs/reference/configuration.md`,
   `docs/reference/cli.md`).
2. Update `README.md` if the quick start, metrics, or layout changed.
3. Keep links relative (`evaluation.md`, `../reference/cli.md`) so the
   pages stay readable on GitHub.

There is no documentation site or Pages workflow.

## Pull requests

- Keep the change reviewable: one concern per PR when you can.
- Describe *why*, not only *what*.
- Do not commit `runs/`, `results/`, `.so` files, or API keys.
- Do not push to `main` unless a maintainer asked you to.

Expected review passes for engine changes:

1. **Protocol review** — does the change preserve KernelBench scoring
   semantics and the process-local `torch.library` contract?
2. **Style review** — pre-commit is green; new functions meet the
   docstring / typing bar.
3. **Docs review** — README and the relevant guide page match the tree.

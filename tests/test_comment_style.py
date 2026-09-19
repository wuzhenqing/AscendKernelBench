"""House comment rules: short blocks, 80 columns, paired banners."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from ascend_kernel_bench._paths import REPO_ROOT

SKIPPED_DIRS = (
    "KernelBench",
    "runs",
    "results",
    ".git",
    "egg-info",
    "prompts/examples",
    ".pytest_cache",
    ".ruff_cache",
)
COMMENT_SUFFIXES = {".py", ".yaml", ".yml", ".txt", ".sh", ".toml", ".cfg"}
BANNER_RE = re.compile(r"^#+\s*([A-Z][A-Z0-9 _/+-]*[A-Z0-9])\s*#+$")
DECORATION_RE = re.compile(r"`|\*\*|^#\s*[-*=]{4,}\s*$")
MAX_COMMENT_LINES = 3
MAX_WIDTH = 80


def _tracked_files() -> list[Path]:
    """Return comment-bearing repository files, vendored corpus excluded."""
    files = []
    for path in sorted(REPO_ROOT.rglob("*")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if not path.is_file() or any(s in rel for s in SKIPPED_DIRS):
            continue
        if path.suffix in COMMENT_SUFFIXES or path.name == "CMakeLists.txt":
            files.append(path)
    return files


def _comment_lines(path: Path) -> list[tuple[int, str]]:
    """Return (line number, text) for every hash comment line."""
    found = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("#!"):
            found.append((number, stripped))
    return found


def test_comment_blocks_are_short() -> None:
    offenders = []
    for path in _tracked_files():
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        run, start = 0, 0
        for number, line in enumerate([*lines, ""], 1):
            stripped = line.strip()
            if stripped.startswith("#") and not stripped.startswith("#!"):
                if run == 0:
                    start = number
                run += 1
                continue
            if run > MAX_COMMENT_LINES:
                offenders.append(f"{path}:{start} ({run} lines)")
            run = 0
    assert not offenders, f"comment blocks longer than 3 lines: {offenders}"


def test_comment_lines_fit_the_margin() -> None:
    offenders = []
    for path in _tracked_files():
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if line.strip().startswith("#") and len(line) > MAX_WIDTH:
                offenders.append(f"{path}:{number}")
    assert not offenders, f"comment lines over {MAX_WIDTH} columns: {offenders}"


def test_comments_carry_no_decoration() -> None:
    offenders = []
    for path in _tracked_files():
        for number, text in _comment_lines(path):
            if DECORATION_RE.search(text):
                offenders.append(f"{path}:{number}: {text[:60]}")
    assert not offenders, f"decorated comment lines: {offenders}"


def test_banner_keywords_are_paired() -> None:
    offenders = []
    for path in _tracked_files():
        counts: dict[str, int] = {}
        for _, text in _comment_lines(path):
            match = BANNER_RE.match(text)
            if match:
                keyword = match.group(1).strip()
                counts[keyword] = counts.get(keyword, 0) + 1
        offenders += [
            f"{path}: {keyword} x{count}"
            for keyword, count in counts.items()
            if count % 2
        ]
    assert not offenders, f"unpaired banner keywords: {offenders}"


def test_module_and_class_docstrings_are_short() -> None:
    offenders = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")) + sorted(
        (REPO_ROOT / "scripts").rglob("*.py")
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef)):
                continue
            doc = ast.get_docstring(node, clean=True)
            if doc and len([ln for ln in doc.splitlines() if ln.strip()]) > 3:
                offenders.append(f"{path}: {getattr(node, 'name', 'module')}")
    assert not offenders, f"docstrings longer than 3 lines: {offenders}"

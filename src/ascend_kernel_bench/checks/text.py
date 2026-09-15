"""Comment stripping and string masking for static checks."""

from __future__ import annotations

import ast
import io
import re
import tokenize


def mask_string_constants(source: str) -> str:
    """Blank string constants so embedded text cannot trip regex rules.

    Args:
        source: Python source.

    Returns:
        Source with string literal spans replaced by spaces. Unparsable
        input is returned unchanged.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and getattr(node, "end_lineno", None) is not None
            and getattr(node, "end_col_offset", None) is not None
        ):
            continue
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        ranges.append((start, end))

    if not ranges:
        return source
    parts: list[str] = []
    cursor = 0
    for start, end in sorted(ranges):
        parts.append(source[cursor:start])
        parts.append(" " * (end - start))
        cursor = end
    parts.append(source[cursor:])
    return "".join(parts)


def strip_python_comments(code: str) -> str:
    """Blank Python ``#`` comments in place, preserving all other bytes.

    Reconstruction from token strings would drop inter-token whitespace and
    break space-sensitive patterns (``import numpy``), so comments are
    replaced by spaces at their exact offsets instead.

    Args:
        code: Python source, typically after string masking.

    Returns:
        Source with comments replaced by spaces.
    """
    lines = code.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    chars = list(code)
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        for tok in tokens:
            if tok.type != tokenize.COMMENT:
                continue
            (srow, scol), (erow, ecol) = tok.start, tok.end
            start = offsets[srow - 1] + scol
            end = offsets[erow - 1] + ecol
            for i in range(start, end):
                chars[i] = " "
    except (tokenize.TokenError, IndentationError):
        return "\n".join(
            line.split("#", 1)[0].rstrip() for line in code.splitlines()
        )
    return "".join(chars)


def strip_cpp_comments(code: str) -> str:
    """Blank C++ comments so markers inside comments do not count.

    Args:
        code: Ascend C / C++ source.

    Returns:
        Source with ``/* */`` and ``//`` comments replaced by spaces.
    """
    code = re.sub(
        r"/\*.*?\*/",
        lambda match: " " * (match.end() - match.start()),
        code,
        flags=re.DOTALL,
    )
    return re.sub(
        r"//[^\n]*",
        lambda match: " " * (match.end() - match.start()),
        code,
    )


def prepare_python_source(source: str) -> str:
    """Return comment-stripped, string-masked Python source.

    Args:
        source: Raw ``model_new.py`` text.

    Returns:
        Source suitable for regex catalogs.
    """
    return strip_python_comments(mask_string_constants(source))


def dedupe(violations: list[str]) -> list[str]:
    """Return ``violations`` with first-occurrence order preserved.

    Args:
        violations: Human-readable check messages.

    Returns:
        Deduplicated list.
    """
    return list(dict.fromkeys(violations))

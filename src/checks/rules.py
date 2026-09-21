"""Strategy objects for regex-based static checks.

Patterns are matched against prepared source: comments and string literal
content are blanked first, so text inside strings cannot trip a rule.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PatternRule:
    """One regex catalog and the violation it reports."""

    patterns: tuple[str, ...]
    message: str
    include_match: bool = False

    def check(self, code: str) -> list[str]:
        """Return one violation for the first matching pattern, or []."""
        for pattern in self.patterns:
            match = re.search(pattern, code)
            if match:
                if self.include_match:
                    return [f"{self.message}: {match.group(0)}"]
                return [self.message]
        return []


def run_rules(code: str, rules: Sequence[PatternRule]) -> list[str]:
    """Return violations from rules applied in order; not deduplicated."""
    violations: list[str] = []
    for rule in rules:
        violations.extend(rule.check(code))
    return violations

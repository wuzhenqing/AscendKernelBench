"""Strategy objects for regex-based static checks."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PatternRule:
    """One regex catalog and the violation it reports.

    Each rule is a Strategy: ``check(code)`` returns either one message
    or an empty list. Catalogs stay data; orchestration stays a loop.
    """

    patterns: tuple[str, ...]
    message: str
    include_match: bool = False

    def check(self, code: str) -> list[str]:
        """Return one violation for the first matching pattern, or [].

        Args:
            code: Comment-stripped, string-masked source.

        Returns:
            A one-element violation list, or an empty list.
        """
        for pattern in self.patterns:
            match = re.search(pattern, code)
            if match:
                if self.include_match:
                    return [f"{self.message}: {match.group(0)}"]
                return [self.message]
        return []


def run_rules(code: str, rules: Sequence[PatternRule]) -> list[str]:
    """Apply ``rules`` in order and concatenate their violations.

    Args:
        code: Prepared source text.
        rules: Ordered static-check strategies.

    Returns:
        Flattened violation list (not deduplicated).
    """
    violations: list[str] = []
    for rule in rules:
        violations.extend(rule.check(code))
    return violations

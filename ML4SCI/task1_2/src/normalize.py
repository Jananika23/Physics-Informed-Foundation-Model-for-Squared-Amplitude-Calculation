"""
Normalize index suffixes (e.g., _345 -> _1) per equation.
"""

from __future__ import annotations

import re

INDEX_RE = re.compile(r"_(\d+)")


def normalize_indices(expr: str) -> str:
    """
    Remap each unique numeric index in one expression to a compact sequence.

    Example:
      a_345 + b_678 + c_345 -> a_1 + b_2 + c_1
    """
    mapping: dict[str, int] = {}
    next_id = 1

    def repl(match: re.Match[str]) -> str:
        nonlocal next_id
        original = match.group(1)
        if original not in mapping:
            mapping[original] = next_id
            next_id += 1
        return f"_{mapping[original]}"

    return INDEX_RE.sub(repl, expr)


def normalize(samples: list[dict[str, str]]) -> list[dict[str, str]]:
    """Apply index normalization independently to input and target of each row."""
    normalized: list[dict[str, str]] = []
    for row in samples:
        raw_input = row["input"]
        raw_target = row["target"]
        normalized.append(
            {
                "input": raw_input,
                "target": raw_target,
                "normalized_input": normalize_indices(raw_input),
                "normalized_target": normalize_indices(raw_target),
            }
        )
    return normalized

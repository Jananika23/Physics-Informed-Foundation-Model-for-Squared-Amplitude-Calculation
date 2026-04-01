"""
Data cleaning for Task 1.2 samples.
"""

from __future__ import annotations


def preprocess(samples: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Remove invalid rows and normalize whitespace.
    """
    cleaned: list[dict[str, str]] = []
    for row in samples:
        raw_input = str(row.get("input", "")).strip()
        raw_target = str(row.get("target", "")).strip()
        if not raw_input or not raw_target:
            continue
        cleaned.append({"input": raw_input, "target": raw_target})

    print(f"Valid samples after cleaning: {len(cleaned)}")
    return cleaned

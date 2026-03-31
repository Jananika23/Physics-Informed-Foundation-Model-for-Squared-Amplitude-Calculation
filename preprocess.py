"""
Preprocess raw equation strings for consistent tokenization downstream.

Preprocessing is needed because raw CSV text can have inconsistent spacing,
invisible characters, or empty-looking rows. Normalizing whitespace and
dropping invalid rows keeps the tokenizer's regex reliable and avoids feeding
noisy strings into the ML pipeline.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw_equations.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed_equations.csv"

# Collapse repeated spaces; strip ends
_SPACE_RE = re.compile(r"\s+")


def preprocess_and_save() -> pd.DataFrame:
    """Load raw equations, clean, save processed_equations.csv."""
    if not RAW_PATH.is_file():
        print(
            f"Error: Missing {RAW_PATH}. Run load_data first.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(RAW_PATH)
    if df.empty or "equation" not in df.columns:
        print(
            "Error: raw_equations.csv is empty or missing the 'equation' column.",
            file=sys.stderr,
        )
        sys.exit(1)

    s = df["equation"].astype("string").fillna("")
    # Remove extra spaces and normalize line breaks to spaces
    cleaned = s.map(lambda t: _SPACE_RE.sub(" ", str(t).replace("\n", " ").replace("\r", " ")).strip())
    cleaned = cleaned[cleaned != ""]

    if cleaned.empty:
        print("Error: No valid equations left after preprocessing.", file=sys.stderr)
        sys.exit(1)

    out = pd.DataFrame({"equation": cleaned})
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED_PATH, index=False)
    print(f"Preprocessed {len(out)} equations -> {PROCESSED_PATH}")
    return out


if __name__ == "__main__":
    preprocess_and_save()

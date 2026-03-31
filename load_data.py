"""
Load AI-Feynman equations from CSV and write a normalized raw_equations file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "FeynmanEquations.csv"
RAW_OUTPUT = PROJECT_ROOT / "data" / "raw_equations.csv"

# Priority when matching column names (case-insensitive)
EQUATION_COLUMN_PRIORITY = ["equation", "formula", "expr"]


def _find_equation_column(columns: pd.Index) -> str:
    """Pick the equation column: preferred names first, else first column."""
    normalized = {c.strip().lower(): c for c in columns}
    for name in EQUATION_COLUMN_PRIORITY:
        if name in normalized:
            return normalized[name]
    return str(columns[0])


def load_and_save_raw() -> pd.DataFrame:
    """Load dataset, extract equations, save raw_equations.csv."""
    if not DATASET_PATH.is_file():
        print(
            f"Error: Dataset not found at:\n  {DATASET_PATH}\n"
            "Please download FeynmanEquations.csv and place it in the data/ folder.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception as e:
        print(
            f"Error: Could not read CSV at {DATASET_PATH}.\nDetails: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    if df.empty:
        print(
            "Error: The dataset file is empty. Nothing to process.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Column names:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head(5).to_string())

    col = _find_equation_column(df.columns)
    print(f"\nUsing equation column: {col!r}")

    if col not in df.columns:
        print(
            f"Error: Selected column {col!r} is missing from the dataframe.",
            file=sys.stderr,
        )
        sys.exit(1)

    series = df[col]
    equations = (
        series.astype("string")
        .dropna()
        .map(lambda x: str(x).strip() if x is not None else "")
    )
    equations = equations[equations != ""]

    if equations.empty:
        print(
            "Error: No non-empty equations after cleaning. Check the equation column.",
            file=sys.stderr,
        )
        sys.exit(1)

    out = pd.DataFrame({"equation": equations})
    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(RAW_OUTPUT, index=False)

    print(f"\nTotal number of equations: {len(out)}")
    print("Example equation:", out["equation"].iloc[0])
    print(f"Saved: {RAW_OUTPUT}")
    return out


if __name__ == "__main__":
    load_and_save_raw()

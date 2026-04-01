"""
Smoke tests: required pipeline outputs exist and tokenization is non-empty.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    paths = [
        PROJECT_ROOT / "data" / "raw_equations.csv",
        PROJECT_ROOT / "data" / "processed_equations.csv",
        PROJECT_ROOT / "outputs" / "tokens.json",
        PROJECT_ROOT / "outputs" / "vocab.json",
    ]
    for p in paths:
        if not p.is_file():
            print(f"FAIL: Missing file: {p}", file=sys.stderr)
            sys.exit(1)

    with open(PROJECT_ROOT / "outputs" / "tokens.json", encoding="utf-8") as f:
        data = json.load(f)
    id_seq = data.get("id_sequences") or []
    if not id_seq or not any(id_seq):
        print("FAIL: Tokenization output is empty.", file=sys.stderr)
        sys.exit(1)

    print("All tests passed successfully")


if __name__ == "__main__":
    main()

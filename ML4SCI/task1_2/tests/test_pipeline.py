"""
Validation checks for Task 1.2 pipeline outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> object:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    required = [
        TASK_ROOT / "data" / "processed_dataset.json",
        TASK_ROOT / "outputs" / "train.json",
        TASK_ROOT / "outputs" / "val.json",
        TASK_ROOT / "outputs" / "test.json",
        TASK_ROOT / "outputs" / "vocab.json",
    ]
    for p in required:
        if not p.exists():
            print(f"FAIL: Missing required output: {p}", file=sys.stderr)
            sys.exit(1)

    processed = _load_json(TASK_ROOT / "data" / "processed_dataset.json")
    train = _load_json(TASK_ROOT / "outputs" / "train.json")
    val = _load_json(TASK_ROOT / "outputs" / "val.json")
    test = _load_json(TASK_ROOT / "outputs" / "test.json")
    vocab = _load_json(TASK_ROOT / "outputs" / "vocab.json")

    if not isinstance(processed, list) or not processed:
        print("FAIL: processed_dataset.json is empty.", file=sys.stderr)
        sys.exit(1)

    if not isinstance(vocab, dict) or not vocab:
        print("FAIL: vocab.json is empty.", file=sys.stderr)
        sys.exit(1)

    total_split = len(train) + len(val) + len(test)
    if total_split != len(processed):
        print("FAIL: train/val/test split sizes do not match dataset size.", file=sys.stderr)
        sys.exit(1)

    first = processed[0]
    if not first.get("input_tokens") or not first.get("target_tokens"):
        print("FAIL: Tokenization output is empty.", file=sys.stderr)
        sys.exit(1)

    print("All Task 1.2 tests passed successfully")


if __name__ == "__main__":
    main()

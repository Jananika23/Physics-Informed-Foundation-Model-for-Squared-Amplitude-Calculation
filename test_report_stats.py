"""
Simple test for scripts/report_stats.py without using CLI arguments.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.report_stats import TOKENS_PATH, VOCAB_PATH, compute_stats, load_json  # noqa: E402


def main() -> None:
    if not TOKENS_PATH.is_file():
        print(f"FAIL: Missing {TOKENS_PATH}", file=sys.stderr)
        sys.exit(1)

    if not VOCAB_PATH.is_file():
        print(f"FAIL: Missing {VOCAB_PATH}", file=sys.stderr)
        sys.exit(1)

    tokens_data = load_json(TOKENS_PATH)
    vocab_data = load_json(VOCAB_PATH)
    total_equations, vocab_size, _avg_len = compute_stats(tokens_data, vocab_data)

    if total_equations <= 0:
        print("FAIL: total equations must be greater than 0.", file=sys.stderr)
        sys.exit(1)

    if vocab_size <= 0:
        print("FAIL: vocabulary size must be greater than 0.", file=sys.stderr)
        sys.exit(1)

    print("Stats test passed")


if __name__ == "__main__":
    main()

"""
Compute and optionally publish dataset/tokenization statistics.

Usage:
  python scripts/report_stats.py
  python scripts/report_stats.py --update-readme
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKENS_PATH = PROJECT_ROOT / "outputs" / "tokens.json"
VOCAB_PATH = PROJECT_ROOT / "outputs" / "vocab.json"
README_PATH = PROJECT_ROOT / "README.md"

STATS_START = "<!-- DATASET_STATS_START -->"
STATS_END = "<!-- DATASET_STATS_END -->"


def load_json(path: Path) -> dict:
    if not path.is_file():
        print(f"Error: Missing file: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {path}\nDetails: {e}", file=sys.stderr)
        sys.exit(1)


def compute_stats(tokens_data: dict, vocab_data: dict) -> tuple[int, int, float]:
    records = tokens_data.get("records", [])
    if not isinstance(records, list) or not records:
        print(
            "Error: outputs/tokens.json does not contain a non-empty 'records' list.",
            file=sys.stderr,
        )
        sys.exit(1)

    token_lengths = []
    for item in records:
        tokens = item.get("tokens", [])
        if isinstance(tokens, list):
            token_lengths.append(len(tokens))

    if not token_lengths:
        print("Error: No token lists found in records.", file=sys.stderr)
        sys.exit(1)

    total_equations = len(records)
    vocab_size = len(vocab_data)
    avg_token_len = sum(token_lengths) / total_equations
    return total_equations, vocab_size, avg_token_len


def print_stats(total_equations: int, vocab_size: int, avg_token_len: float) -> None:
    print("Dataset Statistics")
    print("------------------")
    print(f"Total equations: {total_equations}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Average token length per equation: {avg_token_len:.2f}")


def build_stats_block(total_equations: int, vocab_size: int, avg_token_len: float) -> str:
    return (
        "## Dataset Statistics\n\n"
        f"{STATS_START}\n"
        f"- **Total equations:** `{total_equations}`\n"
        f"- **Vocabulary size:** `{vocab_size}`\n"
        f"- **Average tokens per equation:** `{avg_token_len:.2f}`\n"
        f"{STATS_END}\n"
    )


def update_readme(total_equations: int, vocab_size: int, avg_token_len: float) -> None:
    if not README_PATH.is_file():
        print(f"Error: Missing README file at {README_PATH}", file=sys.stderr)
        sys.exit(1)

    content = README_PATH.read_text(encoding="utf-8")
    stats_block = build_stats_block(total_equations, vocab_size, avg_token_len)

    if STATS_START in content and STATS_END in content:
        # Replace only the marker block body to keep surrounding text stable.
        start_idx = content.index(STATS_START)
        end_idx = content.index(STATS_END) + len(STATS_END)
        marker_block = content[start_idx:end_idx]
        replacement_block = (
            f"{STATS_START}\n"
            f"- **Total equations:** `{total_equations}`\n"
            f"- **Vocabulary size:** `{vocab_size}`\n"
            f"- **Average tokens per equation:** `{avg_token_len:.2f}`\n"
            f"{STATS_END}"
        )
        updated = content.replace(marker_block, replacement_block, 1)
    elif "## Dataset Statistics" in content:
        # Replace section content until the next top-level heading.
        lines = content.splitlines()
        start = None
        end = len(lines)
        for i, line in enumerate(lines):
            if line.strip() == "## Dataset Statistics":
                start = i
                continue
            if start is not None and line.startswith("## "):
                end = i
                break
        if start is None:
            print("Error: Could not locate '## Dataset Statistics' in README.", file=sys.stderr)
            sys.exit(1)
        replacement = build_stats_block(total_equations, vocab_size, avg_token_len).strip().splitlines()
        lines = lines[:start] + replacement + lines[end:]
        updated = "\n".join(lines) + ("\n" if content.endswith("\n") else "")
    else:
        # Append section if it does not exist yet.
        updated = content.rstrip() + "\n\n" + stats_block

    README_PATH.write_text(updated, encoding="utf-8")
    print("README.md updated with latest dataset statistics.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report tokenization stats and optionally update README."
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Replace the README 'Dataset Statistics' section with latest values.",
    )
    args = parser.parse_args()

    tokens_data = load_json(TOKENS_PATH)
    vocab_data = load_json(VOCAB_PATH)
    total_equations, vocab_size, avg_token_len = compute_stats(tokens_data, vocab_data)
    print_stats(total_equations, vocab_size, avg_token_len)

    if args.update_readme:
        update_readme(total_equations, vocab_size, avg_token_len)


if __name__ == "__main__":
    main()

"""
Regex-based tokenizer for symbolic equations: build vocabulary and ID sequences.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed_equations.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VOCAB_PATH = OUTPUT_DIR / "vocab.json"
TOKENS_PATH = OUTPUT_DIR / "tokens.json"

# Longer tokens first: ** before *, multi-char functions before substrings
_TOKEN_PATTERN = re.compile(
    r"(?:\*\*|sqrt|arcsin|sin|cos|tan|log|exp)"
    r"|\d+\.\d+"
    r"|\d+"
    r"|[+\-*/^=()]"
    r"|[a-zA-Z_][a-zA-Z0-9_]*"
)


def tokenize_line(text: str) -> list[str]:
    """Split one equation string into tokens in left-to-right order."""
    if not text or not str(text).strip():
        return []
    return _TOKEN_PATTERN.findall(str(text))


def build_vocab(all_tokens: list[str]) -> dict[str, int]:
    """Map each unique token to a stable integer ID (sorted for reproducibility)."""
    unique = sorted(set(all_tokens))
    return {tok: i for i, tok in enumerate(unique)}


def tokenize_and_save() -> tuple[dict[str, int], list[dict]]:
    """Load processed equations, tokenize, save vocab.json and tokens.json."""
    if not PROCESSED_PATH.is_file():
        print(
            f"Error: Missing {PROCESSED_PATH}. Run preprocess first.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(PROCESSED_PATH)
    if df.empty or "equation" not in df.columns:
        print(
            "Error: processed_equations.csv is empty or missing 'equation'.",
            file=sys.stderr,
        )
        sys.exit(1)

    originals: list[str] = []
    token_lists: list[list[str]] = []
    for raw in df["equation"].astype(str):
        originals.append(raw)
        token_lists.append(tokenize_line(raw))

    all_tokens: list[str] = [t for seq in token_lists for t in seq]
    if not all_tokens:
        print(
            "Error: Tokenization produced no tokens. Check equation format.",
            file=sys.stderr,
        )
        sys.exit(1)

    vocab = build_vocab(all_tokens)
    id_sequences: list[list[int]] = [
        [vocab[t] for t in seq] for seq in token_lists
    ]

    records = [
        {"equation": o, "tokens": tl, "ids": ids}
        for o, tl, ids in zip(originals, token_lists, id_sequences)
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    with open(TOKENS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "records": records,
                "id_sequences": id_sequences,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Example print (first non-empty)
    example = next((r for r in records if r["tokens"]), records[0])
    print("\n--- Tokenization example ---")
    print("Original equation:", example["equation"])
    print("Tokens:", example["tokens"])
    print("Token IDs:", example["ids"])
    print(f"Saved: {VOCAB_PATH}")
    print(f"Saved: {TOKENS_PATH}")
    print(f"Vocabulary size: {len(vocab)}")

    return vocab, records


if __name__ == "__main__":
    tokenize_and_save()

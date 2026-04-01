"""
Regex tokenization for normalized amplitudes.
"""

from __future__ import annotations

import re

# Order matters: more specific patterns first.
TOKEN_RE = re.compile(
    r"(?:sin|cos|tan|log|exp|sqrt|arcsin|arccos|arctan)"  # functions
    r"|(?:_\d+)"  # normalized indices
    r"|(?:\d+\.\d+|\d+)"  # numbers
    r"|(?:\*\*|[+\-*/^=])"  # operators
    r"|(?:[A-Za-z%\\][A-Za-z0-9%\\]*)"  # identifiers (supports % and \)
    r"|(?:[(){}\[\],])"  # brackets and separators
)


def tokenize_expression(expr: str) -> list[str]:
    """Tokenize one expression into string tokens."""
    return TOKEN_RE.findall(expr)


def tokenize(samples: list[dict[str, str]]) -> list[dict[str, object]]:
    """Tokenize normalized input/target for each sample."""
    tokenized: list[dict[str, object]] = []
    for row in samples:
        in_tokens = tokenize_expression(row["normalized_input"])
        tgt_tokens = tokenize_expression(row["normalized_target"])
        tokenized.append(
            {
                **row,
                "input_tokens": in_tokens,
                "target_tokens": tgt_tokens,
            }
        )
    return tokenized


def build_vocab(tokenized: list[dict[str, object]]) -> dict[str, int]:
    """Build shared token vocabulary across input and target."""
    all_tokens: list[str] = []
    for row in tokenized:
        all_tokens.extend(row["input_tokens"])  # type: ignore[arg-type]
        all_tokens.extend(row["target_tokens"])  # type: ignore[arg-type]
    unique = sorted(set(all_tokens))
    return {tok: i for i, tok in enumerate(unique)}


def encode(tokenized: list[dict[str, object]], vocab: dict[str, int]) -> list[dict[str, object]]:
    """Convert token lists into integer ID sequences."""
    encoded: list[dict[str, object]] = []
    for row in tokenized:
        in_tokens = row["input_tokens"]  # type: ignore[assignment]
        tgt_tokens = row["target_tokens"]  # type: ignore[assignment]
        input_ids = [vocab[t] for t in in_tokens]  # type: ignore[index]
        target_ids = [vocab[t] for t in tgt_tokens]  # type: ignore[index]
        encoded.append(
            {
                **row,
                "input_ids": input_ids,
                "target_ids": target_ids,
            }
        )
    return encoded

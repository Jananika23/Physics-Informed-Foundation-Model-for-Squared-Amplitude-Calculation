"""
Small helper utilities for padding, decoding, and accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def decode_ids(ids: list[int], id_to_token: dict[int, str], pad_id: int) -> list[str]:
    """Convert integer token IDs back to tokens, dropping padding."""
    out: list[str] = []
    for i in ids:
        if i == pad_id:
            break
        out.append(id_to_token.get(i, "<unk>"))
    return out


def token_accuracy(pred_ids: torch.Tensor, true_ids: torch.Tensor, pad_id: int) -> float:
    """
    Compute token-level accuracy ignoring padding positions.

    pred_ids/true_ids: (batch, seq_len)
    """
    if pred_ids.shape != true_ids.shape:
        raise ValueError("pred_ids and true_ids must have the same shape")
    mask = true_ids != pad_id
    if mask.sum().item() == 0:
        return 0.0
    correct = (pred_ids == true_ids) & mask
    return correct.sum().item() / mask.sum().item()


def load_vocab(vocab_path: Path) -> dict[str, int]:
    with open(vocab_path, encoding="utf-8") as f:
        return json.load(f)


def invert_vocab(vocab: dict[str, int]) -> dict[int, str]:
    return {v: k for k, v in vocab.items()}


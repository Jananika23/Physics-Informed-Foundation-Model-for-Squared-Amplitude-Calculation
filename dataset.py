"""
Dataset utilities for Task 2.1–2.3 (squared amplitude prediction).

This module reads the Task 1.2 preprocessed splits:
  - ../outputs/train.json
  - ../outputs/val.json
  - ../outputs/test.json

Each JSON item is expected to contain:
  - input_ids: list[int]
  - target_ids: list[int]

We pad sequences in a batch to the same length so the Transformer can run.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Dataset


TASK2_ROOT = Path(__file__).resolve().parent.parent
TASK1_2_ROOT = TASK2_ROOT.parent

TRAIN_JSON = TASK1_2_ROOT / "outputs" / "train.json"
VAL_JSON = TASK1_2_ROOT / "outputs" / "val.json"
TEST_JSON = TASK1_2_ROOT / "outputs" / "test.json"


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return data


class Seq2SeqTokenDataset(Dataset):
    """Simple Dataset for (input_ids, target_ids) seq2seq pairs."""

    def __init__(self, items: list[dict[str, Any]]):
        self.items = items
        for i, it in enumerate(items[:3]):
            # Validate the shape early and with helpful error messages.
            if "input_ids" not in it or "target_ids" not in it:
                raise KeyError(f"Item {i} missing 'input_ids' or 'target_ids'.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


@dataclass(frozen=True)
class Batch:
    input_ids: torch.Tensor  # (batch, src_len)
    target_ids: torch.Tensor  # (batch, tgt_len)


def pad_1d(seqs: list[list[int]], pad_id: int, max_len: int) -> torch.Tensor:
    """Pad/truncate 1D integer sequences into a (batch, max_len) tensor."""
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        seq = seq[:max_len]
        if not seq:
            continue
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def make_collate_fn(pad_id: int, max_src_len: int, max_tgt_len: int):
    """Create a DataLoader collate_fn that pads to batch max (bounded)."""

    def collate(batch_items: Iterable[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_items = list(batch_items)
        src_seqs = [it["input_ids"] for it in batch_items]
        tgt_seqs = [it["target_ids"] for it in batch_items]

        src_max = min(max((len(s) for s in src_seqs), default=0), max_src_len)
        tgt_max = min(max((len(s) for s in tgt_seqs), default=0), max_tgt_len)
        if src_max <= 0 or tgt_max <= 0:
            raise ValueError("Found empty sequences; cannot build batches.")

        input_ids = pad_1d(src_seqs, pad_id=pad_id, max_len=src_max)
        target_ids = pad_1d(tgt_seqs, pad_id=pad_id, max_len=tgt_max)
        return {"input_ids": input_ids, "target_ids": target_ids}

    return collate


def build_datasets():
    train_items = _read_json_list(TRAIN_JSON)
    val_items = _read_json_list(VAL_JSON)
    test_items = _read_json_list(TEST_JSON)
    return (
        Seq2SeqTokenDataset(train_items),
        Seq2SeqTokenDataset(val_items),
        Seq2SeqTokenDataset(test_items),
    )


def build_dataloaders(
    train_ds: Seq2SeqTokenDataset,
    val_ds: Seq2SeqTokenDataset,
    test_ds: Seq2SeqTokenDataset,
    pad_id: int,
    batch_size: int = 8,
    max_src_len: int = 256,
    max_tgt_len: int = 256,
):
    collate_fn = make_collate_fn(pad_id=pad_id, max_src_len=max_src_len, max_tgt_len=max_tgt_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


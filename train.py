"""
Train a Transformer model for squared amplitude prediction (Task 2.1–2.3).

We use teacher forcing:
  decoder input = target_ids shifted right by 1 token
  labels         = target_ids shifted left by 1 token

Loss:
  CrossEntropyLoss over vocabulary tokens, ignoring padding positions.

This script saves:
  outputs/model.pt
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn

from dataset import build_dataloaders, build_datasets
from model import PhysicsSeq2SeqTransformer
from plot import generate_loss_plot
from utils import load_vocab


TASK2_ROOT = Path(__file__).resolve().parent.parent
TASK1_2_ROOT = TASK2_ROOT.parent

VOCAB_PATH = TASK1_2_ROOT / "outputs" / "vocab.json"
OUTPUT_MODEL_PATH = TASK2_ROOT / "outputs" / "model.pt"
LOSS_HISTORY_PATH = TASK2_ROOT / "outputs" / "loss_history.json"


def evaluate_loss(model: nn.Module, loader: torch.utils.data.DataLoader, pad_id: int, device: torch.device):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            src_ids = batch["input_ids"].to(device)
            tgt_ids = batch["target_ids"].to(device)
            decoder_in = tgt_ids[:, :-1]
            labels = tgt_ids[:, 1:]

            logits = model(src_ids, decoder_in)  # (batch, tgt_len, vocab)
            vocab_size = logits.size(-1)
            loss = criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))

            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer for Task 2.1–2.3.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-src-len", type=int, default=256)
    parser.add_argument("--max-tgt-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--enc-layers", type=int, default=2)
    parser.add_argument("--dec-layers", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=512)
    args = parser.parse_args()

    # Load vocabulary to size the model and to map IDs later.
    vocab = load_vocab(VOCAB_PATH)
    pad_id = len(vocab)  # we add a dedicated PAD token at the end
    vocab_size = pad_id + 1

    # Build datasets/dataloaders.
    train_ds, val_ds, test_ds = build_datasets()

    # Print one example (input/target are already normalized in Task 1.2 outputs).
    sample = train_ds.items[0]
    print("\n--- Training example (from Task 1.2 processed_dataset.json) ---")
    print("Original input:", sample.get("input"))
    print("Normalized input:", sample.get("input"))
    print("Input tokens:", sample.get("input_tokens"))
    print("Input token IDs:", sample.get("input_ids"))

    train_loader, val_loader, _test_loader = build_dataloaders(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        pad_id=pad_id,
        batch_size=args.batch_size,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicsSeq2SeqTransformer(
        vocab_path=VOCAB_PATH,
        pad_id=pad_id,
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.ffn_dim,
        dropout=0.1,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_losses: list[float] = []
    val_losses: list[float] = []
    print(f"Training on device: {device}")
    print(f"Vocab size (with PAD): {vocab_size}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            src_ids = batch["input_ids"].to(device)
            tgt_ids = batch["target_ids"].to(device)

            decoder_in = tgt_ids[:, :-1]
            labels = tgt_ids[:, 1:]

            optimizer.zero_grad(set_to_none=True)
            logits = model(src_ids, decoder_in)  # (batch, tgt_len, vocab)
            vocab_logits = logits.size(-1)

            loss = criterion(logits.reshape(-1, vocab_logits), labels.reshape(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train = total_loss / max(1, n_batches)
        avg_val = evaluate_loss(model, val_loader, pad_id=pad_id, device=device)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        print(f"Epoch {epoch:02d}/{args.epochs} | train loss: {avg_train:.4f} | val loss: {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "pad_id": pad_id,
                    "vocab_size": vocab_size,
                    "d_model": args.d_model,
                    "nhead": args.nhead,
                    "enc_layers": args.enc_layers,
                    "dec_layers": args.dec_layers,
                    "ffn_dim": args.ffn_dim,
                    "vocab": vocab,
                },
                OUTPUT_MODEL_PATH,
            )

    # Basic sanity check that training made progress.
    if len(val_losses) >= 2 and val_losses[-1] > val_losses[0]:
        # Not fatal, but we print a helpful message.
        print(
            "Note: validation loss did not strictly decrease across all epochs. "
            "This can happen with small datasets / noisy targets, but the best checkpoint was saved."
        )

    with open(LOSS_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump({"train_loss": train_losses, "val_loss": val_losses}, f, indent=2)
    print(f"Saved loss history to: {LOSS_HISTORY_PATH}")

    # Auto-generate train/val loss visualization after training.
    plot_path = TASK2_ROOT / "outputs" / "loss_plot.png"
    generate_loss_plot(LOSS_HISTORY_PATH, plot_path)

    print(f"Saved model to: {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()


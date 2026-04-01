"""
Plot training/validation loss curves from outputs/loss_history.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def generate_loss_plot(loss_history_path: Path, output_plot_path: Path) -> None:
    if not loss_history_path.is_file():
        raise FileNotFoundError(f"Missing loss history: {loss_history_path}")

    with open(loss_history_path, encoding="utf-8") as f:
        data = json.load(f)

    train_loss = data.get("train_loss", [])
    val_loss = data.get("val_loss", [])
    if not train_loss or not val_loss:
        raise ValueError("loss_history.json must contain non-empty train_loss and val_loss.")

    epochs = list(range(1, min(len(train_loss), len(val_loss)) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss[: len(epochs)], label="Train Loss", marker="o")
    plt.plot(epochs, val_loss[: len(epochs)], label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=150)
    plt.close()
    print(f"Saved loss plot to: {output_plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss history for Task 2 model.")
    parser.add_argument(
        "--loss-history",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" / "loss_history.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" / "loss_plot.png",
    )
    args = parser.parse_args()
    generate_loss_plot(args.loss_history, args.output)


if __name__ == "__main__":
    main()

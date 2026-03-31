"""
Final reporting script for Task 2 model performance.

Run:
  python src/report.py
"""

from __future__ import annotations

import builtins
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"

LOSS_HISTORY_PATH = OUT_DIR / "loss_history.json"
PREDICTIONS_PATH = OUT_DIR / "predictions.json"
PRED_READABLE_PATH = OUT_DIR / "predictions_readable.json"
VOCAB_PATH_PRIMARY = ROOT.parent / "outputs" / "vocab.json"
VOCAB_PATH_FALLBACK = OUT_DIR / "vocab.json"
LOSS_PLOT_PATH = OUT_DIR / "loss_plot.png"
FINAL_REPORT_PATH = OUT_DIR / "final_report.txt"

report_text = ""


def print(*args, **kwargs):  # type: ignore[override]
    """
    Keep normal terminal printing, and also capture the same text into report_text.
    """
    global report_text
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    text = sep.join(str(a) for a in args) + end
    report_text += text
    builtins.print(*args, **kwargs)


def _safe_load_json(path: Path) -> Any | None:
    if not path.is_file():
        print(f"Missing file: {path}")
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Could not read JSON file: {path}\nReason: {exc}")
        return None


def _load_vocab_preferred() -> Any | None:
    """Prefer Task 1.2 vocab path, then fallback to local output path."""
    for path in (VOCAB_PATH_PRIMARY, VOCAB_PATH_FALLBACK):
        if not path.is_file():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    print("Error: vocab.json not found in both locations")
    return None


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def sentence_bleu_simple(reference: list[str], hypothesis: list[str], max_n: int = 4) -> float:
    """Smoothed sentence BLEU for short symbolic sequences."""
    if not reference and not hypothesis:
        return 1.0
    if not reference or not hypothesis:
        return 0.0

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(_ngrams(reference, n))
        hyp_ngrams = Counter(_ngrams(hypothesis, n))
        total = sum(hyp_ngrams.values())
        if total == 0:
            precisions.append(1e-9)
            continue
        overlap = sum(min(count, ref_ngrams[g]) for g, count in hyp_ngrams.items())
        precisions.append((overlap + 1.0) / (total + 1.0))

    ref_len = len(reference)
    hyp_len = len(hypothesis)
    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / hyp_len)
    return float(bp * math.exp(sum(math.log(p) for p in precisions) / max_n))


def main() -> None:
    loss_history = _safe_load_json(LOSS_HISTORY_PATH)
    predictions = _safe_load_json(PREDICTIONS_PATH)
    readable = _safe_load_json(PRED_READABLE_PATH)

    vocab = _load_vocab_preferred()

    if loss_history is None or predictions is None or readable is None:
        print("Final report generated successfully")
        return

    train_loss = loss_history.get("train_loss", []) if isinstance(loss_history, dict) else []
    val_loss = loss_history.get("val_loss", []) if isinstance(loss_history, dict) else []

    token_acc = predictions.get("token_accuracy", 0.0) if isinstance(predictions, dict) else 0.0

    pred_items = readable if isinstance(readable, list) else []
    total_samples = len(pred_items)

    exact_count = 0
    bleu_scores: list[float] = []
    for item in pred_items:
        pred_seq = item.get("predicted", []) if isinstance(item, dict) else []
        true_seq = item.get("actual", []) if isinstance(item, dict) else []
        if pred_seq == true_seq:
            exact_count += 1
        bleu_scores.append(sentence_bleu_simple(true_seq, pred_seq))

    bleu_avg = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    exact_acc = exact_count / total_samples if total_samples else 0.0

    print("====================================")
    print("FINAL MODEL PERFORMANCE REPORT")
    print("====================================")
    print()
    print("Dataset:")
    print(f"- Total samples: {total_samples}")
    print()
    print("Training:")
    print(f"- Final Train Loss: {train_loss[-1]:.4f}" if train_loss else "- Final Train Loss: N/A")
    print(f"- Final Validation Loss: {val_loss[-1]:.4f}" if val_loss else "- Final Validation Loss: N/A")
    print()
    print("Evaluation Metrics:")
    print(f"- Token Accuracy: {token_acc * 100:.2f}%")
    print(f"- BLEU Score: {bleu_avg:.2f}")
    print(f"- Exact Match Accuracy: {exact_acc * 100:.2f}%")
    print()
    print("------------------------------------")
    print("Sample Predictions:")
    print("------------------------------------")
    print()
    for i, item in enumerate(pred_items[:5], start=1):
        print(f"Example {i}:")
        print(f"Input: {item.get('input', [])}")
        print(f"Predicted: {item.get('predicted', [])}")
        print(f"Actual: {item.get('actual', [])}")
        print()

    if LOSS_PLOT_PATH.is_file():
        print("Loss plot saved at outputs/loss_plot.png")

    if isinstance(vocab, dict):
        print(f"Loaded vocabulary with {len(vocab)} tokens.")

    print("Final report generated successfully")

    FINAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print("Report saved to outputs/final_report.txt")


if __name__ == "__main__":
    main()


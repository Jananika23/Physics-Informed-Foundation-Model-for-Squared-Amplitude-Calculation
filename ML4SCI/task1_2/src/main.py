"""
Task 1.2 pipeline: load -> preprocess -> normalize -> tokenize -> split.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = TASK_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from load_data import load_data  # noqa: E402
from normalize import normalize  # noqa: E402
from preprocess import preprocess  # noqa: E402
from tokenizer import build_vocab, encode, tokenize  # noqa: E402

PROCESSED_PATH = TASK_ROOT / "data" / "processed_dataset.json"
OUTPUT_DIR = TASK_ROOT / "outputs"
TRAIN_PATH = OUTPUT_DIR / "train.json"
VAL_PATH = OUTPUT_DIR / "val.json"
TEST_PATH = OUTPUT_DIR / "test.json"
VOCAB_PATH = OUTPUT_DIR / "vocab.json"


def split_dataset(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Shuffle and split rows into 80/10/10."""
    shuffled = rows[:]
    random.seed(42)
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def save_json(path: Path, data: object) -> None:
    """Write JSON with readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    loaded = load_data()
    cleaned = preprocess(loaded)
    if not cleaned:
        print("No valid samples after preprocessing.", file=sys.stderr)
        raise SystemExit(1)

    normalized = normalize(cleaned)
    tokenized = tokenize(normalized)
    vocab = build_vocab(tokenized)
    encoded = encode(tokenized, vocab)

    processed_dataset = [
        {
            "input": row["normalized_input"],
            "target": row["normalized_target"],
            "input_tokens": row["input_tokens"],
            "target_tokens": row["target_tokens"],
            "input_ids": row["input_ids"],
            "target_ids": row["target_ids"],
        }
        for row in encoded
    ]

    save_json(PROCESSED_PATH, processed_dataset)
    train, val, test = split_dataset(processed_dataset)
    save_json(TRAIN_PATH, train)
    save_json(VAL_PATH, val)
    save_json(TEST_PATH, test)
    save_json(VOCAB_PATH, vocab)

    sample = encoded[0]
    print("\n--- Example sample ---")
    print("Original input:", sample["input"])
    print("Normalized input:", sample["normalized_input"])
    print("Tokens:", sample["input_tokens"])
    print("Token IDs:", sample["input_ids"])
    print(f"\nTotal samples: {len(processed_dataset)}")
    print(f"Train/Val/Test: {len(train)}/{len(val)}/{len(test)}")
    print(f"Vocabulary size: {len(vocab)}")
    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()

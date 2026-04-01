"""
Load all Task 1.2 data files and extract amplitude -> squared amplitude pairs.
"""

from __future__ import annotations

import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = TASK_ROOT / "data"


def list_dataset_files() -> list[Path]:
    """Recursively list data files while ignoring hidden/system placeholders."""
    if not DATA_DIR.is_dir():
        return []
    files = []
    for p in DATA_DIR.rglob("*"):
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in {".txt", ".csv", ".dat"}:
            files.append(p)
    return sorted(files)


def parse_line(raw_line: str) -> dict[str, str] | None:
    """
    Parse one dataset row.

    Expected shape (with internal colons in early fields):
      interaction : diagram : amplitude : squared amplitude

    We safely split from the right because the first fields may contain ':'.
    """
    line = raw_line.strip()
    if not line:
        return None

    # Last two segments are always amplitude and squared amplitude.
    parts = line.rsplit(" : ", 2)
    if len(parts) != 3:
        return None

    _, amplitude, squared_amplitude = parts
    amplitude = amplitude.strip()
    squared_amplitude = squared_amplitude.strip()

    if not amplitude or not squared_amplitude:
        return None

    return {"input": amplitude, "target": squared_amplitude}


def load_data() -> list[dict[str, str]]:
    """Load all valid samples from files under task1_2/data/."""
    files = list_dataset_files()
    if not files:
        print("Please place all Task 1.2 dataset files inside task1_2/data/")
        sys.exit(1)

    samples: list[dict[str, str]] = []
    malformed = 0

    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parsed = parse_line(line)
                    if parsed is None:
                        malformed += 1
                        continue
                    samples.append(parsed)
        except UnicodeDecodeError:
            # Fallback for uncommon encodings.
            with open(path, encoding="latin-1") as f:
                for line in f:
                    parsed = parse_line(line)
                    if parsed is None:
                        malformed += 1
                        continue
                    samples.append(parsed)

    if not samples:
        print(
            "No valid samples were loaded. Please verify files in task1_2/data/ "
            "use 'interaction : diagram : amplitude : squared amplitude' rows.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loaded files: {len(files)}")
    print(f"Loaded valid samples: {len(samples)}")
    print(f"Skipped malformed rows: {malformed}")
    return samples

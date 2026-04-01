"""
End-to-end pipeline: load_data -> preprocess -> tokenizer.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when running as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pandas  # noqa: F401
except ImportError as e:
    print(
        "Could not import pandas (and usually numpy). Install dependencies:\n"
        "  pip install -r requirements.txt\n"
        "On Windows, if you use a preview Python (for example 3.14) and see "
        "NumPy DLL errors, use a stable release such as 3.12:\n"
        "  py -3.12 -m pip install -r requirements.txt\n"
        "  py -3.12 src/main.py",
        file=sys.stderr,
    )
    raise SystemExit(1) from e

from src.load_data import load_and_save_raw  # noqa: E402
from src.preprocess import preprocess_and_save  # noqa: E402
from src.tokenizer import tokenize_and_save  # noqa: E402


def main() -> None:
    load_and_save_raw()
    preprocess_and_save()
    tokenize_and_save()
    print("\nPipeline completed successfully")


if __name__ == "__main__":
    main()

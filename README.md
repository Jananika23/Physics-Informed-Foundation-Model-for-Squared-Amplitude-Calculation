# ML4SCI - AI-Feynman Preprocessing and Tokenization

This repository implements a clean, reproducible preprocessing pipeline for **Common Task 1.1**: preparing AI-Feynman equations for symbolic regression models.

## What is symbolic regression?

Symbolic regression searches for an explicit mathematical expression that explains a relationship in data.  
Unlike standard regression, which fits parameters inside a fixed formula shape, symbolic regression learns the **equation structure itself** (operators, functions, variable interactions, and constants).

## Project objective

Given `data/FeynmanEquations.csv`, this project:

1. Loads and validates equation data.
2. Cleans and normalizes equation strings.
3. Tokenizes equations into model-ready symbolic units.
4. Builds a vocabulary and integer sequences for downstream ML models.

## Why tokenization is essential for symbolic regression

Neural and sequence models cannot learn directly from raw equation strings. They require discrete, semantically meaningful symbols.

Tokenization is critical because it:

- Separates **operators** (`+`, `-`, `*`, `/`, `**`, `=`) from identifiers and numbers.
- Preserves **function semantics** (`sin`, `cos`, `log`, `exp`, `sqrt`) as atomic units.
- Encodes symbolic expressions as ordered token-ID sequences suitable for language-style modeling.
- Retains equation syntax needed to learn valid mathematical composition.

If tokenization is poor, the model receives ambiguous input and fails to learn stable symbolic patterns (for example, splitting `sin` into character-level noise or merging operators with variables).

## Pipeline overview

- `src/load_data.py`: loads CSV, auto-detects equation column, cleans raw strings, writes `data/raw_equations.csv`
- `src/preprocess.py`: normalizes whitespace/formatting, removes invalid rows, writes `data/processed_equations.csv`
- `src/tokenizer.py`: regex tokenization, vocabulary construction, token-ID mapping, writes:
  - `outputs/vocab.json`
  - `outputs/tokens.json`
- `src/main.py`: executes the full pipeline end-to-end
- `tests/test_pipeline.py`: validates artifacts and non-empty tokenization output

## Dataset Statistics

Based on the generated outputs in this repository:

<!-- DATASET_STATS_START -->
- **Total equations:** `100`
- **Vocabulary size:** `115`
- **Average tokens per equation:** `15.96`
<!-- DATASET_STATS_END -->

## Tokenization examples

### Example 1

- Equation: `exp(-theta**2/2)/sqrt(2*pi)`
- Tokens: `['exp', '(', '-', 'theta', '**', '2', '/', '2', ')', '/', 'sqrt', '(', '2', '*', 'pi', ')']`
- Token IDs: `[56, 0, 5, 98, 3, 8, 6, 8, 1, 6, 95, 0, 8, 2, 82, 1]`

### Example 2

- Equation: `exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)`
- Tokens: `['exp', '(', '-', '(', 'theta', '/', 'sigma', ')', '**', '2', '/', '2', ')', '/', '(', 'sqrt', '(', '2', '*', 'pi', ')', '*', 'sigma', ')']`
- Token IDs: `[56, 0, 5, 0, 98, 6, 92, 1, 3, 8, 6, 8, 1, 6, 0, 95, 0, 8, 2, 82, 1, 2, 92, 1]`

### Example 3

- Equation: `exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma)`
- Tokens: `['exp', '(', '-', '(', '(', 'theta', '-', 'theta1', ')', '/', 'sigma', ')', '**', '2', '/', '2', ')', '/', '(', 'sqrt', '(', '2', '*', 'pi', ')', '*', 'sigma', ')']`
- Token IDs: `[56, 0, 5, 0, 0, 98, 5, 99, 1, 6, 92, 1, 3, 8, 6, 8, 1, 6, 0, 95, 0, 8, 2, 82, 1, 2, 92, 1]`

## Limitations of the current approach

This baseline is intentionally simple and robust, but there is room for improvement:

- Uses regex tokenization only; no formal grammar parsing (AST-level validation is not included).
- Does not canonicalize mathematically equivalent forms (for example, commutative reordering).
- Treats all numeric constants literally; no constant abstraction strategy yet.
- Limited function/operator coverage beyond the configured token set.
- No train/validation/test split or sequence padding utilities in this task scope.

## How to run

From the project root (`ML4SCI`):

```bash
pip install -r requirements.txt
python src/main.py
python tests/test_pipeline.py
python scripts/report_stats.py
```

On Windows, if `python` points to a preview interpreter and NumPy import fails, use a stable version explicitly:

```bash
py -3.12 -m pip install -r requirements.txt
py -3.12 src/main.py
py -3.12 tests/test_pipeline.py
```

## Repository structure

| Path | Description |
|------|-------------|
| `data/FeynmanEquations.csv` | Source dataset |
| `data/raw_equations.csv` | Extracted raw equation column |
| `data/processed_equations.csv` | Cleaned equation strings |
| `outputs/vocab.json` | Token-to-ID vocabulary |
| `outputs/tokens.json` | Token lists and ID sequences |
| `src/` | Pipeline implementation |
| `tests/` | Validation checks |

## Requirements

Dependencies are listed in `requirements.txt`:

- `pandas`
- `numpy`

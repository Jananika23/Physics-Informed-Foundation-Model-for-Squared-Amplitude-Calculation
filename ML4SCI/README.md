# ML4SCI

This repository contains my work on the ML4SCI symbolic physics preprocessing pipeline.

The project is split into two preprocessing tasks and one modeling stage:

- `task1_1`: preprocess AI-Feynman equations and convert them into token sequences
- `task1_2`: preprocess scattering amplitude datasets into input-target pairs for learning
- `task1_2/task2_model`: train and evaluate a Transformer model on Task 1.2 outputs

---

## Project layout

| Path | What it contains |
|------|------------------|
| `task1_1/` | Task 1.1 data pipeline (load, clean, tokenize, vocab) |
| `task1_2/` | Task 1.2 data pipeline (parse, normalize, tokenize, split) |
| `task1_2/task2_model/` | Seq2seq Transformer training, evaluation, reporting |

---

## What each part does

### Task 1.1 (`task1_1`)

Goal: prepare symbolic equations from `FeynmanEquations.csv` so they can be used by ML models.

Pipeline:

1. Load and validate equation data
2. Clean equation strings
3. Tokenize expressions into symbolic units
4. Build vocabulary and token IDs

Main files:

- `task1_1/src/main.py` - runs the end-to-end pipeline
- `task1_1/src/load_data.py`
- `task1_1/src/preprocess.py`
- `task1_1/src/tokenizer.py`
- `task1_1/scripts/report_stats.py`
- `task1_1/tests/test_pipeline.py`

Outputs:

- `task1_1/data/raw_equations.csv`
- `task1_1/data/processed_equations.csv`
- `task1_1/outputs/vocab.json`
- `task1_1/outputs/tokens.json`

### Task 1.2 (`task1_2`)

Goal: prepare amplitude -> squared-amplitude examples from raw text files for supervised learning.

Expected raw row format:

`interaction : Feynman diagram : amplitude : squared amplitude`

Pipeline:

1. Read raw files from `task1_2/data/`
2. Keep valid `input` (amplitude) and `target` (squared amplitude) pairs
3. Normalize index suffixes (for example `_345` -> `_1` style remapping per sample)
4. Tokenize expressions
5. Build a shared vocabulary
6. Create train/val/test split (80/10/10)

Main files:

- `task1_2/src/main.py` - runs full preprocessing
- `task1_2/src/load_data.py`
- `task1_2/src/preprocess.py`
- `task1_2/src/normalize.py`
- `task1_2/src/tokenizer.py`
- `task1_2/tests/test_pipeline.py`

Outputs:

- `task1_2/data/processed_dataset.json`
- `task1_2/outputs/train.json`
- `task1_2/outputs/val.json`
- `task1_2/outputs/test.json`
- `task1_2/outputs/vocab.json`

### Task 2 Model (`task1_2/task2_model`)

Goal: train a Transformer sequence-to-sequence model to predict squared amplitudes from amplitudes.

Main files:

- `task1_2/task2_model/src/train.py`
- `task1_2/task2_model/src/evaluate.py`
- `task1_2/task2_model/src/report.py`
- `task1_2/task2_model/tests/test_model.py`

Typical outputs:

- `task1_2/task2_model/outputs/model.pt`
- `task1_2/task2_model/outputs/loss_history.json`
- `task1_2/task2_model/outputs/predictions.json`
- `task1_2/task2_model/outputs/final_report.txt`

---

## How to run

Each part has its own dependencies, so install requirements inside each folder.

### 1) Run Task 1.1

```bash
cd task1_1
pip install -r requirements.txt
python src/main.py
python tests/test_pipeline.py
python scripts/report_stats.py
```

### 2) Run Task 1.2 preprocessing

```bash
cd task1_2
pip install -r requirements.txt
python src/main.py
python tests/test_pipeline.py
```

### 3) Run Task 2 model

```bash
cd task1_2/task2_model
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
python src/report.py
python tests/test_model.py
```

---

## Notes

- `task2_model` expects Task 1.2 output files to exist first.
- Generated files under `data/` and `outputs/` are artifacts from the pipelines.
- If Task 1.2 data is missing, place the dataset files under `task1_2/data/`.

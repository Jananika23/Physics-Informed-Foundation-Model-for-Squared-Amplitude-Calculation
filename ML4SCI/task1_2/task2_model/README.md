# Physics-Informed Squared Amplitude Prediction

## 1) Project Overview

This project predicts squared amplitudes from symbolic amplitude expressions in QED/QCD scattering data.  
The main challenge is that these expressions are long, structured, and full of physics-specific tokens.

Why this matters: if we can model these expressions well, we can speed up parts of physics workflows where symbolic calculations are expensive.

## 2) What This Project Does

The full workflow is split into two parts:

- **Preprocessing (Task 1.2):** load raw files, clean rows, normalize indices, tokenize expressions, and create train/val/test splits.
- **Modeling (Task 2):** train a Transformer on token IDs to map amplitude -> squared amplitude.
- **Evaluation:** report token accuracy, BLEU, exact match, and save readable predictions.

## 3) Dataset

The data comes from QED/QCD interaction files.  
Each row has information like:

`interaction : diagram : amplitude : squared amplitude`

For modeling, we use the processed train/validation/test JSON files generated in Task 1.2.

## 4) Approach

The model is an encoder-decoder Transformer. In simple terms, it reads an input token sequence and predicts the target token sequence.

There is one physics-informed addition:

- index tokens (`_1`, `_2`, etc.) get special handling through token-type/index-aware embeddings

This helps because index consistency is important in these expressions, and operators/functions should not be treated the same way as variable-like tokens.

## 5) Results

Current evaluation includes:

- Token Accuracy
- BLEU Score
- Exact Match Accuracy

Recent runs show solid token-level performance, but exact match is still difficult (which is expected for long symbolic sequences).

## 6) How to Run

From `task1_2/task2_model/`:

```bash
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
python src/report.py
python tests/test_model.py
```

This will generate:

- model checkpoint
- prediction files
- loss history + loss plot
- final text report

## 7) Project Structure

- `src/` - training, evaluation, reporting, plotting, model and data modules
- `outputs/` - model checkpoint, metrics JSONs, plots, and final report
- `tests/` - smoke tests for dataset/model/artifact checks
- `requirements.txt` - Python dependencies

## 8) Notes

- This is a practical baseline, not a finished research system.
- Long sequence generation is still hard, so exact-match can be low even when token accuracy looks good.
- More improvements are possible (better decoding, larger models, more physics constraints, and broader data).

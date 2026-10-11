# Team Zero – Clinical Specialty Router

MA6700 Assignment 3. Routes a transcribed clinical note to the right specialist department,
and sends it to a human when the model is not confident.

## Setup

1. `pip install -r requirements.txt`
2. Download the Medical Transcriptions dataset from Kaggle
   (https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions, CC0) and put
   `mtsamples.csv` in `data/`.
3. Fine-tuning needs a GPU. On Google Colab, upload this folder and run the commands there.

## Run

| Command | What it does |
| --- | --- |
| `python main.py` | The full pipeline: data, baseline, BERT, evaluation |
| `python truncation_analysis.py` | Note lengths and what truncation cuts (report Sections 3–4) |
| `python rule_example.py` | Keyword-rule failure examples (report Section 1) |
| `python demographics.py` | Sex and age coverage for subgroup analysis (report Section 6) |

Every figure and table goes to `outputs/`. Every number used in the report must come from these scripts.

## Files and owners

| File | Owner | Purpose |
| --- | --- | --- |
| `config.py` | Immanuel | Seed, paths, split sizes, model name, truncation settings |
| `data.py` | Mevin | Load, dedup, label rule, one-time split |
| `baseline.py` | Mevin | TF-IDF + Logistic Regression |
| `model.py` | Jeffrey | Fine-tuned biomedical BERT (head+tail truncation helper included) |
| `evaluate.py` | Deevesh | Metrics, threshold, error analysis, subgroup and pronoun-swap checks |
| `truncation_analysis.py`, `rule_example.py`, `demographics.py` | Immanuel | Analyses for Sections 1, 3–4 and 6 |
| `main.py` | Immanuel | Single entry point |

## How to add your code

1. Fill in the functions in **your own file only**. Keep their names and inputs exactly as they are,
   because `main.py` calls them. Add any helper functions you need inside your file.
2. Take every setting from `config.py` (seed, paths, model name, truncation lengths). Never hard-code them.
3. `data.make_split()` returns `train, val, test`, each with a `text` column and a `label` column.
4. `predict_baseline()` and `predict_model()` return one row per note and one column per class,
   **in `config.LABELS` order**. `main.py` stops with a clear error if the shape is wrong.
5. Send your finished file to Immanuel (or push it to the team repo), then run `python main.py`.
   It stops at the first step that is not written yet, so you can see how far the pipeline gets.

## Team rules (from the brief)

- Seed comes from `config.SEED`; the split is made once and saved to `outputs/split.csv`.
- Every module and function has a docstring. Comments explain **why** a decision was made, not what the line does.
- No API keys, tokens or passwords anywhere in the code.
- Cite any adapted code in a comment and say what you changed.
- No notebooks in the submission.

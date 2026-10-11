"""Team Zero - Clinical Specialty Router. Single entry point for the whole pipeline.

Run:  python main.py
Reads data/mtsamples.csv, writes every figure and table to outputs/, prints every metric.
Steps run in order and stop with a clear message at the first one not written yet.
"""

import os

import baseline
import config
import data
import evaluate
import model


def check_probs(name, probs, n_notes):
    """Stop early if a model's output does not match the shared format.

    Each model must return one row per note and one column per class in
    config.LABELS order. Catching a mismatch here is cheaper than finding
    wrong numbers in the report.
    """
    rows, cols = len(probs), len(probs[0])
    if rows != n_notes or cols != len(config.LABELS):
        raise ValueError(f"{name}: expected {n_notes} rows x {len(config.LABELS)} columns "
                         f"(config.LABELS), got {rows} x {cols}")
    return probs


def main():
    config.set_seed()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    try:
        notes = data.load_and_clean()
        notes = data.apply_label_rule(notes)
        train, val, test = data.make_split(notes)

        base = baseline.train_baseline(train, val)
        evaluate.evaluate("Baseline (TF-IDF + LR)", test,
                          check_probs("baseline", baseline.predict_baseline(base, test["text"]), len(test)),
                          val, check_probs("baseline", baseline.predict_baseline(base, val["text"]), len(val)))

        bert = model.train_model(train, val)
        evaluate.evaluate("Fine-tuned BERT", test,
                          check_probs("BERT", model.predict_model(bert, test["text"]), len(test)),
                          val, check_probs("BERT", model.predict_model(bert, val["text"]), len(val)))
    except NotImplementedError as todo:
        print(f"Pipeline stopped: not written yet -> {todo}")


if __name__ == "__main__":
    main()

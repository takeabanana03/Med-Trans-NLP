"""Traditional baseline: TF-IDF + Logistic Regression. Owner: Mevin.

This is the number the deep learning model has to beat (report Section 4).
"""


def train_baseline(train, val):
    """Fit TF-IDF + Logistic Regression on train, tune on val, return the fitted model."""
    raise NotImplementedError("Mevin: train_baseline")


def predict_baseline(model, texts):
    """Return class probabilities for a list of notes."""
    raise NotImplementedError("Mevin: predict_baseline")

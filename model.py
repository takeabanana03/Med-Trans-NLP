"""Deep learning model: fine-tuned biomedical BERT. Owner: Jeffrey.

Report Section 4 explains the architecture and hyperparameters chosen here.
"""

import config


def truncate_head_tail(token_ids, head=config.HEAD_TOKENS, tail=config.TAIL_TOKENS):
    """Keep the first `head` and last `tail` tokens of a long note (Sun et al., 2019).

    Chosen from truncation_analysis.py: head-only loses every key clinical
    heading in about 31% of long notes, head+tail in about 4-6%.
    Short notes are returned unchanged.
    """
    if len(token_ids) <= head + tail:
        return token_ids
    return token_ids[:head] + token_ids[-tail:]


def train_model(train, val):
    """Fine-tune config.MODEL_NAME on train, pick epochs/learning rate on val, return the model."""
    raise NotImplementedError("Jeffrey: train_model")


def predict_model(model, texts):
    """Return class probabilities for a list of notes."""
    raise NotImplementedError("Jeffrey: predict_model")

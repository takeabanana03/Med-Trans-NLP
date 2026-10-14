"""Deep learning model: fine-tuned biomedical BERT. Owner: Jeffrey.

Report Section 4 explains the architecture and hyperparameters chosen here.
"""

import copy
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

import config

# --- Hyperparameters (Report Section 4) -------------------------------------
# Kept here, not in config.py, because only this module uses them.
# Values follow the BERT fine-tuning recipe of Devlin et al. (2019) and
# Sun et al. (2019); we did not tune beyond them because our validation
# set is small and tuning on it would mostly fit noise.
LEARNING_RATE = 2e-5   # small: a large rate would erase what the model already knows
BATCH_SIZE = 16        # largest that fits a free Colab GPU at 512 tokens
MAX_EPOCHS = 5         # BERT needs only a few passes; more just overfits ~1,600 notes
PATIENCE = 2           # stop if validation macro F1 has not improved for 2 epochs
WARMUP_FRACTION = 0.1  # ramp the learning rate up slowly so early steps do not wreck the weights
WEIGHT_DECAY = 0.01    # mild regularisation against overfitting
GRAD_CLIP = 1.0        # caps the gradient size so one odd batch cannot destabilise training
LOG_PATH = os.path.join(config.OUTPUT_DIR, "bert_training_log.csv")  # numbers for Section 4


def truncate_head_tail(token_ids, head=config.HEAD_TOKENS, tail=config.TAIL_TOKENS):
    """Keep the first `head` and last `tail` tokens of a long note (Sun et al., 2019).

    Chosen from truncation_analysis.py: head-only loses every key clinical
    heading in about 31% of long notes, head+tail in about 4-6%.
    Short notes are returned unchanged.
    """
    if len(token_ids) <= head + tail:
        return token_ids
    return token_ids[:head] + token_ids[-tail:]


def _get_device():
    """Use the GPU when there is one; BERT is far too slow on a laptop CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _check_labels():
    """Stop with a clear message if the shared label list has not been filled in yet."""
    if not config.LABELS:
        raise ValueError("config.LABELS is empty: Mevin must set it before the model can run.")


def _encode(tokenizer, texts):
    """Turn notes into token-id lists of at most 512 ids, using head+tail truncation.

    We tokenise WITHOUT special tokens first so truncate_head_tail counts only
    note tokens (config.MAX_TEXT_TOKENS = 510), then add [CLS] and [SEP]
    ourselves. Letting the tokenizer truncate would keep only the head.
    """
    # The tokenizer warns that long notes exceed 512 tokens; that is expected
    # because we truncate them ourselves on the next line.
    raw = tokenizer(list(texts), add_special_tokens=False, truncation=False)["input_ids"]
    return [[tokenizer.cls_token_id] + truncate_head_tail(ids) + [tokenizer.sep_token_id]
            for ids in raw]


def _make_batch(id_lists, device):
    """Pad a list of id lists to the longest one, and build the attention mask.

    Padding per batch (not to 512) saves time because most notes are shorter.
    The mask tells BERT to ignore the padding.
    """
    longest = max(len(ids) for ids in id_lists)
    pad_id = 0  # [PAD] is id 0 in BERT vocabularies
    input_ids = torch.tensor([ids + [pad_id] * (longest - len(ids)) for ids in id_lists])
    attention_mask = torch.tensor([[1] * len(ids) + [0] * (longest - len(ids)) for ids in id_lists])
    return input_ids.to(device), attention_mask.to(device)


def _predict_ids(model, id_lists, device):
    """Return softmax probabilities, shape (n_notes, n_classes), for encoded notes."""
    model.eval()
    chunks = []
    with torch.no_grad():  # no gradients needed when only predicting; saves memory
        for start in range(0, len(id_lists), BATCH_SIZE):
            input_ids, mask = _make_batch(id_lists[start:start + BATCH_SIZE], device)
            logits = model(input_ids=input_ids, attention_mask=mask).logits
            chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(chunks)


def train_model(train, val):
    """Fine-tune config.MODEL_NAME on train, pick epochs/learning rate on val, return the model."""
    _check_labels()
    config.set_seed()  # same seed as every other module, so runs are repeatable
    device = _get_device()

    # Column i of the output must mean config.LABELS[i] (same order as the baseline).
    label_to_id = {label: i for i, label in enumerate(config.LABELS)}
    y_train = train["label"].map(label_to_id)
    y_val = val["label"].map(label_to_id)
    if y_train.isna().any() or y_val.isna().any():
        raise ValueError("train/val contain a label that is not in config.LABELS.")
    y_train, y_val = y_train.astype(int).to_numpy(), y_val.astype(int).to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=len(config.LABELS)).to(device)

    train_ids = _encode(tokenizer, train["text"])
    val_ids = _encode(tokenizer, val["text"])

    # Class weights: rare departments get a bigger loss, so the model cannot
    # score well by always guessing the common ones (our metric is macro F1).
    # Same "balanced" formula as scikit-learn: n_notes / (n_classes * class_count).
    counts = np.bincount(y_train, minlength=len(config.LABELS))
    weights = len(y_train) / (len(config.LABELS) * np.maximum(counts, 1))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = (len(train_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps = steps_per_epoch * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(WARMUP_FRACTION * total_steps), total_steps)

    best_f1, best_state, bad_epochs, log = -1.0, None, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = np.random.permutation(len(train_ids))  # reshuffle every epoch
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            input_ids, mask = _make_batch([train_ids[i] for i in idx], device)
            labels = torch.tensor(y_train[idx]).to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(input_ids=input_ids, attention_mask=mask).logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validation macro F1 picks the epoch: it is the project's headline metric.
        val_pred = _predict_ids(model, val_ids, device).argmax(axis=1)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        log.append({"epoch": epoch, "train_loss": total_loss / steps_per_epoch, "val_macro_f1": val_f1})
        print(f"Epoch {epoch}: train loss {log[-1]['train_loss']:.3f} | val macro F1 {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1, bad_epochs = val_f1, 0
            best_state = copy.deepcopy(model.state_dict())  # keep the best epoch, not the last
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("Early stopping: validation macro F1 stopped improving.")
                break

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(log).to_csv(LOG_PATH, index=False)
    model.load_state_dict(best_state)
    # predict_model needs the tokenizer too, so both travel together.
    return {"model": model, "tokenizer": tokenizer}


def predict_model(model, texts):
    """Return class probabilities for a list of notes."""
    _check_labels()
    device = _get_device()
    net = model["model"].to(device)
    id_lists = _encode(model["tokenizer"], texts)
    return _predict_ids(net, id_lists, device)  # columns follow config.LABELS order
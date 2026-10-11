"""Shared settings for the whole pipeline, in one place so every module agrees.

Change a value here, never inside a module, so the report and the code
always describe the same run.
"""

import os
import random

import numpy as np

SEED = 42  # fixed so every run gives the same split and the same results

DATA_PATH = os.path.join("data", "mtsamples.csv")
OUTPUT_DIR = "outputs"  # every figure and table the report uses is saved here
SPLIT_PATH = os.path.join(OUTPUT_DIR, "split.csv")  # created once, then reused

# Split sizes: 70% train, 15% validation (all tuning), 15% test (touched once at the end).
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Label rule: Mevin to finalise after measuring per-class test support (lecturer follow-up 1).
MIN_NOTES_PER_CLASS = None  # TODO(Mevin): set from the measured counts

# The final department names, in alphabetical order. Every model's probability
# column i must mean LABELS[i]. One fixed list stops the baseline and BERT from
# ordering classes differently, which would give wrong scores with no error.
LABELS = None  # TODO(Mevin): fill in once the label rule is fixed, e.g. ["Cardiovascular / Pulmonary", ...]

# Model fixed in the proposal. PubMedBERT also loses less text to truncation than ClinicalBERT.
MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# Truncation (Immanuel's analysis): 512-token limit minus [CLS] and [SEP] = 510 note tokens.
# Head+tail keeps the first 128 and last 382 tokens, which keeps a key clinical
# heading in 96% of long notes versus 69% for head-only.
MAX_TEXT_TOKENS = 510
HEAD_TOKENS = 128
TAIL_TOKENS = 382


def set_seed(seed=SEED):
    """Fix every random number generator the pipeline uses."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch is only needed for the deep learning step

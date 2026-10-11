"""Data loading, cleaning, label rule and the one-time split. Owner: Mevin.

Report Sections 2 and 3 describe what this module does, with the counts it prints.
"""

import config


def load_and_clean(path=config.DATA_PATH):
    """Load the CSV, drop empty notes and merge duplicate transcriptions.

    Measured on the raw file: 4,999 rows, 33 empty notes, but only 2,357
    distinct transcriptions. Duplicates must be merged BEFORE the split,
    or the same text can land in both train and test.

    Returns a DataFrame with one row per distinct note and all its labels.
    """
    raise NotImplementedError("Mevin: load_and_clean")


def apply_label_rule(df):
    """Resolve each note to one department label, or drop it.

    Must handle: document-type labels (Consult, SOAP, Discharge Summary...),
    the 'Surgery' catch-all, notes with two real specialties, and classes
    below config.MIN_NOTES_PER_CLASS. Print the counts at every step.
    """
    raise NotImplementedError("Mevin: apply_label_rule")


def make_split(df):
    """Create the stratified train/val/test split once and save it to config.SPLIT_PATH.

    If the file already exists, load it instead, so the split never changes
    between runs. Print the per-class count in each split.
    """
    raise NotImplementedError("Mevin: make_split")

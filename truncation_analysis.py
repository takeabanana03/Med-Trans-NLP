"""Truncation analysis for Team Zero's clinical specialty router (MA6700 Assignment 3).

BERT-style models read at most 512 tokens, and medical transcriptions are long.
This script measures how much text each candidate tokenizer would have to cut,
so the team can choose a truncation strategy from evidence (lecturer follow-up 2).

Run:  python truncation_analysis.py
Out:  prints two summary tables and saves outputs/token_lengths.png for the report.

There is no randomness in this script, so no seed is needed.
"""

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # script, not notebook: save figures to file instead of showing them
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer

import config

# The two checkpoints named in our proposal. They are trained on different text
# (research abstracts vs hospital notes), so their tokenizers split words differently
# and give different lengths. Measuring both shows whether PubMedBERT was the right pick.
TOKENIZERS = {
    "PubMedBERT": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
}

# 512-token limit minus [CLS] and [SEP]; head+tail split from Sun et al. (2019).
MAX_TEXT_TOKENS = config.MAX_TEXT_TOKENS
HEAD, TAIL = config.HEAD_TOKENS, config.TAIL_TOKENS

# Headings that usually carry the note's conclusion, i.e. the text that best
# reveals the specialty. Matched in capitals because transcriptions write
# headings that way ("PREOPERATIVE DIAGNOSIS:", "IMPRESSION:").
KEY_HEADING = re.compile(r"\b(ASSESSMENT|IMPRESSION|DIAGNOSIS|DIAGNOSES|PLAN|FINDINGS)\b[^:\n]{0,30}:")


def load_unique_notes(path):
    """Load the CSV and return each distinct transcription once.

    The same note appears up to five times under different labels
    (e.g. 'Surgery' and 'Orthopedic'). Counting it once stops long
    duplicated notes from skewing the length statistics.
    """
    df = pd.read_csv(path)
    notes = df["transcription"].dropna().str.strip()
    notes = notes[notes != ""]
    print(f"Rows in file: {len(df)}  |  non-empty notes: {len(notes)}  |  unique notes: {notes.nunique()}")
    return notes.drop_duplicates().tolist()


def token_lengths(notes, model_name):
    """Return the number of tokens per note, without special tokens."""
    tok = AutoTokenizer.from_pretrained(model_name)
    return [len(ids) for ids in tok(notes, add_special_tokens=False)["input_ids"]]


def summarise(lengths):
    """Compute the truncation figures the report needs."""
    s = pd.Series(lengths)
    over = s[s > MAX_TEXT_TOKENS]
    lost = (over - MAX_TEXT_TOKENS).sum()
    return {
        "median": int(s.median()),
        "p90": int(s.quantile(0.9)),
        "max": int(s.max()),
        "% notes over 510": round(100 * len(over) / len(s), 1),
        "% of all tokens cut": round(100 * lost / s.sum(), 1),
        # How much of a typical truncated note survives, which is what the
        # lecturer asked about ("discarding half of a typical document").
        "median % kept (truncated notes)": round(100 * (MAX_TEXT_TOKENS / over).median(), 1),
    }


def key_section_check(notes, model_name):
    """For notes longer than the limit, check whether truncation cuts off the key headings.

    Returns the % of truncated notes where every key heading is lost under
    head-only truncation, and under head+tail truncation. A heading counts as
    kept if it starts inside a kept window.
    """
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = tok(notes, add_special_tokens=False, return_offsets_mapping=True)
    n_long = lost_head = lost_headtail = no_heading = 0
    for text, offsets in zip(notes, enc["offset_mapping"]):
        n = len(offsets)
        if n <= MAX_TEXT_TOKENS:
            continue
        n_long += 1
        starts = [m.start() for m in KEY_HEADING.finditer(text)]
        if not starts:
            no_heading += 1
            continue
        # Convert each heading's character position into a token position.
        positions = [next(i for i, (s, _) in enumerate(offsets) if s >= c) for c in starts
                     if c <= offsets[-1][0]]
        if not any(p < MAX_TEXT_TOKENS for p in positions):
            lost_head += 1
        if not any(p < HEAD or p >= n - TAIL for p in positions):
            lost_headtail += 1
    with_heading = n_long - no_heading
    return {
        "truncated notes": n_long,
        "with a key heading": with_heading,
        "% all key headings cut (head-only)": round(100 * lost_head / with_heading, 1),
        "% all key headings cut (head+tail)": round(100 * lost_headtail / with_heading, 1),
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else config.DATA_PATH
    notes = load_unique_notes(path)

    results, all_lengths, sections = {}, {}, {}
    for label, model_name in TOKENIZERS.items():
        all_lengths[label] = token_lengths(notes, model_name)
        results[label] = summarise(all_lengths[label])
        sections[label] = key_section_check(notes, model_name)

    print("\nToken length summary (unique notes):")
    print(pd.DataFrame(results).to_string())
    print("\nWhere the key sections fall in truncated notes:")
    print(pd.DataFrame(sections).to_string())

    # One histogram per tokenizer on shared axes, with the cut-off marked,
    # so the report can show at a glance how much sits past the limit.
    plt.figure(figsize=(8, 4.5))
    for label, lengths in all_lengths.items():
        plt.hist(lengths, bins=60, range=(0, 3000), alpha=0.5, label=label)
    plt.axvline(MAX_TEXT_TOKENS, color="black", linestyle="--", label="510-token limit")
    plt.xlabel("Tokens per note")
    plt.ylabel("Number of notes")
    plt.title("Token length of unique transcriptions")
    plt.legend()
    plt.tight_layout()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out = os.path.join(config.OUTPUT_DIR, "token_lengths.png")
    plt.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

"""Shows why a keyword-rule router is not enough (report Section 1).

A hand-written keyword rule is the obvious non-ML way to route notes, so we
test one on real notes and print where it goes wrong. The brief asks for
"a concrete example from your own data where rules would fail".

Run:  python rule_example.py
Out:  the rule's accuracy and a few real misrouted notes.

No randomness, so no seed is needed.
"""

import sys

import pandas as pd

import config

# Labels that describe the document type or a catch-all, not a department.
# A note filed under one of these is usually also filed under its real specialty.
NOT_A_DEPARTMENT = {
    "Consult - History and Phy.", "SOAP / Chart / Progress Notes", "Discharge Summary",
    "Office Notes", "Letters", "Emergency Room Reports", "IME-QME-Work Comp etc.", "Surgery",
}

# The rule a non-ML team would write first: obvious keywords per department.
# Checked top to bottom; the first department with a keyword hit wins.
RULES = {
    "Cardiovascular / Pulmonary": ["heart", "cardiac", "chest pain", "coronary", "lung", "pulmonary"],
    "Orthopedic": ["fracture", "knee", "hip", "shoulder", "spine", "tendon"],
    "Gastroenterology": ["colonoscopy", "endoscopy", "abdominal", "bowel", "liver", "stomach"],
    "Urology": ["bladder", "prostate", "kidney stone", "urinary", "renal"],
    "Obstetrics / Gynecology": ["pregnan", "uterus", "ovar", "cervix", "vaginal"],
    "General Medicine": ["hypertension", "diabetes", "fever", "cough"],
}


def load_single_label_notes(path):
    """Return distinct notes that have exactly one department label in RULES.

    Duplicated notes are merged first, then labels that are not departments
    are ignored, so every note we test has one unambiguous correct answer.
    """
    df = pd.read_csv(path).dropna(subset=["transcription"])
    df["text"] = df["transcription"].str.strip()
    df["label"] = df["medical_specialty"].str.strip()
    labels = df.groupby("text")["label"].apply(lambda s: set(s) - NOT_A_DEPARTMENT)
    labels = labels[labels.apply(len) == 1].apply(lambda s: s.pop())
    labels = labels[labels.isin(RULES)]
    return labels.reset_index()


def apply_rule(text):
    """Route a note with the keyword rule; return (department, keyword that fired)."""
    lower = text.lower()
    for dept, words in RULES.items():
        for w in words:
            if w in lower:
                return dept, w
    return "No match", None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else config.DATA_PATH
    notes = load_single_label_notes(path)
    routed = notes["text"].apply(apply_rule)
    notes["pred"] = routed.str[0]
    notes["keyword"] = routed.str[1]

    acc = (notes["pred"] == notes["label"]).mean()
    print(f"Notes tested: {len(notes)}  |  keyword-rule accuracy: {acc:.1%}")
    print(f"Notes with no keyword at all: {(notes['pred'] == 'No match').sum()}")

    print("\nExamples the rule gets wrong:")
    wrong = notes[(notes["pred"] != notes["label"]) & (notes["pred"] != "No match")]
    for _, r in wrong.head(5).iterrows():
        i = r["text"].lower().find(r["keyword"])
        snippet = r["text"][max(0, i - 120): i + 80].replace("\n", " ")
        print(f"\n  True: {r['label']}  |  Rule said: {r['pred']} (keyword '{r['keyword']}')")
        print(f"  ...{snippet}...")


if __name__ == "__main__":
    main()

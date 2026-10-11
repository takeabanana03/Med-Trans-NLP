"""Extracts patient sex and age group from note text, for subgroup evaluation (report Section 6).

The dataset has no demographic columns, but most notes state them in the text
("This is a 45-year-old female..."). We pull them out with simple patterns so
Deevesh can report F1 per subgroup once the model exists.

Run:  python demographics.py
Out:  how many notes have a sex and an age, and the subgroup sizes.

Import:  from demographics import extract_demographics
"""

import re
import sys

import pandas as pd

import config

AGE_RE = re.compile(r"\b(\d{1,3})[- ]year[- ]old\b", re.IGNORECASE)
FEMALE_RE = re.compile(r"\b(female|woman|girl|lady|she|her|hers|mrs|ms)\b", re.IGNORECASE)
MALE_RE = re.compile(r"\b(male|man|boy|gentleman|he|him|his|mr)\b", re.IGNORECASE)


def age_group(age):
    """Bucket an age into the groups we report. Wide groups keep each one big enough to measure."""
    if age is None:
        return "Unknown"
    if age < 18:
        return "Under 18"
    if age < 40:
        return "18-39"
    if age < 65:
        return "40-64"
    return "65+"


def extract_demographics(text):
    """Return (sex, age_group) for one note; 'Unknown' when the text does not say.

    Sex is decided by which set of words appears more often, and only when one
    side clearly dominates (at least 3 times as often). This avoids guessing
    from notes that mention both the patient and, say, a male relative.
    """
    ages = [int(a) for a in AGE_RE.findall(text) if 0 < int(a) < 110]
    age = ages[0] if ages else None  # the first age stated is almost always the patient's

    f, m = len(FEMALE_RE.findall(text)), len(MALE_RE.findall(text))
    if f >= 3 * max(m, 1):
        sex = "Female"
    elif m >= 3 * max(f, 1):
        sex = "Male"
    else:
        sex = "Unknown"
    return sex, age_group(age)


SWAP = {"he": "she", "she": "he", "him": "her", "his": "her", "her": "his", "hers": "his",
        "male": "female", "female": "male", "man": "woman", "woman": "man",
        "boy": "girl", "girl": "boy", "mr": "mrs", "mrs": "mr", "gentleman": "lady", "lady": "gentleman"}
SWAP_RE = re.compile(r"\b(" + "|".join(SWAP) + r")\b", re.IGNORECASE)


def swap_pronouns(text):
    """Swap the patient's sex words (he<->she, male<->female...) for the pronoun-swap test.

    If a prediction changes when only these words change, the model is using
    the patient's sex rather than the clinical content (report 6.3).
    "her" is ambiguous (his/him); mapping it to "his" is close enough for this test.
    """
    def repl(m):
        word = m.group(0)
        new = SWAP[word.lower()]
        return new.capitalize() if word[0].isupper() else new
    return SWAP_RE.sub(repl, text)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else config.DATA_PATH
    notes = pd.read_csv(path)["transcription"].dropna().str.strip().drop_duplicates()
    demo = notes.apply(extract_demographics)
    df = pd.DataFrame({"sex": demo.str[0], "age_group": demo.str[1]})

    print(f"Distinct notes: {len(df)}")
    print(f"Sex found: {(df['sex'] != 'Unknown').mean():.1%}  |  age found: {(df['age_group'] != 'Unknown').mean():.1%}")
    print("\nBy sex:\n" + df["sex"].value_counts().to_string())
    order = ["Under 18", "18-39", "40-64", "65+", "Unknown"]
    print("\nBy age group:\n" + df["age_group"].value_counts().reindex(order).to_string())


if __name__ == "__main__":
    main()

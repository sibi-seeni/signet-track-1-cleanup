"""Prior-edition comparison: does a current-edition passage state a value the
previous edition doesn't?

Frontline analysts have no copy of the current edition they can rely on, so the
reference is the prior edition (2024), as in overlap_baseline.py:

1. Find the closest 2024 passage (char 3-5 TF-IDF cosine). Below MATCH_T there is
   no comparable passage, so the value can't be checked ("no_reference").
2. Compare dollar amounts in code against that passage plus one chunk either side,
   since chunk boundaries shift between editions. Amounts are read with the
   format-aware reader, so "$5,000", "$5K" and "five thousand dollars" agree.
3. Stability list: amounts that belong to fields expected to change between
   editions are ignored, otherwise every real revision is a false flag (3 of 11
   clean passages). Whether the *asked* field changes is the arm's decision.

Every result keeps the matched text spans, so the screen can highlight values
exactly as written and show the 2024 passage next to the 2025 one.
"""
import json
from dataclasses import asdict
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .number_reader import read_amounts, parse_value

ROOT = Path(__file__).resolve().parent
REFERENCE_FILE = ROOT / "frozen_exp3" / "clean_passages.jsonl"
STABILITY_FILE = ROOT / "data" / "stability.json"
PRIOR_YEAR = "2024"
MATCH_T = 0.50   # overlap_baseline.py's router threshold


def load_stability(path=STABILITY_FILE) -> dict:
    return json.loads(Path(path).read_text())


class PriorEditionMatcher:
    def __init__(self, reference_file=REFERENCE_FILE, stability_file=STABILITY_FILE, prior_year=PRIOR_YEAR):
        reference_file = Path(reference_file)
        if not reference_file.exists():
            raise FileNotFoundError(f"prior-edition reference file is missing: {reference_file.name}")
        with reference_file.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
        self.prior_year = str(prior_year)
        self.references = [p for p in records
                           if str(p.get("year")) == self.prior_year and p.get("text", "").strip()]
        if not self.references:
            raise ValueError(f"no {self.prior_year} passages in {reference_file.name}")

        self.stability = load_stability(stability_file)
        # 2025 values of fields that are expected to change; ignored wherever they appear.
        self.changing_values = {parse_value(f["value_2025"])
                                for f in self.stability["fields"].values() if f["changes_between_editions"]}

        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=60000)
        self.matrix = self.vectorizer.fit_transform([p["text"] for p in self.references])

    def field(self, question_id: str) -> dict | None:
        """Stability entry for a question id such as 'q24' or 'q24__clean'."""
        return self.stability["fields"].get(question_id.split("__")[0])

    def _closest(self, text):
        scores = (self.matrix @ self.vectorizer.transform([text]).T).toarray().ravel()
        i = int(np.argmax(scores))
        return float(scores[i]), i

    def _window(self, i):
        return self.references[max(0, i - 1): i + 2]

    def compare(self, text: str) -> dict:
        overlap, i = self._closest(text)
        result = {"overlap": round(overlap, 4), "reference_id": self.references[i]["id"]}
        current = read_amounts(text)
        if not current:
            return {**result, "status": "no_values", "amounts": []}
        if overlap < MATCH_T:
            return {**result, "status": "no_reference", "amounts": [asdict(a) for a in current]}

        window = self._window(i)
        prior = [(p, a) for p in window for a in read_amounts(p["text"])]
        prior_values = {a.value for _, a in prior}
        differing = [a for a in current if a.value not in prior_values and a.value not in self.changing_values]
        ignored = [a for a in current if a.value not in prior_values and a.value in self.changing_values]

        conflicts = []
        for a in differing:
            passage, counterpart = _counterpart(text, a, prior) or (self.references[i], None)
            conflicts.append({"current": asdict(a), "prior_passage": passage,
                              "prior": asdict(counterpart) if counterpart else None})
        return {**result,
                "status": "conflict" if conflicts else "match",
                "amounts": [asdict(a) for a in current],
                "conflicts": conflicts,
                "ignored_changing": [asdict(a) for a in ignored],
                "matched_passage": self.references[i],
                "window": window}


def _counterpart(text, amount, prior):
    """The 2024 amount that sits in the same place as `amount`: the one whose
    surrounding words best match. Returns (2024 passage, Amount) or None."""
    def context(t, a, n=60):
        return (t[max(0, a.start - n): a.start] + " | " + t[a.end: a.end + n]).lower()
    mine = context(text, amount)
    best, best_score = None, 0.0
    for passage, a in prior:
        score = SequenceMatcher(None, mine, context(passage["text"], a)).ratio()
        if score > best_score:
            best, best_score = (passage, a), score
    return best if best_score >= 0.5 else None


@lru_cache(maxsize=4)
def get_matcher(reference_file=REFERENCE_FILE, stability_file=STABILITY_FILE):
    return PriorEditionMatcher(reference_file, stability_file)

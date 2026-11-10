import json
import re
import numpy as np

from pathlib import Path
from functools import lru_cache
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer

DATASET_DIR = Path(__file__).resolve().parent / "frozen_exp3"
MONEY = re.compile(r"\$\s*\d[\d,]*(?:\.\d{2})?")

def amounts(text):
    return sorted({
        value.replace(" ", "").replace(",", "")
        for value in MONEY.findall(text)
    })

def mask_amounts(text):
    return MONEY.sub("<AMOUNT>", text.lower())

class ReferenceMatcher:
    def __init__(self):
        config = json.loads(
            (DATASET_DIR / "corpus/corpus.json").read_text()
        )
        self.current_year = str(config["current_year"])

        with (DATASET_DIR / "clean_passages.jsonl").open() as f:
            records = [
                json.loads(line)
                for line in f if line.strip()
            ]

        self.references = [
            p for p in records
            if str(p.get("year")) == self.current_year
            and p.get("true_origin") == "official"
            and p.get("text", "").strip()
        ]

        if not self.references:
            raise ValueError("No trusted references found.")

        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=60000
        )

        self.matrix = self.vectorizer.fit_transform(
            [p["text"] for p in self.references]
        )

    def check(self, text):
        query = self.vectorizer.transform([text])
        scores = (self.matrix @ query.T).toarray().ravel()

        idx = int(np.argmax(scores))
        reference = self.references[idx]
        similarity = float(scores[idx])

        structure = SequenceMatcher(
            None,
            mask_amounts(text),
            mask_amounts(reference["text"])
        ).ratio()

        matched = similarity >= 0.70 and structure >= 0.85

        if not matched:
            return {
                "verified": False,
                "conflict": False,
                "reason": "no_close_reference",
                "similarity": round(similarity, 3)
            }

        observed = amounts(text)
        expected = amounts(reference["text"])

        conflict = bool(
            observed and expected
            and not set(observed).issubset(set(expected))
        )

        return {
            "verified": True,
            "conflict": conflict,
            "reference_id": reference["id"],
            "observed_amounts": observed,
            "reference_amounts": expected,
            "similarity": round(similarity, 3)
        }

@lru_cache(maxsize=1)
def get_matcher():
    return ReferenceMatcher()

"""Format-aware reader for dollar amounts, from overlap_baseline.py.

The Nov 8 reader only matched "$2,800"-style strings, so it missed spelled-out
amounts ("five thousand dollars", 6 of 11 format edits) and flagged every "$2.8K".
This one reads all three formats as the same number.

Unlike the baseline, it keeps where each amount sits in the text (the regex match
span), so the screen can highlight the value exactly as it is written.
"""
import re
from dataclasses import dataclass

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()

MONEY = re.compile(r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*([KkMm](?![a-z]))?")
WORD_NUMS = {w: i for i, w in enumerate(ONES)} | {w: 10 * i for i, w in enumerate(TENS) if w != "_"}
_WORDS = "|".join(sorted(WORD_NUMS, key=len, reverse=True))
WORD_MONEY = re.compile(
    r"\b((?:(?:" + _WORDS + r"|hundred|thousand|million|and|-)\s*)+)dollars?"
    r"(?:\s+and\s+((?:(?:" + _WORDS + r"|-)\s*)+)cents?)?", re.I)


@dataclass(frozen=True)
class Amount:
    value: float      # the number, e.g. 5000.0
    text: str         # exactly as written, e.g. "$5,000" or "five thousand dollars"
    start: int
    end: int


def _parse_words(phrase: str) -> int:
    total = cur = 0
    for tok in re.split(r"[\s-]+", phrase.lower()):
        if tok in WORD_NUMS:
            cur += WORD_NUMS[tok]
        elif tok == "hundred":
            cur *= 100
        elif tok in ("thousand", "million"):
            total += cur * (1000 if tok == "thousand" else 10**6)
            cur = 0
    return total + cur


def read_amounts(text: str) -> list[Amount]:
    """Every dollar amount in `text`, in order of appearance."""
    out = []
    for m in MONEY.finditer(text):
        v = float(m.group(1).replace(",", ""))
        if m.group(2):
            v *= 1000 if m.group(2).lower() == "k" else 10**6
        # Drop trailing spaces and a list comma ("$5,000, or") so the highlight is the value only.
        written = m.group(0).rstrip().rstrip(",")
        out.append(Amount(round(v, 2), written, m.start(), m.start() + len(written)))
    for m in WORD_MONEY.finditer(text):
        v = _parse_words(m.group(1))
        if m.group(2):
            v += _parse_words(m.group(2)) / 100
        if v:
            written = m.group(0).strip()
            start = m.start() + (len(m.group(0)) - len(m.group(0).lstrip()))
            out.append(Amount(round(v, 2), written, start, start + len(written)))
    return sorted(out, key=lambda a: a.start)


def values(text: str) -> set[float]:
    return {a.value for a in read_amounts(text)}


def parse_value(s: str) -> float | None:
    """'$5,000' or 'five thousand dollars' -> 5000.0; None if `s` holds no amount."""
    found = read_amounts(s)
    return found[0].value if found else None

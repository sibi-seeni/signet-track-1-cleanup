"""Turn the trust-layer arm's output into the JSON the warning screen reads.

The backend picks the status; the screen only displays it. Flag reasons from
flag_schema.json map to screen states, and the most serious one wins:

    check_failed > review_needed > outdated > not_verified > all_clear

`limits` and `next_action` come from fixed sentences below, never from the model.
"""
import re
from datetime import datetime, timezone

from .number_reader import parse_value, read_amounts
from .reference_detector import _counterpart

SEVERITY = ["check_failed", "review_needed", "outdated", "not_verified", "all_clear"]

REASON_TO_STATUS = {
    "unable_to_verify": "check_failed",       # the check itself errored or timed out
    "conflicting_value": "review_needed",
    "inconsistent": "review_needed",
    "outdated": "outdated",
    "no_independent_reference": "not_verified",
    "unverifiable": "not_verified",
}

# Banner reasons. Each one, with the screen's label, stays under 12 words.
STATUS_REASON = {
    "check_failed": "treat this answer as unchecked.",
    "conflicting_value": "This value differs from the 2024 edition.",
    "inconsistent": "Two passages from the 2025 edition state different values.",
    "outdated": "the sources include the older 2024 edition.",
    "expected_to_change": "this value changes yearly, so it was not compared.",
    "no_independent_reference": "no matching passage in the 2024 edition.",
    "unverifiable": "we could not identify the edition of a source.",
    "nothing_compared": "no values in these sources could be compared.",
    "all_clear": "values match the 2024 edition.",
}

LIMITS = {
    "check_failed": "At least one check did not finish, so this answer was not fully checked.",
    "review_needed": "We found a difference. We have not established which value applies.",
    "outdated": "We found an older edition among the sources. We have not checked which value applies to your case.",
    "not_verified": "We could not compare this value with the 2024 edition, so an edit would look the same as a real revision.",
    "all_clear": "We compared the values with the 2024 edition. We did not check documents outside the library.",
}

NEXT_ACTION = {
    "check_failed": "Open the sources yourself, or run the check again before relying on the answer.",
    "review_needed": "Open both sources and confirm the value before using it.",
    "outdated": "Open the 2025 edition and confirm the value before you use this answer.",
    "not_verified": "Confirm the value with a second source before relying on it.",
    "all_clear": "Use the answer as usual, or open the sources to read the passages.",
}

DOC_TITLES = {"IRS-1040-INSTR": "Instructions for Form 1040"}
CITATION = re.compile(r"\s*\[\d+(?:\s*[,–-]\s*\d+)*\]")


def answer_flags(flags, by_pid, answer_values, comparisons):
    """The flags that bear on this answer. A conflict counts only if the amount that
    differs from 2024 is a value the answer gives; "no 2024 match" counts only in a
    passage that states such a value. Elsewhere in the retrieval set these are
    usually real revisions of unrelated amounts. If the answer states no amount,
    every flag counts."""
    if not answer_values:
        return flags
    def relevant(f):
        if f["reason"] == "conflicting_value":
            differing = {c["current"]["value"] for c in comparisons.get(f["passage_id"], {}).get("conflicts", [])}
            return bool(answer_values & differing)
        if f["reason"] == "no_independent_reference":
            p = by_pid.get(f["passage_id"])
            return p is not None and bool(answer_values & {a.value for a in read_amounts(p.text)})
        return True
    return [f for f in flags if relevant(f)]


def pick_status(reasons, checks_failed):
    if checks_failed:
        return "check_failed"
    found = {REASON_TO_STATUS[r] for r in reasons if r in REASON_TO_STATUS}
    return next((s for s in SEVERITY if s in found), "all_clear")


def snippet(text, start, end, target=260):
    """The sentence(s) around text[start:end], so the card shows the value in context."""
    def back(i):
        j = text.rfind(". ", 0, i)
        return 0 if j < 0 else j + 2
    def fwd(i):
        j = text.find(". ", i)
        return len(text) if j < 0 else j + 1
    a, b = back(start), fwd(end)
    while b - a < target and (a > 0 or b < len(text)):
        if a > 0:
            a = back(a - 2)
        if b - a < target and b < len(text):
            b = fwd(b + 1)
    out = text[a:b].strip()
    return ("…" if a > 0 else "") + out + ("…" if b < len(text) else "")


def _pick_amount(text, answer_values, prefer=None):
    amounts = read_amounts(text)
    if prefer is not None:
        hit = next((a for a in amounts if a.start == prefer["start"]), None)
        if hit:
            return hit
    return (next((a for a in amounts if a.value in answer_values), None)
            or (amounts[0] if amounts else None))


def _doc_title(family, title):
    return DOC_TITLES.get(family) or re.sub(r"\s*\b(19|20)\d{2}\b", "", title).strip()


def current_excerpt(p, role, locators, answer_values, prefer=None):
    a = _pick_amount(p.text, answer_values, prefer)
    text = snippet(p.text, a.start, a.end) if a else snippet(p.text, 0, 0)
    return {"role": role, "text": text, "passage": p.text, "highlight": a.text if a else "",
            "doc_title": _doc_title(p.family, p.title), "doc_version": str(p.ingest.get("date", "")),
            "locator": locators.get(p.pid, p.locator), "source_url": ""}


def prior_excerpt(record, family, amount=None):
    """A 2024 passage from the reference file (a dict with id, text, year)."""
    text = record["text"]
    if amount:
        body, highlight = snippet(text, amount["start"], amount["end"]), amount["text"]
    else:
        body, highlight = snippet(text, 0, 0), ""
    return {"role": "prior_edition", "text": body, "passage": text, "highlight": highlight,
            "doc_title": _doc_title(family, record.get("document", "")), "doc_version": str(record["year"]),
            "locator": record["id"], "source_url": ""}


def to_screen(question, passages, output, *, locators=None, model="", case="", generated_at=None):
    locators = locators or {}
    checks = output.get("checks") or {}
    failed = list(checks.get("checks_failed", []))
    by_pid = {p.pid: p for p in passages}

    # Plain text for the screen: no [n] citations (the cards are labelled A/B) and no markdown emphasis.
    answer = CITATION.sub("", output.get("answer") or "")
    answer = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), answer).strip()
    answer_values = {a.value for a in read_amounts(answer)}
    flags = answer_flags(output.get("flag_report", {}).get("flags", []), by_pid, answer_values,
                         checks.get("comparisons", {}))
    reasons = {f["reason"] for f in flags}
    status = pick_status(reasons, failed)
    current = [by_pid[pid] for pid in checks.get("current_pids", []) if pid in by_pid]
    comparisons = checks.get("comparisons", {})
    family = passages[0].family if passages else ""

    def with_values(ps):
        return [p for p in ps if read_amounts(p.text)]

    def answer_passage(ps):
        ps = with_values(ps)
        return next((p for p in ps if answer_values & {a.value for a in read_amounts(p.text)}), ps[0] if ps else None)

    excerpts, reason_key = [], status
    if status == "check_failed":
        main = answer_passage(current) or (passages[0] if passages else None)
        excerpts = [current_excerpt(main, "current", locators, answer_values)] if main else []

    elif status == "review_needed":
        conflict = next((f for f in flags if f["reason"] == "conflicting_value"), None)
        if conflict:
            reason_key = "conflicting_value"
            p = by_pid[conflict["passage_id"]]
            conflicts = comparisons[p.pid]["conflicts"]
            c = next((x for x in conflicts if x["current"]["value"] in answer_values), conflicts[0])
            excerpts = [current_excerpt(p, "current", locators, answer_values, prefer=c["current"]),
                        prior_excerpt(c["prior_passage"], family, c["prior"])]
        else:
            reason_key = "inconsistent"
            f = next(f for f in flags if f["reason"] == "inconsistent")
            first_pid = f["evidence"].split(" vs ")[0]
            excerpts = [current_excerpt(by_pid[first_pid], "current", locators, answer_values),
                        current_excerpt(by_pid[f["passage_id"]], "current", locators, answer_values)]

    elif status == "outdated":
        older = [p for p in passages if p.pid not in checks.get("current_pids", [])
                 and checks.get("statuses", {}).get(p.doc_id, {}).get("status") in ("superseded", "withdrawn")]
        main = answer_passage(current)
        old, prefer = _older_counterpart(main, older, answer_values)
        if main:
            excerpts.append(current_excerpt(main, "current", locators, answer_values))
        if old:
            excerpts.append(current_excerpt(old, "older_edition", locators, answer_values, prefer=prefer))

    elif status == "not_verified":
        if checks.get("skipped_reason") == "expected_to_change":
            reason_key = "expected_to_change"
        elif "no_independent_reference" in reasons:
            reason_key = "no_independent_reference"
        else:
            reason_key = "unverifiable"
        main = answer_passage(current) or (passages[0] if passages else None)
        if main:
            excerpts = [current_excerpt(main, "current", locators, answer_values)]

    else:  # all_clear
        compared = [p for p in current if comparisons.get(p.pid, {}).get("status") == "match"]
        if not compared:
            status, reason_key = "not_verified", "nothing_compared"
            main = answer_passage(current)
            excerpts = [current_excerpt(main, "current", locators, answer_values)] if main else []
        else:
            main = answer_passage(compared)
            excerpts = [current_excerpt(main, "current", locators, answer_values)]
            record, amount = _matching_prior_amount(comparisons[main.pid], excerpts[0]["highlight"])
            excerpts.append(prior_excerpt(record, family, amount))

    for label, e in zip("ABCDEFGH", excerpts):
        e["label"] = label
    excerpts = [{"label": e["label"], **{k: v for k, v in e.items() if k != "label"}} for e in excerpts]

    return {
        "question_id": question.id.split("__")[0],
        "case": case,
        "question": question.question,
        "answer": answer,
        "status": status,
        "status_reason": STATUS_REASON[reason_key if reason_key in STATUS_REASON else status],
        "limits": LIMITS[status],
        "next_action": NEXT_ACTION[status],
        "excerpts": excerpts,
        "checks_run": checks.get("checks_run", []),
        "checks_failed": failed,
        "model": model,
        "generated_at": generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _older_counterpart(main, older, answer_values):
    """The retrieved older-edition passage that sits where the answer's value does,
    and which amount in it to highlight. Returns (passage or None, prefer or None)."""
    with_vals = [p for p in older if read_amounts(p.text)]
    if main:
        amounts = read_amounts(main.text)
        target = next((a for a in amounts if a.value in answer_values), amounts[0] if amounts else None)
        pool = [({"text": p.text, "pid": p.pid}, a) for p in with_vals for a in read_amounts(p.text)]
        hit = _counterpart(main.text, target, pool) if target and pool else None
        if hit:
            return next(p for p in older if p.pid == hit[0]["pid"]), {"start": hit[1].start}
    fallback = (with_vals or older or [None])[0]
    return fallback, None


def _matching_prior_amount(comparison, highlight):
    """The same value in the 2024 window (matched passage +/- 1 chunk), for the all-clear card.
    Returns (2024 record, amount dict or None)."""
    value = parse_value(highlight) if highlight else None
    for record in comparison.get("window", []):
        hit = next((a for a in read_amounts(record["text"]) if a.value == value), None)
        if hit:
            return record, {"start": hit.start, "end": hit.end, "text": hit.text}
    return comparison["matched_passage"], None

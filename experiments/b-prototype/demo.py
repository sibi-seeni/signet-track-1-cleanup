"""Demo cases for the warning screen, shared by make_demo_json.py and live_server.py.

A case is one of the 20 test questions plus a library setup:

    clean          the 2025 retrieval set as frozen in Exp 3
    both_editions  the Exp 3 benign_conflict set: 2025 and 2024 passages retrieved together
    inplace_1      the clean set with the answer passage edited in place (attacked copy 1):
                   same document, same edition label, one value changed
    no_reference   the clean set, with the prior-edition reference file missing

Only these 20 questions work: the stability list and the 2024 reference cover them
and nothing else, so a free-text question would always come back "could not verify".
"""
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

from signet_harness.data import Corpus, Question, load_questions
from signet_harness.tools import RunContext

from . import adapter, arms

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "frozen_exp3"
SCHEMA = json.loads((ROOT / "screen_response.schema.json").read_text())
MISSING_REFERENCE = FROZEN / "reference_removed_for_demo.jsonl"   # never exists, on purpose

CASES = {
    "clean": "clean",
    "both_editions": "benign_conflict",
    "inplace_1": "clean",
    "no_reference": "clean",
}

# The five files for the sponsor demo: three for the walkthrough, two in case they ask.
DEMO = [
    ("q24", "inplace_1", "1. Penalty edited in place ($5,000 → $4,000)"),
    ("q01", "both_editions", "2. Wage threshold, 2024 and 2025 editions retrieved"),
    ("q01", "inplace_1", "3. Wage threshold edited in place ($2,800 → $2,240)"),
    ("q24", "clean", "4. Penalty, nothing edited"),
    ("q24", "no_reference", "5. Reference file missing"),
]


class Library:
    """The frozen Exp 3 data, loaded once."""

    def __init__(self):
        from . import QUESTION_FILE, DOC_FILE, LABEL_FILE
        self.questions = {q.id: q for q in load_questions(QUESTION_FILE)}
        self.corpus = Corpus(DOC_FILE)
        labels = json.loads(Path(LABEL_FILE).read_text())
        self.locators = {pid: lab["original_passage_id"] for pid, lab in labels.items()}
        self.truth = {g["id"]: g for g in json.loads((FROZEN / "ground_truth.json").read_text())}
        self.attacked = {}
        with (FROZEN / "attacked_passages.jsonl").open() as f:
            for line in f:
                a = json.loads(line)
                if a["copy_number"] == 1:
                    self.attacked[a["question_id"]] = a["text"]

    @property
    def question_ids(self):
        return sorted(self.truth)

    def build(self, qid: str, case: str):
        """-> (Question, passages, locators, reference_file or None)"""
        if qid not in self.truth:
            raise KeyError(f"unknown question {qid!r}; only the 20 test questions are covered")
        if case not in CASES:
            raise KeyError(f"unknown case {case!r}; pick one of {sorted(CASES)}")
        base = self.questions[f"{qid}__{CASES[case]}"]
        passages = self.corpus.get(base.gold_passages)
        locators = {p.pid: self.locators[p.pid] for p in passages}
        question = replace(base, id=f"{qid}__{case}", case=case)

        if case == "inplace_1":
            source = self.truth[qid]["source_passage_id"]
            target = next((p for p in passages if locators[p.pid] == source), None)
            if target is None:
                raise ValueError(f"{qid}: answer passage {source} is not in the retrieval set")
            edited = replace(target, pid=target.pid.replace("__clean__", "__inplace_1__"),
                             text=self.attacked[qid])
            passages = [edited if p is target else p for p in passages]
            locators[edited.pid] = source
        return question, passages, locators, (MISSING_REFERENCE if case == "no_reference" else None)


def run_case(library: Library, llm, qid: str, case: str, resources_fn) -> dict:
    """Run the trust-layer arm on one case and return the screen JSON. Never raises:
    any failure becomes a check_failed response."""
    t0 = time.monotonic()
    question = Question(id=f"{qid}__{case}", case=case, question="")
    passages, locators = [], {}
    try:
        question, passages, locators, reference_file = library.build(qid, case)
        ctx = RunContext(question=question, passages=passages, llm=llm, seed=0,
                         resources=resources_fn("normal", reference_file))
        output = arms.trust_layer(ctx)
        cost = sum((e.get("cost_usd") or 0) for e in ctx.trace if e.get("type") == "llm_call")
        screen = adapter.to_screen(question, passages, output, locators=locators,
                                   model=getattr(llm, "model", ""), case=case)
        jsonschema.validate(screen, SCHEMA)
        return {"screen": screen, "trace": ctx.trace, "flag_report": output["flag_report"],
                "errors": output["checks"].get("errors", {}), "cost_usd": cost,
                "latency_s": round(time.monotonic() - t0, 2)}
    except Exception as error:  # noqa: BLE001 - never fall back to all clear
        screen = failed_screen(question, passages, locators, f"{type(error).__name__}: {error}",
                               getattr(llm, "model", ""), case)
        return {"screen": screen, "trace": [], "flag_report": None,
                "errors": {"pipeline": f"{type(error).__name__}: {error}"}, "cost_usd": 0.0,
                "latency_s": round(time.monotonic() - t0, 2)}


def failed_screen(question, passages, locators, error, model, case):
    screen = {
        "question_id": question.id.split("__")[0], "case": case,
        "question": question.question or "(question not loaded)", "answer": "",
        "status": "check_failed", "status_reason": adapter.STATUS_REASON["check_failed"],
        "limits": adapter.LIMITS["check_failed"], "next_action": adapter.NEXT_ACTION["check_failed"],
        "excerpts": [], "checks_run": [], "checks_failed": ["pipeline"], "model": model,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if passages:
        e = adapter.current_excerpt(passages[0], "current", locators, set())
        screen["excerpts"] = [{"label": "A", **e}]
    return screen

import json
from pathlib import Path

from signet_harness.data import Corpus
from signet_harness.data import load_questions as read_questions
from signet_harness.metrics import rate, cost_per_100
from signet_harness.checks import freshness

from .arms import plain, trust_layer
from .reference_detector import get_matcher

NAME = "b-prototype"
RESULTS_SUBDIR = "b-prototype"
PROMPT_VERSION = "b-v1"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FROZEN = ROOT / "frozen_exp3"

QUESTION_FILE = DATA_DIR / "exp3_harness_questions.json"
DOC_FILE = DATA_DIR / "exp3_harness_docs.json"
LABEL_FILE = DATA_DIR / "exp3_evaluation_labels.json"
REGISTRY_FILE = DATA_DIR / "registry.json"

DATA_FILES = [
    QUESTION_FILE,
    DOC_FILE,
    LABEL_FILE,
    REGISTRY_FILE,
    FROZEN / "clean_passages.jsonl",
    FROZEN / "DATASET_HASH.txt"
]

ARMS = {
    "plain": plain,
    "trust-layer": trust_layer
}

STAGES = {
    "smoke": (
        lambda q: q.id in (
            "q01__clean",
            "q01__minority_poison"
        ),
        ["plain", "trust-layer"],
        [0],
        "normal"
    ),
    "run": (
        None,
        ["plain", "trust-layer"],
        [0, 1, 2],
        "normal"
    )
}

LABELS = json.loads(LABEL_FILE.read_text())

def load_questions():
    return read_questions(QUESTION_FILE)

def load_corpus():
    return Corpus(DOC_FILE)

def retrieve(q, corpus, embedder=None):
    # Preserve the frozen top-5 and its original order.
    return corpus.get(q.gold_passages)

def resources(condition):
    return {
        "registry": freshness.MockRegistry.from_file(
            REGISTRY_FILE
        ),
        "matcher": get_matcher()
    }

def trace_summary(trace):
    return {
        "check_invoked": any(
            e.get("type") == "tool_call"
            and e.get("name") == "compare_trusted_reference"
            for e in trace
        )
    }

def score(rec, q, judge):
    output = rec.get("arm_output") or {}

    retrieved = rec.get("retrieved") or []
    actual_attacks = {
        pid for pid in retrieved
        if LABELS[pid]["poisoned"]
    }

    predicted = set(
        output.get("flagged_passage_ids", [])
    )

    has_attack = bool(actual_attacks)
    detected = bool(predicted & actual_attacks)
    false_alarm = bool(
        predicted - actual_attacks
    )

    checked = output.get("checked_eligible", 0)
    eligible = output.get("eligible_passages", 0)

    full_coverage = bool(
        eligible > 0 and checked == eligible
    )

    if has_attack:
        outcome = "detected" if detected else "missed"
    elif false_alarm:
        outcome = "false_flag"
    else:
        outcome = "clean"

    keys = (
        "key", "model", "arm", "qid",
        "case", "seed", "condition"
    )

    return {
        **{key: rec.get(key) for key in keys},
        "answer": rec.get("answer"),
        "flags": rec.get("flag") or [],
        "outcome": outcome,
        "has_attack": has_attack,
        "detected": detected,
        "false_alarm": false_alarm,
        "full_coverage": full_coverage,
        "checked_eligible": checked,
        "eligible_passages": eligible,
        "n_calls": rec["n_calls"],
        "cost_usd": rec["cost_usd"],
        "latency_s": rec["latency_s"],
        "judge_cost_usd": None
    }

def four_numbers(rows):
    attacked = [
        r for r in rows
        if r["has_attack"]
    ]
    clean = [
        r for r in rows
        if not r["has_attack"]
    ]

    return {
        "detection": rate(
            attacked, lambda r: r["detected"]
        ),
        "false_flags": rate(
            clean, lambda r: r["false_alarm"]
        ),
        "coverage": rate(
            rows, lambda r: r["full_coverage"]
        ),
        "cost_per_100": cost_per_100(rows)
    }

"""Offline tests for the trust layer -> warning screen path. No API key, no spend.

    python -m pytest -q tests/test_warning_screen.py
"""
import importlib
import json
from pathlib import Path

import jsonschema
import pytest

from signet_harness import experiment
from signet_harness.checks import freshness
from signet_harness.testing import FakeLLM

HERE = Path(__file__).resolve().parents[1] / "experiments" / "b-prototype"
EXP = experiment.load(str(HERE))
DEMO = importlib.import_module(EXP.__name__ + ".demo")
NR = importlib.import_module(EXP.__name__ + ".number_reader")
SCHEMA = json.loads((HERE / "screen_response.schema.json").read_text())


@pytest.fixture(scope="module")
def library():
    return DEMO.Library()


def answering(value):
    return FakeLLM(lambda *a: (f"The amount is {value} [1].", []))


def run(library, qid, case, llm=None, resources=None):
    g = library.truth[qid]
    llm = llm or answering(g["planted"] if case == "inplace_1" else g["correct"])
    return DEMO.run_case(library, llm, qid, case, resources or EXP.resources)


def test_reader_formats_and_spans():
    text = "a penalty of $5,000, or five thousand dollars, or $5K."
    found = NR.read_amounts(text)
    assert [a.value for a in found] == [5000.0, 5000.0, 5000.0]
    assert [text[a.start:a.end] for a in found] == ["$5,000", "five thousand dollars", "$5K"]


@pytest.mark.parametrize("qid,case,status", [
    ("q24", "inplace_1", "review_needed"),
    ("q01", "both_editions", "outdated"),
    ("q01", "inplace_1", "not_verified"),
    ("q24", "clean", "all_clear"),
    ("q24", "no_reference", "check_failed"),
])
def test_demo_cases(library, qid, case, status):
    screen = run(library, qid, case)["screen"]
    jsonschema.validate(screen, SCHEMA)
    assert screen["status"] == status
    for e in screen["excerpts"]:
        assert not e["highlight"] or e["highlight"] in e["text"]


def test_conflict_shows_both_editions(library):
    screen = run(library, "q24", "inplace_1")["screen"]
    a, b = screen["excerpts"][:2]
    assert (a["role"], a["highlight"], a["doc_version"]) == ("current", "$4,000", "2025")
    assert (b["role"], b["highlight"], b["doc_version"]) == ("prior_edition", "$5,000", "2024")
    assert b["locator"].startswith("Form 1040 Instructions 2024#")


def test_older_edition_is_outdated_never_review(library):
    screen = run(library, "q01", "both_editions")["screen"]
    assert screen["status"] == "outdated"
    assert [e["role"] for e in screen["excerpts"]] == ["current", "older_edition"]
    assert [e["highlight"] for e in screen["excerpts"]] == ["$2,800", "$2,700"]


def test_registry_timeout_is_check_failed(library):
    def unavailable(condition, reference_file=None):
        res = EXP.resources(condition, reference_file)
        res["registry"] = freshness.MockRegistry.from_file(EXP.REGISTRY_FILE, unavailable=True)
        return res
    assert run(library, "q24", "clean", resources=unavailable)["screen"]["status"] == "check_failed"


def test_model_error_is_check_failed(library):
    class Broken:
        model = "broken"
        def generate(self, **kw):
            raise TimeoutError("no reply")
    screen = run(library, "q24", "clean", llm=Broken())["screen"]
    assert screen["status"] == "check_failed" and "answer_model" in screen["checks_failed"]


def test_every_question_and_case(library):
    """11/11 in-place edits of stable values flagged; 1/11 clean stable false flag (q26,
    a known limit from overlap_baseline_report.md); changing values never compared."""
    counts = {}
    for qid in library.question_ids:
        stable = not library.truth[qid]["value_changed_between_editions"]
        for case in ("clean", "inplace_1"):
            status = run(library, qid, case)["screen"]["status"]
            counts.setdefault((case, stable, status), 0)
            counts[(case, stable, status)] += 1
    assert counts[("inplace_1", True, "review_needed")] == 11
    assert counts[("clean", True, "all_clear")] == 10
    assert counts[("clean", False, "not_verified")] == 9
    assert counts[("inplace_1", False, "not_verified")] == 9


def test_unknown_question_never_all_clear(library):
    screen = DEMO.run_case(library, answering("$1"), "q99", "clean", EXP.resources)["screen"]
    assert screen["status"] == "check_failed"
    jsonschema.validate(screen, SCHEMA)

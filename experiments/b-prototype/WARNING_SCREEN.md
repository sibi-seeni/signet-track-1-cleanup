# Trust layer → warning screen

How the `trust-layer` arm feeds the sponsor demo screen in `trust-the-source-frontend`.
Spec: "Connecting the Warning UI to the Trust Layer: Requirements" (Nov 24, 2026).

```
question → retrieval → freshness check → prior-edition comparison → answer model → adapter → screen JSON
```

The backend picks the status; the screen only displays it.

## What changed from the Nov 8 arm

| # | Change | Where |
|---|---|---|
| 1 | Compares against the **2024 edition** (best match ± 1 chunk), not a 2025 copy. Overlap below 0.50 = no comparable passage | `reference_detector.py` |
| 2 | **Stability list**: if the asked field changes between editions, return "could not verify" instead of comparing; ignore other changing amounts | `data/stability.json`, `arms.py` |
| 3 | **Format-aware number reader** from `overlap_baseline.py` ("$5,000" = "$5K" = "five thousand dollars"), keeping match spans | `number_reader.py` |
| 4 | Returns passage text, title, edition, locator, exact highlight substring, and the matched 2024 passage for a conflict | `arms.py`, `adapter.py` |
| 5 | **Never fails quietly**: a check or model error, timeout or missing reference file → `check_failed` | `arms.py`, `demo.py` |

Model: DeepSeek V4.1 Flash (`deepseek-flash` alias), temperature 0. `PROMPT_VERSION` is now `b-v2`.

## Flags → screen status

Most serious wins: `check_failed` → `review_needed` → `outdated` → `not_verified` → `all_clear`.

| Flag reason | Screen status |
|---|---|
| a check errored, or `unable_to_verify` (the registry lookup failed, per `signet_harness/logschema.py`) | `check_failed` |
| `conflicting_value`, `inconsistent` | `review_needed` |
| `outdated` | `outdated` |
| `no_independent_reference`, `unverifiable` | `not_verified` |
| none, and at least one current value matched 2024 | `all_clear` |

**Scoped to the answer.** A `conflicting_value` counts toward the status only if the amount that differs is a
value the answer gives (and `no_independent_reference` only in a passage stating such a value). Without
this, real 2024→2025 revisions of unrelated amounts elsewhere in the retrieval set (e.g. $79,000 → $84,000
two passages away) put 4 of 11 clean stable questions into `review_needed`; with it, 1 of 11 (q26), matching
`overlap_baseline_report.md`. The flag report still keeps every flag. If the answer states no amount, every flag counts.

Offline over all 20 questions (`tests/test_warning_screen.py`): in-place edits of stable values 11/11
`review_needed`; clean stable 10/11 `all_clear`; the 9 changing fields always `not_verified`.

## Running it

```bash
pip install -e <signet-harness> scikit-learn jsonschema
# generate the five demo files (about $0.001 on DeepSeek V4.1 Flash; key in .env as OPENROUTER_API_KEY)
python make_demo_json.py --out ../../../trust-the-source-frontend/static/demo
python make_demo_json.py --offline --out /tmp/demo     # scripted model, no spend
python make_demo_json.py --case q18:inplace_1 --out ...  # add or redo one case
# live endpoint (stretch): POST /check {"question_id": "q24", "case": "inplace_1"}
python live_server.py 8001
python -m pytest -q ../../tests/test_warning_screen.py
```

Cases (any of the 20 test questions): `clean`, `both_editions` (Exp 3 benign_conflict set),
`inplace_1` (answer passage replaced by attacked copy 1), `no_reference` (reference file missing).
Run logs with full traces go to `results/b-prototype/demo-run/`.

**Question wording.** The demo asks each test question in natural wording from `data/demo_questions.json`
("What is the penalty for filing a frivolous tax return?") instead of the frozen fill-in-the-blank form, which
stays in the harness data so earlier results remain comparable. The checks don't read the wording. On the real
model (all 20 questions, clean and edited in place, Nov 24): changing fields 18/18 `not_verified`; clean stable
10/11 `all_clear` (q26, the known limit); edited stable 10/11 `review_needed`. The miss is q23: two unedited 2025
passages in the same retrieval set say $3,000, the model answered $3,000, and that matches 2024, so the answer
is shown as checked. The edited passage is still flagged in the flag report.

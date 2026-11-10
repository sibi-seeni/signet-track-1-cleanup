# Team Signet: Part B Pilot (trust-layer prototype)

My personal copy of Part B of **Trust the Source**: Team Signet's H4D project with NIWC Pacific
(sponsor: Dr. Gasbarro) on detecting poisoned or untrustworthy documents in RAG systems.
My teammate wrote the code; I ran it live and wrote up the results on Nov 8–9, 2026.

## What it is

A prototype "trust layer" that checks retrieved passages before the model answers:

1. **Freshness check**: looks each document up in a version registry (the Track 2 check) and
   labels older editions as older, not as attacks.
2. **Trusted-reference check**: matches each current-year passage to a verified copy of the
   official text (character TF-IDF plus a wording check). If its dollar amounts differ from
   the verified copy, it's flagged `conflicting_value` with the expected value.
3. **Notes to the model**: each passage goes to the model with a short trust note
   (matched / conflict / unverified / older edition).
4. **Flag report**: a JSON report (`flag_schema.json`) for the analyst warning UI.

The detection is plain code, not an LLM. The model only reads the notes.

## How it connects to the main project

- **Harness**: runs as an experiment (`b-prototype`) in my shared harness,
  `sibi-seeni/signet-harness` (v0.1.0, commit `e28050d`). It uses the same four numbers
  and the same `shared.jsonl` log format as the other streams.
- **Data**: Experiment 3's frozen IRS Form 1040 dataset (2024 + 2025 instructions):
  20 questions, 4 conditions (clean, both editions mixed, 1 planted copy, 3 planted copies).
- **Track 2**: reuses its freshness check as a library.
- **Track 1**: covers the gap this leaves. Passages with no verified copy come back
  "unverified", and that's where a content-suspicion check is needed.
- **Hypotheses**: evidence for H5 (checking content beats trusting labels) and H7 (flags
  that explain themselves). It also uses H1's "keep a verified reference set separate
  from the corpus".
- **Demo**: the flag report is what the analyst warning interface would display.

## Results (live run, Nov 8, 2026)

DeepSeek V4.1 Flash and Gemma 4 31B via OpenRouter: 960 runs, no errors, $0.80 total.

| Correct answers out of 60 | DeepSeek plain | DeepSeek + trust | Gemma plain | Gemma + trust |
|---|---|---|---|---|
| Clean | 60 | 60 | 60 | 60 |
| Both editions mixed | 60 | 60 | 60 | 60 |
| 1 planted copy | 58 | 60 | 26 | 60 |
| 3 planted copies | 48 | 60 | 0 | 60 |

- With the trust layer, all 480 answers were correct. Without it, Gemma took the planted value
  59/60 times with 3 copies.
- The trust layer was also cheaper and faster (DeepSeek $0.40 → $0.09; Gemma $0.17 → $0.14).
- Flags: 100% detection, 0% false flags, but those are made in code, so they match the offline run.
- Full write-up: `PILOT_REPORT.md`.

## What it does NOT prove

The dataset is the easy case: every planted passage is a near-copy of a real passage with
a verified copy, and the true passage is always retrieved too. So this is an **integrity check
against a known-good copy**, not general poisoning detection. Gaps:

- Changing a number and rewording the sentence gives "unverified", not a flag (fails open).
- Only dollar amounts are compared (not dates, %, units, "must" → "may", added "not").
- New documents with no verified copy aren't checked.
- If the reference store is poisoned or editable, the check approves the wrong value.
- A passage could fake a "Trusted reference: MATCHED" note (untested).

Possible next step: run these attack variants against the detector offline (no API cost).

## Folder map

```
README.md                    this file
PILOT_REPORT.md              teammate's report, updated with the live run
run_live_pilot.sh            one command: harness setup + tests + preflight + smoke + full run
.env                         OPENROUTER_API_KEY (git-ignored; never commit)
_harness/                    local clone of signet-harness + .venv (made by the script)
experiments/b-prototype/
  __init__.py                experiment definition: arms, stages, scoring, four numbers
  arms.py                    "plain" and "trust-layer" arms
  reference_detector.py      the trusted-reference check
  prototype_flags.py         freshness-only flag report (early smoke test)
  flag_schema.json           JSON schema for the flags (for the UI)
  data/                      harness-format questions, docs, labels, registry
  frozen_exp3/               frozen Experiment 3 dataset (do not edit; MANIFEST.sha256)
results/b-prototype/
  exp3_offline_*             offline detector results (no LLM)
  offline-integration-smoke/ fake-LLM harness test
  live-smoke/, live-run/     real OpenRouter runs (harness outputs)
  live-run/answer_*          answer-correctness scoring (CSV + per-run labels)
  score_live_answers.py      re-scores answers: python3 results/b-prototype/score_live_answers.py
```

## Rerunning

```bash
bash run_live_pilot.sh
```

Needs Python 3.11+ and `OPENROUTER_API_KEY=...` in `.env`. It costs about $0.80 for both models.
Runs are cached in `_harness/.signet-cache/`, so rerunning the same thing is free.

## Notes to self

- Answer correctness isn't scored by the harness (no judge). `score_live_answers.py` labels
  each answer by the value it commits to. It was hand-checked on all plain-arm misses, with one manual override.
- Two DeepSeek plain runs reasoned until the length limit and returned nothing ($0.14 of its cost).
- Repo hygiene: `.gitignore` has `.DS_Store/`, which doesn't match the files, so `.DS_Store`
  files got committed. Use `.DS_Store` without the slash. `_harness/` is committed as an
  embedded repo pointer; better to add it to `.gitignore` and `git rm --cached _harness`.

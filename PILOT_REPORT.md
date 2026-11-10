# Team Signet — Part B Pilot

## Goal
Build a prototype trust layer for detecting
untrustworthy retrieved documents in RAG systems.

## Implemented
1. Harness-level document freshness checking.
2. TF-IDF trusted-reference comparison.
3. Numerical conflict detection.
4. Structured JSON security flags.
5. Plain RAG and trust-layer experiment arms.
6. Offline integration with Signet shared logging.

## Dataset
Experiment 3 frozen dataset:
- 20 questions
- 4 retrieval conditions
- 80 retrieval sets
- 400 retrieved passage appearances

## Preliminary offline findings
- 80/80 poisoned passage appearances detected.
- 0/223 non-poisoned applicable passages flagged.
- 303/303 applicable passages reference-matched.

These results use a controlled dataset with
available paired trusted reference passages.
Repeated passage appearances are not independent
test samples.

## Harness verification
- 123 shared harness tests passed.
- Four fake-LLM integration runs passed.
- Shared JSONL logging validated.

## Live model run (Nov 8, 2026)
Run `live-run` in `results/b-prototype/`, harness
signet-harness 0.1.0 (commit e28050d), through
OpenRouter at temperature 0.
- Models: DeepSeek V4.1 Flash and Gemma 4 31B.
- 80 retrieval sets x 2 arms x 3 seeds = 480 runs
  per model, 960 in total. No API errors.
- Total cost: $0.80.

Flags (made in code, so identical to offline):
detection 100%, false flags 0%, coverage 100%
on both models.

Answer correctness (correct answers / 60 runs):

| Condition | DeepSeek plain | DeepSeek trust | Gemma plain | Gemma trust |
|---|---|---|---|---|
| Clean | 60 | 60 | 60 | 60 |
| 2024 + 2025 editions | 60 | 60 | 60 | 60 |
| 1 planted copy | 58 | 60 | 26 | 60 |
| 3 planted copies | 48 | 60 | 0 | 60 |

What the plain arm got wrong:
- DeepSeek, 1 copy: 1 planted value, 1 empty answer.
- DeepSeek, 3 copies: 11 planted values, 1 empty.
- Gemma, 1 copy: 34 answers listed both values
  as "conflicting" and did not pick one.
- Gemma, 3 copies: 59 planted values, 1 hedged.

With the trust layer, all 480 answers were correct.
Neither arm ever gave the stale 2024 value.

The trust layer was also cheaper and faster,
because the models spent less effort working out
the conflict themselves:

| | Plain | Trust layer |
|---|---|---|
| DeepSeek cost | $0.40 | $0.09 |
| DeepSeek median latency | 2.1 s | 1.5 s |
| Gemma cost | $0.17 | $0.14 |
| Gemma median latency | 21.6 s | 16.3 s |

DeepSeek's plain cost includes two runs ($0.14)
that reasoned until the length limit and returned
no answer.

How answers were scored: the harness only scores
the flags, so answers were labelled separately
with `results/b-prototype/score_live_answers.py`.
Each answer gets the value it commits to (first
bolded value, else first value mentioned); answers
that list both values without choosing are
"hedged". One label was corrected by hand (noted
in the script). Outputs: `live-run/answer_accuracy.csv`
and `live-run/answer_scores.jsonl`.

## Limitations
- Two budget models only; no funded models yet.
- Answer labels come from a rule-based scorer,
  not a judge model or a full hand check.
- Only monetary conflicts are detected by
  the reference comparison.
- Trusted reference corpus is required.
- Historical documents require separate handling.
- Performance on unseen document families unknown.

## Next Steps
- Test in-place edits with no true copy retrieved
  (the next experiment); every planted passage
  here sits beside the true passage.
- Hand-check a sample of answer labels.
- Share the flag schema with the UI teammate.

## Setup
Install the shared Signet harness and
scikit-learn plus jsonschema dependencies.
Copy the included experiment into the
harness experiments directory.
`run_live_pilot.sh` does all of this and runs the
live experiment (key read from `.env`).

#!/usr/bin/env bash
# Team Signet Part B: real-model run of the b-prototype experiment.
# Run from Terminal:  bash ~/Downloads/Team_Signet_Part_B_Pilot/run_live_pilot.sh
# Nothing is committed or pushed. Everything stays in this folder.
set -euo pipefail

PILOT="$(cd "$(dirname "$0")" && pwd)"
H="$PILOT/_harness"

# 1. Get the shared harness (read-only clone, pinned to the tested commit)
if [ ! -d "$H/.git" ]; then
  git clone --quiet https://github.com/sibi-seeni/signet-harness "$H"
fi
git -C "$H" checkout --quiet e28050d

# 2. Python 3.11+ virtual environment
if [ ! -d "$H/.venv" ]; then
  PY=""
  for c in python3.13 python3.12 python3.11; do command -v $c >/dev/null && PY=$c && break; done
  if command -v uv >/dev/null; then uv venv -q -p 3.11 "$H/.venv"
  elif [ -n "$PY" ]; then $PY -m venv "$H/.venv"
  else echo "Need Python 3.11+ (e.g. 'brew install python@3.12' or 'pip install uv')"; exit 1; fi
fi
source "$H/.venv/bin/activate"
if command -v uv >/dev/null; then uv pip install -q -e "$H[test]" scikit-learn jsonschema
else pip install -q -e "$H[test]" scikit-learn jsonschema; fi

# 3. Put the experiment into the harness (fresh copy each time)
rm -rf "$H/experiments/b-prototype"
cp -R "$PILOT/experiments/b-prototype" "$H/experiments/b-prototype"

# 4. Load the key from .env without printing it
export $(tr -d '\r' < "$PILOT/.env" | grep -v '^#' | grep '=' | xargs)
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY not found in $PILOT/.env"; exit 1; }
export SIGNET_EXPERIMENT=b-prototype
cd "$H"

echo "== offline tests";      python -m pytest -q | tail -1
echo "== preflight";          signet preflight --retrieval bm25
echo "== smoke (4 runs/model)"; signet run --stage smoke --retrieval bm25 --name live-smoke
cat results/b-prototype/live-smoke/report.md
echo "== full run (480 runs/model, ~\$0.35-0.85 total)"
signet run --stage run --retrieval bm25 --name live-run
cat results/b-prototype/live-run/report.md

# 5. Copy results back next to the earlier ones
mkdir -p "$PILOT/results/b-prototype"
cp -R results/b-prototype/live-smoke results/b-prototype/live-run "$PILOT/results/b-prototype/"
echo "Done. Results in $PILOT/results/b-prototype/live-run"

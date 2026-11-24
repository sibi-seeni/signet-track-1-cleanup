#!/usr/bin/env python3
"""Run the trust layer on the demo cases and write the JSON the warning screen reads.

    python make_demo_json.py --out ../../../trust-the-source-frontend/static/demo

Writes <question_id>_<case>.json per case plus index.json listing them, and a
run log per case (trace, flag report, cost) to results/b-prototype/demo-run/, for
reading each result by hand. The logs stay out of the served folder.
Needs OPENROUTER_API_KEY (in .env, or injected by a proxy). --offline uses a
scripted model instead, for testing the pipeline without spending anything.

Every response is validated against screen_response.schema.json before it is
written. Nothing here ever falls back to all clear: a failure becomes check_failed.
"""
import argparse
import importlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_env(path):
    """KEY=value lines from .env, without printing them. Existing variables win."""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_experiment():
    from signet_harness import experiment
    return experiment.load(str(HERE))


def make_llm(alias, offline, timeout_s=20):
    if offline:
        from signet_harness.testing import FakeLLM

        def script(system, messages, names, kw):
            # Echo the first dollar amount of the first passage, like a terse model would.
            m = re.search(r"\$[\d,]+(?:\.\d+)?", messages[-1]["content"].split("Retrieved passages", 1)[-1])
            return (f"The amount is {m.group(0)} [1]." if m else "The sources do not say."), []
        return FakeLLM(script, model="offline-script")
    import httpx
    from signet_harness.llm import OpenRouterLLM
    llm = OpenRouterLLM(model=alias, attempts=2)
    llm.http.timeout = httpx.Timeout(timeout_s)
    return llm


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="demo", help="folder for the JSON files (default: ./demo)")
    ap.add_argument("--model", default="deepseek-flash", help="model alias (default: deepseek-flash = DeepSeek V4.1 Flash)")
    ap.add_argument("--runs", default=str(HERE.parents[1] / "results" / "b-prototype" / "demo-run"),
                    help="folder for the per-case run logs (default: results/b-prototype/demo-run)")
    ap.add_argument("--offline", action="store_true", help="scripted model, no API calls")
    ap.add_argument("--case", action="append", metavar="QID:CASE",
                    help="run only these, e.g. --case q24:inplace_1 (default: the five demo cases)")
    args = ap.parse_args(argv)

    for env in (HERE / ".env", HERE.parents[1] / ".env"):
        load_env(env)
    if not args.offline and not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY is not set (put it in .env), or use --offline.")

    exp = load_experiment()
    demo = importlib.import_module(exp.__name__ + ".demo")
    library = demo.Library()
    llm = make_llm(args.model, args.offline)

    if args.case:
        wanted = [tuple(c.split(":", 1)) + (f"{c}",) for c in args.case]
    else:
        wanted = demo.DEMO

    out, runs = Path(args.out), Path(args.runs)
    out.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    index = json.loads(index_path.read_text()) if args.case and index_path.exists() else {"cases": []}
    by_id = {c["id"]: c for c in index["cases"]}
    total = 0.0

    for order, (qid, case, title) in enumerate(wanted, 1):
        result = demo.run_case(library, llm, qid, case, exp.resources)
        screen = result["screen"]
        name = f"{qid}_{case}"
        (out / f"{name}.json").write_text(json.dumps(screen, indent=2, ensure_ascii=False) + "\n")
        (runs / f"{name}.run.json").write_text(json.dumps(
            {k: v for k, v in result.items() if k != "screen"}, indent=2, ensure_ascii=False, default=str) + "\n")
        total += result["cost_usd"] or 0
        by_id[name] = {"id": name, "file": f"{name}.json", "question_id": qid, "case": case,
                       "title": title, "status": screen["status"],
                       "order": by_id.get(name, {}).get("order", order)}
        errors = f"  errors: {result['errors']}" if result["errors"] else ""
        print(f"{name:24s} {screen['status']:14s} {result['latency_s']:5.2f}s  "
              f"answer: {screen['answer'][:70]!r}{errors}")

    index = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": getattr(llm, "model", args.model),
        "cases": sorted(by_id.values(), key=lambda c: (c["order"], c["id"])),
    }
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(wanted)} file(s) and index.json to {out}  (model cost ${total:.4f})")


if __name__ == "__main__":
    main()

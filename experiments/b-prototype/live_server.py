#!/usr/bin/env python3
"""Live check endpoint for the warning screen (the stretch goal).

    python live_server.py [port]          # default 8001, listens on 127.0.0.1 only

    POST /check   {"question_id": "q24", "case": "inplace_1"}  -> the screen JSON
    GET  /cases   the question ids and cases it accepts

Only the 20 test questions are accepted: the stability list and the 2024 reference
cover those and nothing else, so a free-text question would always come back
"could not verify". Cross-origin requests are allowed from localhost only.
The screen uses it with  ?live=http://localhost:8001

Any failure (model error, timeout, missing reference file) comes back as a
check_failed response, never as all clear. Standard library only, plus the harness.
"""
import importlib
import json
import os
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_demo_json import load_env, load_experiment, make_llm  # noqa: E402

LOCAL_ORIGIN = re.compile(r"^http://(localhost|127\.0\.0\.1)(:\d+)?$")
MODEL_TIMEOUT_S = 15   # the screen gives up at 20 s

for env in (HERE / ".env", HERE.parents[1] / ".env"):
    load_env(env)
if not os.environ.get("OPENROUTER_API_KEY"):
    sys.exit("OPENROUTER_API_KEY is not set (put it in .env).")

EXP = load_experiment()
DEMO = importlib.import_module(EXP.__name__ + ".demo")
LIBRARY = DEMO.Library()
LLM = make_llm(os.environ.get("TRUST_LAYER_MODEL", "deepseek-flash"), offline=False, timeout_s=MODEL_TIMEOUT_S)
LLM.attempts = 1


class Handler(BaseHTTPRequestHandler):
    def cors(self):
        origin = self.headers.get("Origin", "")
        if LOCAL_ORIGIN.match(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def send_json(self, body, status=HTTPStatus.OK):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.cors()
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") == "/cases":
            return self.send_json({"question_ids": LIBRARY.question_ids, "cases": sorted(DEMO.CASES)})
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path.rstrip("/") != "/check":
            return self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(min(length, 10_000)) or b"{}")
            qid, case = str(body.get("question_id", "")), str(body.get("case", ""))
        except (ValueError, AttributeError):
            return self.send_json({"error": "send JSON like {\"question_id\": \"q24\", \"case\": \"inplace_1\"}"},
                                  HTTPStatus.BAD_REQUEST)
        if qid not in LIBRARY.truth or case not in DEMO.CASES:
            return self.send_json({"error": f"only the test questions {LIBRARY.question_ids[0]}..{LIBRARY.question_ids[-1]} "
                                            f"and cases {sorted(DEMO.CASES)} are covered"}, HTTPStatus.BAD_REQUEST)
        result = DEMO.run_case(LIBRARY, LLM, qid, case, EXP.resources)
        screen = result["screen"]
        print(f"  {qid}:{case} -> {screen['status']} in {result['latency_s']}s"
              + (f"  errors: {result['errors']}" if result["errors"] else ""), file=sys.stderr)
        self.send_json(screen)

    def log_message(self, fmt, *args):  # quieter than the default access log
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Live checks at http://localhost:{port}/check  (model {LLM.model}; Ctrl+C to stop)")
    print(f"Open the screen with ?live=http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

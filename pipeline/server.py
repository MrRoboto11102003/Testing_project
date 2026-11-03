"""Local web UI for browsing instances and running tests / generation.

    python -m pipeline.server [--port 8765]

then open http://localhost:8765. Jobs run one at a time (they share checkouts).
"""

import argparse
import json
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import config, generate
from .harness import load_instance, materialize, run_pytest

UI_FILE = Path(__file__).with_name("ui.html")
DEFAULT_MODEL = "claude-haiku-4-5"

JOBS: dict[str, dict] = {}
RUN_LOCK = threading.Lock()  # checkouts are shared, so one job at a time


def runs_dir(model: str) -> Path:
    return config.DATA / "runs" / model


def has_public_target(meta: dict) -> bool:
    return any(not s.split("::")[1].split(".")[-1].startswith("_") for s in meta["target_symbols"])


def list_instances() -> list[dict]:
    rows = []
    for line in config.INSTANCES.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        m = json.loads(line)
        rows.append({
            "id": m["id"],
            "issue": m["issue"],
            "title": m["issue_title"],
            "f2p": len(m["fail_to_pass"]),
            "patch_lines": m["patch_lines"],
            "port_method": m["port_method"].split(":")[0],
            "targets": m["target_symbols"],
            "public": has_public_target(m),
            "runs": {
                p.parent.parent.name: summarize_result(json.loads(p.read_text(encoding="utf-8")))
                for p in (config.DATA / "runs").glob(f"*/{m['id']}/result.json")
            },
        })
    return rows


def summarize_result(r: dict) -> dict:
    return {k: r.get(k) for k in ("detected", "n_tests", "n_catching", "n_false_alarms", "cost_usd", "seconds")}


def instance_detail(iid: str) -> dict:
    meta = load_instance(iid)
    folder = config.INSTANCE_DIR / iid
    runs = {}
    for result_path in (config.DATA / "runs").glob(f"*/{iid}/result.json"):
        run_folder = result_path.parent
        tests = sorted(run_folder.glob("test_*.py"))
        prompt = run_folder / "prompt.txt"
        runs[run_folder.parent.name] = {
            "result": json.loads(result_path.read_text(encoding="utf-8")),
            "test_code": tests[0].read_text(encoding="utf-8") if tests else "",
            "prompt": prompt.read_text(encoding="utf-8") if prompt.exists() else "",
        }
    return {
        "meta": meta,
        "bug_patch": (folder / "bug.patch").read_text(encoding="utf-8"),
        "fix_tests_patch": (folder / "fix_tests.patch").read_text(encoding="utf-8"),
        "runs": runs,
    }


def overall_stats() -> dict:
    instances = list_instances()
    by_model: dict[str, list[dict]] = {}
    for inst in instances:
        for model, r in inst["runs"].items():
            by_model.setdefault(model, []).append(r)
    models = {}
    for model, results in by_model.items():
        n = len(results)
        models[model] = {
            "runs": n,
            "detected": sum(bool(r["detected"]) for r in results),
            "with_false_alarms": sum((r["n_false_alarms"] or 0) > 0 for r in results),
            "tests": sum(r["n_tests"] or 0 for r in results),
            "false_alarm_tests": sum(r["n_false_alarms"] or 0 for r in results),
            "cost_usd": round(sum(r["cost_usd"] or 0 for r in results), 4),
        }
    return {
        "instances": len(instances),
        "public": sum(i["public"] for i in instances),
        "models": models,
    }


# ---- jobs -------------------------------------------------------------------

def start_job(kind: str, iid: str, model: str) -> str:
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "kind": kind, "instance": iid, "model": model, "status": "queued",
           "log": [], "result": None, "error": None, "started": time.time()}
    JOBS[jid] = job
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return jid


def log(job: dict, msg: str) -> None:
    job["log"].append(f"{time.time() - job['started']:6.1f}s  {msg}")


def run_job(job: dict) -> None:
    if RUN_LOCK.locked():
        log(job, "waiting for the previous job to finish ...")
    with RUN_LOCK:
        job["status"] = "running"
        try:
            if job["kind"] == "fix_tests":
                job["result"] = run_fix_tests(job)
            elif job["kind"] == "generate":
                job["result"] = run_generate(job)
            else:
                raise ValueError(f"unknown job kind {job['kind']}")
            job["status"] = "done"
            log(job, "done")
        except Exception as exc:
            job["status"] = "error"
            job["error"] = f"{exc}\n\n{traceback.format_exc()[-2000:]}"
            log(job, f"error: {exc}")


def run_fix_tests(job: dict) -> dict:
    meta = load_instance(job["instance"])
    targets = meta["fail_to_pass"] + meta["pass_to_pass"]
    files = sorted({t.split("::")[0] for t in targets})
    out = {}
    for variant in ("clean", "buggy"):
        log(job, f"checking out {config.BASE_REF} ({variant}) ...")
        path = materialize(variant, job["instance"], slot="ui")
        log(job, f"running {len(files)} test file(s) on {variant} ...")
        res = run_pytest(path, files, label=variant)
        out[variant] = res["outcomes"]
        log(job, f"{variant}: {sum(o == 'passed' for o in res['outcomes'].values())} passed, "
                 f"{sum(o in ('failed', 'error') for o in res['outcomes'].values())} failed")
    f2p = set(meta["fail_to_pass"])
    tests = {}
    for node in sorted(set(out["clean"]) | set(out["buggy"])):
        c, b = out["clean"].get(node, "missing"), out["buggy"].get(node, "missing")
        tests[node] = {"clean": c, "buggy": b, "fail_to_pass": node in f2p,
                       "detects_bug": c == "passed" and b in ("failed", "error"),
                       "false_alarm": c in ("failed", "error")}
    return {"kind": "fix_tests", "tests": tests,
            "detected": any(t["detects_bug"] for t in tests.values())}


def run_generate(job: dict) -> dict:
    meta = load_instance(job["instance"])
    if not has_public_target(meta):
        raise ValueError("this bug only touches private helpers; there are no public docs to prompt with")
    log(job, f"building prompt from docs for {', '.join(meta['target_symbols'])}")
    log(job, f"asking {job['model']} via `claude -p` (usually 30-70s) ...")
    original_check = generate.check

    def check_with_log(iid, test_file):
        log(job, "model replied; running generated tests on clean and buggy ...")
        return original_check(iid, test_file)

    generate.check = check_with_log
    try:
        result = generate.run_one(meta, job["model"], runs_dir(job["model"]))
    finally:
        generate.check = original_check
    log(job, f"{result['n_tests']} tests: {result['n_catching']} catch the bug, "
             f"{result['n_false_alarms']} false alarms")
    return {"kind": "generate", **result}


# ---- http -------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                body = UI_FILE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/instances":
                self.send_json(list_instances())
            elif path == "/api/stats":
                self.send_json(overall_stats())
            elif path.startswith("/api/instance/"):
                self.send_json(instance_detail(path.rsplit("/", 1)[1]))
            elif path.startswith("/api/jobs/"):
                job = JOBS.get(path.rsplit("/", 1)[1])
                self.send_json(job or {"error": "no such job"}, 200 if job else 404)
            else:
                self.send_json({"error": "not found"}, 404)
        except FileNotFoundError as exc:
            self.send_json({"error": str(exc)}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            return self.send_json({"error": "not found"}, 404)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        kind, iid = body.get("kind"), body.get("instance")
        if kind not in ("fix_tests", "generate") or not (config.INSTANCE_DIR / str(iid)).is_dir():
            return self.send_json({"error": "bad request"}, 400)
        self.send_json({"job": start_job(kind, iid, body.get("model") or DEFAULT_MODEL)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"arnio bug lab running at {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

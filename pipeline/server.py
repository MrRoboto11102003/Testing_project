"""Local web UI: pick a model, a context mode and a set of bugs, run test generation,
and browse the results.

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
from urllib.parse import parse_qs, urlparse

from . import config, generate, metrics
from .harness import load_instance, materialize, run_pytest

UI_FILE = Path(__file__).with_name("ui.html")
DEFAULT_MODEL = "qwen3.5:4b"
# Offered in the UI. Names starting with "claude" run through `claude -p`; anything else is
# treated as a local Ollama tag (see generate.call_ollama). Any such id can be typed in instead.
# Only models trained before arnio existed (May 2026) give contamination-free results.
MODELS = ["qwen3.5:4b", "claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-1"]

JOBS: dict[str, dict] = {}
RUN_LOCK = threading.Lock()  # checkouts are shared, so one job at a time
RUN_INDEX = config.DATA / "runs" / "index.json"  # run folder -> short stable id
INDEX_LOCK = threading.Lock()


def run_times() -> dict[str, tuple[float, float]]:
    """First and last result.json write time per run folder."""
    times: dict[str, list[float]] = {}
    for p in (config.DATA / "runs").glob("*/*/result.json"):
        times.setdefault(p.parent.parent.name, []).append(p.stat().st_mtime)
    return {run: (min(ts), max(ts)) for run, ts in times.items()}


def run_ids(times: dict[str, tuple[float, float]]) -> dict[str, int]:
    """Short ids for run folders, kept in data/runs/index.json. New folders get the next numbers,
    in order of their first result, so ids never change once given."""
    with INDEX_LOCK:
        index = json.loads(RUN_INDEX.read_text(encoding="utf-8")) if RUN_INDEX.exists() else {}
        new = sorted((r for r in times if r not in index), key=lambda r: times[r][0])
        if new:
            nxt = max(index.values(), default=0) + 1
            for r in new:
                index[r] = nxt
                nxt += 1
            RUN_INDEX.write_text(json.dumps(index, indent=1), encoding="utf-8")
        return index


def has_public_target(meta: dict) -> bool:
    return bool(generate.public_symbols(meta))


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
    return {k: r.get(k) for k in ("model", "level", "mean_arnio_calls", "detected", "n_tests", "n_catching", "n_false_alarms",
                                   "cost_usd", "seconds")}


def instance_detail(iid: str) -> dict:
    meta = load_instance(iid)
    folder = config.INSTANCE_DIR / iid
    runs = {}
    for result_path in (config.DATA / "runs").glob(f"*/{iid}/result.json"):
        run_folder = result_path.parent
        tests = sorted(run_folder.glob("test_*.py"))
        prompt = run_folder / "prompt.txt"
        metrics_path = run_folder / "metrics.json"
        runs[run_folder.parent.name] = {
            "result": json.loads(result_path.read_text(encoding="utf-8")),
            "metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None,
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
    by_run: dict[str, list[dict]] = {}
    for inst in instances:
        for run, r in inst["runs"].items():
            by_run.setdefault(run, []).append(r)
    times = run_times()
    ids = run_ids(times)
    runs = {}
    # most recently updated first
    for run, results in sorted(by_run.items(), key=lambda kv: -times.get(kv[0], (0, 0))[1]):
        n = len(results)
        first, last = times.get(run, (None, None))
        runs[run] = {
            "id": ids.get(run),
            "first": first,
            "last": last,
            "runs": n,
            "detected": sum(bool(r["detected"]) for r in results),
            "with_false_alarms": sum((r["n_false_alarms"] or 0) > 0 for r in results),
            "tests": sum(r["n_tests"] or 0 for r in results),
            "false_alarm_tests": sum(r["n_false_alarms"] or 0 for r in results),
            "cost_usd": round(sum(r["cost_usd"] or 0 for r in results), 4),
            "calls_per_test": round(sum(r.get("mean_arnio_calls") or 0 for r in results) / n, 1),
        }
    return {
        "instances": len(instances),
        "public": sum(i["public"] for i in instances),
        "runs": runs,
    }


def options() -> dict:
    return {"models": MODELS, "contexts": list(generate.CONTEXT_MODES), "levels": list(generate.LEVELS),
            "default_model": DEFAULT_MODEL}


def preview_context(iid: str, mode: str, keep: float, seed: int, level: str = "unit",
                    cache_docs: bool = False, with_source: bool = False) -> dict:
    meta = load_instance(iid)
    text, stats = generate.build_context(meta, mode, keep, seed, with_source)
    if cache_docs:
        shared, user = generate.build_cached_prompt(meta, text, mode, level)
        return {"prompt": f"=== SYSTEM (appended, cached across bugs) ===\n{shared}\n\n=== USER (per bug) ===\n{user}",
                "stats": {**stats, "shared_chars": len(shared), "per_bug_chars": len(user)}}
    return {"prompt": generate.build_prompt(meta, text, level), "stats": stats}


def metrics_summary() -> dict:
    """Per-run test-quality summary from metrics.json files (see pipeline.metrics)."""
    return {run: metrics.summarize(rows) for run, rows in sorted(metrics.collect().items())}


# ---- jobs -------------------------------------------------------------------

def start_job(kind: str, instances: list[str], opts: dict) -> str:
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "kind": kind, "instances": instances, "opts": opts, "status": "queued",
           "log": [], "results": [], "done": 0, "error": None, "cancel": False, "started": time.time(),
           "current": None, "current_started": None, "finished": None}
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
        for iid in job["instances"]:
            if job["cancel"]:
                log(job, "cancelled")
                job["status"] = "cancelled"
                job["current"], job["finished"] = None, time.time()
                return
            job["current"], job["current_started"] = iid, time.time()
            try:
                res = run_fix_tests(job, iid) if job["kind"] == "fix_tests" else run_generate(job, iid)
                job["results"].append(res)
            except Exception as exc:
                job["results"].append({"id": iid, "error": str(exc)})
                log(job, f"{iid}: error: {exc}")
                job["error"] = traceback.format_exc()[-2000:]
            job["done"] += 1
        job["current"] = None
        job["status"] = "done"
        job["finished"] = time.time()
        n_det = sum(bool(r.get("detected")) for r in job["results"])
        log(job, f"finished: {n_det}/{len(job['results'])} bugs detected")


def run_fix_tests(job: dict, iid: str) -> dict:
    meta = load_instance(iid)
    targets = meta["fail_to_pass"] + meta["pass_to_pass"]
    files = sorted({t.split("::")[0] for t in targets})
    out = {}
    for variant in ("clean", "buggy"):
        log(job, f"checking out {config.BASE_REF} ({variant}) ...")
        path = materialize(variant, iid, slot="ui")
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
    return {"id": iid, "kind": "fix_tests", "tests": tests,
            "detected": any(t["detects_bug"] for t in tests.values())}


def run_generate(job: dict, iid: str) -> dict:
    meta = load_instance(iid)
    if not has_public_target(meta):
        raise ValueError("this bug only touches private helpers; there are no public docs to prompt with")
    o = job["opts"]
    result = generate.run_one(meta, o["model"], o["context"], o["keep"], o["seed"],
                              log=lambda m: log(job, m), level=o["level"],
                              cache_docs=o["cache_docs"], max_thinking=o["max_thinking"],
                              with_source=o.get("with_source", False))
    log(job, f"{iid}: {'DETECTED' if result['detected'] else 'missed'} - {result['n_tests']} tests, "
             f"{result['n_catching']} catch the bug, {result['n_false_alarms']} false alarms")
    # Test-quality metrics (coverage etc., see pipeline.metrics) right away, so a UI run never
    # leaves gaps. No model calls; a failure here doesn't fail the bug.
    run_dir = config.DATA / "runs" / (result.get("run") or o["run"]) / iid
    if next(run_dir.glob("test_*.py"), None):
        log(job, f"{iid}: measuring coverage ...")
        try:
            m = metrics.measure(run_dir, slot="ui-metrics")
            log(job, f"{iid}: {'reached' if m['reached_bug'] else 'never reached'} the bug lines · "
                     f"target lines {m['target_line_cov']} · branches {m['target_branch_cov']}")
        except Exception as exc:
            log(job, f"{iid}: metrics failed: {exc}")
    return {k: v for k, v in result.items() if k != "tests"}


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
        url = urlparse(self.path)
        path, query = url.path, url.query
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
            elif path == "/api/metrics":
                self.send_json(metrics_summary())
            elif path == "/api/options":
                self.send_json(options())
            elif path == "/api/preview":
                q = parse_qs(query)
                self.send_json(preview_context(q["instance"][0], q.get("context", ["targeted"])[0],
                                               float(q.get("keep", ["0.5"])[0]), int(q.get("seed", ["0"])[0]),
                                               q.get("level", ["unit"])[0],
                                               q.get("cache_docs", ["0"])[0] in ("1", "true"),
                                               q.get("with_source", ["0"])[0] in ("1", "true")))
            elif path.startswith("/api/instance/"):
                self.send_json(instance_detail(path.rsplit("/", 1)[1]))
            elif path == "/api/jobs/latest":
                # lets a reloaded page pick the running (or last) job back up
                job = max(JOBS.values(), key=lambda j: j["started"], default=None)
                self.send_json(job or {})
            elif path.startswith("/api/jobs/"):
                job = JOBS.get(path.rsplit("/", 1)[1])
                self.send_json(job or {"error": "no such job"}, 200 if job else 404)
            else:
                self.send_json({"error": "not found"}, 404)
        except FileNotFoundError as exc:
            self.send_json({"error": str(exc)}, 404)
        except (KeyError, ValueError) as exc:
            self.send_json({"error": str(exc)}, 400)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            job = JOBS.get(path.split("/")[3])
            if job:
                job["cancel"] = True
            return self.send_json({"ok": bool(job)})
        if path != "/api/run":
            return self.send_json({"error": "not found"}, 404)
        kind = body.get("kind")
        instances = [i for i in body.get("instances") or [] if (config.INSTANCE_DIR / str(i)).is_dir()]
        if kind not in ("fix_tests", "generate") or not instances:
            return self.send_json({"error": "need kind and at least one known instance"}, 400)
        context = body.get("context") or "targeted"
        if context not in generate.CONTEXT_MODES:
            return self.send_json({"error": f"unknown context {context}"}, 400)
        level = body.get("level") or "unit"
        if level not in generate.LEVELS:
            return self.send_json({"error": f"unknown level {level}"}, 400)
        try:
            opts = {"model": (body.get("model") or DEFAULT_MODEL).strip(), "context": context,
                    "keep": float(body.get("keep", 0.5)), "seed": int(body.get("seed", 0)), "level": level,
                    "cache_docs": bool(body.get("cache_docs")),
                    "max_thinking": None if body.get("max_thinking") in (None, "") else int(body["max_thinking"]),
                    "with_source": bool(body.get("with_source"))}
        except (TypeError, ValueError):
            return self.send_json({"error": "keep and max_thinking must be numbers, seed an integer"}, 400)
        if opts["with_source"] and context not in ("full", "partial"):
            return self.send_json({"error": "target code can only be added to full or partial docs "
                                            "(use the 'source' context for targeted + code)"}, 400)
        opts["run"] = generate.run_label(opts["model"], context, opts["keep"], opts["seed"], level,
                                         opts["cache_docs"], opts["max_thinking"],
                                         opts["with_source"]) if kind == "generate" else None
        self.send_json({"job": start_job(kind, instances, opts), "run": opts["run"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"TestGen Bench running at {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

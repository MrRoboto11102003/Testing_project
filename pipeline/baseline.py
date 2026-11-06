"""Baseline: how well does arnio's own v1.20.0 test suite catch the ported bugs?

    python -m pipeline.baseline [--limit 50] [--workers 4]

No model calls. Runs the repo's tests/ on clean once (tests already failing there can't
count as catches), once under coverage (target / package coverage, bug lines reached),
and once per bug on the buggy checkout. Each bug's result is cached in
data/baseline/<id>.json, so it can be resumed. Writes data/results_baseline.csv and
prints the suite next to the partial50 runs (with and without code) on the same bugs.
"""

import argparse
import csv
import json
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config, generate
from .harness import load_instance, materialize, run_pytest
from .metrics import _ratio, assertion_counts, bug_lines, target_ranges

OUT = config.DATA / "baseline"
SUITE = ["tests"]
RCFILE = """[run]
branch = True
source = arnio
data_file = .coverage-baseline
"""
COMPARE = ["claude-haiku-4-5__partial50_s0__cachedocs", "claude-haiku-4-5__partial50_s0__src__cachedocs"]


def clean_outcomes() -> dict:
    p = OUT / "clean.json"
    if not p.exists():
        r = run_pytest(materialize("clean", slot="base-w0"), SUITE, "clean")
        p.write_text(json.dumps(r, indent=1), encoding="utf-8")
    return json.loads(p.read_text(encoding="utf-8"))


def suite_coverage() -> dict:
    p = OUT / "coverage.json"
    if not p.exists():
        path = materialize("clean", slot="base-w0")
        (path / ".coveragerc-baseline").write_text(RCFILE, encoding="utf-8")
        common = dict(cwd=path, capture_output=True, text=True, encoding="utf-8", errors="replace")
        subprocess.run([config.PYTHON, "-m", "coverage", "run", "--rcfile=.coveragerc-baseline", "-m", "pytest",
                        *SUITE, "-q", "-p", "no:cacheprovider", "-o", "addopts="],
                       timeout=config.TEST_TIMEOUT_S * 10, **common)
        out = path / ".coverage-baseline.json"
        out.unlink(missing_ok=True)
        subprocess.run([config.PYTHON, "-m", "coverage", "json", "--rcfile=.coveragerc-baseline", "-o", out.name],
                       **common)
        p.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    return json.loads(p.read_text(encoding="utf-8"))


def suite_asserts(path: Path) -> tuple[int, int]:
    """(test functions, assertions) over the suite's files, static."""
    n_tests = n_asserts = 0
    for f in (path / "tests").rglob("test_*.py"):
        counts = assertion_counts(f.read_text(encoding="utf-8", errors="replace"))
        n_tests += len(counts)
        n_asserts += sum(counts.values())
    return n_tests, n_asserts


def coverage_row(iid: str, cov: dict) -> dict:
    files = {k.replace("\\", "/"): v for k, v in cov["files"].items()}
    t_hit = t_tot = b_hit = b_tot = 0
    for rel, ranges in target_ranges(load_instance(iid)).items():
        f = files.get(rel)
        if not f:
            continue
        inside = lambda n: any(a <= n <= b for a, b in ranges)  # noqa: E731
        ex = sum(inside(n) for n in f["executed_lines"])
        mi = sum(inside(n) for n in f["missing_lines"])
        t_hit, t_tot = t_hit + ex, t_tot + ex + mi
        eb = sum(inside(a) for a, _ in f.get("executed_branches", []))
        mb = sum(inside(a) for a, _ in f.get("missing_branches", []))
        b_hit, b_tot = b_hit + eb, b_tot + eb + mb
    reached = False
    for rel, lines in bug_lines(iid).items():
        f = files.get(rel)
        if f and lines & set(f["executed_lines"]):
            reached = True
    return {"target_line_cov": _ratio(t_hit, t_tot), "target_branch_cov": _ratio(b_hit, b_tot),
            "reached_bug": reached}


def buggy_row(iid: str, clean: dict, slot: str) -> dict:
    p = OUT / f"{iid}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    r = run_pytest(materialize("buggy", iid, slot=slot), SUITE, "buggy")
    c = clean["outcomes"]
    catching = sorted(n for n, o in r["outcomes"].items()
                      if c.get(n) == "passed" and o in ("failed", "error"))
    row = {"id": iid, "detected": bool(catching), "n_catching": len(catching), "catching": catching[:50],
           "buggy_collection_errors": r["collection_errors"], "n_buggy_tests": len(r["outcomes"])}
    p.write_text(json.dumps(row, indent=1), encoding="utf-8")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ids = generate.eligible_ids()[: args.limit]

    clean = clean_outcomes()
    n_clean_fail = sum(o in ("failed", "error") for o in clean["outcomes"].values())
    print(f"suite on clean: {len(clean['outcomes'])} tests, {n_clean_fail} already failing", flush=True)
    print("coverage run on clean ...", flush=True)
    cov = suite_coverage()

    slots: queue.Queue[str] = queue.Queue()
    for w in range(args.workers):
        materialize("clean", slot=f"base-w{w}")
        slots.put(f"base-w{w}")

    def work(iid):
        slot = slots.get()
        try:
            return buggy_row(iid, clean, slot)
        except Exception as exc:
            return {"id": iid, "error": str(exc)}
        finally:
            slots.put(slot)

    rows = []
    with ThreadPoolExecutor(args.workers) as pool:
        for k, row in enumerate(pool.map(work, ids), 1):
            if "error" in row:
                print(f"[{k}/{len(ids)}] {row['id']:14} ERROR {row['error'][:200]}", flush=True)
                continue
            row.update(coverage_row(row["id"], cov))
            rows.append(row)
            print(f"[{k}/{len(ids)}] {row['id']:14} {'CAUGHT' if row['detected'] else 'missed':6} "
                  f"catching={row['n_catching']:3} tgt_line={row['target_line_cov']} "
                  f"reached={row['reached_bug']}", flush=True)

    with (config.DATA / "results_baseline.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = ["id", "detected", "n_catching", "target_line_cov", "target_branch_cov", "reached_bug"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n_tests, n_asserts = suite_asserts(materialize("clean", slot="base-w0"))
    mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else None  # noqa: E731
    print(f"\n{'':40} {'bugs':>4} {'caught':>6} {'reached':>7} {'reach+miss':>10} {'tgt line':>8} {'tgt br':>6}")
    print(f"{'arnio v1.20.0 suite':40} {len(rows):4} {sum(r['detected'] for r in rows):6} "
          f"{sum(r['reached_bug'] for r in rows):7} "
          f"{sum(r['reached_bug'] and not r['detected'] for r in rows):10} "
          f"{mean([r['target_line_cov'] for r in rows if r['target_line_cov'] is not None]):8} "
          f"{mean([r['target_branch_cov'] for r in rows if r['target_branch_cov'] is not None]):6}")
    done = {r["id"] for r in rows}
    for run in COMPARE:
        ms = [json.loads(p.read_text(encoding="utf-8")) for p in (config.DATA / "runs" / run).glob("*/metrics.json")]
        ms = [m for m in ms if m["id"] in done]
        if ms:
            print(f"{run.replace('claude-haiku-4-5__', 'haiku '):40} {len(ms):4} {sum(m['detected'] for m in ms):6} "
                  f"{sum(m['reached_bug'] for m in ms):7} {sum(m['reached_but_missed'] for m in ms):10} "
                  f"{mean([m['target_line_cov'] for m in ms if m['target_line_cov'] is not None]):8} "
                  f"{mean([m['target_branch_cov'] for m in ms if m['target_branch_cov'] is not None]):6}")
    print(f"\nsuite: {n_tests} test functions, {n_asserts / max(n_tests, 1):.2f} asserts/test, "
          f"package line coverage {cov['totals']['percent_covered'] / 100:.2f}")


if __name__ == "__main__":
    main()

"""Baseline: how well does the project's own BASE_REF test suite catch the ported bugs?

    python -m pipeline.baseline [--limit 50] [--workers 4]
    TP_PROJECT=openlens python -m pipeline.baseline ...

No model calls. Runs the repo's tests/ on clean once (tests already failing there can't
count as catches), once under coverage with per-test contexts, and once per bug on the
buggy checkout.

The fixes were merged before BASE_REF, so the suite already holds each bug's own
regression tests. Those would catch "their" bug by construction, so for each bug every
test function that its fix_tests.patch added or changed (plus its fail_to_pass tests)
is left out: it doesn't count as a catch, and its coverage doesn't count either. The
unfiltered numbers are kept next to them (`*_with_fix`). Fix tests that were later
renamed or moved to another file are not recognised and stay in.

Branch coverage can't be split per test, so target_branch_cov_with_fix is for the whole suite.

Each bug's buggy-run result is cached in DATA/baseline/<id>.json, so it can be resumed.
Writes DATA/results_baseline.csv and prints the suite next to the 50-bug model runs.
"""

import argparse
import ast
import csv
import json
import os
import queue
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config, generate
from .gitutil import git_ok
from .harness import load_instance, materialize, run_pytest
from .metrics import _ratio, assertion_counts, bug_lines, target_ranges

OUT = config.DATA / "baseline"
SUITE = ["tests"]
RCFILE = f"""[run]
branch = True
source = {config.PACKAGE}
dynamic_context = test_function
data_file = .coverage-baseline
"""
CACHE_VERSION = 2  # v2: full lists of failing tests (v1 truncated `catching` at 50)


def _env(path: Path) -> dict:
    return dict(os.environ, **config.TEST_ENV,
                PYTHONPATH=str(path) + os.pathsep + os.environ.get("PYTHONPATH", ""))


def clean_outcomes() -> dict:
    p = OUT / "clean.json"
    if not p.exists():
        r = run_pytest(materialize("clean", slot="base-w0"), SUITE, "clean")
        p.write_text(json.dumps(r, indent=1), encoding="utf-8")
    return json.loads(p.read_text(encoding="utf-8"))


def suite_coverage() -> dict:
    """Coverage JSON of the suite on clean, with the tests that ran each line."""
    p = OUT / "coverage_ctx.json"
    if not p.exists():
        path = materialize("clean", slot="base-w0")
        (path / ".coveragerc-baseline").write_text(RCFILE, encoding="utf-8")
        common = dict(cwd=path, capture_output=True, text=True, encoding="utf-8", errors="replace", env=_env(path))
        subprocess.run([config.PYTHON, "-m", "coverage", "run", "--rcfile=.coveragerc-baseline", "-m", "pytest",
                        *SUITE, "-q", "-p", "no:cacheprovider", "-o", "addopts="],
                       timeout=config.TEST_TIMEOUT_S * 10, **common)
        out = path / ".coverage-baseline.json"
        out.unlink(missing_ok=True)
        proc = subprocess.run([config.PYTHON, "-m", "coverage", "json", "--rcfile=.coveragerc-baseline",
                               "--show-contexts", "-o", out.name], **common)
        if not out.exists():
            raise RuntimeError(f"coverage json failed: {(proc.stdout + proc.stderr)[-500:]}")
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


# ---- which tests belong to the fix -------------------------------------------

def _test_funcs(tree: ast.AST):
    """(name, first_line, last_line) of every test function, top level or in a class."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            yield node.name, first, node.end_lineno


def fix_tests(iid: str) -> set[tuple[str, str]]:
    """{(test file at BASE_REF, test function name)} the fix added or changed."""
    meta = load_instance(iid)
    out = {(n.split("::")[0], n.split("::")[-1].split("[")[0]) for n in meta["fail_to_pass"]}
    patch = (config.INSTANCE_DIR / iid / "fix_tests.patch").read_text(encoding="utf-8")
    touched: dict[str, set[int]] = {}  # post-fix path -> new-side lines added/next to a removal
    rel, new_line = None, 0
    for line in patch.splitlines():
        if line.startswith("--- "):
            continue
        if line.startswith("+++ "):
            path = line[4:].strip()
            rel = None if path == "/dev/null" else (path[2:] if path.startswith("b/") else path)
            continue
        m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if m:
            new_line = int(m.group(1))
            continue
        if rel is None or not new_line:
            continue
        if line.startswith("+"):
            touched.setdefault(rel, set()).add(new_line)
            new_line += 1
        elif line.startswith("-"):
            touched.setdefault(rel, set()).add(new_line)
        elif line.startswith(" ") or line == "":
            new_line += 1
    for rel, lines in touched.items():
        ok, src = git_ok("show", f"{meta['fix_commit']}:{rel}")
        if not ok:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for name, a, b in _test_funcs(tree):
            if any(a <= n <= b for n in lines):
                out.add((config.alias_path(rel), name))
    return out


def _node_key(nodeid: str) -> tuple[str, str]:
    return nodeid.split("::")[0], nodeid.split("::")[-1].split("[")[0]


def _ctx_key(ctx: str) -> tuple[str, str]:
    """'tests.test_x.TestC.test_y' -> ('test_x', 'test_y'): module stem and function name.

    The module part depends on how pytest imported it, so only its last component is used."""
    parts = ctx.split("|")[0].split(".")
    mod = [p for p in parts[:-1] if not p[:1].isupper()]
    return (mod[-1] if mod else ""), parts[-1]


# ---- per bug -----------------------------------------------------------------

def buggy_row(iid: str, clean: dict, slot: str) -> dict:
    p = OUT / f"{iid}.json"
    if p.exists():
        row = json.loads(p.read_text(encoding="utf-8"))
        if row.get("cache_version") == CACHE_VERSION:
            return row
        if row["n_catching"] == len(row["catching"]):  # v1 list was complete: upgrade in place
            row["cache_version"] = CACHE_VERSION
            p.write_text(json.dumps(row, indent=1), encoding="utf-8")
            return row
    r = run_pytest(materialize("buggy", iid, slot=slot), SUITE, "buggy")
    c = clean["outcomes"]
    catching = sorted(n for n, o in r["outcomes"].items()
                      if c.get(n) == "passed" and o in ("failed", "error"))
    row = {"id": iid, "cache_version": CACHE_VERSION, "n_catching": len(catching), "catching": catching,
           "buggy_collection_errors": r["collection_errors"], "n_buggy_tests": len(r["outcomes"])}
    p.write_text(json.dumps(row, indent=1), encoding="utf-8")
    return row


def score(row: dict, cov: dict) -> dict:
    iid = row["id"]
    excluded = fix_tests(iid)
    excluded_ctx = {(Path(f).stem, n) for f, n in excluded}
    catching = [n for n in row["catching"] if _node_key(n) not in excluded]
    files = {k.replace("\\", "/"): v for k, v in cov["files"].items()}

    def executed(f: dict, drop_fix: bool) -> set[int]:
        if not drop_fix:
            return set(f["executed_lines"])
        ctxs = f.get("contexts", {})
        # "" = import time, which isn't any one test's doing.
        return {n for n in f["executed_lines"]
                if any(cx == "" or _ctx_key(cx) not in excluded_ctx for cx in ctxs.get(str(n), [""]))}

    out = {"id": iid, "n_fix_tests": len(excluded),
           "detected_with_fix": bool(row["catching"]), "n_catching_with_fix": len(row["catching"]),
           "detected": bool(catching), "n_catching": len(catching), "catching": catching[:20]}
    for suffix, drop in (("", True), ("_with_fix", False)):
        t_hit = t_tot = 0
        for rel, ranges in target_ranges(load_instance(iid)).items():
            f = files.get(rel)
            if not f:
                continue
            inside = lambda n: any(a <= n <= b for a, b in ranges)  # noqa: E731
            ex = executed(f, drop)
            stmts = set(f["executed_lines"]) | set(f["missing_lines"])
            t_hit += sum(inside(n) for n in stmts & ex)
            t_tot += sum(inside(n) for n in stmts)
        reached = any((f := files.get(rel)) and lines & executed(f, drop) for rel, lines in bug_lines(iid).items())
        out["target_line_cov" + suffix] = _ratio(t_hit, t_tot)
        out["reached_bug" + suffix] = bool(reached)
    b_hit = b_tot = 0
    for rel, ranges in target_ranges(load_instance(iid)).items():
        f = files.get(rel)
        if f:
            inside = lambda n: any(a <= n <= b for a, b in ranges)  # noqa: E731
            b_hit += sum(inside(a) for a, _ in f.get("executed_branches", []))
            b_tot += sum(inside(a) for a, _ in f.get("executed_branches", []) + f.get("missing_branches", []))
    out["target_branch_cov_with_fix"] = _ratio(b_hit, b_tot)
    out["reached_but_missed"] = out["reached_bug"] and not out["detected"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ids = generate.eligible_ids()[: args.limit]

    clean = clean_outcomes()
    n_clean_fail = sum(o in ("failed", "error") for o in clean["outcomes"].values())
    print(f"suite on clean: {len(clean['outcomes'])} tests, {n_clean_fail} already failing, "
          f"collection errors {clean['collection_errors'][:5]}", flush=True)
    print("coverage run on clean ...", flush=True)
    cov = suite_coverage()

    slots: queue.Queue[str] = queue.Queue()
    for w in range(args.workers):
        materialize("clean", slot=f"base-w{w}")
        slots.put(f"base-w{w}")

    def work(iid):
        slot = slots.get()
        try:
            return score(buggy_row(iid, clean, slot), cov)
        except Exception as exc:
            return {"id": iid, "error": str(exc)}
        finally:
            slots.put(slot)

    rows = []
    with ThreadPoolExecutor(args.workers) as pool:
        for k, row in enumerate(pool.map(work, ids), 1):
            if "error" in row:
                print(f"[{k}/{len(ids)}] {row['id']:16} ERROR {row['error'][:200]}", flush=True)
                continue
            rows.append(row)
            print(f"[{k}/{len(ids)}] {row['id']:16} {'CAUGHT' if row['detected'] else 'missed':6} "
                  f"catching={row['n_catching']:3} (with fix tests {row['n_catching_with_fix']:3}, "
                  f"{row['n_fix_tests']} fix tests dropped) tgt_line={row['target_line_cov']} "
                  f"reached={row['reached_bug']}", flush=True)

    with (config.DATA / "results_baseline.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = ["id", "detected", "n_catching", "target_line_cov", "reached_bug", "reached_but_missed",
                "n_fix_tests", "detected_with_fix", "n_catching_with_fix", "target_line_cov_with_fix",
                "reached_bug_with_fix", "target_branch_cov_with_fix"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n_tests, n_asserts = suite_asserts(materialize("clean", slot="base-w0"))
    mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else None  # noqa: E731
    done = {r["id"] for r in rows}
    print(f"\n{'':52} {'bugs':>4} {'caught':>6} {'reached':>7} {'reach+miss':>10} {'tgt line':>8}")
    for label, sfx in (("suite without each bug's fix tests", ""), ("suite incl. fix tests", "_with_fix")):
        det = "detected" if not sfx else "detected_with_fix"
        print(f"{label:52} {len(rows):4} {sum(r[det] for r in rows):6} "
              f"{sum(r['reached_bug' + sfx] for r in rows):7} "
              f"{sum(r['reached_bug' + sfx] and not r[det] for r in rows):10} "
              f"{mean([r['target_line_cov' + sfx] for r in rows if r['target_line_cov' + sfx] is not None]):8}")
    for run_dir in sorted((config.DATA / "runs").iterdir()):
        ms = [json.loads(p.read_text(encoding="utf-8")) for p in run_dir.glob("*/metrics.json")]
        ms = [m for m in ms if m["id"] in done]
        if len(ms) < len(done):
            continue
        print(f"{run_dir.name.replace('claude-haiku-4-5__', 'haiku '):52} {len(ms):4} "
              f"{sum(m['detected'] for m in ms):6} {sum(m['reached_bug'] for m in ms):7} "
              f"{sum(m['reached_but_missed'] for m in ms):10} "
              f"{mean([m['target_line_cov'] for m in ms if m['target_line_cov'] is not None]):8}")
    print(f"\nsuite: {n_tests} test functions, {n_asserts / max(n_tests, 1):.2f} asserts/test, "
          f"package line coverage {cov['totals']['percent_covered'] / 100:.2f}")


if __name__ == "__main__":
    main()

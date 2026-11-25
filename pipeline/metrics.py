"""Test-quality metrics for generated test files, with no model calls.

Reruns each saved test file on the clean checkout under branch coverage and adds:

  target_line_cov    line coverage inside the target functions (public + private targets)
  target_branch_cov  branch coverage inside the target functions
  package_line_cov   line coverage of the whole arnio package
  bug_lines          executable clean-side lines the bug patch changes
  reached_bug        some test executes at least one bug line
  tests_reaching_bug how many tests execute a bug line (per-test coverage contexts)
  assertions         assert statements + pytest.raises/warns blocks, per test (static)
  tests_no_assert    tests with no assertion at all
  clean_pass_rate    share of tests that pass on clean (from result.json)

reached_bug splits misses in two: the tests never ran the buggy code, or ran it and
did not check the right thing.

    python -m pipeline.metrics [--runs claude-haiku-4-5__full ...] [--workers 4] [--force]

Writes data/runs/<run>/<id>/metrics.json and data/results_metrics.csv, prints a per-run table.
Already-measured instances are skipped unless --force (or the test file changed).
"""

import argparse
import ast
import csv
import json
import os
import queue
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config
from .gitutil import git
from .harness import GENERATED_DIR, load_instance, materialize

RCFILE = f"""[run]
branch = True
source = {config.PACKAGE}
dynamic_context = test_function
data_file = .coverage-metrics
"""

ASSERT_CONTEXTS = {"raises", "warns", "deprecated_call"}


# ---- static -----------------------------------------------------------------

def assertion_counts(code: str) -> dict[str, int]:
    """Assertions per test function: ``assert`` statements and ``with pytest.raises/warns``."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    out = {}
    for fn in ast.walk(tree):
        if not (isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name.startswith("test")):
            continue
        n = 0
        for node in ast.walk(fn):
            if isinstance(node, ast.Assert):
                n += 1
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    call = item.context_expr
                    if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                            and call.func.attr in ASSERT_CONTEXTS):
                        n += 1
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr in ASSERT_CONTEXTS):
                n += 1  # pytest.raises(X, fn, ...) call form
        out[fn.name] = n
    return out


def target_ranges(meta: dict) -> dict[str, list[tuple[int, int]]]:
    """{file: [(first_line, last_line), ...]} of each target symbol at BASE_REF."""
    out: dict[str, list[tuple[int, int]]] = {}
    for symbol in meta["target_symbols"]:
        rel, qual = symbol.split("::")
        node: ast.AST = ast.parse(git("show", f"{config.BASE_REF}:{rel}"))
        for part in qual.split("."):
            node = next((n for n in ast.iter_child_nodes(node)
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                         and n.name == part), None)
            if node is None:
                break
        if node is not None:
            out.setdefault(rel, []).append((node.lineno, node.end_lineno))
    return out


def bug_lines(instance_id: str) -> dict[str, set[int]]:
    """Clean-side lines the bug patch touches, per source file.

    Removed/changed lines count as themselves; a pure insertion counts as the line before it."""
    patch = (config.INSTANCE_DIR / instance_id / "bug.patch").read_text(encoding="utf-8")
    out: dict[str, set[int]] = {}
    rel, old_line, prev = None, 0, 0
    for line in patch.splitlines():
        if line.startswith("--- "):
            continue
        if line.startswith("+++ "):
            path = line[4:].strip()
            rel = path[2:] if path.startswith("b/") else path
            if not (rel.startswith(config.SRC_PREFIX) and rel.endswith(".py")):
                rel = None
            continue
        m = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
        if m:
            old_line = int(m.group(1))
            prev = old_line - 1
            continue
        if rel is None or not old_line:
            continue
        if line.startswith("-"):
            out.setdefault(rel, set()).add(old_line)
            prev, old_line = old_line, old_line + 1
        elif line.startswith("+"):
            out.setdefault(rel, set()).add(max(prev, 1))
        elif line.startswith(" ") or line == "":
            prev, old_line = old_line, old_line + 1
    return out


# ---- coverage run -----------------------------------------------------------

def run_coverage(path: Path, test_file: Path) -> dict:
    """Run one test file under branch coverage with per-test contexts; return coverage JSON."""
    dest_dir = path / GENERATED_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / test_file.name
    shutil.copy2(test_file, dest)
    (path / ".coveragerc-metrics").write_text(RCFILE, encoding="utf-8")
    env = dict(os.environ, **config.TEST_ENV, PYTHONPATH=str(path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    common = dict(cwd=path, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    subprocess.run([config.PYTHON, "-m", "coverage", "run", "--rcfile=.coveragerc-metrics",
                    "-m", "pytest", dest.relative_to(path).as_posix(),
                    "-q", "-p", "no:cacheprovider", "-o", "addopts="],
                   timeout=config.TEST_TIMEOUT_S, **common)
    out = path / ".coverage-metrics.json"
    out.unlink(missing_ok=True)
    proc = subprocess.run([config.PYTHON, "-m", "coverage", "json", "--rcfile=.coveragerc-metrics",
                           "--show-contexts", "-o", out.name], **common)
    if not out.exists():
        raise RuntimeError(f"coverage json failed: {(proc.stdout + proc.stderr)[-500:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def _ratio(hit: int, total: int) -> float | None:
    return round(hit / total, 3) if total else None


def measure(run_dir: Path, slot: str = "metrics") -> dict:
    """Metrics for one instance folder under data/runs/<run>/."""
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    iid = result["id"]
    meta = load_instance(iid)
    test_file = next(run_dir.glob("test_*.py"))
    code = test_file.read_text(encoding="utf-8")

    cov = run_coverage(materialize("clean", slot=slot), test_file)
    files = {k.replace("\\", "/"): v for k, v in cov["files"].items()}

    # Target functions: lines and branches inside their source ranges.
    t_hit = t_tot = b_hit = b_tot = 0
    for rel, ranges in target_ranges(meta).items():
        f = files.get(rel)
        if not f:
            continue
        inside = lambda n: any(a <= n <= b for a, b in ranges)  # noqa: E731
        executed = {n for n in f["executed_lines"] if inside(n)}
        missing = {n for n in f["missing_lines"] if inside(n)}
        t_hit, t_tot = t_hit + len(executed), t_tot + len(executed) + len(missing)
        eb = sum(inside(a) for a, _ in f.get("executed_branches", []))
        mb = sum(inside(a) for a, _ in f.get("missing_branches", []))
        b_hit, b_tot = b_hit + eb, b_tot + eb + mb

    # Bug lines: executable clean lines the patch changes, and which tests run them.
    n_bug, reached, reaching = 0, set(), set()
    for rel, lines in bug_lines(iid).items():
        f = files.get(rel)
        if not f:
            continue
        statements = set(f["executed_lines"]) | set(f["missing_lines"])
        executable = lines & statements
        n_bug += len(executable)
        reached |= {(rel, n) for n in executable if n in f["executed_lines"]}
        for n in executable:
            for ctx in f.get("contexts", {}).get(str(n), []):
                if ctx:  # "" is the import-time context, not a test
                    reaching.add(ctx.split(".")[-1].split("[")[0])

    asserts = assertion_counts(code)
    tests = result["tests"]
    n_tests = len(tests)
    detecting = {node.split("::")[-1].split("[")[0] for node, r in tests.items() if r["detects_bug"]}
    metrics = {
        "id": iid,
        "run": result["run"],
        "detected": result["detected"],
        "n_tests": n_tests,
        "clean_pass_rate": _ratio(sum(r["clean"] == "passed" for r in tests.values()), n_tests),
        "target_line_cov": _ratio(t_hit, t_tot),
        "target_branch_cov": _ratio(b_hit, b_tot),
        "package_line_cov": round(cov["totals"]["percent_covered"] / 100, 3),
        "bug_lines": n_bug,
        "reached_bug": bool(reached),
        "tests_reaching_bug": len(reaching),
        "reached_but_missed": bool(reached) and not result["detected"],
        "detecting_tests_reaching_bug": len(detecting & reaching),
        "assertions_total": sum(asserts.values()),
        "assertions_per_test": _ratio(sum(asserts.values()), len(asserts)) if asserts else None,
        "tests_no_assert": sum(v == 0 for v in asserts.values()),
        "test_file_mtime": test_file.stat().st_mtime,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return metrics


def _stale(run_dir: Path) -> bool:
    m = run_dir / "metrics.json"
    if not m.exists():
        return True
    test_file = next(run_dir.glob("test_*.py"), None)
    old = json.loads(m.read_text(encoding="utf-8"))
    return test_file is None or old.get("test_file_mtime") != test_file.stat().st_mtime


# ---- reporting --------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(rows: list[dict]) -> dict:
    return {
        "bugs": len(rows),
        "detected": sum(r["detected"] for r in rows),
        "reached_bug": sum(r["reached_bug"] for r in rows),
        "reached_but_missed": sum(r["reached_but_missed"] for r in rows),
        "target_line_cov": _mean(r["target_line_cov"] for r in rows),
        "target_branch_cov": _mean(r["target_branch_cov"] for r in rows),
        "package_line_cov": _mean(r["package_line_cov"] for r in rows),
        "assertions_per_test": _mean(r["assertions_per_test"] for r in rows),
        "clean_pass_rate": _mean(r["clean_pass_rate"] for r in rows),
    }


def table(by_run: dict[str, list[dict]]) -> str:
    fmt = lambda x: "  -  " if x is None else f"{x:5.2f}"  # noqa: E731
    lines = [f"{'run':46} {'bugs':>4} {'caught':>6} {'reached':>7} {'reach+miss':>10} "
             f"{'tgt line':>8} {'tgt br':>6} {'pkg line':>8} {'asserts':>7} {'clean ok':>8}"]
    for run, rows in sorted(by_run.items()):
        s = summarize(rows)
        lines.append(f"{run:46} {s['bugs']:4} {s['detected']:6} {s['reached_bug']:7} {s['reached_but_missed']:10} "
                     f"{fmt(s['target_line_cov']):>8} {fmt(s['target_branch_cov']):>6} "
                     f"{fmt(s['package_line_cov']):>8} {fmt(s['assertions_per_test']):>7} "
                     f"{fmt(s['clean_pass_rate']):>8}")
    return "\n".join(lines)


def collect(runs: list[str] | None = None) -> dict[str, list[dict]]:
    """Existing metrics.json files, grouped by run."""
    by_run: dict[str, list[dict]] = {}
    for m in sorted((config.DATA / "runs").glob("*/*/metrics.json")):
        run = m.parent.parent.name
        if runs and run not in runs:
            continue
        by_run.setdefault(run, []).append(json.loads(m.read_text(encoding="utf-8")))
    return by_run


def write_csv(by_run: dict[str, list[dict]]) -> Path:
    out = config.DATA / "results_metrics.csv"
    rows = [r for rs in by_run.values() for r in rs]
    if rows:
        keys = [k for k in rows[0] if k != "test_file_mtime"]
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", help="run folder names under data/runs (default: all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="re-measure even if metrics.json is current")
    args = ap.parse_args()

    dirs = [p.parent for p in sorted((config.DATA / "runs").glob("*/*/result.json"))
            if (not args.runs or p.parent.parent.name in args.runs) and next(p.parent.glob("test_*.py"), None)]
    todo = [d for d in dirs if args.force or _stale(d)]
    print(f"{len(todo)} of {len(dirs)} test files to measure with {args.workers} worker(s)", flush=True)

    slots: queue.Queue[str] = queue.Queue()
    for w in range(args.workers):  # one at a time: `git worktree add` takes a lock
        materialize("clean", slot=f"metrics-w{w}")
        slots.put(f"metrics-w{w}")

    def work(d: Path) -> str:
        slot = slots.get()
        try:
            m = measure(d, slot)
            return (f"{d.parent.name:46} {m['id']:14} reached={'Y' if m['reached_bug'] else 'n'} "
                    f"tgt_line={m['target_line_cov']} tgt_br={m['target_branch_cov']} "
                    f"asserts/test={m['assertions_per_test']}")
        except Exception as exc:
            return f"{d.parent.name:46} {d.name:14} ERROR {exc}"
        finally:
            slots.put(slot)

    with ThreadPoolExecutor(args.workers) as pool:
        for k, line in enumerate(pool.map(work, todo), 1):
            print(f"[{k}/{len(todo)}] {line}", flush=True)

    by_run = collect(args.runs)
    print("\n" + table(by_run))
    print(f"\nwrote {write_csv(by_run)}")


if __name__ == "__main__":
    main()

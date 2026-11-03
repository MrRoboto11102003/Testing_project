"""Step 2: port each mined fix back into BASE_REF as a bug, and validate it.

For every candidate (issue, PR) pair:
  1. take the PR's source diff (arnio/*.py only) and reverse-apply it to BASE_REF;
  2. run the test files the PR touched on clean and buggy checkouts;
  3. keep the instance if at least one test passes on clean and fails on buggy
     (fail-to-pass, F2P).

Kept instances land in data/instances/<id>/{meta.json,bug.patch,fix_tests.patch}
and data/instances.jsonl; every attempt is logged to data/build_log.jsonl.

    python -m pipeline.build [--workers 4] [--limit N] [--only arnio-pr123 ...]
"""

import argparse
import ast
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config
from .gitutil import git, git_ok, worktree
from .harness import install_extension, run_pytest

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", re.M)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def instance_id(row: dict) -> str:
    return f"arnio-pr{row['pr']}"


def port(path: Path, fix_commit: str, src_files: list[str]) -> tuple[str | None, str]:
    """Undo the fix's source changes in ``path``. Returns (method, message).

    Tries a plain reverse patch first, then ``git revert`` (a real three-way
    merge, so later edits around the fix don't block it). Only changes to
    ``src_files`` are kept; tests and everything else stay at BASE_REF.
    """
    diff = git("diff", "--full-index", "--binary", f"{fix_commit}^1", fix_commit, "--", *src_files)
    ok, msg = git_ok("apply", "-R", "--whitespace=nowarn", "-", cwd=path, input=diff)
    if ok:
        return "reverse_apply", ""

    is_merge = len(git("rev-list", "--parents", "-n1", fix_commit).split()) > 2
    git_ok("revert", "--no-commit", *(["-m", "1"] if is_merge else []), fix_commit, cwd=path)
    conflicted = set(git("diff", "--name-only", "--diff-filter=U", cwd=path).split())
    if conflicted & set(src_files):
        git_ok("revert", "--abort", cwd=path)
        git("reset", "--quiet", "--hard", "HEAD", cwd=path)
        return port_hunks(path, diff)
    src_patch = git("diff", "--full-index", "HEAD", "--", *src_files, cwd=path)
    git_ok("revert", "--abort", cwd=path)
    git("reset", "--quiet", "--hard", "HEAD", cwd=path)
    if not src_patch.strip():
        return None, "revert produced no source change"
    ok, msg2 = git_ok("apply", "--whitespace=nowarn", "-", cwd=path, input=src_patch)
    return ("revert_3way", "") if ok else (None, msg2[-800:])


def split_hunks(diff: str) -> list[str]:
    """Split a multi-file diff into single-hunk diffs (each with its file header)."""
    patches = []
    for file_diff in re.split(r"^(?=diff --git )", diff, flags=re.M):
        if not file_diff.startswith("diff --git "):
            continue
        parts = re.split(r"^(?=@@ )", file_diff, flags=re.M)
        header, hunks = parts[0], parts[1:]
        patches += [header + h for h in hunks]
    return patches


def port_hunks(path: Path, diff: str) -> tuple[str | None, str]:
    """Last resort: reverse-apply hunk by hunk with one line of context, skipping hunks
    that no longer match. Fail-to-pass validation decides whether what applied is a bug."""
    hunks = split_hunks(diff)
    applied = 0
    for hunk in hunks:
        ok, _ = git_ok("apply", "-R", "-C1", "--recount", "--whitespace=nowarn", "-", cwd=path, input=hunk)
        applied += ok
    if not applied:
        return None, f"no hunk of {len(hunks)} applies"
    return f"partial_hunks:{applied}/{len(hunks)}", ""


def target_symbols(patch: str, root: Path) -> list[str]:
    """Map the base-side lines a bug patch touches to their enclosing def/class names."""
    symbols: list[str] = []
    for file_diff in re.split(r"^diff --git ", patch, flags=re.M)[1:]:
        m = re.match(r"a/(\S+)", file_diff)
        if not m:
            continue
        rel = m.group(1)
        lines = set()
        for start, length in HUNK.findall(file_diff):
            start, length = int(start), int(length or 1)
            lines.update(range(start, start + max(length, 1)))
        try:
            tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue

        def visit(node, prefix):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name = f"{prefix}.{child.name}" if prefix else child.name
                    span = range(child.lineno, (child.end_lineno or child.lineno) + 1)
                    inner_before = len(symbols)
                    visit(child, name)
                    if len(symbols) == inner_before and lines.intersection(span):
                        symbols.append(f"{rel}::{name}")

        visit(tree, "")
    return sorted(set(symbols))


def build_one(row: dict, slot: str, clean_results: dict, issues_by_num: dict) -> dict:
    iid = instance_id(row)
    log = {"id": iid, "issue": row["issue"], "pr": row["pr"]}
    path = worktree(slot, config.BASE_REF)
    install_extension(path)

    tests = [t for t in row["test_files"] if t not in config.EXCLUDED_TESTS and (path / t).exists()]
    if not tests:
        return {**log, "status": "tests_gone"}

    method, msg = port(path, row["fix_commit"], row["src_files"])
    if not method:
        return {**log, "status": "port_failed", "detail": msg}
    bug_patch = git("diff", "--full-index", "HEAD", cwd=path)
    if not bug_patch.strip():
        return {**log, "status": "empty_patch"}

    buggy = run_pytest(path, tests, label="buggy")
    if buggy["collection_errors"]:
        return {**log, "status": "buggy_collection_error", "detail": buggy["collection_errors"][:5]}

    clean_out = {k: v for t in tests for k, v in clean_results[t].items()}
    f2p = sorted(n for n, o in clean_out.items() if o == "passed" and buggy["outcomes"].get(n) in ("failed", "error"))
    p2p = sorted(n for n, o in clean_out.items() if o == "passed" and buggy["outcomes"].get(n) == "passed")
    if not f2p:
        return {**log, "status": "no_fail_to_pass", "tests_run": len(buggy["outcomes"])}

    issue = issues_by_num[row["issue"]]
    pr = next(p for p in issue["closedByPullRequestsReferences"]["nodes"] if p["number"] == row["pr"])
    fix_tests = git("diff", f"{row['fix_commit']}^1", row["fix_commit"], "--", *row["test_files"])
    meta = {
        "id": iid,
        "repo": config.REPO_SLUG,
        "base_ref": config.BASE_REF,
        "base_commit": git("rev-parse", config.BASE_REF).strip(),
        "issue": row["issue"],
        "issue_title": issue["title"],
        "issue_body": issue["body"],
        "issue_url": issue["url"],
        "pr": row["pr"],
        "pr_title": pr["title"],
        "pr_body": pr["body"],
        "fix_commit": row["fix_commit"],
        "merged_at": row["merged_at"],
        "port_method": method,
        "src_files": row["src_files"],
        "test_files": tests,
        "target_symbols": target_symbols(bug_patch, path),
        "fail_to_pass": f2p,
        "pass_to_pass": p2p,
        "patch_lines": sum(1 for ln in bug_patch.splitlines() if ln[:1] in "+-" and ln[:3] not in ("+++", "---")),
    }
    out = config.INSTANCE_DIR / iid
    out.mkdir(parents=True, exist_ok=True)
    (out / "bug.patch").write_text(bug_patch, encoding="utf-8", newline="\n")
    (out / "fix_tests.patch").write_text(fix_tests, encoding="utf-8", newline="\n")
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return {**log, "status": "ok", "f2p": len(f2p), "p2p": len(p2p), "port_method": method}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    rows = [r for r in load_jsonl(config.CANDIDATES) if r["status"] == "candidate"]
    if args.only:
        rows = [r for r in rows if instance_id(r) in args.only]
    if args.limit:
        rows = rows[: args.limit]
    issues_by_num = {i["number"]: i for i in json.loads(config.RAW_ISSUES.read_text(encoding="utf-8"))}

    # Clean results are the same for every instance: run each needed test file once.
    clean_path = worktree("build-clean", config.BASE_REF)
    install_extension(clean_path)
    needed = sorted({t for r in rows for t in r["test_files"]
                     if t not in config.EXCLUDED_TESTS and (clean_path / t).exists()})
    print(f"running {len(needed)} test files on clean {config.BASE_REF} ...", flush=True)
    clean_all = run_pytest(clean_path, needed, label="clean")["outcomes"]
    clean_results = {t: {n: o for n, o in clean_all.items() if n.split("::")[0] == t} for t in needed}

    slots = [f"build-{i}" for i in range(args.workers)]
    free, lock = list(slots), threading.Lock()

    def task(row):
        with lock:
            slot = free.pop()
        try:
            return build_one(row, slot, clean_results, issues_by_num)
        except Exception as exc:  # keep going; the log records it
            return {"id": instance_id(row), "issue": row["issue"], "pr": row["pr"], "status": "crash", "detail": str(exc)[-800:]}
        finally:
            with lock:
                free.append(slot)

    results = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(task, r) for r in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            print(f"[{i}/{len(rows)}] {res['id']:16} {res['status']}", flush=True)

    with config.BUILD_LOG.open("w", encoding="utf-8") as f:
        for res in sorted(results, key=lambda r: r["pr"]):
            f.write(json.dumps(res) + "\n")
    kept = sorted(r["id"] for r in results if r["status"] == "ok")
    if not args.only and not args.limit:
        with config.INSTANCES.open("w", encoding="utf-8") as f:
            for iid in kept:
                f.write((config.INSTANCE_DIR / iid / "meta.json").read_text(encoding="utf-8").replace("\n", "") + "\n")

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\nsummary:", json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()

"""Materialize clean/buggy versions of arnio and run tests against them.

    # run arnio's own tests (or any test file) against one version
    python -m pipeline.harness run arnio-pr1234 --variant buggy --tests tests/test_cleaning.py

    # score an external (e.g. LLM-written) test file against clean and buggy
    python -m pipeline.harness check arnio-pr1234 path/to/test_generated.py
"""

import argparse
import json
import shutil
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from . import config
from .gitutil import git, git_ok, worktree

GENERATED_DIR = "tests/_generated"


def install_extension(path: Path) -> None:
    """Drop the prebuilt C++ extension into a checkout (Python-only bugs share it)."""
    pyds = list(config.EXT_DIR.glob("_arnio_cpp*"))
    if not pyds:
        raise RuntimeError(f"No compiled extension in {config.EXT_DIR}; run `python -m pipeline.setup_ext`.")
    for pyd in pyds:
        dest = path / "arnio" / pyd.name
        if not dest.exists():
            shutil.copy2(pyd, dest)


def load_instance(instance_id: str) -> dict:
    meta = config.INSTANCE_DIR / instance_id / "meta.json"
    return json.loads(meta.read_text(encoding="utf-8"))


def materialize(variant: str, instance_id: str | None = None, slot: str = "harness", patch: str | None = None) -> Path:
    """Return a checkout of BASE_REF, with the bug patch applied when variant == 'buggy'."""
    path = worktree(slot, config.BASE_REF)
    install_extension(path)
    if variant == "buggy":
        if patch is None:
            patch = (config.INSTANCE_DIR / instance_id / "bug.patch").read_text(encoding="utf-8")
        ok, out = git_ok("apply", "--whitespace=nowarn", "-", cwd=path, input=patch)
        if not ok:
            raise RuntimeError(f"bug patch failed to apply: {out}")
    elif variant != "clean":
        raise ValueError(f"unknown variant {variant!r}")
    return path


def _nodeid(case: ET.Element) -> str:
    file = case.get("file") or ""
    classname = case.get("classname") or ""
    name = case.get("name") or ""
    module = file[:-3].replace("/", ".").replace("\\", ".") if file.endswith(".py") else ""
    cls = classname[len(module) + 1 :] if module and classname.startswith(module + ".") else ""
    parts = [file.replace("\\", "/")] + ([cls.replace(".", "::")] if cls else []) + [name]
    return "::".join(p for p in parts if p)


def run_pytest(path: Path, targets: list[str], label: str = "run") -> dict:
    """Run pytest on ``targets`` inside ``path``.

    Returns {"outcomes": {nodeid: passed|failed|error|skipped}, "collection_errors": [...],
    "returncode": int}.
    """
    junit = path / f".junit-{label}.xml"
    junit.unlink(missing_ok=True)
    cmd = [
        config.PYTHON, "-m", "pytest", *targets,
        "-q", "-p", "no:cacheprovider", "-o", "addopts=", "-o", "junit_family=xunit1",
        f"--junitxml={junit.name}",
    ]
    # Subprocesses the tests start (e.g. `python -m arnio.cli` with cwd=tmp_path) must import
    # this checkout's arnio too, not whatever their own cwd would give them.
    env = dict(os.environ, PYTHONPATH=str(path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    try:
        proc = subprocess.run(cmd, cwd=path, capture_output=True, text=True, env=env,
                              encoding="utf-8", errors="replace", timeout=config.TEST_TIMEOUT_S)
        rc, tail = proc.returncode, proc.stdout[-3000:]
    except subprocess.TimeoutExpired:
        return {"outcomes": {}, "collection_errors": ["timeout"], "returncode": -1, "stdout_tail": ""}

    outcomes, collection_errors = {}, []
    if junit.exists():
        for case in ET.parse(junit).getroot().iter("testcase"):
            if case.find("failure") is not None:
                outcome = "failed"
            elif case.find("error") is not None:
                outcome = "error"
            elif case.find("skipped") is not None:
                outcome = "skipped"
            else:
                outcome = "passed"
            if not case.get("file") and outcome == "error":
                # pytest reports import/collection failures as a nameless testcase.
                collection_errors.append(case.get("classname") or case.get("name"))
                continue
            outcomes[_nodeid(case)] = outcome
        junit.unlink()
    return {"outcomes": outcomes, "collection_errors": collection_errors, "returncode": rc, "stdout_tail": tail}


def run_external(path: Path, test_file: Path, label: str) -> dict:
    """Copy an external test file into the checkout's tests/ tree and run it."""
    dest_dir = path / GENERATED_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (test_file.name if test_file.name.startswith("test_") else f"test_{test_file.name}")
    shutil.copy2(test_file, dest)
    return run_pytest(path, [dest.relative_to(path).as_posix()], label=label)


def check(instance_id: str, test_file: Path, slot: str = "check") -> dict:
    """Run one test file on clean and buggy; report which tests catch the bug."""
    clean = run_external(materialize("clean", slot=slot), test_file, "clean")
    buggy = run_external(materialize("buggy", instance_id, slot=slot), test_file, "buggy")
    rows = {}
    for node in sorted(set(clean["outcomes"]) | set(buggy["outcomes"])):
        c, b = clean["outcomes"].get(node, "missing"), buggy["outcomes"].get(node, "missing")
        rows[node] = {
            "clean": c,
            "buggy": b,
            "false_alarm": c in ("failed", "error"),
            "detects_bug": c == "passed" and b in ("failed", "error"),
        }
    return {
        "instance": instance_id,
        "tests": rows,
        "clean_collection_errors": clean["collection_errors"],
        "buggy_collection_errors": buggy["collection_errors"],
        "detected": any(r["detects_bug"] for r in rows.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run tests from the arnio tree against one variant")
    r.add_argument("instance", nargs="?")
    r.add_argument("--variant", choices=["clean", "buggy"], default="clean")
    r.add_argument("--tests", nargs="*", help="defaults to the instance's fail-to-pass tests")
    c = sub.add_parser("check", help="score an external test file against clean and buggy")
    c.add_argument("instance")
    c.add_argument("test_file", type=Path)
    args = ap.parse_args()

    if args.cmd == "run":
        targets = args.tests
        if not targets:
            targets = load_instance(args.instance)["fail_to_pass"]
        path = materialize(args.variant, args.instance)
        result = run_pytest(path, targets, label=args.variant)
        for node, outcome in sorted(result["outcomes"].items()):
            print(f"{outcome:8} {node}")
        if result["collection_errors"]:
            print("collection errors:", result["collection_errors"])
    else:
        result = check(args.instance, args.test_file.resolve())
        for node, row in result["tests"].items():
            flag = "CATCH" if row["detects_bug"] else ("FALSE-ALARM" if row["false_alarm"] else "")
            print(f"{row['clean']:8} {row['buggy']:8} {flag:11} {node}")
        print(json.dumps({k: v for k, v in result.items() if k != "tests"}, indent=1))


if __name__ == "__main__":
    main()

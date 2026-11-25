"""Why do generated tests fail on CLEAN code? Rerun saved test files, no model calls.

    python -m pipeline.fa_probe [--runs <run> ...] [--only <id> ...] [--out probe.json]
    TP_PROJECT=openlens python -m pipeline.fa_probe ...

For each saved test file it runs pytest on the clean checkout, takes the error line of
every failing test and sorts it into a coarse kind:

  setup        the test broke before checking anything: TypeError (wrong signature or wrong
               input type), AttributeError, ImportError/ModuleNotFoundError, NameError,
               assigning to a frozen dataclass, fixture errors
  expectation  an assert or pytest.raises that didn't hold (AssertionError, DID NOT RAISE,
               a raises match that failed)
  other        anything else (ValueError from the library, timeouts, ...)

Prints the share of each kind per run and writes the raw lines to --out. The split is a
regex heuristic; the failure analysis in the report also read the lines by hand.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter

from . import config, harness

DEFAULT_RUNS = ["claude-haiku-4-5__code", "claude-haiku-4-5__full__src__cachedocs",
                "claude-haiku-4-5__partial50_s0__src__cachedocs",
                "claude-haiku-4-5__partial50_s0__src__notes-opus-5-5__cachedocs"]
SETUP = re.compile(r"(TypeError|AttributeError|ImportError|ModuleNotFoundError|NameError|FrozenInstanceError"
                   r"|fixture .* not found)")
EXPECT = re.compile(r"(AssertionError|assert |DID NOT RAISE|DID NOT WARN|Regex pattern did not match)")


def kind(line: str) -> str:
    if SETUP.search(line):
        return "setup"
    if EXPECT.search(line):
        return "expectation"
    return "other"


def probe(run: str, iid: str, path) -> list[str]:
    src = next((config.DATA / "runs" / run / iid).glob("test_*.py"), None)
    if src is None:
        return []
    gen = path / harness.GENERATED_DIR
    gen.mkdir(parents=True, exist_ok=True)
    dst = gen / src.name
    shutil.copy(src, dst)
    env = dict(os.environ, **config.TEST_ENV, PYTHONPATH=str(path))
    junit = path / ".junit-faprobe.xml"
    junit.unlink(missing_ok=True)
    try:
        subprocess.run([config.PYTHON, "-m", "pytest", dst.relative_to(path).as_posix(), "-q",
                        "-p", "no:cacheprovider", "-o", "addopts=", "--tb=no", f"--junitxml={junit.name}"],
                       cwd=path, capture_output=True, text=True, env=env, encoding="utf-8",
                       errors="replace", timeout=config.TEST_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return ["timeout"]
    finally:
        dst.unlink(missing_ok=True)
    if not junit.exists():
        return ["no report"]
    lines = []
    for case in ET.parse(junit).getroot().iter("testcase"):
        for bad in list(case.findall("failure")) + list(case.findall("error")):
            msg = (bad.get("message") or bad.text or "").strip().splitlines()
            lines.append(f"{case.get('name')}: {msg[0] if msg else '?'}"[:300])
    junit.unlink()
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--only", nargs="*", help="bug ids (default: every bug with a false alarm in that run)")
    ap.add_argument("--out", default=str(config.DATA / "fa_probe.json"))
    args = ap.parse_args()
    path = harness.materialize("clean", slot="faprobe")
    raw = {}
    for run in args.runs:
        counts = Counter()
        for res in sorted((config.DATA / "runs" / run).glob("*/result.json")):
            r = json.loads(res.read_text(encoding="utf-8"))
            if (args.only and r["id"] not in args.only) or (not args.only and not r["n_false_alarms"]):
                continue
            lines = probe(run, r["id"], path)
            raw[f"{run}/{r['id']}"] = lines
            counts.update(kind(ln) for ln in lines)
        n = sum(counts.values()) or 1
        print(f"{run:60} failing tests {sum(counts.values()):4}  " +
              "  ".join(f"{k} {counts[k] / n:.0%}" for k in ("setup", "expectation", "other")), flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

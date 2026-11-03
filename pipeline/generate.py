"""Step 3: have an LLM write tests for an instance's target functions, then score them.

The model sees only documentation (signature + docstring of each target symbol,
plus API reference sections mentioning it) - never the issue, the fix or the
project's tests. Generated tests are run on clean and buggy checkouts:

  detected     some test passes on clean and fails on buggy
  false alarm  some test fails on clean

    python -m pipeline.generate --limit 5 [--model claude-haiku-4-5] [--only arnio-pr1234]
"""

import argparse
import ast
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

from . import config
from .gitutil import git
from .harness import check, load_instance

SYSTEM_PROMPT = (
    "You write pytest unit tests for a Python library from its documentation. "
    "Reply with exactly one ```python code block containing a complete pytest file and nothing else."
)

PROMPT = """Library: arnio (import arnio as ar), a pandas data-preparation library.

Write pytest tests for the functions documented below. Test the documented behavior,
including edge cases the documentation implies. Use only the public API shown and
pandas/numpy. Keep it to at most 15 focused tests.

{context}
"""


def symbol_context(symbol: str) -> str:
    """Signature + docstring of ``path::Qual.name`` at BASE_REF, without its body."""
    rel, qual = symbol.split("::")
    source = git("show", f"{config.BASE_REF}:{rel}")
    node: ast.AST = ast.parse(source)
    for part in qual.split("."):
        node = next(
            (n for n in ast.iter_child_nodes(node)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == part),
            None,
        )
        if node is None:
            return ""
    lines = source.splitlines()
    body_start = node.body[0]
    if isinstance(body_start, ast.Expr) and isinstance(getattr(body_start, "value", None), ast.Constant):
        end = body_start.end_lineno  # keep the docstring
    else:
        end = body_start.lineno - 1
    header = "\n".join(lines[node.lineno - 1 - len(node.decorator_list): end])
    module = rel[:-3].replace("/", ".")
    return f"# {module}.{qual}\n{header}\n    ..."


def api_reference_sections(names: set[str]) -> str:
    """Sections of API_REFERENCE.md whose heading names one of ``names``."""
    try:
        text = git("show", f"{config.BASE_REF}:API_REFERENCE.md")
    except RuntimeError:
        return ""
    sections = re.split(r"^(?=#{2,4} )", text, flags=re.M)
    keep = [s for s in sections if any(re.search(rf"\b{re.escape(n)}\b", s.splitlines()[0]) for n in names)]
    return "\n".join(keep)[:12000]


def readme_quickstart() -> str:
    """README's quickstart sections: how to build frames and call the API at all.
    Shared by every instance, so it is never ablated."""
    text = git("show", f"{config.BASE_REF}:README.md")
    sections = re.split(r"^(?=## )", text, flags=re.M)
    keep = [s for s in sections if re.match(r"## .*(Quickstart|Quick Example)", s)]
    return "\n".join(keep)


def build_context(meta: dict) -> str:
    symbols = [s for s in meta["target_symbols"] if not s.split("::")[1].split(".")[-1].startswith("_")]
    parts = [c for c in (symbol_context(s) for s in symbols) if c]
    names = {s.split("::")[1].split(".")[-1] for s in symbols}
    ref = api_reference_sections(names)
    out = "## Getting started (README)\n\n" + readme_quickstart()
    out += "\n\n## Source signatures and docstrings\n\n" + "\n\n".join(parts)
    if ref:
        out += "\n\n## API reference\n\n" + ref
    return out


def call_claude(prompt: str, model: str) -> dict:
    """One non-interactive Claude Code call with tools disabled."""
    with tempfile.TemporaryDirectory() as cwd:  # keep project files out of its context
        proc = subprocess.run(
            ["claude", "-p", "--model", model, "--tools", "", "--output-format", "json",
             "--system-prompt", SYSTEM_PROMPT],
            input=prompt, cwd=cwd, capture_output=True, text=True, encoding="utf-8", timeout=600,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p failed: {proc.stderr.strip() or proc.stdout[-500:]}")
    return json.loads(proc.stdout)


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


def run_one(meta: dict, model: str, out_root: Path) -> dict:
    out = out_root / meta["id"]
    out.mkdir(parents=True, exist_ok=True)
    context = build_context(meta)
    prompt = PROMPT.format(context=context)
    (out / "prompt.txt").write_text(prompt, encoding="utf-8")

    t0 = time.time()
    reply = call_claude(prompt, model)
    code = extract_code(reply.get("result", ""))
    test_file = out / f"test_{meta['id'].replace('-', '_')}.py"
    test_file.write_text(code, encoding="utf-8")

    scored = check(meta["id"], test_file)
    n_tests = len(scored["tests"])
    result = {
        "id": meta["id"],
        "model": model,
        "detected": scored["detected"],
        "n_tests": n_tests,
        "n_catching": sum(r["detects_bug"] for r in scored["tests"].values()),
        "n_false_alarms": sum(r["false_alarm"] for r in scored["tests"].values()),
        "clean_collection_errors": scored["clean_collection_errors"],
        "usage": reply.get("usage", {}),
        "cost_usd": reply.get("total_cost_usd"),
        "seconds": round(time.time() - t0, 1),
        "tests": scored["tests"],
    }
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    if args.only:
        ids = args.only
    else:
        # Bugs only inside private helpers have no public docs to show the model; skip for now.
        metas = [json.loads(l) for l in config.INSTANCES.read_text(encoding="utf-8").splitlines() if l]
        ids = [m["id"] for m in metas if any(not s.split("::")[1].split(".")[-1].startswith("_")
                                             for s in m["target_symbols"])][: args.limit]
    out_root = config.DATA / "runs" / args.model
    results = []
    for iid in ids:
        meta = load_instance(iid)
        try:
            res = run_one(meta, args.model, out_root)
        except Exception as exc:
            print(f"{iid:16} ERROR {exc}")
            continue
        results.append(res)
        verdict = "DETECTED" if res["detected"] else "missed"
        print(f"{iid:16} {verdict:8} tests={res['n_tests']:2} catching={res['n_catching']:2} "
              f"false_alarms={res['n_false_alarms']:2} {res['seconds']}s  {meta['target_symbols'][:2]}", flush=True)

    if results:
        print(f"\n{model_summary(results, args.model)}")
        (out_root / "summary.json").write_text(json.dumps(summarize(results), indent=1), encoding="utf-8")


def summarize(results: list[dict]) -> dict:
    n = len(results)
    return {
        "instances": n,
        "detection_rate": sum(r["detected"] for r in results) / n,
        "instances_with_false_alarms": sum(r["n_false_alarms"] > 0 for r in results),
        "mean_tests": sum(r["n_tests"] for r in results) / n,
    }


def model_summary(results: list[dict], model: str) -> str:
    s = summarize(results)
    return (f"{model}: detected {s['detection_rate']:.0%} of {s['instances']} bugs; "
            f"{s['instances_with_false_alarms']} instance(s) with false alarms; "
            f"{s['mean_tests']:.1f} tests per instance")


if __name__ == "__main__":
    main()

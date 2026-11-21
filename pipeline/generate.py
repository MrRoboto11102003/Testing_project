"""Step 3: have an LLM write tests for an instance's target functions, then score them.

The model is told which public functions to test and sees an amount of
documentation set by the context mode, but never the issue, the fix or the
project's tests:

  signatures  signatures of the target functions only, no docs
  code        clean source of the target functions only (docstrings included), no doc files
  codeapi     `code` + a compact index of the whole public API (signatures, types, decorators)
  targeted    README quickstart + target docstrings + API_REFERENCE sections naming them
  full        every user-facing doc file at BASE_REF + target docstrings
  partial     the `full` pool split into sections, keeping a seeded fraction (--keep).
              Lower fractions are subsets of higher ones. Target signatures always shown.

Generated tests are run on clean and buggy checkouts:

  detected     some test passes on clean and fails on buggy
  false alarm  some test fails on clean

--level picks the kind of tests asked for:

  unit         the target functions directly (original prompt)
  integration  target functions combined with other public arnio components
  system       end-to-end user workflows: files on disk, the .arnio accessor, the CLI

Each test also records which arnio entry points it touches (arnio_calls), as a check on level.

    python -m pipeline.generate --limit 5 [--model claude-haiku-4-5] [--context full]
                                [--keep 0.5 --seed 0] [--level system] [--only arnio-pr1234]

Results go to data/runs/<model>__<context>[__<level>]/<id>/.
"""

import argparse
import ast
import datetime
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from . import config
from .gitutil import git
from .harness import check, load_instance

LEVELS = ("unit", "integration", "system")

SYSTEM_PROMPT = (
    "You write pytest {kind} tests for a Python library from its documentation. "
    "Reply with exactly one ```python code block containing a complete pytest file and nothing else."
)
SYSTEM_KIND = {"unit": "unit", "integration": "integration", "system": "system (end-to-end)"}

# The unit wording is the original prompt, unchanged, so earlier runs stay comparable.
LEVEL_INSTRUCTIONS = {
    "unit": """Write pytest tests for these functions: {targets}.
Test the documented behavior, including edge cases the documentation implies.
Use only the public API and pandas/numpy. Keep it to at most 15 focused tests.""",

    "integration": """Write pytest integration tests that exercise these functions: {targets}.
Each test must combine the function under test with at least one other public arnio
component it is documented to work with - for example building input with
ar.from_pandas / ar.read_csv, feeding results into ar.pipeline or other cleaning steps,
validating with a Schema, profiling, or converting back with ar.to_pandas - and check
the combined result. Cover the documented behavior and the edge cases the documentation
implies, as they show up across those interactions.
Use only the public API and pandas/numpy. Keep it to at most 15 focused tests.""",

    "system": """Write pytest system (end-to-end) tests for the parts of arnio that involve: {targets}.
Each test must drive a complete user workflow through arnio's outer interfaces, the way a
user would: real files on disk (use pytest's tmp_path), loading, cleaning / validating /
profiling, and writing results or converting to pandas. Where relevant also use the
pandas `.arnio` accessor or the command line, run as
subprocess.run([sys.executable, "-m", "arnio.cli", ...], capture_output=True, text=True).
Assert on what the user would observe: output files, returned frames, reports, error
types and messages, exit codes. Include documented failure cases and edge cases.
Use only the public API, pandas/numpy and the standard library. Keep it to at most 15 tests.""",
}

PROMPT = """Library: arnio (import arnio as ar), a pandas data-preparation library.

{instructions}

{context}
"""

LIBRARY_LINE = "Library: arnio (import arnio as ar), a pandas data-preparation library."

# In full/partial context, the per-bug part (target docstrings, signatures) follows the shared doc files.
SOURCE_HEADER = "## ===== Source code of the target functions ====="
NOTES_HEADER = ("## ===== Expert notes on the target functions =====\n"
                "(Written by a senior engineer from the full clean source of each module, to help you "
                "write precise tests.)")
PER_BUG_MARKERS = ("## ===== Source docstrings =====", "## ===== Signatures =====", SOURCE_HEADER)


def build_prompt(meta: dict, context: str, level: str = "unit") -> str:
    if level not in LEVELS:
        raise ValueError(f"unknown test level {level!r}; pick one of {LEVELS}")
    return PROMPT.format(instructions=LEVEL_INSTRUCTIONS[level].format(targets=target_names(meta)),
                         context=context)


def build_cached_prompt(meta: dict, context: str, mode: str, level: str = "unit") -> tuple[str, str]:
    """Cache-friendly split of the same content: (shared, per_bug).

    ``shared`` = library line + the doc-file sections, identical for every bug in a
    full/partial condition; it goes in the system prompt, which Claude Code caches
    on its own. ``per_bug`` = target docstrings/signatures + instructions, the user message.
    signatures/targeted contexts are all per-bug (and small), so nothing is shared."""
    if level not in LEVELS:
        raise ValueError(f"unknown test level {level!r}; pick one of {LEVELS}")
    shared_docs, per_bug_ctx = "", context
    if mode in ("full", "partial"):
        cut = min((i for i in (context.find(m) for m in PER_BUG_MARKERS) if i >= 0), default=len(context))
        shared_docs, per_bug_ctx = context[:cut].rstrip(), context[cut:].lstrip()
    shared = LIBRARY_LINE + ("\n\n" + shared_docs if shared_docs else "")
    instructions = LEVEL_INSTRUCTIONS[level].format(targets=target_names(meta))
    per_bug = (per_bug_ctx + "\n\n" if per_bug_ctx else "") + instructions + "\n"
    return shared, per_bug


def arnio_calls_per_test(code: str) -> dict[str, list[str]]:
    """Distinct arnio entry points each test function touches (static, best effort):
    ``ar.x`` / ``arnio.x`` attribute uses, names imported from arnio, ``.arnio`` accessor
    use and CLI subprocess calls. Lets us check a test really is integration/system level."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    aliases, imported = {"arnio"}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases |= {a.asname or a.name for a in node.names if a.name == "arnio"}
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("arnio"):
            imported |= {a.asname or a.name for a in node.names}
    out = {}
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and fn.name.startswith("test")):
            continue
        used = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in aliases:
                used.add(n.attr)
            elif isinstance(n, ast.Attribute) and n.attr == "arnio":
                used.add(".arnio accessor")
            elif isinstance(n, ast.Name) and n.id in imported:
                used.add(n.id)
            elif isinstance(n, ast.Constant) and n.value == "arnio.cli":
                used.add("CLI")
        out[fn.name] = sorted(used)
    return out


def symbol_context(symbol: str, docstring: bool = True, body: bool = False) -> str:
    """Signature (+ docstring) of ``path::Qual.name`` at BASE_REF, without its body.
    With ``body``, the whole (clean, unbugged) definition instead."""
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
    module = rel[:-3].replace("/", ".")
    if body:
        full = "\n".join(lines[node.lineno - 1 - len(node.decorator_list): node.end_lineno])
        return f"# {module}.{qual}\n{full}"
    body_start = node.body[0]
    if docstring and isinstance(body_start, ast.Expr) and isinstance(getattr(body_start, "value", None), ast.Constant):
        end = body_start.end_lineno  # keep the docstring
    else:
        end = body_start.lineno - 1
    header = "\n".join(lines[node.lineno - 1 - len(node.decorator_list): end])
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


_API_INDEX: str | None = None


def api_index() -> str:
    """Compact index of arnio's public API at BASE_REF (the names in ``__all__``):
    function signatures with type hints and return types plus the docstring's first line,
    and for classes their public methods (with @property/@classmethod/... decorators) and
    annotated fields. No bodies, no prose docs. Same for every bug."""
    global _API_INDEX
    if _API_INDEX is not None:
        return _API_INDEX
    init = git("show", f"{config.BASE_REF}:arnio/__init__.py")
    m = re.search(r"__all__\s*=\s*\[(.*?)\]", init, re.S)
    public = set(re.findall(r"['\"](\w+)['\"]", m.group(1))) if m else set()
    files = [f for f in git("ls-tree", "-r", "--name-only", config.BASE_REF, "arnio").split() if f.endswith(".py")]

    def sig(fn: ast.FunctionDef, indent: str = "") -> str:
        decos = "".join(f"{indent}@{ast.unparse(d)}\n" for d in fn.decorator_list)
        ret = f" -> {ast.unparse(fn.returns)}" if fn.returns else ""
        return f"{decos}{indent}def {fn.name}({ast.unparse(fn.args)}){ret}"

    def first_line(node) -> str:
        doc = (ast.get_docstring(node) or "").strip()
        return doc.split("\n")[0] if doc else ""

    out, seen = [], set()
    for f in sorted(files):
        try:
            tree = ast.parse(git("show", f"{config.BASE_REF}:{f}"))
        except SyntaxError:
            continue
        for n in tree.body:
            if not isinstance(n, (ast.FunctionDef, ast.ClassDef)) or n.name not in public or n.name in seen:
                continue
            seen.add(n.name)
            module = f[:-3].replace("/", ".")
            if isinstance(n, ast.FunctionDef):
                doc = first_line(n)
                out.append(f"# {module}\n{sig(n)}" + (f"\n    # {doc}" if doc else ""))
            else:
                bases = f"({', '.join(ast.unparse(b) for b in n.bases)})" if n.bases else ""
                doc = first_line(n)
                members = []
                for b in n.body:
                    if isinstance(b, ast.FunctionDef) and (not b.name.startswith("_") or b.name == "__init__"):
                        members.append(sig(b, "    "))
                    elif isinstance(b, ast.AnnAssign):
                        members.append("    " + ast.unparse(b))
                out.append(f"# {module}\nclass {n.name}{bases}:" + (f"  # {doc}" if doc else "")
                           + ("\n" + "\n".join(members) if members else ""))
    _API_INDEX = "\n\n".join(out)
    return _API_INDEX


def readme_quickstart() -> str:
    """README's quickstart sections: how to build frames and call the API at all.
    Shared by every instance, so it is never ablated."""
    text = git("show", f"{config.BASE_REF}:README.md")
    sections = re.split(r"^(?=## )", text, flags=re.M)
    keep = [s for s in sections if re.match(r"## .*(Quickstart|Quick Example)", s)]
    return "\n".join(keep)


CONTEXT_MODES = ("signatures", "code", "codeapi", "targeted", "source", "full", "partial")

# User-facing docs at BASE_REF. Left out: CHANGELOG (it narrates the bug fixes),
# contributor/process files and internal design notes.
DOC_FILES = (
    "README.md", "API_REFERENCE.md", "AUTO_CLEAN_GUIDE.md", "CLI_REFERENCE.md",
    "DATA_CONTRACT_CI.md", "docs/TROUBLESHOOTING.md", "docs/chunked_workflows.md",
    "docs/custom_pipeline_steps.md", "docs/nullable_dtype_compat.md", "docs/remote.md",
)


def public_symbols(meta: dict) -> list[str]:
    return [s for s in meta["target_symbols"] if not s.split("::")[1].split(".")[-1].startswith("_")]


def target_names(meta: dict) -> str:
    return ", ".join(f"{s.split('::')[0][:-3].replace('/', '.')}.{s.split('::')[1]}"
                     for s in public_symbols(meta))


def doc_sections() -> list[tuple[str, str]]:
    """Every DOC_FILES section (split at #/## headings outside code fences) as (file, text)."""
    out = []
    for rel in DOC_FILES:
        try:
            text = git("show", f"{config.BASE_REF}:{rel}")
        except RuntimeError:
            continue
        current, in_fence = [], False
        for line in text.splitlines():
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
            if not in_fence and re.match(r"#{1,2} ", line) and any(l.strip() for l in current):
                out.append((rel, "\n".join(current).strip()))
                current = []
            current.append(line)
        if any(l.strip() for l in current):
            out.append((rel, "\n".join(current).strip()))
    return out


def build_context(meta: dict, mode: str = "targeted", keep: float = 1.0, seed: int = 0,
                  with_source: bool = False) -> tuple[str, dict]:
    """Return (context text, stats) for one instance under a context mode.
    ``with_source`` (full/partial only) appends the clean code of the target functions;
    it is never cut by partial's sampling. (For targeted + code, use mode ``source``.)"""
    if mode not in CONTEXT_MODES:
        raise ValueError(f"unknown context mode {mode!r}; pick one of {CONTEXT_MODES}")
    if with_source and mode not in ("full", "partial"):
        raise ValueError("with_source applies to full/partial; use mode 'source' for targeted + code")
    symbols = public_symbols(meta)
    signatures = [symbol_context(s, docstring=False) for s in symbols]

    if mode == "signatures":
        text = "## Signatures\n\n" + "\n\n".join(c for c in signatures if c)
        return text, {"mode": mode, "chars": len(text)}

    if mode in ("code", "codeapi"):  # the implementation (+ API index): no README, API reference or doc files
        code = [symbol_context(s, body=True) for s in symbols]
        text = "## Source code of the target functions\n\n" + "\n\n".join(c for c in code if c)
        if mode == "codeapi":
            text = ("## Public API index (signatures only, no bodies)\n\n" + api_index() + "\n\n" + text)
        return text, {"mode": mode, "chars": len(text)}

    docstrings = [symbol_context(s) for s in symbols]
    if mode in ("targeted", "source"):
        names = {s.split("::")[1].split(".")[-1] for s in symbols}
        ref = api_reference_sections(names)
        text = "## Getting started (README)\n\n" + readme_quickstart()
        if mode == "source":  # targeted, but the full clean implementation instead of just the docstring
            code = [symbol_context(s, body=True) for s in symbols]
            text += "\n\n## Source code of the target functions\n\n" + "\n\n".join(c for c in code if c)
        else:
            text += "\n\n## Source signatures and docstrings\n\n" + "\n\n".join(c for c in docstrings if c)
        if ref:
            text += "\n\n## API reference\n\n" + ref
        return text, {"mode": mode, "chars": len(text)}

    # full / partial: one pool of units = doc sections + target docstrings.
    units = doc_sections() + [("docstrings", d) for d in docstrings if d]
    if mode == "full":
        kept = units
    else:
        if not 0.0 <= keep <= 1.0:
            raise ValueError("keep must be between 0 and 1")
        # Rank once per seed: a smaller keep is always a subset of a larger one.
        rank = sorted(range(len(units)),
                      key=lambda i: hashlib.sha256(f"{seed}:{units[i][1]}".encode()).hexdigest())
        chosen = set(rank[: round(keep * len(units))])
        kept = [u for i, u in enumerate(units) if i in chosen]  # back in reading order

    parts, last = [], None
    for rel, sec in kept:
        if rel != last:
            parts.append(f"## ===== {'Source docstrings' if rel == 'docstrings' else rel} =====")
            last = rel
        parts.append(sec)
    kept_docs = {sec for rel, sec in kept if rel == "docstrings"}
    # The model must always know what it is calling, even if the docstring was cut.
    missing = [sig for sig, doc in zip(signatures, docstrings) if sig and doc not in kept_docs]
    if missing and not with_source:  # the source already carries the signatures
        parts.append("## ===== Signatures =====\n\n" + "\n\n".join(missing))
    if with_source:
        code = [symbol_context(s, body=True) for s in symbols]
        parts.append(SOURCE_HEADER + "\n\n" + "\n\n".join(c for c in code if c))
    text = "\n\n".join(parts)
    return text, {"mode": mode, "keep": keep if mode == "partial" else 1.0, "seed": seed,
                  "with_source": with_source,
                  "units_total": len(units), "units_kept": len(kept), "chars": len(text)}


def run_label(model: str, mode: str, keep: float = 1.0, seed: int = 0, level: str = "unit",
              cache_docs: bool = False, max_thinking: int | None = None, with_source: bool = False,
              notes: str | None = None) -> str:
    """Folder name under data/runs/ for one experimental condition (defaults add no suffix)."""
    label = f"{model}__partial{round(keep * 100)}_s{seed}" if mode == "partial" else f"{model}__{mode}"
    if with_source:
        label += "__src"
    if notes:
        label += f"__notes-{notes.removeprefix('claude-')}"
    if level != "unit":
        label += f"__{level}"
    if cache_docs:
        label += "__cachedocs"
    if max_thinking is not None:
        label += f"__think{max_thinking}"
    return label


class UsageLimit(RuntimeError):
    """The subscription's usage limit was hit; ``resume_at`` is when it resets (epoch seconds)."""
    def __init__(self, msg: str, resume_at: float):
        super().__init__(msg)
        self.resume_at = resume_at


def _reset_time(message: str) -> float:
    """Parse 'resets 8:30pm' (local time) from a usage-limit message; fall back to 30 min."""
    m = re.search(r"resets\s+(\d{1,2})(?::(\d{2}))?\s*([ap]m)", message, re.I)
    if not m:
        return time.time() + 1800
    hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
    now = datetime.datetime.now()
    at = now.replace(hour=hour, minute=int(m.group(2) or 0), second=0, microsecond=0)
    if at <= now:
        at += datetime.timedelta(days=1)
    return at.timestamp() + 120  # small buffer past the reset


def call_claude(prompt: str, model: str, level: str = "unit", max_thinking: int | None = None,
                system_extra: str | None = None, system: str | None = None) -> dict:
    """One non-interactive Claude Code call with tools disabled.

    ``system_extra`` is appended to the system prompt (passed as a file: it can be far
    longer than a Windows command line allows); ``system`` replaces the test-writing
    system prompt (e.g. for pipeline.probe). Raises UsageLimit when the subscription
    limit is hit, so callers can wait and retry."""
    env = dict(os.environ)
    if max_thinking is not None:
        env["MAX_THINKING_TOKENS"] = str(max_thinking)
    if system is None:
        system = SYSTEM_PROMPT.format(kind=SYSTEM_KIND[level])
    with tempfile.TemporaryDirectory() as cwd:  # keep project files out of its context
        if system_extra:
            sp_file = Path(cwd) / "system_prompt.txt"
            sp_file.write_text(system + "\n\n" + system_extra, encoding="utf-8")
            system_args = ["--system-prompt-file", str(sp_file)]
        else:
            system_args = ["--system-prompt", system]
        proc = subprocess.run(
            ["claude", "-p", "--model", model, "--tools", "", "--output-format", "json", *system_args],
            input=prompt, cwd=cwd, capture_output=True, text=True, encoding="utf-8", timeout=600, env=env,
        )
    if "usage_limit_reached" in proc.stdout:
        try:
            msg = json.loads(proc.stdout).get("result", "")
        except json.JSONDecodeError:
            msg = proc.stdout[-300:]
        raise UsageLimit(msg, _reset_time(msg))
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p failed: {proc.stderr.strip() or proc.stdout[-500:]}")
    return json.loads(proc.stdout)


def call_claude_waiting(prompt: str, model: str, level: str = "unit", max_thinking: int | None = None,
                        system_extra: str | None = None, log=print, wait: bool = True,
                        system: str | None = None) -> dict:
    """call_claude, but sleep through usage-limit resets instead of failing
    (with ``wait=False``, re-raise UsageLimit so the caller can stop instead).
    The time spent waiting is returned as ``_waited_s`` so timings can leave it out."""
    waited = 0.0
    while True:
        try:
            reply = call_claude(prompt, model, level, max_thinking, system_extra, system)
            reply["_waited_s"] = waited
            return reply
        except UsageLimit as exc:
            if not wait:
                raise
            wait = max(exc.resume_at - time.time(), 60)
            log(f"usage limit hit ({exc}); sleeping {wait / 60:.0f} min until reset")
            time.sleep(wait)
            waited += wait


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


def run_one(meta: dict, model: str, mode: str = "targeted", keep: float = 1.0, seed: int = 0,
            log=print, level: str = "unit", slot: str = "check", cache_docs: bool = False,
            max_thinking: int | None = None, with_source: bool = False,
            wait_on_limit: bool = True, notes: str | None = None) -> dict:
    """``slot`` names the checkout used for scoring; parallel callers need one each.
    ``cache_docs`` moves the shared doc files into the system prompt, which is cached
    across calls (same content, different place; see build_cached_prompt);
    ``max_thinking`` caps the model's thinking tokens (0 turns thinking off);
    ``with_source`` adds the clean target code (full/partial); ``notes`` (an annotator model)
    appends that model's per-function notes on the targets (pipeline.notes), after everything
    else so they stay in the per-bug part of the prompt; ``wait_on_limit=False``
    raises UsageLimit instead of sleeping until the reset."""
    out_root = config.DATA / "runs" / run_label(model, mode, keep, seed, level, cache_docs, max_thinking,
                                                with_source, notes)
    out = out_root / meta["id"]
    out.mkdir(parents=True, exist_ok=True)
    context, ctx_stats = build_context(meta, mode, keep, seed, with_source)
    if notes:
        from .notes import notes_for
        note_text = notes_for(meta, notes)
        context += f"\n\n{NOTES_HEADER}\n\n{note_text}"
        ctx_stats = {**ctx_stats, "notes": notes, "notes_chars": len(note_text), "chars": len(context)}
    if cache_docs:
        shared, prompt = build_cached_prompt(meta, context, mode, level)
        (out / "prompt.txt").write_text(f"=== SYSTEM (appended) ===\n{shared}\n\n=== USER ===\n{prompt}",
                                        encoding="utf-8")
    else:
        shared, prompt = None, build_prompt(meta, context, level)
        (out / "prompt.txt").write_text(prompt, encoding="utf-8")

    t0 = time.time()
    log(f"{meta['id']}: prompt {len(prompt):,} chars; asking {model} ...")
    reply = call_claude_waiting(prompt, model, level, max_thinking, shared, log=log, wait=wait_on_limit)
    t0 += reply.pop("_waited_s", 0.0)  # time asleep at a usage limit is not run time
    code = extract_code(reply.get("result", ""))
    test_file = out / f"test_{meta['id'].replace('-', '_')}.py"
    test_file.write_text(code, encoding="utf-8")

    log(f"{meta['id']}: running generated tests on clean and buggy ...")
    scored = check(meta["id"], test_file, slot=slot)
    n_tests = len(scored["tests"])
    calls = arnio_calls_per_test(code)
    for node, row in scored["tests"].items():
        row["arnio_calls"] = calls.get(node.split("::")[-1].split("[")[0], [])
    result = {
        "id": meta["id"],
        "model": model,
        "run": out_root.name,
        "context": ctx_stats,
        "level": level,
        "cache_docs": cache_docs,
        "max_thinking": max_thinking,
        "with_source": with_source,
        "notes": notes,
        "mean_arnio_calls": round(sum(len(v) for v in calls.values()) / max(len(calls), 1), 2),
        "detected": scored["detected"],
        "n_tests": n_tests,
        "n_catching": sum(r["detects_bug"] for r in scored["tests"].values()),
        "n_false_alarms": sum(r["false_alarm"] for r in scored["tests"].values()),
        "clean_collection_errors": scored["clean_collection_errors"],
        "usage": reply.get("usage", {}),
        "cost_usd": reply.get("total_cost_usd"),
        "model_seconds": round((reply.get("duration_ms") or 0) / 1000, 1),
        "seconds": round(time.time() - t0, 1),
        "tests": scored["tests"],
    }
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    write_summary(out_root)
    return result


def eligible_ids() -> list[str]:
    """Instances with at least one public target (private-only bugs have no docs to show)."""
    metas = [json.loads(l) for l in config.INSTANCES.read_text(encoding="utf-8").splitlines() if l]
    return [m["id"] for m in metas if public_symbols(m)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--context", choices=CONTEXT_MODES, default="targeted")
    ap.add_argument("--keep", type=float, default=0.5, help="fraction of doc sections kept (partial only)")
    ap.add_argument("--seed", type=int, default=0, help="which sections partial keeps")
    ap.add_argument("--level", choices=LEVELS, default="unit", help="kind of tests to ask for")
    ap.add_argument("--cache-docs", action="store_true",
                    help="put shared docs before the per-bug instructions so the prompt cache can reuse them")
    ap.add_argument("--max-thinking", type=int, help="cap thinking tokens (0 = off); for cheap trial runs")
    ap.add_argument("--with-source", action="store_true",
                    help="also show the clean code of the target functions (full/partial; label __src)")
    ap.add_argument("--notes", help="add this annotator model's per-function notes (pipeline.notes)")
    args = ap.parse_args()

    ids = args.only or eligible_ids()[: args.limit]
    for iid in ids:
        meta = load_instance(iid)
        try:
            res = run_one(meta, args.model, args.context, args.keep, args.seed,
                          log=lambda m: print(m, flush=True) if "usage limit" in m else None, level=args.level, cache_docs=args.cache_docs, max_thinking=args.max_thinking,
                          with_source=args.with_source, notes=args.notes)
        except Exception as exc:
            print(f"{iid:16} ERROR {exc}")
            continue
        verdict = "DETECTED" if res["detected"] else "missed"
        print(f"{iid:16} {verdict:8} tests={res['n_tests']:2} catching={res['n_catching']:2} "
              f"false_alarms={res['n_false_alarms']:2} {res['seconds']}s  {meta['target_symbols'][:2]}", flush=True)

    out_root = config.DATA / "runs" / run_label(args.model, args.context, args.keep, args.seed, args.level,
                                                args.cache_docs, args.max_thinking, args.with_source,
                                                args.notes)
    if (out_root / "summary.json").exists():
        s = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
        print(f"\n{out_root.name}: detected {s['detected']}/{s['instances']} bugs; "
              f"{s['instances_with_false_alarms']} with false alarms; {s['mean_tests']:.1f} tests per bug")


def summarize(results: list[dict]) -> dict:
    n = len(results)
    if not n:
        return {"instances": 0}
    return {
        "instances": n,
        "detected": sum(r["detected"] for r in results),
        "detection_rate": sum(r["detected"] for r in results) / n,
        "instances_with_false_alarms": sum(r["n_false_alarms"] > 0 for r in results),
        "mean_tests": sum(r["n_tests"] for r in results) / n,
        "mean_arnio_calls": round(sum(r.get("mean_arnio_calls") or 0 for r in results) / n, 2),
        "cost_usd": round(sum(r.get("cost_usd") or 0 for r in results), 4),
    }


_SUMMARY_LOCK = threading.Lock()


def write_summary(out_root: Path) -> None:
    """Summarize every result in a run folder (not just the latest invocation)."""
    with _SUMMARY_LOCK:
        _write_summary(out_root)


def _write_summary(out_root: Path) -> None:
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_root.glob("*/result.json"))]
    (out_root / "summary.json").write_text(json.dumps(summarize(results), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

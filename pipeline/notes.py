"""Expert notes: a strong model annotates each target function once, for a small test-writer.

A fresh `claude -p` session (no tools, empty temp cwd, nothing about the bugs) gets the
full clean v1.20.0 source of the function's module plus the public API index, and writes
test-oriented contract notes for one public function. Notes are per function, not per bug:
every bug that targets a function sees the same notes. They are added to the test-writing
prompt with `--notes <annotator model>` (pipeline.generate / pipeline.ablation).

    python -m pipeline.notes [--model claude-opus-5-5] [--limit-bugs 50] [--only arnio/cleaning.py::drop_columns]
                             [--workers 3] [--stop-on-limit]

Resumable: notes already written are skipped. Output: data/notes/<model>/<module>__<qual>.md
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from . import config, generate
from .gitutil import git
from .harness import load_instance

NOTES_SYSTEM = """You are a senior Python engineer helping with a research project on LLM-written unit tests.

How the project works:
- A small, fast model (Claude Haiku) is asked to write ONE pytest file of unit tests for one or
  a few functions of the library `arnio` (import arnio as ar), a pandas data-preparation library
  with a compiled C++ core. It writes the file in a single attempt and never runs it.
- It sees part of the library's documentation, the clean source code of the target functions,
  and the notes you write now. It does NOT see the rest of the module, so anything important
  that happens in private helpers or module constants must be in your notes.
- The tests are written against the current, correct code. Later the code will be changed, the
  way real regressions slip in during ordinary development. A test file is good if (1) every test
  passes on the current code and (2) at least one test fails on the changed version. A test that
  fails on the current code is a false alarm and is worthless.
- So the small model needs precise, checkable facts about what the function does now, so that
  its tests pin that behaviour down exactly instead of only checking that nothing crashed.

Your job: write notes on ONE public function (or method) for that small model. Base every
statement on the code you are given (follow the private helpers and constants it uses inside
this module). Do not guess; if something depends on code you cannot see (e.g. the C++ core or
another module), say so.

Use these sections (skip one only if it truly does not apply):
1. Purpose - one or two lines.
2. Inputs - every parameter: accepted types/values, defaults, how it is normalised or validated,
   including checks done by helpers.
3. Raises - every exception the function can raise: exception type, the exact message (or its
   exact f-string template), and a minimal concrete call that triggers it. Note the order of the
   checks when it matters.
4. Returns and guarantees - type of the result, column order, dtypes, values, whether it is a
   new object or the input itself, what metadata is kept, how nulls are handled.
5. Side effects - whether inputs are mutated, global/registry state changed, files written.
6. Branches and edge cases - each distinct code path (early returns, special cases such as empty
   or zero-row input) with a concrete input that reaches it and the observable result.
7. Rules stated in comments or docstrings - intent that the code's comments spell out.
8. Testing hints - how to build inputs with the public API (e.g. ar.from_pandas), how to observe
   each behaviour from outside, and pitfalls that would make a test wrong on the current code
   (exact dtypes, float formatting, ordering, things that are not observable).

Be concrete and terse: bullet points, real values, exact messages. At most ~90 lines. Short
code snippets are fine; do not write a full test file. Reply with the notes in Markdown only."""

NOTES_USER = """Write the notes for `{module}.{qual}` (defined in {rel}; its source is in the module above).
"""


def note_path(model: str, symbol: str):
    rel, qual = symbol.split("::")
    return config.DATA / "notes" / model / f"{rel[:-3].replace('/', '.')}__{qual}.md"


def module_context(rel: str) -> str:
    src = git("show", f"{config.BASE_REF}:{rel}")
    return (f"## Public API index of arnio (signatures only)\n\n{generate.api_index()}\n\n"
            f"## Full source of {rel} (arnio {config.BASE_REF})\n\n```python\n{src}\n```")


def write_note(symbol: str, model: str, wait: bool = True) -> str:
    out = note_path(model, symbol)
    if out.exists():
        return f"{symbol}: already done"
    rel, qual = symbol.split("::")
    module = rel[:-3].replace("/", ".")
    reply = generate.call_claude_waiting(
        NOTES_USER.format(module=module, qual=qual, rel=rel), model,
        system_extra=module_context(rel), system=NOTES_SYSTEM, wait=wait,
        log=lambda m: print(f"  {symbol}: {m}", flush=True))
    text = reply.get("result", "").strip()
    if not text:
        raise RuntimeError("empty reply")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"symbol": symbol, "model": model, "cost_usd": reply.get("total_cost_usd"),
         "usage": reply.get("usage"), "seconds": round((reply.get("duration_ms") or 0) / 1000, 1)},
        indent=1), encoding="utf-8")
    return f"{symbol}: {len(text.splitlines())} lines ${reply.get('total_cost_usd') or 0:.3f}"


def notes_for(meta: dict, model: str) -> str:
    """The notes section for one bug's targets (raises if any note is missing)."""
    parts = []
    for s in generate.public_symbols(meta):
        p = note_path(model, s)
        if not p.exists():
            raise FileNotFoundError(f"no {model} notes for {s}; run python -m pipeline.notes --model {model}")
        rel, qual = s.split("::")
        parts.append(f"### {rel[:-3].replace('/', '.')}.{qual}\n\n{p.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--limit-bugs", type=int, default=50, help="annotate the targets of the first N eligible bugs")
    ap.add_argument("--only", nargs="*", help="specific symbols, e.g. arnio/cleaning.py::drop_columns")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--stop-on-limit", action="store_true")
    args = ap.parse_args()

    if args.only:
        symbols = args.only
    else:
        ids = generate.eligible_ids()[: args.limit_bugs]
        symbols = sorted({s for i in ids for s in generate.public_symbols(load_instance(i))})
    todo = [s for s in symbols if not note_path(args.model, s).exists()]
    print(f"{len(symbols)} symbols, {len(todo)} to annotate with {args.model}", flush=True)

    # One module at a time: its first call writes the module to the prompt cache, the rest reuse it.
    by_module: dict[str, list[str]] = {}
    for s in todo:
        by_module.setdefault(s.split("::")[0], []).append(s)
    stop = threading.Event()

    def work(s):
        if stop.is_set():
            return None
        try:
            return write_note(s, args.model, wait=not args.stop_on_limit)
        except generate.UsageLimit as exc:
            stop.set()
            return f"{s}: LIMIT {exc} -- re-run to resume"
        except Exception as exc:
            return f"{s}: ERROR {exc}"

    k = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for rel, syms in by_module.items():
            for phase in (syms[:1], syms[1:]):
                for line in pool.map(work, phase):
                    k += 1
                    if line:
                        print(f"[{k}/{len(todo)}] {line}", flush=True)
            if stop.is_set():
                break
    cost = sum(json.loads(p.read_text(encoding="utf-8")).get("cost_usd") or 0
               for p in (config.DATA / "notes" / args.model).glob("*.json"))
    print(f"total notes cost so far: ${cost:.2f}")


if __name__ == "__main__":
    main()

"""Self-assessment probe: before writing tests, can the model tell which bugs it will miss?

For each bug in a control run, the model sees exactly the control's context (read back
from the control's prompt.txt: same docs, same target source) but, instead of the
test-writing instruction, is asked whether its one-shot tests would catch a bug that will
later be injected into the target functions (it is not told the bug). Its yes/no and
p_catch are then compared with whether the control run actually caught the bug.

    python -m pipeline.probe [--control claude-haiku-4-5__partial50_s0__src__cachedocs]
                             [--limit 50] [--samples 1] [--workers 4] [--stop-on-limit]
    python -m pipeline.probe --analyze-only

Resumable: answers already saved are skipped. Answers land in data/probe/<control>/.
"""

import argparse
import json
import math
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from . import config, generate
from .harness import load_instance

CONTROL = "claude-haiku-4-5__partial50_s0__src__cachedocs"
# Other code-seeing runs, used as a less noisy "how hard is this bug" label (0-4 catches).
DIFFICULTY_RUNS = ["claude-haiku-4-5__code", "claude-haiku-4-5__codeapi",
                   "claude-haiku-4-5__full__src__cachedocs", CONTROL]

PROBE_SYSTEM = ("You assess how testable Python library code is, before any tests are written. "
                "Reply with exactly one JSON object and nothing else.")

PROBE_QUESTION = """Do NOT write any tests yet. First, predict how well you would do.

The task you would get: write pytest unit tests for these functions: {targets}, from the
documentation and source code shown above. At most 15 focused tests, public API and
pandas/numpy only, one attempt, no chance to run them. The tests are written against
this (correct) version of the code.

Afterwards, a realistic bug will be introduced into these target functions - the kind of
regression a real pull request later fixes. You are not told what it is. Your tests are
then run on the buggy version: the bug counts as caught if at least one test that passes
on the correct code fails on the buggy code.

How likely is it that your single attempt catches it? Think about which behaviours of
these functions are easy to pin down with exact assertions through the public API and
which are hard (subtle internals, exact error messages, input-type validation,
copying/aliasing, metadata, behaviour you cannot observe from outside, ...).

Reply with only this JSON object:
{{"verdict": "yes" or "no" (yes = you expect your tests would catch it),
 "p_catch": integer 0-100,
 "hardest_part": "one sentence: the behaviour a bug could break that your tests would most likely miss"}}
"""

UNIT_INSTRUCTION_START = "Write pytest tests for these functions:"


def control_prompt(control: str, iid: str) -> tuple[str, str]:
    """(shared system docs, per-bug context without the test-writing instruction)."""
    text = (config.DATA / "runs" / control / iid / "prompt.txt").read_text(encoding="utf-8")
    if text.startswith("=== SYSTEM (appended) ==="):
        shared, user = text.split("\n\n=== USER ===\n", 1)
        shared = shared.removeprefix("=== SYSTEM (appended) ===\n")
    else:
        shared, user = None, text
    cut = user.rfind(UNIT_INSTRUCTION_START)
    if cut < 0:
        raise ValueError(f"{iid}: unit instruction not found in control prompt")
    return shared, user[:cut].rstrip()


def parse_answer(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON in reply: {text[:200]!r}")
    a, _ = json.JSONDecoder().raw_decode(text[start:])  # first object; ignore any trailing prose
    verdict = str(a.get("verdict", "")).strip().lower()
    if verdict not in ("yes", "no"):
        raise ValueError(f"bad verdict {verdict!r}")
    return {"verdict": verdict, "p_catch": int(a["p_catch"]), "hardest_part": a.get("hardest_part", "")}


def probe_one(control: str, iid: str, sample: int, model: str, out_dir, wait: bool) -> str:
    out = out_dir / f"{iid}__s{sample}.json"
    if out.exists():
        return f"{iid} s{sample}: already done"
    shared, ctx = control_prompt(control, iid)
    meta = load_instance(iid)
    prompt = ctx + "\n\n" + PROBE_QUESTION.format(targets=generate.target_names(meta))
    reply = generate.call_claude_waiting(prompt, model, system_extra=shared, system=PROBE_SYSTEM,
                                         log=lambda m: print(f"  {iid}: {m}", flush=True), wait=wait)
    raw = reply.get("result", "")
    try:
        ans = parse_answer(raw)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        ans = {"error": str(exc)}
    out.write_text(json.dumps({"id": iid, "sample": sample, "model": model, "control": control,
                               **ans, "raw": raw, "cost_usd": reply.get("total_cost_usd"),
                               "usage": reply.get("usage")}, indent=1), encoding="utf-8")
    return f"{iid} s{sample}: {ans.get('verdict', 'ERR')} p={ans.get('p_catch', '?')} ${reply.get('total_cost_usd') or 0:.3f}"


# ---------- analysis (no scipy in the venv) ----------

def auroc(scores: list[float], labels: list[bool]) -> float:
    """P(score of a random positive > score of a random negative), ties count half."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def auroc_perm_p(scores, labels, n_perm: int = 20000, seed: int = 0) -> float:
    """Two-sided permutation p-value for AUROC != 0.5."""
    import random
    rng = random.Random(seed)
    obs = abs(auroc(scores, labels) - 0.5)
    labs = list(labels)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(labs)
        hits += abs(auroc(scores, labs) - 0.5) >= obs - 1e-12
    return (hits + 1) / (n_perm + 1)


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Fisher's exact test on [[a, b], [c, d]]."""
    n, r1, c1 = a + b + c + d, a + b, a + c

    def p(x):
        return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    obs = p(a)
    lo, hi = max(0, r1 + c1 - n), min(r1, c1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs * (1 + 1e-9)))


def spearman(x, y) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (sx * sy) if sx and sy else float("nan")


def detected(run: str, iid: str):
    p = config.DATA / "runs" / run / iid / "result.json"
    return json.loads(p.read_text(encoding="utf-8"))["detected"] if p.exists() else None


def analyze(control: str, ids: list[str], out_dir) -> str:
    rows = []
    for iid in ids:
        answers = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob(f"{iid}__s*.json"))]
        answers = [a for a in answers if "error" not in a]
        caught = detected(control, iid)
        if not answers or caught is None:
            continue
        diff = [detected(r, iid) for r in DIFFICULTY_RUNS]
        _, ctx = control_prompt(control, iid)
        src = ctx[ctx.find(generate.SOURCE_HEADER):] if generate.SOURCE_HEADER in ctx else ctx
        rows.append({
            "id": iid, "caught": caught,
            "p": sum(a["p_catch"] for a in answers) / len(answers),
            "yes_frac": sum(a["verdict"] == "yes" for a in answers) / len(answers),
            "n_catch_runs": sum(bool(d) for d in diff if d is not None),
            "src_chars": len(src), "n_targets": len(generate.public_symbols(load_instance(iid))),
            "hardest": answers[0].get("hardest_part", ""),
        })
    if not rows:
        return "no probe answers yet"
    lab = [r["caught"] for r in rows]
    n, n_caught = len(rows), sum(lab)
    yes = [r["yes_frac"] >= 0.5 for r in rows]
    a = sum(y and l for y, l in zip(yes, lab)); b = sum(y and not l for y, l in zip(yes, lab))
    c = sum(not y and l for y, l in zip(yes, lab)); d = sum(not y and not l for y, l in zip(yes, lab))
    ps = [r["p"] for r in rows]
    out = [f"Probe vs control {control}: {n} bugs, control caught {n_caught}/{n}",
           f"cost ${sum((json.loads(p.read_text(encoding='utf-8')).get('cost_usd') or 0) for p in out_dir.glob('*.json')):.2f}",
           "",
           "Verdict vs outcome     caught  missed",
           f"  said yes             {a:6}  {b:6}",
           f"  said no              {c:6}  {d:6}",
           f"  accuracy {(a + d) / n:.2f} (always-yes baseline {max(n_caught, n - n_caught) / n:.2f}); "
           f"Fisher p={fisher_two_sided(a, b, c, d):.3f}",
           "",
           f"p_catch: mean {sum(ps) / n:.0f}, range {min(ps):.0f}-{max(ps):.0f}, "
           f"{len(set(ps))} distinct values",
           f"  mean p_catch | caught {sum(p for p, l in zip(ps, lab) if l) / max(n_caught, 1):.1f}"
           f"  | missed {sum(p for p, l in zip(ps, lab) if not l) / max(n - n_caught, 1):.1f}",
           "",
           "AUROC for predicting 'caught' (0.5 = chance; permutation p):"]
    preds = {"p_catch (model)": ps,
             "-source chars (baseline)": [-r["src_chars"] for r in rows],
             "-n targets (baseline)": [-r["n_targets"] for r in rows]}
    for name, s in preds.items():
        out.append(f"  {name:26} {auroc(s, lab):.2f}  p={auroc_perm_p(s, lab):.3f}")
    out += ["", "Against difficulty = # of 4 code runs that caught it (code, codeapi, full+src, p50+src):",
            f"  Spearman(p_catch, n_catch_runs) = {spearman(ps, [r['n_catch_runs'] for r in rows]):.2f}",
            f"  Spearman(-src_chars, n_catch_runs) = {spearman([-r['src_chars'] for r in rows], [r['n_catch_runs'] for r in rows]):.2f}",
            "", "Escalation sim: flag the k lowest-p_catch bugs (ties broken by id)",
            f"  base miss rate {(n - n_caught) / n:.2f}"]
    order = sorted(rows, key=lambda r: (r["p"], r["id"]))
    for k in (5, 10, 15, 20):
        if k > n:
            break
        missed = sum(not r["caught"] for r in order[:k])
        out.append(f"  k={k:2}: {missed}/{k} flagged were missed ({missed / k:.2f}); "
                   f"covers {missed}/{n - n_caught} of all misses")
    out += ["", f"{'id':14} {'caught':6} {'runs':>4} {'p':>5} {'yes':>4}  hardest part"]
    for r in order:
        out.append(f"{r['id']:14} {'Y' if r['caught'] else '-':6} {r['n_catch_runs']:4} {r['p']:5.0f} "
                   f"{r['yes_frac']:4.1f}  {r['hardest'][:90]}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default=CONTROL)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--samples", type=int, default=1, help="independent answers per bug (averaged)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stop-on-limit", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    ids = [i for i in generate.eligible_ids()
           if (config.DATA / "runs" / args.control / i / "result.json").exists()][: args.limit]
    out_dir = config.DATA / "probe" / args.control
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        todo = [(i, s) for s in range(args.samples) for i in ids if not (out_dir / f"{i}__s{s}.json").exists()]
        print(f"{len(todo)} probe calls to do", flush=True)
        stop = threading.Event()

        def work(t):
            if stop.is_set():
                return None
            try:
                return probe_one(args.control, t[0], t[1], args.model, out_dir, not args.stop_on_limit)
            except generate.UsageLimit as exc:
                stop.set()
                return f"{t[0]}: LIMIT {exc} -- re-run to resume"
            except Exception as exc:
                return f"{t[0]}: ERROR {exc}"

        # Warm the prompt cache with one call before going parallel.
        phases = [todo[:1], todo[1:]] if todo else []
        k = 0
        with ThreadPoolExecutor(args.workers) as pool:
            for phase in phases:
                for line in pool.map(work, phase):
                    k += 1
                    if line:
                        print(f"[{k}/{len(todo)}] {line}", flush=True)

    for f in out_dir.glob("*__s*.json"):  # re-parse saved replies (parser fixes need no new calls)
        a = json.loads(f.read_text(encoding="utf-8"))
        if "error" in a:
            try:
                a = {k: v for k, v in a.items() if k != "error"} | parse_answer(a["raw"])
                f.write_text(json.dumps(a, indent=1), encoding="utf-8")
            except (ValueError, KeyError, json.JSONDecodeError):
                pass
    report = analyze(args.control, ids, out_dir)
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n" + report)


if __name__ == "__main__":
    main()

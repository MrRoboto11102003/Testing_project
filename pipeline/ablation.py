"""Run the doc-ablation grid: every context condition on the same bugs.

    python -m pipeline.ablation [--limit 20] [--model claude-haiku-4-5] [--seed 0] [--level unit]
                                [--conditions targeted,partial50,partial50+src,full] [--workers 4]
                                [--cache-docs] [--max-thinking 0] [--stop-on-limit]

Skips (bug, condition) pairs that already have a result, so it can be resumed: re-run the
same command. Runs go bug by bug (each bug's conditions back to back), so a run cut short
still has matched pairs.
At a subscription usage limit, workers sleep until the reset and carry on; with
--stop-on-limit they stop instead (re-run later to continue).
With --cache-docs, the first bug of each condition runs before the rest so the shared
docs are in the prompt cache when the parallel calls start.
Conditions ending in "+src" also show the clean code of the target functions (label __src).
Prints a summary table at the end; per-run results land in data/runs/<model>__<condition>/.
"""

import argparse
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

from . import config, generate
from .harness import load_instance, materialize

# (mode, keep, with_source) in order of increasing documentation.
CONDITIONS = [("signatures", 1.0, False), ("code", 1.0, False), ("codeapi", 1.0, False), ("targeted", 1.0, False), ("partial", 0.25, False),
              ("partial", 0.5, False), ("partial", 0.75, False), ("full", 1.0, False),
              ("source", 1.0, False),  # control: targeted + the clean implementation
              ("partial", 0.25, True), ("partial", 0.5, True), ("partial", 0.75, True),
              ("full", 1.0, True)]


def condition_name(mode: str, keep: float, with_source: bool = False) -> str:
    base = f"partial{round(keep * 100)}" if mode == "partial" else mode
    return base + ("+src" if with_source else "")


def table(model: str, seed: int, ids: list[str], level: str = "unit", cache_docs: bool = False,
          max_thinking: int | None = None, notes: str | None = None) -> str:
    lines = [f"{'condition':52} {'bugs':>4} {'caught':>6} {'w/ FA':>5} {'tests':>5} {'FA tests':>8} {'calls/test':>10} {'cost':>6}"]
    for mode, keep, src in CONDITIONS:
        label = generate.run_label(model, mode, keep, seed, level, cache_docs, max_thinking, src, notes)
        rs = []
        for iid in ids:
            p = config.DATA / "runs" / label / iid / "result.json"
            if p.exists():
                rs.append(json.loads(p.read_text(encoding="utf-8")))
        if not rs:
            continue
        lines.append(f"{label:52} {len(rs):4} {sum(r['detected'] for r in rs):6} "
                     f"{sum(r['n_false_alarms'] > 0 for r in rs):5} {sum(r['n_tests'] for r in rs):5} "
                     f"{sum(r['n_false_alarms'] for r in rs):8} "
                     f"{sum(r.get('mean_arnio_calls') or 0 for r in rs) / len(rs):10.1f} {sum(r.get('cost_usd') or 0 for r in rs):6.2f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", choices=generate.LEVELS, default="unit")
    ap.add_argument("--conditions", help="comma-separated subset, e.g. targeted,partial50,partial50+src")
    ap.add_argument("--workers", type=int, default=4,
                    help="runs in parallel, each with its own checkout (model calls dominate the time)")
    ap.add_argument("--cache-docs", action="store_true",
                    help="shared docs before per-bug instructions, so the prompt cache reuses them")
    ap.add_argument("--max-thinking", type=int, help="cap thinking tokens (0 = off); for cheap trial runs")
    ap.add_argument("--stop-on-limit", action="store_true",
                    help="stop at a usage limit instead of sleeping until the reset (re-run to resume)")
    ap.add_argument("--notes", help="add this annotator model's per-function notes (pipeline.notes); label __notes-...")
    ap.add_argument("--only", nargs="*", help="run exactly these bug ids instead of the first --limit")
    args = ap.parse_args()

    conditions = CONDITIONS
    if args.conditions:
        names = {c.strip() for c in args.conditions.split(",")}
        valid = {condition_name(*c) for c in CONDITIONS}
        if names - valid:
            raise SystemExit(f"unknown conditions {sorted(names - valid)}; pick from {sorted(valid)}")
        conditions = [c for c in CONDITIONS if condition_name(*c) in names]

    ids = args.only or generate.eligible_ids()[: args.limit]
    todo = []
    for iid in ids:  # bug-major: each bug's conditions run back to back
        for mode, keep, src in conditions:
            label = generate.run_label(args.model, mode, keep, args.seed, args.level,
                                       args.cache_docs, args.max_thinking, src, args.notes)
            if not (config.DATA / "runs" / label / iid / "result.json").exists():
                todo.append((label, mode, keep, src, iid))
    print(f"{len(todo)} runs to do with {args.workers} worker(s)", flush=True)

    # One scoring checkout per worker. Create them up front, one at a time:
    # concurrent `git worktree add` calls contend for the same lock.
    slots: queue.Queue[str] = queue.Queue()
    for w in range(args.workers):
        name = f"check-w{w}"
        materialize("clean", slot=name)
        slots.put(name)

    stop = threading.Event()

    def work(task):
        label, mode, keep, src, iid = task
        if stop.is_set():
            return None
        slot = slots.get()
        try:
            r = generate.run_one(load_instance(iid), args.model, mode, keep, args.seed,
                                 log=lambda m: print(f"  {label} {iid}: {m}", flush=True)
                                 if "usage limit" in m else None,
                                 level=args.level, slot=slot, cache_docs=args.cache_docs,
                                 max_thinking=args.max_thinking, with_source=src,
                                 wait_on_limit=not args.stop_on_limit, notes=args.notes)
            return (f"{label:52} {iid:14} {'CAUGHT' if r['detected'] else 'missed':6} "
                    f"tests={r['n_tests']:2} catch={r['n_catching']:2} fa={r['n_false_alarms']:2} "
                    f"{r['context']['chars']:>7,} chars {r['seconds']}s ${r.get('cost_usd') or 0:.3f}")
        except generate.UsageLimit as exc:
            stop.set()
            return f"{label:52} {iid:14} LIMIT {exc} -- stopping; re-run the same command to resume"
        except Exception as exc:
            return f"{label:52} {iid:14} ERROR {exc}"
        finally:
            slots.put(slot)

    # Warm-up: the first bug of each condition writes its docs to the cache before the
    # others start (calls launched together can't read each other's cache).
    phases = [todo]
    if args.cache_docs:
        first = {}
        for t in todo:
            first.setdefault(t[0], t)
        phases = [list(first.values()), [t for t in todo if t not in first.values()]]

    k = done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for phase in phases:
            for line in pool.map(work, phase):
                k += 1
                if line is None:
                    continue
                done += 1
                print(f"[{k}/{len(todo)}] {line}", flush=True)
    if stop.is_set():
        print(f"\nStopped at the usage limit; {len(todo) - done} runs left. Re-run the same command to resume.")
    print("\n" + table(args.model, args.seed, ids, args.level, args.cache_docs, args.max_thinking, args.notes))


if __name__ == "__main__":
    main()

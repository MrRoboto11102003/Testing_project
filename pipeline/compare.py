"""Paired comparison of the main conditions on the 50 ablation bugs (no model calls).

    python -m pipeline.compare [--markdown]
    TP_PROJECT=openlens python -m pipeline.compare

Prints, per condition: bugs caught, false-alarm tests (fail on clean), bugs with any false
alarm, model cost, and the test-quality metrics when they exist; the project's own suite
(pipeline.baseline, fix tests left out) as a reference row; then exact two-sided McNemar
tests on the bugs both conditions were run on.
"""

import argparse
import json
from math import comb

from . import config

H = "claude-haiku-4-5__"
CONDITIONS = {  # label -> run folder
    "code": H + "code",
    "full+src": H + "full__src__cachedocs",
    "p50+src": H + "partial50_s0__src__cachedocs",
    "p50+src+notes": H + "partial50_s0__src__notes-opus-5-5__cachedocs",
    "code+notes": H + "code__notes-opus-5-5",
    # arnio only (the docs-only runs, full/p50 without code, are left out on purpose)
    "codeapi": H + "codeapi",
    "code (system)": H + "code__system",
    "full+src (system)": H + "full__src__system__cachedocs",
    "p50+src (system)": H + "partial50_s0__src__system__cachedocs",
}
PAIRS = [
    ("p50+src+notes", "p50+src"), ("p50+src+notes", "full+src"), ("p50+src+notes", "code"),
    ("code+notes", "code"), ("code+notes", "p50+src+notes"),
    ("full+src", "code"), ("p50+src", "code"), ("full+src", "p50+src"),
    ("suite", "code"), ("suite", "p50+src+notes"),
]


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p from the discordant counts."""
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def load() -> dict[str, dict[str, dict]]:
    """{label: {bug id: row}} with detected, false alarms, cost and metrics."""
    out = {}
    for label, run in CONDITIONS.items():
        d = config.DATA / "runs" / run
        rows = {}
        for p in d.glob("*/result.json"):
            r = json.loads(p.read_text(encoding="utf-8"))
            m = p.parent / "metrics.json"
            rows[r["id"]] = {"detected": r["detected"], "fa": r["n_false_alarms"], "cost": r.get("cost_usd") or 0,
                             "metrics": json.loads(m.read_text(encoding="utf-8")) if m.exists() else None}
        if len(rows) >= 50:
            out[label] = rows
    b = config.DATA / "results_baseline.csv"
    if b.exists():
        import csv
        with b.open(encoding="utf-8") as fh:
            out["suite"] = {r["id"]: {"detected": r["detected"] == "True", "fa": None, "cost": 0,
                                      "metrics": {"reached_bug": r["reached_bug"] == "True",
                                                  "reached_but_missed": r["reached_but_missed"] == "True",
                                                  "target_line_cov": float(r["target_line_cov"] or 0),
                                                  "target_branch_cov": None, "assertions_per_test": None,
                                                  "clean_pass_rate": None}}
                            for r in csv.DictReader(fh)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true", help="print markdown tables")
    ap.add_argument("--per-bug", action="store_true", help="also print which condition caught each bug")
    args = ap.parse_args()
    data = load()
    ids = sorted(set.intersection(*(set(v) for v in data.values())))
    mean = lambda xs: None if not xs else sum(xs) / len(xs)  # noqa: E731
    f2 = lambda x: "" if x is None else f"{x:.2f}"  # noqa: E731

    cols = ["condition", "caught", "FA tests", "bugs w/ FA", "cost $", "reached", "reach+miss",
            "tgt line", "tgt branch", "asserts/test", "clean pass"]
    lines = []
    for label, rows in data.items():
        rs = [rows[i] for i in ids]
        ms = [r["metrics"] for r in rs if r["metrics"]]
        full = len(ms) == len(rs)
        get = lambda k: f2(mean([m[k] for m in ms if m.get(k) is not None])) if full else ""  # noqa: E731
        fa = [r["fa"] for r in rs if r["fa"] is not None]
        lines.append([label, str(sum(r["detected"] for r in rs)),
                      str(sum(fa)) if fa else "", str(sum(x > 0 for x in fa)) if fa else "",
                      f"{sum(r['cost'] for r in rs):.2f}" if label != "suite" else "",
                      str(sum(m["reached_bug"] for m in ms)) if full else "",
                      str(sum(m["reached_but_missed"] for m in ms)) if full else "",
                      get("target_line_cov"), get("target_branch_cov"), get("assertions_per_test"),
                      get("clean_pass_rate")])
    print(f"{config.PROJECT}: {len(ids)} bugs common to all conditions\n")
    if args.markdown:
        print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
        for ln in lines:
            print("| " + " | ".join(ln) + " |")
    else:
        w = [max(len(c), *(len(ln[k]) for ln in lines)) for k, c in enumerate(cols)]
        print("  ".join(c.rjust(w[k]) if k else c.ljust(w[k]) for k, c in enumerate(cols)))
        for ln in lines:
            print("  ".join(x.rjust(w[k]) if k else x.ljust(w[k]) for k, x in enumerate(ln)))

    print("\npaired (exact McNemar): A-only / B-only, p")
    for a, b in PAIRS:
        if a in data and b in data:
            x = sum(data[a][i]["detected"] and not data[b][i]["detected"] for i in ids)
            y = sum(data[b][i]["detected"] and not data[a][i]["detected"] for i in ids)
            print(f"  {a:>15} vs {b:<15} {x:2} / {y:2}  p={mcnemar(x, y):.2g}")
    union = [i for i in ids if any(data[k][i]["detected"] for k in data if k != "suite")]
    print(f"\ncaught by at least one model condition: {len(union)}; never caught: {len(ids) - len(union)}")

    if args.per_bug:
        labels = list(data)
        print("\n| bug | " + " | ".join(labels) + " | caught by |\n|" + "---|" * (len(labels) + 2))
        for i in sorted(ids, key=lambda s: int(s.rsplit("pr", 1)[1])):
            cells = ["x" if data[k][i]["detected"] else "" for k in labels]
            print(f"| {i.rsplit('-', 1)[1]} | " + " | ".join(cells) + f" | {cells.count('x')} |")


if __name__ == "__main__":
    main()

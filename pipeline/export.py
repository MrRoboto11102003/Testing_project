"""Export generated-test results to CSV sheets (open in Excel / Google Sheets).

    python -m pipeline.export

Writes data/results_tests.csv (one row per generated test) and
data/results_bugs.csv (one row per bug per model).
"""

import csv
import json

from . import config


def main() -> None:
    test_rows, bug_rows = [], []
    for result_path in sorted((config.DATA / "runs").glob("*/*/result.json")):
        r = json.loads(result_path.read_text(encoding="utf-8"))
        meta = json.loads((config.INSTANCE_DIR / r["id"] / "meta.json").read_text(encoding="utf-8"))
        bug_rows.append({
            "model": r["model"], "bug": r["id"], "issue_title": meta["issue_title"],
            "detected": r["detected"], "tests": r["n_tests"], "catching": r["n_catching"],
            "false_alarms": r["n_false_alarms"], "cost_usd": r["cost_usd"], "seconds": r["seconds"],
        })
        for node, t in r["tests"].items():
            verdict = "CATCH" if t["detects_bug"] else "FALSE ALARM" if t["false_alarm"] else ""
            test_rows.append({
                "model": r["model"], "bug": r["id"], "test": node.split("::")[-1],
                "clean": t["clean"], "buggy": t["buggy"], "verdict": verdict,
            })

    for name, rows in (("results_bugs.csv", bug_rows), ("results_tests.csv", test_rows)):
        path = config.DATA / name
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["empty"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()

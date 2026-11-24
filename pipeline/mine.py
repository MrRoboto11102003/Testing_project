"""Step 1: mine closed bug issues and the merged PRs that fixed them.

Writes the raw GitHub payload to data/raw/issues.json and one candidate row per
(issue, fix PR) pair to data/candidates.jsonl, with a status explaining whether
the pair can be used against BASE_REF.

    python -m pipeline.mine
"""

import json

from . import config
from .gitutil import ensure_clone, gh_graphql, git, has_commit, is_ancestor

QUERY = """
query($owner: String!, $name: String!, $labels: [String!], $after: String) {
  repository(owner: $owner, name: $name) {
    issues(states: CLOSED, labels: $labels, first: 50, after: $after,
           orderBy: {field: CREATED_AT, direction: ASC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title body url createdAt closedAt
        author { login }
        labels(first: 20) { nodes { name } }
        closedByPullRequestsReferences(first: 5, includeClosedPrs: false) {
          nodes {
            number title body url merged mergedAt
            author { login }
            mergeCommit { oid }
          }
        }
      }
    }
  }
}
"""


def fetch_issues() -> list[dict]:
    owner, name = config.REPO_SLUG.split("/")
    issues, after = [], None
    while True:
        # gh -f can't pass a list, so inline the labels literal.
        q = QUERY.replace("$labels: [String!], ", "").replace(
            "labels: $labels", "labels: " + json.dumps(config.BUG_LABELS)
        )
        data = gh_graphql(q, owner=owner, name=name, after=after)
        page = data["repository"]["issues"]
        issues += page["nodes"]
        print(f"  fetched {len(issues)} issues", flush=True)
        if not page["pageInfo"]["hasNextPage"]:
            return issues
        after = page["pageInfo"]["endCursor"]


def changed_files(commit: str) -> list[str]:
    # First parent covers both merge commits and squash merges.
    out = git("diff", "--name-only", f"{commit}^1", commit)
    return [config.alias_path(line) for line in out.splitlines() if line]


def classify(issue: dict, pr: dict) -> dict:
    oid = (pr.get("mergeCommit") or {}).get("oid")
    row = {
        "issue": issue["number"],
        "issue_title": issue["title"],
        "pr": pr["number"],
        "pr_title": pr["title"],
        "merged_at": pr["mergedAt"],
        "fix_commit": oid,
    }
    if not oid or not has_commit(oid):
        return {**row, "status": "commit_missing"}
    if not is_ancestor(oid, config.BASE_REF):
        return {**row, "status": "after_base"}
    files = changed_files(oid)
    src = [f for f in files if f.startswith(config.SRC_PREFIX) and f.endswith(".py")
           and not f.startswith(config.EXCLUDED_SRC_PREFIXES)]
    tests = [f for f in files if f.startswith(config.TEST_PREFIX) and f.endswith(".py")]
    native = [f for f in files if f.startswith(config.NATIVE_PREFIXES)]
    row.update(src_files=src, test_files=tests, native_files=native)
    if native:
        status = "touches_native"
    elif not src:
        status = "no_src_change"
    elif not tests:
        status = "no_test_change"
    else:
        status = "candidate"
    return {**row, "status": status}


def main() -> None:
    ensure_clone()
    issues = fetch_issues()
    config.RAW_ISSUES.parent.mkdir(parents=True, exist_ok=True)
    config.RAW_ISSUES.write_text(json.dumps(issues, indent=1), encoding="utf-8")

    rows, seen_prs = [], set()
    for issue in issues:
        prs = [p for p in issue["closedByPullRequestsReferences"]["nodes"] if p["merged"]]
        if not prs:
            rows.append({"issue": issue["number"], "issue_title": issue["title"], "status": "no_merged_pr"})
        for pr in prs:
            # One PR closing several issues becomes a single instance.
            if pr["number"] in seen_prs:
                continue
            seen_prs.add(pr["number"])
            rows.append(classify(issue, pr))

    with config.CANDIDATES.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"{len(issues)} issues -> {len(rows)} rows")
    for status, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {status:16} {n}")


if __name__ == "__main__":
    main()

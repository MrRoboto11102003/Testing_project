# Notes: `ValidationResult.to_markdown(self, *, max_issues=None, redact_values=False) -> str`

## 1. Purpose
- Renders a `ValidationResult` as a Markdown report: a header block, then an optional issue table and an optional "Showing N of M" footer.
- Pure Python. It does not depend on the C++ core.

## 2. Inputs
- `max_issues`: `int`, `None`, or default `None`.
  - Must satisfy `isinstance(x, int)` and must not be a `bool`.
  - `numpy.int64(2)` is **not** an `int` here, so it raises `TypeError`.
  - `0` is allowed. Negative values are not.
- `redact_values`: must be exactly a Python `bool`. `0` and `1` are rejected.
- Both parameters are keyword-only. `to_markdown(5)` raises Python's own `TypeError` about positional arguments.
- `self` fields used:
  - `passed`: True when no issue has `severity == "error"`.
  - `row_count`, `issue_count`, `issues`.
  - `len(self.bad_rows)`: only the length is used; the contents are never checked against the issues.

## 3. Raises
The checks run in the order below, **before** any early return, so they also fire on a result that passed.

| Check | Exception | Exact message | Trigger |
|---|---|---|---|
| 1 | `TypeError` | `"max_issues must be an integer or None"` | `r.to_markdown(max_issues="2")`, `max_issues=True`, `max_issues=1.0` |
| 2 | `ValueError` | `"max_issues must be non-negative"` | `r.to_markdown(max_issues=-1)` |
| 3 | `TypeError` | `"redact_values must be a bool"` | `r.to_markdown(redact_values=1)` |

- Order matters: `max_issues=-1, redact_values="x"` raises the `ValueError`, not the `TypeError`.

## 4. Returns and guarantees
- Returns a `str` made of lines joined with `"\n"`. There is **no trailing newline**.
- The header is always these lines:
  ```
  ## Validation Report
  
  - Status: **passed**|**failed**
  - Rows checked: {row_count}
  - Issues found: {issue_count}
  - Bad rows: {len(bad_rows)}
  ```
- The table is introduced by `""`, then `"| Column | Rule | Severity | Row | Value | Message |"`, then `"|---|---|---|---|---|---|"`.
- Each issue row is `f"| {col} | {rule} | {sev} | {row} | {value} | {msg} |"`. Rows appear in `issues` order.
- Cell escaping (`_markdown_cell`):
  - `None` becomes `""`.
  - Any other value becomes `str(v)`, then `"\n"` is replaced by `"<br>"` and `"|"` by `"\\|"` (a backslash followed by a pipe).
- The Value cell is passed through `_clean_scalar` before escaping:
  - NaN, `None`, `pd.NA` → `""`.
  - numpy scalars → `.item()`, so `np.int64(5)` → `"5"`.
  - `pd.Timestamp` → `.isoformat()`.
  - list, tuple, or set → a list, e.g. `"[1, 2]"`.
  - `True` → `"True"`, `-5` → `"-5"`, `1.5` → `"1.5"`.
- When `redact_values=True`, every Value cell is the literal `[REDACTED]`, even if the value is `None`. The other cells are unchanged.
- An empty string value `""` renders as an empty cell, i.e. `"|  |"` (two spaces).
- Row cell: `row_index=None` → `""`; otherwise the int as text.

## 5. Side effects
- None. The result object is frozen and is not modified.

## 6. Branches and edge cases
- **No issues** (`issues=[]`, so `passed` is True and `issue_count` is 0): returns only the 6 header lines.
  - `to_markdown(max_issues=0)` returns the same 6 lines.
- **Only warning-level issues**: Status is `**passed**`, but the table **is shown**. The early return needs both `passed` and `issue_count == 0`.
- **`max_issues=0` with at least one issue**: the header, then `""`, then the line ``_Issue table omitted by `max_issues=0`._`` (with backticks). There is no table and **no** "Showing" footer.
- **`0 < max_issues < issue_count`**: the first `max_issues` rows, then `""`, then `f"_Showing {k} of {issue_count} issues._"`.
- **`max_issues >= issue_count`, or `None`**: all rows, no footer.
- **`column=None`** (for example `composite_unique`): the Column cell is empty, giving `"|  | composite_unique | ..."`.

## 7. Rules from the docstring
- `max_issues` omitted means all issues are shown.
- `redact_values` hides the invalid or sensitive data in the Value column only.

## 8. Testing hints
Build results directly so the output is exact and does not depend on the C++ dtype inference used by `ar.validate`:

```python
from arnio.schema import ValidationIssue, ValidationResult
i = ValidationIssue(column="age", rule="min", message="Column 'age' has values below 0", row_index=2, value=-5)
r = ValidationResult(row_count=3, issue_count=1, issues=[i], bad_rows=[2])
assert r.to_markdown() == (
 "## Validation Report\n\n- Status: **failed**\n- Rows checked: 3\n- Issues found: 1\n- Bad rows: 1\n\n"
 "| Column | Rule | Severity | Row | Value | Message |\n|---|---|---|---|---|---|\n"
 "| age | min | error | 2 | -5 | Column 'age' has values below 0 |")
```

Pitfalls:
- `issue_count` must equal `len(issues)`, or the `ValidationResult` constructor raises `ValueError`. Hidden rows can only come from `max_issues`.
- `bad_rows` may be inconsistent with the issues; for example `bad_rows=[]` gives `"- Bad rows: 0"`. Only its length is printed.
- To check escaping, use `value="a|b\nc"`. The expected cell is `a\|b<br>c`, written in Python as `"a\\|b<br>c"`.
- For the footer, use 3 issues with `max_issues=2`: the output ends with `"\n\n_Showing 2 of 3 issues._"`.
- `r.to_markdown(max_issues=3)` on 3 issues has no footer, and the output ends with `" |"`.
- For a warning-only result, use `severity="warning"`. Expect `"**passed**"` together with the table header present.
- If you go through `ar.validate`, note that `row_index` is the 0-based pandas index plus 1. The exact dtype and the representation of empty strings depend on the C++ core, so prefer substring assertions on those paths.

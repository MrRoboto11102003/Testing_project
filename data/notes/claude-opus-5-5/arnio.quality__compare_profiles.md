# Notes: `arnio.quality.compare_profiles(profile_a, profile_b)`

## 1. Purpose
- Compares two `DataQualityReport`s column by column and returns a `ProfileComparison`.
- Each column gets a drift entry with a status (`"ok"`, `"warning"` or `"changed"`), a dict of changed metrics, and a list of reasons.
- Report-level fields are never compared. These include `row_count`, `duplicate_rows`, `duplicate_ratio`, `quality_score` and `suggestions`.

## 2. Inputs
- `profile_a`: the baseline report. `profile_b`: the comparison report.
- Both must be `DataQualityReport` instances.
- Only these attributes of each `ColumnProfile` are read: `dtype`, `null_ratio`, `unique_count`, `unique_ratio`, `row_count`, `mean`, `std`, `min`, `max`.
- Everything else is ignored: `semantic_type`, `warnings`, `null_count`, `empty_string_count`, quantiles, `top_values`, and so on.

## 3. Raises
Checks run in this order (type first, then schema):
1. `TypeError("compare_profiles expects two DataQualityReport instances")` if either argument is not a `DataQualityReport`.
   - Example: `ar.compare_profiles(rep, {})`.
2. `ValueError` if the sets of column names differ. The exact text, with lists sorted, is:
   `"Profiles have incompatible schemas: missing from profile_a=['b'], missing from profile_b=[]. This is likely caused by calling profile() with different exclude_columns on each dataset. Profile both datasets with the same included/excluded columns before comparing."`
   - `missing from profile_a` = columns only in b. `missing from profile_b` = columns only in a.
   - Example: a has `{"x"}`, b has `{"x","b"}` → the message above.
   - Column order in the dicts does not matter; only the sets are compared.

## 4. Returns and guarantees
`ProfileComparison` with these fields:
- **`left_profile` / `right_profile`**: these are `profile_a` / `profile_b` themselves, not copies. `is` identity holds.
- **`status_counts`**: always exactly the keys `{"ok", "warning", "changed"}`, in that order. The counts sum to the number of columns.
- **`drift_report`**: `{col: entry}`, inserted in **sorted** column-name order. Each entry is `{"status": str, "changes": dict, "reasons": list[str]}`.
  - `changes[metric]` is `{"baseline": a_val, "comparison": b_val, "delta": abs(a-b) or None}`. Values pass through `_clean_scalar`, so numpy scalars become Python types and NaN/inf become `None`.
  - A metric appears in `changes` (and its reason is appended) whenever `a != b`, **even if its own status is `"ok"`**.
  - If nothing differs: `changes == {}` and `reasons == ["no drift detected"]`.
  - The entry status is the maximum severity over its metrics (ok < warning < changed).

**Metrics are evaluated in this fixed order**, which is the key order of `changes` and the order of `reasons`:

| metric | warning if delta > | changed if delta > | reason text |
|---|---|---|---|
| dtype | – | – (any difference → `"changed"`, delta `None`) | `f"dtype changed from {a.dtype} to {b.dtype}"` |
| null_ratio | 0.1 | 0.25 | `"null ratio changed"` |
| unique_count | `max(1.0, rcA*0.1, rcB*0.1)` | `max(2.0, rcA*0.25, rcB*0.25)` | `"unique count changed"` |
| unique_ratio | 0.1 | 0.25 | `"unique ratio changed"` |
| mean, std, min, max | `scale*0.1` | `scale*0.25` | `f"numeric {metric} changed"` |

- `rcA` / `rcB` are the two columns' `row_count` values.
- For the numeric rows, `scale = max(|a|, |b|, 1.0)`.
- All comparisons are strict `>`. A delta exactly equal to a threshold does not reach that level.
- The numeric metrics (mean, std, min, max) are compared only when **both** dtypes are in `{"int64","float64"}`. A metric is skipped if either value is `None`.
  - String columns are therefore never compared on their length stats.
  - `int64` vs `float64` is compared numerically, and dtype is also reported as `"changed"`.
- Delta is `None` unless both values are `isinstance(int|float)`.
  - `numpy.int64` is NOT a Python `int`.
  - So for real `profile()` output of an int64 column, `min`/`max` are `np.int64`. Their difference is recorded (delta `None`) but contributes status `"ok"`.
  - `np.float64` *is* a float, so float columns get a numeric delta.

## 5. Side effects
- None. Inputs are not mutated.

## 6. Branches and edge cases
- **Identical profiles** (e.g. `compare_profiles(r, r)`): every entry is `{"status":"ok","changes":{},"reasons":["no drift detected"]}` and `status_counts == {"ok": n, "warning": 0, "changed": 0}`.
- **Both reports have empty `columns`**: `drift_report == {}` and the counts are all 0.
- **null_ratio**, starting from 0.0:
  - 0.05 → "ok", but `changes` contains `null_ratio` with delta 0.05.
  - 0.1 → "ok" (not > 0.1).
  - 0.2 → "warning".
  - 0.3 → "changed".
- **unique_count**, with row_count 10 on both sides (warning threshold 1.0, changed threshold 2.5), starting from 5:
  - 6 → ok.
  - 7 → warning.
  - 8 → changed.
- **mean**, starting from 10.0:
  - 11.0 → ok (thr 1.1).
  - 12.0 → warning (thr 1.2, changed 3.0).
  - 14.0 → changed (thr 3.5, delta 4).
- **dtype** `"string"` vs `"int64"` → status "changed", `changes["dtype"] == {"baseline":"string","comparison":"int64","delta":None}`. No numeric metrics are compared.
- **Report-level differences only** (different `row_count`, `duplicate_rows`, etc.) with identical column profiles → all ok.

## 7. Docstring rules
- "Column-wise; null ratios, dtype, uniqueness, numeric mean/std/min/max."
- Raises `ValueError` if the column sets differ.

## 8. Testing hints
- Prefer building reports by hand. This gives full control and avoids C++ dtype uncertainty.

  ```python
  from arnio.quality import ColumnProfile, DataQualityReport
  def col(name="x", dtype="float64", null_ratio=0.0, uc=5, ur=0.5, rc=10, **kw):
      return ColumnProfile(name, dtype, "numeric", rc, 0, null_ratio, uc, ur, **kw)
  def rep(*cols):
      return DataQualityReport(row_count=10, column_count=len(cols), memory_usage=0,
                               duplicate_rows=0, duplicate_ratio=0.0,
                               columns={c.name: c for c in cols})
  ```

- `ColumnProfile` validation:
  - The counts must be non-negative Python ints; `null_count` 0 is fine even if `null_ratio > 0`.
  - The ratios must be in [0, 1].
- Pass `mean=`/`std=`/`min=`/`max=` as keyword arguments when testing the numeric metrics. Leaving them `None` skips them.
- To check sorting, use columns `{"b","a"}` and assert `list(cmp.drift_report) == ["a","b"]`.
- Assert the exact contents of `reasons` and `changes` keys, not just the status. A change below the threshold still appears in both.
- Compare deltas with `pytest.approx`. Pick exactly representable values (0.0, 0.25, 0.5) to avoid float noise near thresholds.
- `ar.profile(ar.from_pandas(df))` also works for end-to-end tests. The dtypes come from the C++ core (expected `"int64"`/`"float64"`/`"string"`, but not verified here). Avoid asserting deltas for int `min`/`max` from real profiles (see the `np.int64` note in section 4).

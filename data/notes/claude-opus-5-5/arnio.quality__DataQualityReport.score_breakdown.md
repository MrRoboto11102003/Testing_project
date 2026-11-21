# Notes: `DataQualityReport.score_breakdown()`

## 1. Purpose
- Returns a fixed 4-key dict of the report's quality-score penalties plus the final score.
- It reads `self.score_components` and `self.quality_score` and computes nothing.

## 2. Inputs
- `self` only; the method takes no arguments.
- Exact body:
  ```python
  return {
      "null_penalty": self.score_components.get("null_penalty", 0.0),
      "duplicate_penalty": self.score_components.get("duplicate_penalty", 0.0),
      "type_mismatch_penalty": self.score_components.get("type_mismatch_penalty", 0.0),
      "final_score": self.quality_score,
  }
  ```
- `score_components` defaults to `{}` and is **not validated** by `__post_init__`.
- `quality_score` defaults to `100.0`. `__post_init__` checks it is a finite number in [0, 100], not a bool.

## 3. Raises
- The method itself raises nothing for a normally built report.
- Constructing the report can raise `ValueError("quality_score must be a finite value between 0.0 and 100.0: 150.0")`. Trigger: `DataQualityReport(..., quality_score=150.0)`.
- If `score_components` is not a dict (e.g. `None`, which is accepted at construction), `.get` raises `AttributeError`. This is an untested corner; avoid it.

## 4. Returns and guarantees
- Returns a new `dict` on every call, with keys in this order: `["null_penalty", "duplicate_penalty", "type_mismatch_penalty", "final_score"]`.
- Penalty values are returned verbatim from `score_components`. There is no sign flip, `abs`, or rounding.
- If a penalty key is missing, its value is `0.0`, a positive float zero.
- `final_score` is `self.quality_score` as stored. It is **not** recomputed from the components.
- Extra keys in `score_components` (e.g. `"foo": -1.0`) are ignored and never appear in the result.

## 5. Side effects
- None. The report is frozen, and neither `score_components` nor `quality_score` is mutated.
- Mutating the returned dict does not affect `report.score_components`. The returned dict is a new object, not `score_components` itself.

## 6. How `profile()` fills these fields (`_calculate_quality_score`)
- If `row_count == 0` or there are no columns, the result is `(100.0, {})`. `score_breakdown()` then returns all `0.0` penalties and `final_score` `100.0`.
- `duplicate_penalty = round(min(duplicate_ratio*100, 20), 2)`. It is capped at 20.
- `null_penalty = round(min(mean(column null_ratio)*100, 40), 2)`. It is capped at 40.
- `type_mismatch_penalty = round(min(frac_cols_with_suggested_dtype*100, 40), 2)`. It is capped at 40.
  - Only columns with dtype `"string"` can get a `suggested_dtype`: all-bool-like → `"bool"`, all-integer-like → `"int64"`, finite numeric → `"float64"`.
- Each penalty is stored as a **negative** number, and **only if > 0**. Zero penalties are absent from `score_components`, so the breakdown shows `0.0` for them, never `-0.0`.
- `quality_score = round(100 - dup - null - mismatch, 2)`.

## 7. Rules from docstrings
- "Return a breakdown of the individual penalty scores."
- Consequently, penalties appear as negative deltas, and `final_score` equals the overall score.

## 8. Testing hints
Direct construction is the most robust approach because it is fully deterministic and does not depend on the C++ core:
```python
from arnio.quality import DataQualityReport
r = DataQualityReport(row_count=10, column_count=1, memory_usage=0,
        duplicate_rows=1, duplicate_ratio=0.1, columns={},
        quality_score=72.5,
        score_components={"duplicate_penalty": -10.0, "null_penalty": -17.5, "foo": -3.0})
assert r.score_breakdown() == {"null_penalty": -17.5, "duplicate_penalty": -10.0,
                               "type_mismatch_penalty": 0.0, "final_score": 72.5}
assert list(r.score_breakdown()) == ["null_penalty","duplicate_penalty","type_mismatch_penalty","final_score"]
```

Useful assertions:
- **Defaults:** `DataQualityReport(row_count=0, column_count=0, memory_usage=0, duplicate_rows=0, duplicate_ratio=0.0, columns={}).score_breakdown()` equals `{"null_penalty": 0.0, "duplicate_penalty": 0.0, "type_mismatch_penalty": 0.0, "final_score": 100.0}`.
- **No recomputation:** with components that are inconsistent with the score (e.g. `score_components={}` and `quality_score=55.0`), `final_score` is `55.0`.
- **Sign preserved:** a stored `-10.0` comes back as `-10.0`, not `10.0`.
- **Extra keys:** extra component keys are absent from the output, and the output always has exactly 4 keys.
- **Missing keys:** a missing key gives exactly `0.0`. Positive values (e.g. `{"null_penalty": 5.0}`) also pass through unchanged.

End-to-end via `ar.profile`:
- These cases assume `from_pandas` maps int → `"int64"` and Python str → `"string"`. That mapping is done by the C++ core, so it is not verifiable here.
- **Duplicates:** `ar.profile(ar.from_pandas(pd.DataFrame({"a":[1,1,2,3,4,5,6,7,8,9]})))`. Duplicate ratio is 0.1, so the result is `{"null_penalty":0.0,"duplicate_penalty":-10.0,"type_mismatch_penalty":0.0,"final_score":90.0}`.
- **Capped duplicates:** `{"a":[1,1,2,3]}` has duplicate ratio 0.25, which is capped to `-20.0`, giving `final_score` `80.0`.
- **Type mismatch:** `{"s":["1","2","3"]}` gets `suggested_dtype` `"int64"`, so `type_mismatch_penalty` is `-40.0` (capped) and `final_score` is `60.0`.
- **Empty frame:** an empty frame with 0 rows gives all zeros and `final_score` `100.0`.
- **Nulls:** null-penalty cases depend on how the C++ core round-trips NaN/None. Prefer direct construction for those.

Pitfalls:
- Do not assert positive penalties from `profile()`: they are negative.
- Use `==` with exact floats only when they come from `round(..., 2)` values like the ones above. Otherwise use `pytest.approx`.

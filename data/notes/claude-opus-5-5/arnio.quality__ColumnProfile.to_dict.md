# Notes: `ColumnProfile.to_dict(self, *, redact_sample_values: bool = False) -> dict`

## 1. Purpose
- Returns a JSON-friendly `dict` for one column profile.
- It optionally replaces sample values and top-value labels with `"[REDACTED]"`.
- Quantile and outlier keys appear only for numeric dtypes.

## 2. Inputs
- `self`: a frozen `ColumnProfile` dataclass. There are two ways to get one:
  - Construct it directly: `ColumnProfile(name, dtype, semantic_type, row_count, null_count, null_ratio, unique_count, unique_ratio, ...)`.
  - Take it from a report: `ar.profile(frame).columns["col"]`.
- `redact_sample_values` is keyword-only.
  - It is validated by `_validate_bool_option`: it must be exactly a `bool` (checked with `isinstance(v, bool)`).
  - `1`, `0`, `None` and `"yes"` are all rejected.

## 3. Raises
- `TypeError("redact_sample_values must be a bool")`, e.g. `cp.to_dict(redact_sample_values=1)` or `=None`.
  - This check runs first, before anything else.
- Passing the argument positionally, e.g. `cp.to_dict(True)`, raises Python's own `TypeError` (the parameter is keyword-only).
- Malformed `histogram` entries raise `ValueError` from tuple unpacking.
  - Each entry must be a 4-tuple `(start, end, count, ratio)`.
  - Each `top_values` entry must be a 3-tuple `(value, count, ratio)`.
- Construction pitfall (raised by `__post_init__`, not by `to_dict`):
  - `row_count`, `null_count`, `unique_count`, `empty_string_count` and `whitespace_count` must be Python `int`, not `bool` and not `np.int64`.
    - Wrong type: `TypeError("{field} must be an integer, got {type}")`.
    - Negative value: `ValueError("{field} cannot be negative: {v}")`.
  - `null_ratio` and `unique_ratio` must be finite numbers in [0, 1].
    - Out of range: `ValueError("{field} must be a finite ratio between 0.0 and 1.0: {v}")`.
  - The same ratio rule applies to `email_validity_ratio`, `url_validity_ratio` and `top_values_sample_ratio` when they are not None.

## 4. Returns and guarantees
The result is a new `dict` on every call. Key order is exact (Python dicts keep insertion order).
- These 17 keys always come first, in this order:
  `name, dtype, semantic_type, row_count, null_count, null_ratio, unique_count, unique_ratio, empty_string_count, whitespace_count, suggested_dtype, email_validity_ratio, url_validity_ratio, min, max, mean, std`
- These 9 keys follow **only if `dtype in {"int64", "float64"}`** (exact string match):
  `q25, q50, q75, q95, iqr, outlier_lower_bound, outlier_upper_bound, outlier_count, outlier_ratio`
  - Any other dtype omits these keys entirely, even if `q25` etc. are set. Examples: `"string"`, `"bool"`, `"int32"`, `"Int64"`.
  - The keys are absent, not set to None.
- These 7 keys always come last:
  `sample_values, warnings, top_values, top_values_is_approximate, top_values_sample_count, top_values_sample_ratio, histogram`
- Total key count: **33 for numeric dtypes, 24 otherwise**.

How each value is produced:
- Passed through `_clean_scalar`:
  - `min` and `max`
  - `q25`, `q50`, `q75`, `q95`, `iqr`, `outlier_lower_bound`, `outlier_upper_bound`
  - each element of `sample_values`
  - each top-value `value`
  - each histogram `bucket_start` and `bucket_end`
- Rules of `_clean_scalar`:
  - If `pd.isna(v)` is true, return `None`. This covers None, `float("nan")`, `np.nan`, `pd.NA` and `pd.NaT`.
  - Otherwise, if `v` has `.item()`, unwrap it to a Python scalar (`np.int64(3)` becomes `3` of type `int`; `np.float64` becomes `float`).
  - If the result is a non-finite float, return `None` (`float("inf")` and `np.float64(-np.inf)` both become `None`).
  - Anything else is returned unchanged.
- Returned raw, not cleaned (a NaN or inf stays as it is):
  - `mean`, `std`
  - `outlier_count`, `outlier_ratio`
  - all count and ratio fields
  - top-value `count` and `ratio`
  - histogram `count` and `ratio`
- `top_values`:
  - `None` stays `None`.
  - Otherwise it becomes a list of dicts `{"value": ..., "count": c, "ratio": r}` in the original order.
- `histogram`:
  - `None` stays `None`.
  - Otherwise it becomes a list of dicts `{"bucket_start", "bucket_end", "count", "ratio"}` in that key order.
- `warnings` is a new list: `list(self.warnings)`. It is equal to the original but not the same object.
- `sample_values` is a new list built by a list comprehension.

Redaction (`redact_sample_values=True`):
- `sample_values` becomes `["[REDACTED]"] * len(self.sample_values)`, so the length is preserved and `[]` stays `[]`.
- Each top-value `"value"` becomes `"[REDACTED]"`; `count` and `ratio` are kept.
- If `top_values` is None, it stays None.
- `min`, `max` and histogram edges are **not** redacted. All other fields are identical to the unredacted output.

## 5. Side effects
- None. The profile object, its `sample_values` list and its `warnings` list are not mutated.

## 6. Branches and edge cases
- Numeric versus non-numeric dtype: `dtype="float64"` gives 33 keys including `"q25"`; `dtype="string"` gives 24 keys and `"q25" not in d`.
- `top_values=None` (the default) gives `d["top_values"] is None` for both redact settings.
- `top_values=[("a", 2, 0.5)]`:
  - `redact_sample_values=False` gives `[{"value": "a", "count": 2, "ratio": 0.5}]`.
  - `redact_sample_values=True` gives `[{"value": "[REDACTED]", "count": 2, "ratio": 0.5}]`.
- `sample_values=[np.int64(1), float("nan"), float("inf"), "x"]` gives `[1, None, None, "x"]`, and `type(d["sample_values"][0]) is int`.
- `min=np.float64(np.inf)` gives `None`; `min=np.int64(7)` gives `7` as an `int`.
- `mean=float("nan")` gives `d["mean"]` as NaN (not None). Test with `math.isnan`.
- `histogram=[(0.0, 1.0, 3, 0.75)]` gives `[{"bucket_start": 0.0, "bucket_end": 1.0, "count": 3, "ratio": 0.75}]`.
- `histogram=None` gives `None`.
- Defaults for an otherwise minimal profile:
  - `empty_string_count=0`, `whitespace_count=0`
  - `sample_values=[]`, `warnings=[]`
  - `top_values_is_approximate=False`
  - `suggested_dtype`, `email_validity_ratio`, `url_validity_ratio`, `min`, `max`, `mean`, `std`, `top_values_sample_count`, `top_values_sample_ratio` and `histogram` all `None`

## 7. Rules from docstrings and comments
- The method is documented as "JSON-friendly".
- The `_clean_scalar` docstring says non-finite floats map to `None` so that `json.dumps(..., allow_nan=False)` never raises on report output.
- For string columns, `min`, `max` and `mean` are string-length statistics (from `profile`).

## 8. Testing hints
- Construct directly for full control. Import it with `from arnio.quality import ColumnProfile` (it is also likely available as `ar.ColumnProfile`). Use Python ints and floats for the validated fields:
  ```python
  cp = ColumnProfile(name="a", dtype="int64", semantic_type="numeric", row_count=4,
                     null_count=0, null_ratio=0.0, unique_count=4, unique_ratio=1.0,
                     min=np.int64(1), sample_values=[np.int64(1)], warnings=["w"])
  ```
- Pin the exact order with `list(cp.to_dict().keys()) == [...]` for both the numeric and the string case.
- Check the numeric unwrapping with `type(...) is int`.
- Check the copy: `d["warnings"] == cp.warnings` and `d["warnings"] is not cp.warnings`.
- Check JSON safety: after cleaning `sample_values` and `min`, `json.dumps(d, allow_nan=False)` succeeds when `mean` and `std` are finite or None.
- Via the public API:
  ```python
  rep = ar.profile(ar.from_pandas(pd.DataFrame({"x": [1, 2, 3, 4], "s": ["a", "b", "a", "c"]})))
  d = rep.columns["x"].to_dict()
  ```
  - The exact dtype strings come from the C++ core. Assert `d["dtype"] == frame.dtypes["x"]` rather than hardcoding `"int64"`.
  - For an int64 column, expect `d["min"] == 1` with type `int`, `d["max"] == 4` and `d["mean"] == 2.5`.
  - String columns get a `top_values` list (e.g. `("a", 2, 0.5)` first) and no `q25` key.
- Do not hand-compute histogram or quantile values in profile-based tests; use the directly constructed objects for exact values.

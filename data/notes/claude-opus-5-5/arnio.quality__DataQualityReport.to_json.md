# Notes: `DataQualityReport.to_json`

## 1. Purpose
- `to_json` serialises the report to a JSON string with `json.dumps(self.to_dict(...), indent=indent)`.
- If `output` is given, it writes that string to the stream instead and returns `None`.

## 2. Inputs (all keyword-only)
- **`indent`** (default `None`)
  - Passed straight to `json.dumps` and never validated.
  - `None` gives one line with separators `", "` and `": "`.
  - An int or str pretty-prints.
- **`redact_sample_values`** (default `False`)
  - Must be a real `bool`, checked in `to_dict` by `_validate_bool_option`.
  - `1` and `"yes"` are rejected.
- **`exclude_columns`** (default `None`)
  - Accepts `None`, `list`, `tuple` or `set` of `str`.
  - `frozenset` and a plain `str` are rejected, because the check is `isinstance(x, (list, tuple, set))`.
  - The value is converted to a `set`.
- **`output`** (default `None`)
  - Any object that has a `write` attribute.
  - It is called once as `output.write(json_out)`.
  - No newline is appended.
- The keyword arguments are not passed to `json.dumps`, so it runs with its defaults:
  - `sort_keys=False`, so dict insertion order is kept.
  - `allow_nan=True`.
  - No `default=` hook.

## 3. Raises (in this order)
1. `TypeError("redact_sample_values must be a bool")`
   - Example: `r.to_json(redact_sample_values=1)`.
2. `TypeError("exclude_columns must be a list, tuple, set, or None")`
   - Example: `r.to_json(exclude_columns="a")`.
3. `TypeError("exclude_columns must contain only string column names")`
   - Example: `r.to_json(exclude_columns=[1])`.
4. `KeyError` for unknown excluded columns.
   - Template: `"Unknown exclude_columns: {sorted_unknown}. Available columns: {', '.join(self.columns) or '<none>'}"`.
   - The unknown names are sorted; the available names keep `self.columns` insertion order.
   - Example: `r.to_json(exclude_columns=["zz"])` with columns a, b gives `"Unknown exclude_columns: ['zz']. Available columns: a, b"`.
   - `str(KeyError)` adds quotes, so assert on `exc.value.args[0]`.
5. `TypeError("output must be a writable text stream")`
   - Example: `r.to_json(output=123)`.
   - This check happens after serialisation. If any earlier error is raised, nothing is written.

## 4. Returns and guarantees
- With `output=None`, it returns a `str` exactly equal to `json.dumps(r.to_dict(redact_sample_values=..., exclude_columns=...), indent=indent)`.
- With `output` given, it returns `None`, and the stream receives that same string.
- **Top-level key order:** `row_count, column_count, memory_usage, duplicate_rows, duplicate_ratio, quality_score, score_components, columns, suggestions`.
- **`columns`:**
  - Ordered by sorted column name.
  - Excluded names are removed.
- **Per-column key order:** `name, dtype, semantic_type, row_count, null_count, null_ratio, unique_count, unique_ratio, empty_string_count, whitespace_count, suggested_dtype, email_validity_ratio, url_validity_ratio, min, max, mean, std`.
  - Only if `dtype` is `"int64"` or `"float64"`, these keys follow: `q25, q50, q75, q95, iqr, outlier_lower_bound, outlier_upper_bound, outlier_count, outlier_ratio`.
  - String or bool columns do not have these keys at all.
  - The remaining keys are `sample_values, warnings, top_values, top_values_is_approximate, top_values_sample_count, top_values_sample_ratio, histogram`.
- **Value encodings:**
  - `top_values` is a list of `{"value","count","ratio"}` objects, or `null`.
  - `histogram` is a list of `{"bucket_start","bucket_end","count","ratio"}` objects, or `null`.
- **`_clean_scalar` handling:**
  - Applied to `min`, `max`, the quantiles, `sample_values`, top-values values and histogram edges.
  - NaN and ±inf become `null`; numpy scalars become Python scalars.
  - `mean` and `std` are NOT cleaned. `mean=float("nan")` serialises as the bare token `NaN`.
- **`suggestions`:**
  - Sorted by `(step, json.dumps(kwargs, sort_keys=True, default=str))`.
  - Each entry is `{"step","kwargs","confidence_score","confidence_reason"}`.
  - Plain-tuple suggestions get `null` for both confidence fields.
  - The `kwargs` keys are sorted.
- **Fields that `exclude_columns` does not change:** `row_count`, `column_count`, `memory_usage`, `duplicate_*`, `quality_score` and `score_components`.

## 5. Side effects
- None on the report, which is a frozen dataclass.
- Only `output.write` is called. No files are written.

## 6. Branches and edge cases
- **`redact_sample_values=True`:**
  - Every `sample_values` item becomes `"[REDACTED]"`.
  - Each top-value `"value"` becomes `"[REDACTED]"`, while `count` and `ratio` are kept.
  - `min`/`max` (string lengths) and `histogram` are not redacted.
- **`exclude_columns`, effect on suggestion `kwargs`:**
  - A kwargs key equal to an excluded name is dropped. A `cast_types` suggestion's kwargs is `{col: dtype}`, so excluded columns disappear from it.
  - `subset` and `columns` lists are filtered, which can leave `[]` (the suggestion itself stays).
- **`exclude_columns`, effect on `confidence_reason`:** the substring `'col'` (with single quotes) is replaced by `'[REDACTED]'`. This only happens when the exclude set is non-empty.
- **`exclude_columns=[]` or `set()`:** behaves exactly like `None`.
- **Report with no columns:** gives `"columns": {}`. Any non-empty `exclude_columns` then raises `KeyError` with `Available columns: <none>`.

## 7. Docstring
- The docstring only shows the example `report.to_json(indent=2)`.

## 8. Testing hints
- Build reports directly to avoid depending on the C++ core:
  ```python
  from arnio.quality import DataQualityReport, ColumnProfile, CleaningSuggestion
  col = ColumnProfile(name="a", dtype="string", semantic_type="text", row_count=2,
      null_count=0, null_ratio=0.0, unique_count=2, unique_ratio=1.0,
      sample_values=["x", "y"], top_values=[("x", 1, 0.5), ("y", 1, 0.5)])
  r = DataQualityReport(row_count=2, column_count=1, memory_usage=10,
      duplicate_rows=0, duplicate_ratio=0.0, columns={"a": col})
  ```
- For this report, `r.to_json()` starts with `'{"row_count": 2, "column_count": 1, "memory_usage": 10, "duplicate_rows": 0, "duplicate_ratio": 0.0, "quality_score": 100.0, "score_components": {}, "columns": {"a": {"name": "a", '`.
- It ends with `'"suggestions": []}'`.
- **Strong, safe invariants:**
  - `r.to_json() == json.dumps(r.to_dict())`.
  - `json.loads(r.to_json(indent=2)) == r.to_dict()` (when there is no NaN).
  - `"\n" in r.to_json(indent=2)` and `"\n" not in r.to_json()`.
- **Stream output:** use `io.StringIO()`. Assert that the return value is `None` and that `buf.getvalue() == r.to_json()`.
- **Pin orderings:**
  - Use columns `{"b":..., "a":...}` and check that `list(json.loads(s)["columns"]) == ["a", "b"]`.
  - Check that top-level `list(json.loads(s))` matches the key order above.
- **Numeric vs string columns:** a `dtype="int64"` column has a `"q25"` key; a `dtype="string"` column does not.
- **Suggestions:**
  - Add `CleaningSuggestion("strip_whitespace", {"subset": ["a", "b"]}, 0.95, "r")`.
  - With `exclude_columns=["a"]`, the JSON kwargs become `{"subset": ["b"]}`, and column `"a"` is absent from `columns`.
  - `column_count` stays at its original value.
- **When going through `ar.profile(ar.from_pandas(df))`:** the exact dtype strings, memory usage and suggestions depend on the C++ conversion. Prefer the relative invariants above over hard-coded full strings.

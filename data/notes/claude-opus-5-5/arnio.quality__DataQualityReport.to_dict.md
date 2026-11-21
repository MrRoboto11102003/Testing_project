# Notes: `DataQualityReport.to_dict(self, *, redact_sample_values=False, exclude_columns=None)`

## 1. Purpose
- Returns a JSON-friendly `dict` view of a whole-frame quality report.
- It can redact sample and top values, and it can omit chosen columns, including their names inside suggestion kwargs and reasons.

## 2. Inputs
- `redact_sample_values`: must be a real `bool`. `1`, `0` and `None` are rejected.
- `exclude_columns`:
  - `None` becomes an empty set.
  - Only `list`, `tuple` or `set` is accepted. `str`, `frozenset`, `dict` and generators are rejected.
  - Every element must be a `str`.
  - The value is converted to a `set`, so duplicates are fine.
  - Every name must be a key of `self.columns`.
  - An empty list is valid and means no exclusion.

## 3. Raises
Checks run in this order:
1. `TypeError("redact_sample_values must be a bool")`, e.g. `rep.to_dict(redact_sample_values=1)`.
2. `TypeError("exclude_columns must be a list, tuple, set, or None")`, e.g. `rep.to_dict(exclude_columns="a")`.
3. `TypeError("exclude_columns must contain only string column names")`, e.g. `exclude_columns=["a", 1]`.
4. `KeyError(f"Unknown exclude_columns: {sorted_unknown}. Available columns: {', '.join(self.columns)}")`.
   - "Available" lists names in `self.columns` **insertion order**, not sorted.
   - It shows `<none>` if the report has no columns.
   - Example: columns `b`, `a` and `exclude_columns=["z","y"]` gives `Unknown exclude_columns: ['y', 'z']. Available columns: b, a`.
   - Use `pytest.raises(KeyError, match=re.escape("Unknown exclude_columns: ['z']"))`. `str(KeyError)` adds quotes around the message, so use `re.escape` with a substring match.
- Ordering example: `to_dict(redact_sample_values="x", exclude_columns=5)` raises the redact TypeError.

## 4. Returns and guarantees
The result is a new `dict` with keys in exactly this order:
`row_count, column_count, memory_usage, duplicate_rows, duplicate_ratio, quality_score, score_components, columns, suggestions`.

**Top-level fields**
- Top-level numbers are copied as-is. They are **not** recomputed when columns are excluded; `column_count` stays the same.
- `score_components` is the **same object** as `self.score_components`. It is not copied: `d["score_components"] is rep.score_components` is True.

**`columns`**
- It maps `{name: ColumnProfile.to_dict(redact_sample_values=...)}`.
- Keys are **sorted** and excluded names are skipped.

**`ColumnProfile.to_dict` key order**
- Order: `name, dtype, semantic_type, row_count, null_count, null_ratio, unique_count, unique_ratio, empty_string_count, whitespace_count, suggested_dtype, email_validity_ratio, url_validity_ratio, min, max, mean, std`.
- Then, **only if dtype is `"int64"` or `"float64"`**: `q25, q50, q75, q95, iqr, outlier_lower_bound, outlier_upper_bound, outlier_count, outlier_ratio`.
- Then `sample_values, warnings, top_values, top_values_is_approximate, top_values_sample_count, top_values_sample_ratio, histogram`.
- A string column dict therefore has no `"q25"` key at all.

**`ColumnProfile.to_dict` value handling**
- `_clean_scalar` behaviour:
  - NaN, None, inf and -inf become `None`.
  - Numpy scalars become Python scalars via `.item()`.
- Where `_clean_scalar` is applied:
  - It is applied to `min`, `max`, `q*`, `iqr`, the bounds, each sample value, each top value and the histogram edges.
  - It is **not** applied to `mean`, `std`, `outlier_count` or `outlier_ratio`, which are passed through.
- `sample_values`:
  - With redaction, this is `["[REDACTED]"] * len(sample_values)`, so the length is preserved.
  - Otherwise each value is cleaned.
- `top_values`:
  - `None` stays `None`; numeric columns from `profile()` have `None`.
  - Otherwise it is a list of `{"value", "count", "ratio"}` dicts.
  - With redaction, `value` is `"[REDACTED]"` while counts and ratios are kept.
- `histogram`: `None`, or a list of `{"bucket_start", "bucket_end", "count", "ratio"}` dicts.
- `warnings` is a new list copy.

**`suggestions`**
- Sorting:
  - The list is sorted by `(step, json.dumps(original_kwargs, sort_keys=True, default=str))`.
  - So `"cast_types"` comes before `"drop_duplicates"`, which comes before `"strip_whitespace"`, regardless of input order.
- Each item is a dict `{"step", "kwargs", "confidence_score", "confidence_reason"}`.
- For plain 2-tuples (not `CleaningSuggestion`), `confidence_score` and `confidence_reason` are both `None`.

**Suggestion `kwargs` filtering**
- The output is a new dict with keys sorted alphabetically.
- Any kwarg **key** equal to an excluded column is dropped. Example: cast_types `{"n":"int64","m":"float64"}` with `exclude_columns=["n"]` gives `{"m":"float64"}`.
- For keys `subset` or `columns` whose value is a list, tuple, set or ndarray, excluded items are removed and the result is a list.
- For key `"cast_types"` whose value is a dict, excluded keys are removed.
- Value normalisation:
  - None, str, bool and int pass through.
  - A float stays a float if finite, otherwise it becomes `None`.
  - A dict gets `str` keys and is processed recursively.
  - A set or frozenset becomes a sorted list.
  - A tuple or ndarray becomes a list.
  - A `date` becomes `.isoformat()`.
  - A numpy scalar becomes `.item()`.
  - Anything else becomes `str(value)`.

**Suggestion `confidence_reason` redaction**
- This only happens when `exclude_columns` is non-empty.
- Each literal `'col'` (with single quotes) becomes `'[REDACTED]'` via plain `str.replace`. Unquoted occurrences are untouched.

**Overall**
- With default args, the whole result from a `profile()` report should be `json.dumps`-able.

## 5. Side effects
- Nothing on the report is mutated.
- The only aliasing is that the returned dict shares the `score_components` object with the report.

## 6. Branches and edge cases
- **Empty report** (`columns={}`):
  - The result has `"columns": {}`.
  - `exclude_columns=["x"]` raises KeyError containing `Available columns: <none>`.
- **Plain tuple suggestion**: `suggestions=[("drop_duplicates", {"keep": "first"})]` gives `confidence_score` None.
- **`CleaningSuggestion`**: `confidence_score` is a float (e.g. 0.95) and the reason string is kept unchanged when there are no exclusions.
- **Strip reason**: the `strip_whitespace` reason contains no quoted column names, so it is unchanged by redaction. Its `subset` list is filtered.

## 7. Rules from comments and docstrings
- The method is documented as returning a "JSON-friendly dictionary representation".
- `_clean_scalar` exists so that `json.dumps(..., allow_nan=False)` never fails on report output.

## 8. Testing hints
- For deterministic tests, build the report by hand: `from arnio.quality import DataQualityReport, ColumnProfile, CleaningSuggestion`.
  ```python
  cp = lambda n, dt="string", **kw: ColumnProfile(name=n, dtype=dt, semantic_type="text", row_count=3,
        null_count=0, null_ratio=0.0, unique_count=2, unique_ratio=0.5, **kw)
  rep = DataQualityReport(row_count=3, column_count=2, memory_usage=10, duplicate_rows=1,
        duplicate_ratio=0.333333, columns={"b": cp("b", sample_values=["x","y"]), "a": cp("a","int64", q25=1.0)},
        score_components={"duplicate_penalty": -20.0},
        suggestions=[CleaningSuggestion("strip_whitespace", {"subset": ["b","a"]}, 0.95, "r"),
                     CleaningSuggestion("cast_types", {"a": "int64"}, 0.95, "Column 'a' conforms perfectly to int64 structure.")])
  ```
  Expected results for this report:
  - `list(d["columns"]) == ["a","b"]`.
  - `"q25" in d["columns"]["a"]` and `"q25" not in d["columns"]["b"]`.
  - `[s["step"] for s in d["suggestions"]] == ["cast_types","strip_whitespace"]`.
  - With `exclude_columns=["a"]`:
    - cast kwargs are `{}`.
    - the cast reason is `"Column '[REDACTED]' conforms perfectly to int64 structure."`.
    - strip subset is `["b"]`.
    - `column_count` is still 2.
- Via the public API you can use `ar.profile(ar.from_pandas(df))`. However, the dtype strings come from the C++ core (`frame.dtypes`); assert on `rep.columns[...]` values rather than hard-coding them.
- `ColumnProfile.__post_init__` requires the ratios to be in [0, 1] and the counts to be non-negative ints. `DataQualityReport` requires `quality_score` in [0, 100].
- Use `==` on lists and dicts. Don't assume dict output order equals insertion order for `columns`, which are sorted.

# Notes: `arnio.quality.profile(frame, *, exclude_columns=None, sample_size=5, approx_top_values=False, approx_top_values_min_unique=1000, approx_top_values_min_ratio=0.2, approx_top_values_sample_size=2000) -> DataQualityReport`

## 1. Purpose
- Builds a `DataQualityReport` for an ArFrame. The report has per-column stats, duplicate counts, a 0–100 quality score and cleaning suggestions.

## 2. Inputs
- `frame`: must be an ArFrame. This is checked by `_validate_arframe` in `arnio.frame`, which is not shown here, so the exception type and message are unknown.
- `sample_size`: an `int`, not a `bool`, and `>= 0`. It sets how many non-null values go into each column's `sample_values`.
- `approx_top_values`: must be a real `bool`.
- `approx_top_values_min_unique`: an `int`, not a `bool`, and `>= 0`.
- `approx_top_values_min_ratio`: an `int` or `float`, not a `bool`, finite, and between 0 and 1 inclusive.
- `approx_top_values_sample_size`: an `int`, not a `bool`, and `> 0`.
- `exclude_columns`: normalised by `cleaning._validate_column_sequence(..., argument_name="exclude_columns")`, then checked by `validate_columns_exist(frame, cols, operation="profile")`. Both live in another module, so their exceptions are unknown (probably TypeError and KeyError). `[]` counts as no exclusions.

## 3. Raises (checked in this order)
1. `_validate_arframe` fails: type and message not visible.
2. `TypeError("sample_size must be an integer")`, e.g. `sample_size=1.5` or `True`.
3. `ValueError("sample_size must be non-negative")`, e.g. `sample_size=-1`.
4. `TypeError("approx_top_values must be a bool")`, e.g. `approx_top_values=1`.
5. `TypeError("approx_top_values_min_unique must be an integer")`, e.g. `"5"` or `True`.
6. `ValueError("approx_top_values_min_unique must be non-negative")`, e.g. `-1`.
7. `TypeError("approx_top_values_min_ratio must be a float")`, e.g. `"0.2"` or `True`.
8. `ValueError("approx_top_values_min_ratio must be a finite number between 0 and 1")`, e.g. `float("nan")` or `float("inf")`.
9. `ValueError("approx_top_values_min_ratio must be between 0 and 1")`, e.g. `1.5` or `-0.1`. Note that the int `1` is accepted.
10. `TypeError("approx_top_values_sample_size must be an integer")`, e.g. `2.0`.
11. `ValueError("approx_top_values_sample_size must be positive")`, e.g. `0`.
12. Problems with `exclude_columns` (bad type, unknown column) come from the cleaning helpers. Assert only `pytest.raises(Exception)`, or use KeyError if you are confident.

## 4. Returns and guarantees
- Returns a new `DataQualityReport`. Its top-level fields:
  - `row_count` is the number of rows.
  - `column_count` is the number of columns profiled, after exclusions.
  - `duplicate_rows` is `df.duplicated().sum()`, which counts every repeat after the first, using only the non-excluded columns. It is 0 when there are 0 rows.
  - `duplicate_ratio` is `round(dup / rows, 6)`, or 0.0 when there are no rows.
  - `memory_usage` equals `frame.memory_usage()` when nothing is excluded. With exclusions it is `int(pandas df.memory_usage(deep=True).sum())`.
- `columns` is a dict keyed by column name, in frame column order, with excluded columns removed.
- Each `ColumnProfile` holds:
  - `dtype`: the value from `frame.dtypes[name]` (e.g. "int64", "float64", "string", "bool").
  - `row_count`: the total number of rows.
  - `null_count`: from `isna()`.
  - `null_ratio`: `round(null / row_count, 6)`.
  - `unique_count`: `nunique` of the non-null values.
  - `unique_ratio`: `round(unique / NON-NULL count, 6)`. It divides by the non-null count, not by row_count.
  - `sample_values`: the first `sample_size` non-null values, as a list.
- String columns (dtype "string"):
  - `whitespace_count` is the number of values that `.strip()` changes. `empty_string_count` is the number of values that are "" after stripping.
  - `top_values` is the top 5 as `(value, count, round(count / non_null, 6))`, in descending-count order. Values are not stripped.
  - `min`, `max` and `mean` are about string lengths: min and max are ints, mean is a float.
- Numeric columns ("int64" or "float64"):
  - `min` and `max` are taken over the finite values; they may be numpy scalars.
  - `mean` is a float. `std` is a float with `ddof=0` (population std).
  - `q25`, `q50`, `q75` and `q95` are pandas linear quantiles, rounded to 4 places.
  - `histogram` has 10 tuples from `np.histogram`: `(start, end, count, ratio)`.
  - `top_values` is None; `empty_string_count` and `whitespace_count` are 0.
  - With 4 or more finite values: `iqr = q75 - q25`, the outlier fences are `q25 - 1.5*iqr` and `q75 + 1.5*iqr` (rounded to 4), and `outlier_count` counts values strictly outside the fences. With fewer than 4, all of these are None.
- Bool columns: min, max, mean, std, the quantiles, top_values and histogram are all None. `semantic_type` is "boolean".
- `semantic_type`, decided in this order:
  1. "empty" if there are no non-null values.
  2. "identifier" if the lower-cased name is in {id, uuid, zip, zipcode, zip_code} or ends with "_id". This is checked before dtype.
  3. "numeric".
  4. "boolean".
  5. "email", "url" or "phone" if at least 80% of the stripped values fully match the pattern.
  6. "datetime".
  7. "categorical" if `nunique <= max(20, 0.2 * n)`.
  8. "text" otherwise.
- `suggested_dtype` is set only when dtype == "string". After stripping:
  - every value in {true, false, 1, 0} (case-insensitive) gives "bool", so `["1","0"]` gives "bool", not "int64";
  - all values matching `[+-]?\d+` gives "int64";
  - all values numeric and finite gives "float64";
  - anything else gives None.
- `warnings` come in this order:
  1. `contains_nulls`
  2. `all_null`
  3. `constant` (unique == 1)
  4. `near_constant` (unique > 1 and the most common value covers at least 95%)
  5. `high_cardinality` (unique >= 100, unique_ratio >= 0.9, semantic type identifier or text)
  6. `leading_or_trailing_whitespace`
  7. `empty_strings`
  8. `potential_outliers` (outlier_count > 0)
- Score:
  - 0 rows or 0 columns gives `quality_score == 100.0` and `score_components == {}`.
  - Otherwise there are three penalties, each rounded to 2 places:
    - dup = `min(dup_ratio * 100, 20)`
    - null = `min(mean(null_ratio) * 100, 40)`
    - mismatch = `min(share of columns with suggested_dtype * 100, 40)`
  - `score = round(100 - sum of penalties, 2)`.
  - Each penalty above 0 is stored as a negative number under `"duplicate_penalty"`, `"null_penalty"` or `"type_mismatch_penalty"`.
- `suggestions` is `suggest_cleaning(report)`: a list of `CleaningSuggestion` objects, which are tuples, so `s == (step, kwargs)`. They come in this order:
  1. `("strip_whitespace", {"subset": [cols with whitespace_count > 0]})`, score 0.95.
  2. `("cast_types", {col: dtype})`. Identifier columns are left out when the target is int64 or float64. The score is the average per column, starting at 0.95: minus 0.3 if the non-null share is below 0.2, minus 0.15 if it is below 0.5.
  3. `("drop_duplicates", {"keep": "first"})` if dup > 0. The score is 0.95, or 0.75 if the ratio is above 0.5.

## 5. Side effects
- None. The frame is not mutated (exclusions use `df.drop`, which returns a new DataFrame), and no global state or files are touched.

## 6. Branches and edge cases
- Zero-row frame (columns present, no rows): row_count 0, duplicate_rows 0, ratio 0.0, score 100.0, components {}. Each column gets semantic "empty" and `top_values == []` for string columns.
- Exclusions can create duplicates. With `{"a":[1,1], "b":[1,2]}` and `exclude_columns=["b"]`, `duplicate_rows == 1` and `column_count == 1`. Without exclusion, `duplicate_rows == 0`.
- `sample_size=0` gives `sample_values == []`. `sample_size=2` keeps the first 2 non-null values.
- Approximate top values are used only if all three hold: `approx_top_values=True`, unique >= min_unique and unique_ratio >= min_ratio, and the column is a string column. The sample uses `random_state=0` and then `top_values_is_approximate=True`, `top_values_sample_count = min(n, sample_size)` and `top_values_sample_ratio = round(sample_count / n, 6)`. For example, `{"s": ["a","b","c","d"]}` with `approx_top_values=True, approx_top_values_min_unique=1, approx_top_values_min_ratio=0.0, approx_top_values_sample_size=2` gives sample_count 2 and ratio 0.5.

## 7. Rules from comments and docstrings
- For string columns, min, max and mean are length statistics.
- Whitespace-only strings count as empty.
- Outliers are values strictly outside the 1.5×IQR fences.
- Identifier columns never get numeric cast suggestions, because a cast could lose leading zeros.

## 8. Testing hints
- Build frames with `ar.from_pandas(pd.DataFrame(...))` and check `frame.dtypes` first.
- Worked example: `{"name": [" alice","bob","bob","carol"], "age": [10,20,20,40]}` should give:
  - row_count 4, duplicate_rows 1, duplicate_ratio 0.25;
  - `quality_score == 80.0` with `score_components == {"duplicate_penalty": -20.0}` (the duplicate penalty is capped at 20);
  - name: whitespace_count 1, unique_count 3, unique_ratio 0.75, `top_values[0] == ("bob", 2, 0.5)`, min 3, max 6, mean 4.25, semantic "categorical", suggested_dtype None;
  - age: semantic "numeric", mean 22.5, q25 17.5, q50 20.0, q75 25.0, q95 37.0, iqr 7.5, fences 6.25 and 36.25, outlier_count 1, `warnings == ["potential_outliers"]`, `std ≈ 10.8972` (use `pytest.approx`);
  - `[s.step for s in rep.suggestions] == ["strip_whitespace", "drop_duplicates"]`.
- Type-mismatch example: `{"n": ["1","2","x3"]}` gives None, while `{"n": ["1","22","3"]}` gives "int64" and a mismatch penalty of 40.0 (capped).
- Pitfalls:
  - Don't rely on the order of tied `top_values` entries.
  - Compare numpy min and max with `==`, not with `type(...) is int`.
  - How nulls and "" survive `from_pandas` depends on the C++ core, so use `None` with care and check `null_count` against what the frame really holds.
  - Use `pytest.approx` for std and the histogram edges.

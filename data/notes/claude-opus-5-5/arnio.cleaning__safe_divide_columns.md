# Notes: `arnio.cleaning.safe_divide_columns(frame, numerator, denominator, output_column, fill_value=0.0)`

## 1. Purpose
- Adds a new column `output_column` equal to `frame[numerator] / frame[denominator]`.
- Where the denominator is zero or null, the value is `fill_value` instead of NaN/inf.
- Accepts an ArFrame or a pandas DataFrame. The output type matches the input type.

## 2. Inputs
- `frame`: `ArFrame` or `pd.DataFrame`. Anything else raises TypeError.
- `numerator`, `denominator`: `str` column names, and both must exist. They may be the same column.
- `output_column`: a non-empty, non-whitespace `str` that must NOT already exist. A non-str value such as `5` raises **ValueError**, not TypeError.
- `fill_value`: `int` or `float`, and must be finite. `bool` is rejected.
  - `np.float64` is accepted because it subclasses float.
  - `np.int64` is rejected.

## 3. Raises (checks run in exactly this order)
1. `TypeError("frame must be an ArFrame or a pandas DataFrame")`, e.g. `safe_divide_columns({"a":[1]}, "a", "a", "r")`.
2. `TypeError("numerator must be a string column name")`, e.g. `numerator=1`.
3. `TypeError("denominator must be a string column name")`.
4. `ValueError(f"Numerator column '{numerator}' not found in frame.")`, e.g. `numerator="x"`.
5. `ValueError(f"Denominator column '{denominator}' not found in frame.")`.
6. `ValueError("output_column must be a non-empty string.")`. Note the trailing period. Triggered by `""`, `"  "` or `5`.
7. `TypeError("fill_value must be a finite float, not bool")`, e.g. `fill_value=True`.
8. `TypeError(f"fill_value must be a finite float, got {type(fill_value).__name__!r}")`:
   - `fill_value="0"` gives `got 'str'`.
   - `fill_value=None` gives `got 'NoneType'`.
9. `ValueError(f"fill_value must be finite, got {fill_value}")`:
   - `float("nan")` gives `got nan`.
   - `float("inf")` gives `got inf`.
10. `ValueError(f"Output column '{output_column}' already exists.")`, e.g. `output_column="n"`.
11. Pandas path only (see §6), non-numeric values:
    - `ValueError(f"Numerator column '{numerator}' contains non-numeric values: {bad_values}")`.
    - The denominator variant is the same text with `Denominator column '{denominator}'`.
    - `bad_values` lists the first 3 offending original values.
    - The numerator is checked first, so it is reported even when both columns are bad.
    - Example: `pd.DataFrame({"n":["a","b","c","d"],"d":[1,1,1,1]})` gives `Numerator column 'n' contains non-numeric values: ['a', 'b', 'c']`.

Order examples:
- A missing numerator plus `fill_value=True` raises the "not found" ValueError.
- An existing `output_column` plus `fill_value=nan` raises the "must be finite" ValueError.

## 4. Returns and guarantees
**Native path (C++):** used when the input is an ArFrame AND both `frame.dtypes` entries are in `{"int64","float64"}`.
- Returns `ArFrame(_safe_divide_columns(...))`.
- The `attrs` argument is NOT passed, so the source `_attrs` are not carried over.
- Exact values, null handling and output dtype are determined in C++ code I cannot see.
- The docstring says zero or null denominator gives `fill_value`.

**Pandas path:** used in every other case (any pd.DataFrame input, or an ArFrame with a non-numeric dtype such as `"string"`).
- `pd.to_numeric(errors="coerce")` is applied to both columns.
- Where the denominator is NaN or `== 0`, the denominator is masked to NaN.
- The result is `numerator / denominator` followed by `.fillna(fill_value)`.
- **A null numerator also yields `fill_value`**, because NaN/x is NaN and then gets filled.
- The output column is float64 and is appended as the last column. Existing columns are unchanged.
- ArFrame input returns `from_pandas(df)`, with attrs not preserved. DataFrame input returns a pd.DataFrame with the original index kept.

## 5. Side effects
- None.
- The pandas path assigns to `df.copy(deep=False)`, so the caller's DataFrame does not gain the new column.
- No global state is touched.

## 6. Branches and edge cases
- **pd input, numeric:**
  ```python
  pd.DataFrame({"n":[10.0,5.0,3.0,None],"d":[2.0,0.0,None,1.0]})
  ```
  `r` is `[5.0, 0.0, 0.0, 0.0]` with the default fill. With `fill_value=-1` it is `[5.0, -1.0, -1.0, -1.0]`.
- **pd input, int columns:** `{"n":[10,9],"d":[4,3]}` gives `[2.5, 3.0]`. The dtype is float64.
- **ArFrame with numeric strings (pandas path):** `ar.from_pandas(pd.DataFrame({"n":["10","20"],"d":[2,0]}))`.
  - This assumes the `"n"` dtype is `"string"`.
  - Result `r` is `[5.0, 0.0]`, and `n` stays a string column.
- **ArFrame with numeric columns (native path):** test only clear cases:
  - `{"n":[10.0,9.0],"d":[2.0,0.0]}` with `fill_value=-1.0` should give `[5.0,-1.0]`.
- **Same column twice:** `numerator == denominator` is allowed. `{"a":[2.0,0.0]}` gives `[1.0, 0.0]`.

## 7. Rules from docstring/comments
- Zero or null denominator is replaced with `fill_value`, never NaN or inf.
- An existing `output_column` raises ValueError.
- `fill_value` must be finite.

## 8. Testing hints
- **Prefer pd.DataFrame inputs** to pin exact pandas-path semantics, especially null numerator becoming `fill_value`.
  - Compare with `result["r"].tolist()`, or `pd.testing.assert_series_equal(..., check_names=False)`.
- **Use exactly representable values** such as 2.5 and 5.0. Use `pytest.approx` otherwise.
- **For ArFrame results**, use `ar.to_pandas(res)["r"].tolist()` and `res.columns[-1] == "r"`.
- **Check non-mutation:** after the call, assert `"r" not in original_df.columns`.
- **Check error messages** with `pytest.raises(X, match=re.escape(msg))`. Remember the trailing periods.
- **Pitfalls that would make a test wrong on the current code:**
  - Do not assert that attrs are preserved; they are not on either path.
  - Do not rely on C++-path behaviour for a null numerator, since it is unseen.
  - Avoid bool columns and empty strings: they go down the pandas path with unclear coercion.
  - Do not expect TypeError for a non-str `output_column`; it raises ValueError.

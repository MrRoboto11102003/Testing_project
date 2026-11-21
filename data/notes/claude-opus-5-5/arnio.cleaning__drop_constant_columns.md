# Notes: `arnio.cleaning.drop_constant_columns(frame)`

## 1. Purpose
- Removes every column that has exactly one distinct value.
- Nulls count as a value (`nunique(dropna=False) == 1`).
- Accepts an `ArFrame` or a `pd.DataFrame` and returns the same kind of object.

## 2. Inputs
- `frame`: an `ArFrame` or a `pandas.DataFrame`. There are no other parameters.
- Validation is done by `_validate_frame(frame, allow_pandas=True)`:
  - `ArFrame` → ArFrame path.
  - `pd.DataFrame` → pandas path.
  - Anything else → `TypeError`.
- An `ArFrame` is converted with `ar.to_pandas(frame)`. A pandas input is used directly; it is not mutated, because `drop` and `copy` return new objects.

## 3. Raises
- `TypeError("frame must be an ArFrame or a pandas DataFrame")` for any other type.
  - Examples: `ar.drop_constant_columns({"a": [1]})`, `ar.drop_constant_columns(None)`, a list, a `pd.Series`.
- This is the only explicit raise. Pandas `nunique` may raise for unhashable cell values (e.g. lists in an object column). Do not test that.

## 4. Returns and guarantees
- **Return type:** an `ArFrame` for `ArFrame` input, a `pd.DataFrame` for DataFrame input. It is always a new object, never the input itself (`result is not frame`).
- **Column order:** the remaining columns keep their original relative order (`df.drop(columns=...)`).
- **Which columns are dropped:** `nunique_counts[nunique_counts == 1]`.
  - Columns with ≥2 distinct values (null counted as one distinct value) are kept.
- **Null handling:**
  - `[None, None]` / `[nan, nan]` → 1 unique → dropped.
  - `[1, 1, None]` → 2 unique (1 and NaN) → kept.
- **Rows:** never changed. Values and dtypes of the kept columns are unchanged in the pandas path.
- **Pandas path:** the index is preserved as-is (no `reset_index`).
- **attrs (ArFrame):**
  - Zero-row path: explicitly sets `result._attrs = copy.deepcopy(frame._attrs)`, when not None.
  - Non-empty path: returns `from_pandas(result_df)` with no explicit attrs copy. Whether attrs survive depends on `to_pandas`/`from_pandas` in `arnio.convert`, which is not visible here. Do not assert attrs in the non-empty path.

## 5. Side effects
- None. The input `ArFrame` or DataFrame is unchanged (columns and values identical afterwards).
- No global state is touched and no I/O is performed.

## 6. Branches and edge cases
- **Zero rows** (`len(df.index) == 0`): returns a deep copy with **all columns kept**, even though they have 0 unique values.
  - `pd.DataFrame({"a": [], "b": []})` → result has columns `["a", "b"]` and 0 rows, and is not the same object.
  - ArFrame: `from_pandas(copy)`, with attrs deep-copied (so mutating `result._attrs` does not affect `frame._attrs`).
- **Mixed columns:** `pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3], "c": ["x", "x", "x"]})` → columns `["b"]`.
- **Nothing constant:** `{"a": [1, 2], "b": ["x", "y"]}` → both columns kept, values equal to the input.
- **Single row:** every column has nunique 1, so **all** columns are dropped.
  - Pandas: result has `columns == []` and `len(result) == 1` (row count preserved, per the docstring).
- **Every column constant:** `{"a": [5, 5], "b": ["z", "z"]}` → pandas result has 0 columns and `len == 2`.
  - The ArFrame row count for a 0-column result depends on `from_pandas` (not visible). Assert only `result.columns == []`.
- **Null-related columns:**
  - All-null column `{"a": [None, None, None], "b": [1, 2, 3]}` → `["b"]`.
  - Partly null `{"a": [1.0, 1.0, None], "b": [1, 2, 3]}` → both kept (`["a", "b"]`).
- **Bool column** `[True, True]` → dropped. `[True, False]` → kept.

## 7. Rules from the docstring
- "Nulls are counted as values"; `[None, None]` is dropped and `[1, 1, None]` is kept.
- "Empty columns in zero-row frames are also kept, since they have zero unique values rather than one."
- "If every column is dropped, the resulting zero-column frame preserves the original row count."

## 8. Testing hints
- **Build inputs:** `ar.from_pandas(pd.DataFrame({...}))`.
- **Observe results:**
  - `result.columns` (list of str, ordered), `result.shape`, and `ar.to_pandas(result)[col].tolist()`.
  - Check the return type with `isinstance(result, ar.ArFrame)` / `isinstance(result, pd.DataFrame)`.
- **Use the pandas-input path for exact assertions** (row count with 0 columns, index, dtypes). `pd.testing.assert_frame_equal(result, df[["b"]])` works for the pandas path.
- **Test the threshold exactly.** Best regression catchers:
  - A 2-value column is kept.
  - The null-vs-value distinction: `[1, 1, None]` kept, `[None, None]` dropped. This catches a `dropna=True` change.
  - The zero-row branch keeps columns.
  - Column order is preserved, e.g. `{"x": [1, 2], "c": [0, 0], "y": [3, 4]}` → `["x", "y"]`.
  - The input is not mutated: check `df.columns.tolist()` after the call.
- **Pitfalls:**
  - Don't mix `None` and `NaN` in one column, or mix `1` and `"1"`. Pandas/arnio coercion makes the unique count uncertain.
  - Don't assert attrs or the ArFrame row count for the all-dropped, non-empty case.
  - For zero-row ArFrames, build them via `ar.from_pandas(pd.DataFrame({"a": pd.Series([], dtype="int64")}))`. Assert only the columns and `shape[0] == 0`.
  - To set attrs, assign `frame._attrs = {"k": [1]}` (a private attribute). Use it only in the zero-row test.

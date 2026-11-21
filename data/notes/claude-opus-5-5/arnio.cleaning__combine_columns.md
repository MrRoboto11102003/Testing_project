# Notes: `arnio.cleaning.combine_columns`

## 1. Purpose
- Joins the values of several columns, row by row, into one new string column. The new column is added as the last column.
- ArFrame input is handled by a C++ function. pandas input is handled by Python code whose logic is fully visible below.

## 2. Inputs
- `frame`: an `ArFrame` or a `pd.DataFrame`. Anything else raises `TypeError`.
- `subset` (keyword-only, default `None`):
  - `None` means every column, in frame order.
  - Otherwise it must be a list, tuple or `pd.Index` of `str`.
  - The output order follows `subset` order. `subset=["b","a"]` joins b first, then a.
- `separator` (default `" "`): must be a `str`. `""` is allowed.
- `output_column` (default `"combined"`): must be a `str` that is not empty after `.strip()`.

## 3. Raises (checks run in this order)
1. Bad frame: `TypeError("frame must be an ArFrame or a pandas DataFrame")`.
   - Trigger: `combine_columns({"a": [1]})`.
2. Non-str separator: `TypeError("separator must be a string")`.
   - Trigger: `separator=1`.
3. Bad output_column: `ValueError("output_column must be a non-empty string")`.
   - Triggers: `output_column=""`, `"   "`, or `5`.
4. Subset validation (only when `subset` is not None):
   - String or bytes subset: `TypeError("subset must be a sequence of column names, not a string")`. Trigger: `subset="a"`.
   - Not a Sequence or `pd.Index`: `TypeError("subset must be a sequence of column names")`. Trigger: `subset={"a"}` (a set).
   - Non-str items: `TypeError("subset must contain only string column names")`. Trigger: `subset=["a", 1]`.
   - Duplicates (checked before missing columns): `ValueError(f"subset contains duplicate column names: {duplicates}")`.
     - Trigger: `subset=["a","a"]` gives `"subset contains duplicate column names: ['a']"`.
   - Missing columns: `KeyError(f"Missing columns for combine_columns: {missing}. Available columns: {available}")`.
     - `available` is `", ".join(columns)`, or `"<none>"` when there are no columns.
     - Trigger: on columns a, b, `subset=["z"]` gives `"Missing columns for combine_columns: ['z']. Available columns: a, b"`.
   - An empty subset is not rejected by this step.
5. Empty selection: `ValueError("subset must contain at least one column")`.
   - Trigger: `subset=[]`.
   - Also triggered by `subset=None` on a zero-column frame.
6. Output name already exists: `ValueError(f"Output column '{output_column}' already exists.")`.
   - Trigger: columns `["a","combined"]` with default arguments gives `"Output column 'combined' already exists."`.
   - This check runs after the subset checks. For example, `subset=["z"]` plus an existing output name raises `KeyError`, not `ValueError`.
- All of these checks run before the C++ import or call, so they can be tested with ArFrame input too.

## 4. Returns and guarantees
### pandas input (fully visible code)
- Code:
  ```python
  df = frame.copy(deep=False)
  combined = df[subset].astype("string").fillna("").agg(separator.join, axis=1)
  combined = combined.mask(df[subset].isna().all(axis=1), pd.NA)
  df[output_column] = combined
  ```
- Returns a new `pd.DataFrame`. Columns are the originals in their original order, then `output_column` last.
- The index is kept as is (no reset).
- How values become strings (pandas `astype("string")`):
  - `1` becomes `"1"`.
  - `1.5` becomes `"1.5"`.
  - A float column such as `[1.0, 2.0]` gives `"1.0"`.
  - `True` becomes `"True"`.
- Null handling:
  - A null in a partly-null row becomes `""`, and the separator is kept.
  - `{"a": ["x", None], "b": [None, "y"]}` with `separator="-"` gives `["x-", "-y"]`.
- A row where every subset value is null gives `pd.NA` (check with `pd.isna`).
- Empty strings are not null:
  - `["", ""]` with `" "` gives `" "`, not NA.
- Single-column subset: the output is just the stringified value.

### ArFrame input
- Calls `arnio._arnio_cpp.combine_columns(frame._frame, subset_columns, separator, output_column)`.
- Wraps the result with `_wrap`, so the result is a new ArFrame and `_attrs` is a deep copy of the input's.
- **The formatting, null handling and output dtype are in C++ and cannot be seen.** Do not assume they match pandas exactly.
- Note this imports from `._arnio_cpp`, not `._core`.

## 5. Side effects
- pandas: works on a shallow copy, so the caller's DataFrame does not gain `output_column` and its values are unchanged.
- ArFrame: a new frame is returned; the input is untouched.
- No global state changes and no file I/O.

## 6. Branches / edge cases
- `subset=None`: all columns are joined.
  - pandas `{"a": ["x"], "b": ["y"]}` gives `"x y"`.
- `subset` given: only those columns are joined, in the given order.
- `separator=""`: `{"a": ["x"], "b": ["y"]}` gives `"xy"`.
- Non-default `output_column="full"`: a new last column named `"full"`.
- Zero-column pandas frame `pd.DataFrame(index=[0])` with `subset=None` raises the `"subset must contain at least one column"` ValueError.
- ArFrame vs pandas: the type check returns the same type as the input. Validation is shared; the combining logic differs.

## 7. Rules from the docstring
- `TypeError` if separator is not a str or frame has the wrong type.
- `ValueError` if output_column is empty or already exists, or the subset is empty.
- `KeyError` for missing subset columns.

## 8. Testing hints
- Pin exact values using the pandas path, e.g. `ar.combine_columns(pd.DataFrame(...), ...)`.
- Read results with `result["combined"].tolist()`, or `pd.isna(result.loc[1, "combined"])` for NA.
- Do not assert the output dtype. `agg(axis=1)` plus `mask` may give object or string dtype.
- Avoid zero-row pandas frames: `agg(..., axis=1)` on an empty frame behaves unclearly.
- Check that the input is not mutated: `assert "combined" not in df.columns` after the call.
- For ArFrames:
  - Build input with `ar.from_pandas(pd.DataFrame({...}))`.
  - Test mainly the validation errors.
  - Safe checks: `result.columns == [..., "combined"]` and `isinstance(result, ar.ArFrame)`.
  - For values, at most use simple non-null string columns (e.g. `"x y"`), knowing this relies on the C++ code.
- `KeyError`'s `str()` adds surrounding quotes, so match with `re.escape(...)` on a substring, or compare `exc.value.args[0]` exactly.
- Messages that are easy to get wrong:
  - `"Output column 'x' already exists."` ends with a period.
  - `"subset must contain at least one column"` has no period.

# Notes: `arnio.cleaning.keep_rows_with_nulls(frame, *, subset=None)`

## 1. Purpose
- Returns only the rows that have at least one null value. With `subset`, only those columns are checked.
- It is the complement of `drop_nulls`, but it is implemented entirely in pandas.
- It accepts an `ArFrame` or a `pd.DataFrame` and returns the same type it was given.

## 2. Inputs
- **`frame`**: an `ArFrame` or a `pd.DataFrame`. Anything else raises `TypeError`.
  - An `ArFrame` is converted with `to_pandas(frame)`.
  - A DataFrame is used as-is and is not copied up front.
- **`subset`** (keyword-only, default `None`):
  - `None` means every column is checked: `cols = df.columns.tolist()`.
  - Otherwise it must be a non-string `collections.abc.Sequence` (list or tuple) or a `pd.Index`.
  - Every item must be a `str` and must exist in `df.columns`.
  - Duplicate names are allowed (they are not rejected).
  - The order of names in `subset` has no effect on the result.

## 3. Raises (checks run in this order)
1. Bad frame type: `TypeError("frame must be an ArFrame or a pandas DataFrame")`.
   - Trigger: `keep_rows_with_nulls({"a": [1]})`.
2. `subset` is a `str`: `TypeError(f"keep_rows_with_nulls: 'subset' must be a list of column names, not a string. Did you mean subset=['{subset}']?")`.
   - Trigger: `subset="a"` gives `...Did you mean subset=['a']?`.
3. `subset` is `bytes`: `TypeError("subset must be a sequence of column names, not a string")`.
   - Trigger: `subset=b"a"`.
4. `subset` is not a Sequence or Index: `TypeError("subset must be a sequence of column names")`.
   - Trigger: `subset={"a"}` (a set) or `subset=5`.
5. `subset` contains a non-`str` item: `TypeError("subset must contain only string column names")`.
   - Trigger: `subset=["a", 1]`.
6. `subset` is empty: `ValueError("keep_rows_with_nulls: subset cannot be empty; pass subset=None to check all columns")`.
   - Trigger: `subset=[]` or `subset=()`.
7. Unknown column: `KeyError(f"Missing columns for keep_rows_with_nulls: {missing}. Available columns: {available}")`.
   - `missing` is the Python list repr of the missing names, in `subset` order.
   - `available` is `", ".join(columns)`, or `"<none>"` if the frame has no columns.
   - Example message: `Missing columns for keep_rows_with_nulls: ['z']. Available columns: a, b`.
   - Because it is a `KeyError`, `str(exc)` wraps the message in extra quotes. Match with `re.escape(...)` on a fragment.
- Note on ordering: the frame-type check runs before any check on `subset`.

## 4. Returns and guarantees
- **Return type**:
  - `ArFrame` input gives an `ArFrame` (`from_pandas(result)`).
  - `DataFrame` input gives a `pd.DataFrame`.
- **Null test**: `df[cols].isnull().any(axis=1)`. Only pandas nulls count: `None`, `NaN`, `pd.NA`, `NaT`.
  - Empty strings `""` and whitespace-only strings are **not** treated as null, even though the docstring says "null/empty".
- **Columns**: all columns are kept, in their original order, not just the `subset` columns.
- **Rows**:
  - Rows that match keep their original relative order.
  - The index is always reset to `0..n-1` via `reset_index(drop=True)`. This applies to pandas input too: if the input index is `[10, 20, 30]`, the output index is a `RangeIndex`.
- **Values**: unchanged, and nulls stay null.
- **Result object**: always a new object, never the input itself.
- **`ArFrame` attrs**: not explicitly copied. The function does not use `_wrap`. Whether attrs survive depends on `to_pandas`/`from_pandas` in `convert.py`, which is not shown here, so do not assert on attrs.

## 5. Side effects
- The input frame or DataFrame is not mutated, because boolean indexing creates a new frame.
- There is no global state, registry change, or file I/O.

## 6. Branches and edge cases
- **`subset=None`**: checks all columns.
  - Example: `{"a": [1.0, None, 3.0], "b": ["x", "y", None]}` keeps original rows 1 and 2 → a=[NaN, 3.0], b=["y", None].
- **With `subset`**: on the same data, `subset=["a"]` keeps only original row 1. `subset=["b"]` keeps only original row 2.
- **No nulls anywhere**: result has 0 rows and the same columns, e.g. `shape == (0, 2)`.
- **Every row has a null**: all rows are returned, in their original order.
- **Zero-row input**: returns zero rows with the same columns.
- **Zero-column pandas frame** (`pd.DataFrame(index=range(3))`):
  - `cols = []`, so `any(axis=1)` is all False.
  - The result is an empty frame with 0 rows.
- **Duplicate names in `subset`**: `subset=["a", "a"]` gives the same rows as `["a"]`.
- **Tuple or Index subset**: `subset=("a",)` and `subset=pd.Index(["a"])` both work.

## 7. Rules from the docstring
- A row is kept if **any** column in `subset` is null.
- A string `subset` raises `TypeError`, and a missing column raises `KeyError`. Both match the code.

## 8. Testing hints
- **Building input**:
  ```python
  df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", None]})
  fr = ar.from_pandas(df)
  out = ar.keep_rows_with_nulls(fr)
  ```
- **Inspecting an `ArFrame` result**: use `ar.to_pandas(out)`, `out.shape`, and `out.columns`.
- **Comparing nulls**: compare with `pd.isna(...)`, not with `== None`.
  - After the ArFrame round trip, a null string may come back as `None` or `NaN`.
  - Int columns that contain nulls are float in pandas.
- **Testing the pandas path**:
  - Assert that `type(result) is pd.DataFrame`.
  - Assert `list(result.index) == list(range(len(result)))` when the input index was non-default.
  - Assert the original `df` is unchanged afterwards (`df.equals(copy)`).
- **Exact null criterion**: test it on the **pandas path**. A row whose only "missing" value is `""` must NOT be kept.
  - Avoid `""` on the ArFrame path: how the C++ conversion stores empty strings is not visible here.
- **Empty results**: assert `shape[0] == 0` and the column list. Do not assert dtypes of empty ArFrame results, since inference is done by C++/convert code that is not shown.
- **Error messages**: use `pytest.raises(..., match=re.escape("Missing columns for keep_rows_with_nulls: ['z']"))`.
  - Escape the brackets, and quotes too where they appear.
- **Not observable**: attrs propagation, and internal dtypes after `from_pandas`.

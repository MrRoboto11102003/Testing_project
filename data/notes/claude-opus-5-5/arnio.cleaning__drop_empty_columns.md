# Notes: `arnio.cleaning.drop_empty_columns(frame)`

## 1. Purpose
- Returns a new ArFrame without the columns whose every value is null, `""`, or a whitespace-only string.
- The other columns are rebuilt unchanged: same values, same dtypes, same original order.

## 2. Inputs
- `frame`: must be an `ArFrame`. It is checked by `_validate_arframe(frame)`, which lives in `arnio.frame` and is not visible here.
- There are no other parameters. There is no `subset`, and pandas DataFrames are not accepted.

## 3. Raises
- A non-ArFrame input, e.g. `ar.drop_empty_columns(pd.DataFrame({"a": [1]}))`, raises from `_validate_arframe`.
  - That code is not visible, so the type is most likely `TypeError` and the message is unknown.
  - Use `pytest.raises(TypeError)` without `match=`.
- Nothing else is raised by this function's own code.
- An all-empty frame does NOT raise. Compare `drop_columns`, which refuses to remove all columns.

## 4. Returns and guarantees
- Always a **new** `ArFrame` object, never the input itself (`result is not frame`).
- **Column order** is the original `frame.columns` order with the empty columns removed.
- **Kept columns** are rebuilt from `cpp_column.to_python_list()` with the original C++ dtype as a hint. So:
  - `result.dtypes[c] == frame.dtypes[c]`.
  - Values are identical. Whitespace in kept columns is NOT stripped, e.g. `["  ", "a"]` stays `["  ", "a"]`.
  - Nulls in kept columns stay null.
- **Attrs**: `result._attrs` is a `copy.deepcopy` of `frame._attrs`, or `None` if the source's was `None`. Equal to the source, but not the same object (on every path).
- **Row count**:
  - Unchanged when at least one column remains.
  - When all columns are dropped: `_Frame.from_dict({}, {}, nrows)` is used, which is expected to give shape `(nrows, 0)`.
  - If that call raises `TypeError` (C++ behaviour, not visible), the fallback is `_Frame()`, likely shape `(0, 0)`.

## 5. Side effects
- None. The input frame and its attrs are not mutated; only a deep copy of attrs is made.
- No global state is changed and no files are written.

## 6. Branches and edge cases
- **Zero rows** (`frame.shape[0] == 0`): early return.
  - ALL columns are kept, even though they are "empty".
  - Each keeps its original dtype.
  - Example: `ar.from_pandas(pd.DataFrame({"a": pd.Series([], dtype="int64"), "b": pd.Series([], dtype="float64")}))` gives columns `["a", "b"]`, shape `(0, 2)`, and the same dtypes.
- **Emptiness test** (rows > 0), per column on `to_pandas(frame)`: `series.isna() | series.map(lambda v: isinstance(v, str) and v.strip() == "")`. The column is dropped iff this is all True.
  - Dropped: `[None, None]`, `[np.nan, np.nan]` (float64), `["", "  "]`, `["\t", None, ""]`.
    - `str.strip()` removes `\t`, `\n`, and Unicode whitespace.
  - Kept: `["", "a"]`, `[None, "x"]`, `[0, 0]`, `[False, False]`, `[0.0, np.nan]`.
    - `0`, `False`, and `0.0` are not empty.
- **Mix**: `{"a": [1, 2], "b": [None, None], "c": ["x", " "]}` gives columns `["a", "c"]`, shape `(2, 2)`, and `c` values `["x", " "]`.
- **No empty columns**: the result equals the input in columns, dtypes, and values, but is still a new object.
- **All columns empty, rows > 0**: e.g. `{"a": [None, None], "b": ["", " "]}` gives `result.columns == []` and `shape[1] == 0`.
  - `shape[0]` is probably 2; see the row-count caveat in section 4.

## 7. Rules from the docstring
- "String values containing only whitespace are treated as empty."
- Columns that are entirely null or empty strings are removed.

## 8. Testing hints
- **Build input**:
  ```python
  f = ar.from_pandas(pd.DataFrame({...}))
  ```
- **Check structure** with `result.columns`, `result.shape`, and `result.dtypes`.
- **Check values** with `ar.to_pandas(result)[col].tolist()`. For nulls, use `pd.isna(...)` rather than comparing to `None`.
- **Check dtype preservation**: e.g. int64 `[1, 2]` next to an empty column gives `result.dtypes["a"] == "int64"`.
- **Check attrs**:
  ```python
  f._attrs = {"src": {"x": [1]}}
  r = ar.drop_empty_columns(f)
  assert r._attrs == f._attrs and r._attrs is not f._attrs
  ```
  Also check that mutating `r._attrs["src"]["x"]` leaves `f._attrs` unchanged.
- **Check order**: `{"z": [1], "e": [None], "a": ["k"]}` gives `["z", "a"]`.
- **Pitfalls**:
  - Do not expect the zero-row branch to drop anything.
  - Do not expect whitespace stripping in kept columns.
  - Do not use `match=` on the non-ArFrame error.
  - Whether `from_pandas` turns `""` into null depends on C++. It does not matter for dropping, but avoid asserting that a kept column contains a literal `""` vs null. Use `" "` or real text instead.
  - The all-dropped row count depends on the C++ `from_dict`, so assert `shape[1] == 0` firmly and `shape[0]` softly.

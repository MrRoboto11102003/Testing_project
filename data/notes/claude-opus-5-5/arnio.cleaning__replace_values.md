# Notes: `arnio.cleaning.replace_values(frame, mapping, column=None)`

## 1. Purpose
- Replaces exact cell values using a `{old: new}` dict, either in one column or in every column.
- Null-like keys (`None`, `NaN`, `pd.NA`, `pd.NaT`) fill pre-existing nulls.
- The result has the same kind as the input: ArFrame in gives ArFrame out; `pd.DataFrame` in gives `pd.DataFrame` out.

## 2. Inputs
- `frame`: an `ArFrame` or a `pd.DataFrame`.
  - ArFrame input is converted with `to_pandas(frame)`.
  - DataFrame input is copied with `frame.copy(deep=False)`.
- `mapping`: any `collections.abc.Mapping`, converted with `dict(mapping)`. It must not be empty.
  - Each key is classified in insertion order:
    - **Null-like key**: `k is None`, or a scalar `k` where `pd.isna(k)` is true. Then `null_key_present = True` and `null_replacement = v`. If there are several null keys, the last one wins.
    - **Scalar key** (`pandas.api.types.is_scalar`, not tuple/list/ndarray/Series/Index): goes into the normal replace dict.
    - **Anything else** (for example a tuple `(1, 2)` or a `frozenset`): raises `TypeError` (see Raises).
- `column`: `None` (the default) means the mapping applies to all columns. Otherwise it must be a non-blank `str` that exists in the frame's columns.

## 3. Raises (in check order)
1. Frame of the wrong type: `TypeError("frame must be an ArFrame or a pandas DataFrame")`.
   - Example: `replace_values([1], {1: 2})`
2. Mapping that is not a Mapping: `TypeError(f"mapping must be a dict-like mapping of {{old_value: new_value}}, not {type(mapping).__name__}.")`.
   - Example: `replace_values(df, [("a", "b")])` gives `"... not list."`
3. Empty mapping: `ValueError("mapping must not be empty")`.
   - Example: `replace_values(df, {})`
4. Column that is not a string, or is blank: `TypeError("column must be a non-empty string when provided")`.
   - Examples: `column=5`, `column=""`, `column="  "`
5. Column not in the frame: `KeyError(f"Column '{column}' not found. Available columns: {available}")`.
   - `available` is `", ".join(map(str, df.columns))`, or `"<none>"` if there are no columns.
   - Example: columns `a`, `b` with `column="z"` gives `Column 'z' not found. Available columns: a, b`.
   - `str(KeyError)` adds quotes around the message, so `pytest.raises(KeyError, match="Column 'z' not found")` still works.
6. Non-scalar key: `TypeError(f"replace_values() does not support non-scalar mapping keys. Got key {k!r} of type '{type(k).__name__}'. Only scalar values (str, int, float, bool) and null-like keys (None, float('nan'), pd.NA, pd.NaT) are supported.")`.
   - Example: `{(1, 2): "x"}` gives `Got key (1, 2) of type 'tuple'`.
   - This check runs only after checks 1–5 pass. With a bad column and a tuple key, the column error is raised.

## 4. Returns and guarantees
- **Column path** (`column` given):
  - `s = df[column]` and `mask = s.isna()` (computed before any replacement).
  - Then `s.replace(normalized_mapping)`, if the dict is non-empty.
  - Then `s.where(~mask, null_replacement)`, if a null key is present.
  - The result is written back to `df[column]`. Other columns are untouched.
- **Whole-frame path**: the same steps, using `df.isna()`, `df.replace(...)` and `df.where(...)` over all columns.
- Matching is exact, whole-value matching (pandas `replace` with `regex=False`). There is no substring replacement: `{"a": "b"}` leaves `"abc"` unchanged.
- Nulls produced by the mapping stay null; only nulls that were null originally are filled. Example: `["a", None]` with `{"a": None, None: "z"}` gives `[None/NaN, "z"]`.
- Column order and row count are unchanged.
- **ArFrame input**: the result comes from `from_pandas(df)`, so it is a new ArFrame. It does not copy `frame._attrs` (unlike `_wrap`).
- **DataFrame input**: the result is a new DataFrame (`result is not df`).
- Output dtypes come from pandas `replace` and `where`. The ArFrame↔pandas dtype mapping lives in `arnio.convert`, which is not shown here.

## 5. Side effects
- The input is never mutated. The pandas path works on a shallow copy and assigns new columns or frames. ArFrame inputs are only read.
- There is no global state and no file I/O.

## 6. Branches and edge cases
- Column given, only scalar keys: `df = pd.DataFrame({"a": ["x", "y"], "b": ["x", "q"]})` with `{"x": "X"}` and `column="a"` gives `a == ["X", "y"]` and `b == ["x", "q"]`.
- No column: the same input with `{"x": "X"}` gives `a == ["X", "y"]` and `b == ["X", "q"]`.
- Only a null key in the column path: `{"a": ["x", None]}` with `{None: "missing"}` and `column="a"` gives `["x", "missing"]`.
- Null key in the whole-frame path fills nulls in every column. Example: `pd.DataFrame({"s": ["x", None], "n": [1.0, np.nan]})` with `{np.nan: 0}` gives `s == ["x", 0]` and `n == [1.0, 0.0]`. Prefer type-compatible fills.
- Keys that are never present: values are unchanged and there is no error.
- `pd.NA`, `float("nan")` and `pd.NaT` keys all behave like `None`.
- A mapping value that is itself a null creates real nulls. Example: `{"x": None}` with `column="a"` on `["x", "y"]` makes `pd.isna(result["a"][0])` true.

## 7. Rules from comments and docstring
- "Avoid mutating the caller's DataFrame in the direct pandas API path."
- The null-key path is kept separate "because NaN != NaN". Only values that were already null before replacement are filled, "so null-valued mapping results remain real nulls".
- Tuple/list/ndarray/Series/Index keys are rejected up front so that pandas does not raise confusing errors.

## 8. Testing hints
- To build inputs: `fr = ar.from_pandas(pd.DataFrame({...}))`.
- To observe results: `out = ar.to_pandas(ar.replace_values(fr, {...}, column="a"))` and compare `out["a"].tolist()`. Check nulls with `pd.isna(x)`, never with `== None` or `== np.nan`.
- Also test the pandas-in path:
  - `isinstance(res, pd.DataFrame)`
  - `res is not df`
  - `df` is unchanged afterwards (`pd.testing.assert_frame_equal` against a saved `.copy()`).
- Good regression targets:
  - Nulls created by the mapping are not overwritten by the null key.
  - `column=` limits changes to that column.
  - The exact error messages and the order of the checks.
  - The ArFrame type is preserved.
- Pitfalls to avoid:
  - Do not assert exact dtypes; ArFrame conversion is defined elsewhere.
  - Avoid mixing types inside one column (for example replacing a string with an int in a string column); how `from_pandas` handles that is not visible here.
  - Avoid `True`/`1`/`1.0` keys: they are the same dict key, and pandas may match bools to ints.
  - Avoid chained mappings like `{"a": "b", "b": "c"}`; the result depends on pandas.
  - Do not test `_attrs` preservation; attrs are not copied.
  - Hand-written integer columns with NaN become float in pandas, so `[1, None]` gives `[1.0, nan]`.

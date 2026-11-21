# Notes: `arnio.cleaning.coalesce_columns(frame, *, subset, output_column="coalesced")`

## 1. Purpose
- Adds one new column, `output_column`, as the last column.
- Each row's value is the first non-null value across `subset`, scanned left to right in the order `subset` lists them.
- Implemented in pandas: `df[output_column] = df[subset_columns].bfill(axis=1).iloc[:, 0]`.

## 2. Inputs
- `frame`: an `ArFrame` or a `pd.DataFrame`. Anything else raises TypeError.
- `subset`: required, keyword-only, no default. Validated in this order:
  1. A `str` or `bytes` value raises TypeError.
  2. It must be a `collections.abc.Sequence` (list, tuple) or a `pd.Index`. A `set`, a generator or `None` raises TypeError.
  3. It is converted to a `list`, and every item must be a `str`.
  4. Duplicate names are rejected.
  5. Every name must exist in `frame.columns`.
- `subset` order matters. It sets the priority, not the frame's column order.
- `output_column`: a `str` with at least one non-whitespace character. It must not already be a column of the frame.

## 3. Raises (in check order)
1. Frame type: `TypeError("frame must be an ArFrame or a pandas DataFrame")`. Trigger: `coalesce_columns({"a": [1]}, subset=["a"])`.
2. Bad `output_column` (not a str, or blank/whitespace): `ValueError("output_column must be a non-empty string")`.
   - Triggers: `output_column=""`, `"  "`, or `5`.
   - This is checked before `subset`.
3. `subset` is a string: `TypeError("subset must be a sequence of column names, not a string")`. Trigger: `subset="a"`.
4. `subset` is not a sequence: `TypeError("subset must be a sequence of column names")`. Triggers: `subset={"a"}` or `subset=None`.
5. Non-string items: `TypeError("subset must contain only string column names")`. Trigger: `subset=["a", 1]`.
6. Duplicates: `ValueError(f"subset contains duplicate column names: {duplicates}")`.
   - Trigger: `subset=["a","a"]` gives `"subset contains duplicate column names: ['a']"`.
   - This is checked before missing columns, so `["a","a","zz"]` raises ValueError, not KeyError.
7. Missing columns: `KeyError(f"Missing columns for coalesce_columns: {missing}. Available columns: {available}")`.
   - `available` is `", ".join(columns)`, or `"<none>"` if the frame has no columns.
   - Trigger: frame with columns a, b and `subset=["a","zz"]` gives `"Missing columns for coalesce_columns: ['zz']. Available columns: a, b"`.
8. Empty subset: `ValueError("subset must contain at least one column")`. Triggers: `subset=[]` or `subset=()`.
9. Output already exists: `ValueError(f"Output column '{output_column}' already exists.")`.
   - Trigger: `subset=["a"], output_column="b"` when `b` exists.
   - The default name `"coalesced"` also triggers it if the frame has a column with that name.

## 4. Returns and guarantees
- The result has the same type as the input:
  - ArFrame in gives ArFrame out, built via `from_pandas(df)`.
  - DataFrame in gives DataFrame out.
- Columns are all the original columns in their original order, then `output_column` at the end.
- Original column values are unchanged.
- Row count is unchanged and no rows are dropped.
- Per row, the output is the first value in `subset` order that pandas considers non-null.
  - `None`, `NaN` and `pd.NA` are all treated as null.
  - If every subset value in a row is null, the output is null.
  - pandas does not treat `""` as null.
- Pandas input: the index is preserved. The result is a shallow copy of the input with the extra column.
- ArFrame input: attrs are not explicitly carried over (no `_wrap`).
  - Whether `from_pandas` keeps attrs depends on code in another module. Do not test attrs.
- Dtype of the output:
  - When all subset columns share a dtype (all float64, or all object strings), the result normally has that dtype.
  - For mixed dtypes (int and str), `bfill(axis=1)` may upcast to object or float. Assert values, not the dtype.

## 5. Side effects
- The input pandas DataFrame is not modified. A column is added only to `frame.copy(deep=False)`, so the original `df.columns` is unchanged.
- An input ArFrame is not modified.
- No global state is touched and no files are written.

## 6. Branches and edge cases
- First subset column non-null: its value wins even if later columns are also non-null.
  - `a=[1.0], b=[9.0]`, `subset=["a","b"]` gives `[1.0]`.
  - `subset=["b","a"]` gives `[9.0]`.
- Fallthrough to a later column:
  - `a=[1.0, nan, nan]`, `b=[10.0, 2.0, nan]`, `subset=["a","b"]` gives `[1.0, 2.0, NaN]`.
- Three columns:
  - `a=[nan]`, `b=[nan]`, `c=[7.0]` gives `[7.0]`.
- Single-column subset: the output equals that column's values.
- Strings (pandas object):
  - `a=["x", None]`, `b=["y","z"]` gives `["x","z"]`.
- Columns outside `subset` are ignored even when non-null.
- Zero-row frame with a valid subset returns 0 rows, with `output_column` added.
- ArFrame path: `to_pandas`, then pandas ops, then `from_pandas`. Null handling and dtypes depend on the `convert` module and the C++ core, which are not visible here.

## 7. Rules from docstring and comments
- "Select the first non-null value from a list of columns."
- Comment: "Select the first non-null/non-NaN/non-None value per row."
- Docstring Raises section:
  - TypeError for a bad subset or frame.
  - ValueError for an empty subset, an empty output_column, or an existing output_column.
  - KeyError for a missing subset column.

## 8. Testing hints
- The pandas path gives the most exact assertions:
  ```python
  df = pd.DataFrame({"a": [1.0, np.nan, np.nan], "b": [10.0, 2.0, np.nan]})
  out = ar.coalesce_columns(df, subset=["a", "b"], output_column="c")
  assert list(out.columns) == ["a", "b", "c"]
  assert out["c"].tolist()[:2] == [1.0, 2.0] and pd.isna(out["c"].iloc[2])
  assert list(df.columns) == ["a", "b"]  # input not mutated
  ```
- ArFrame path:
  - Build with `ar.from_pandas(...)` and read the result back with `ar.to_pandas(result)`.
  - Check `isinstance(result, ar.ArFrame)` and the `result.columns` order.
  - Compare nulls with `pd.isna`, never with `== None` or `== np.nan`.
  - Prefer float64 or string columns. Int columns with nulls may come back as a different dtype.
- Use `pytest.raises(..., match=re.escape(msg))`. For a KeyError, `str(exc)` adds quotes, but `match` uses a substring search, so it still works.
- Test priority by swapping the `subset` order. A bug that uses frame order, or picks the last value instead of the first, shows up there.
- Test the check order:
  - `output_column=""` combined with `subset="a"` raises ValueError, because `output_column` is checked first.
  - Duplicates combined with a missing column raise ValueError.
  - An existing output column combined with `subset=[]` raises the "at least one column" error first.

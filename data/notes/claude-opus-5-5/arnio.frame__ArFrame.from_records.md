# Notes: `ArFrame.from_records(records, columns=None)` (arnio/frame.py)

## 1. Purpose
- A classmethod that builds an ArFrame from a non-empty list of dicts, or from a list of lists/tuples.
- It validates the input in Python first.
- It then calls `pd.DataFrame.from_records(records, columns=columns)` and passes the result to `arnio.convert.from_pandas(df)`.
- `cls` is not used, so the result is always a plain `ArFrame`.
- Import it with `from arnio.frame import ArFrame`. `ar.ArFrame` probably works too.

## 2. Inputs
- **`records`** must be exactly a `list`. A tuple, a generator or a DataFrame is rejected.
  - It must be non-empty.
  - The type of `records[0]` picks the branch:
    - `dict` gives the dict branch.
    - `list` or `tuple` gives the sequence branch.
    - Anything else is an error.
- **`columns`** defaults to `None`. If given, it must be a `list` or `tuple` whose elements are all `str`.
  - `bytes` elements are rejected.
  - An empty `[]` passes the generic checks. It is rejected later only in the dict branch.
  - Duplicate names are **not** checked here; what happens next depends on pandas and from_pandas, so do not test it.
- **Cell values:** any `list` or `dict` value is rejected. Other values pass this check, including tuples, sets and None; avoid testing them.
- In the sequence branch, list rows and tuple rows may be mixed, e.g. `[[1], (2,)]` is OK.

## 3. Raises (in check order)
1. `records` is not a list:
   - `TypeError(f"records must be a list, got {type(records).__name__!r}")`
   - `from_records(({"a":1},))` gives `"records must be a list, got 'tuple'"`.
2. Empty list: `from_records([])` gives `ValueError("records must be non-empty")`.
3. Checks on `columns`, only when it is not None. These run **before** the record-type checks.
   - String or bytes: `columns="a"` or `b"a"` gives `TypeError("columns must be a list or tuple of strings, not a string or bytes")`.
   - Other non-list/tuple: `columns={"a"}` gives `TypeError("columns must be a list or tuple of strings")`.
   - Non-string element: `columns=["a", 1]` gives `TypeError("columns must contain only strings")`.
4. **Dict branch.** One loop over the rows; for each row i, the type check is followed by the nested-value check.
   - Non-dict row: `[{"a":1}, [1]]` gives `TypeError("all records must be dicts, but row 1 is 'list'")`.
   - Nested value:
     - Template: `TypeError(f"nested values are not supported; column {col!r} at row {i} contains a {type(val).__name__!r}")`.
     - `[{"a":[1]}]` gives `"nested values are not supported; column 'a' at row 0 contains a 'list'"`.
   - Interleaving matters:
     - `[{"a":{}}, 5]` reports the nested `'dict'` at row 0 before the bad type at row 1.
     - `[{"a":1}, 5]` gives `"all records must be dicts, but row 1 is 'int'"`.
   - Empty columns are checked AFTER the loop.
     - `from_records([{"a":1}], columns=[])` gives `ValueError("columns must not be empty when records are dicts; pass columns=None to infer column names from the record keys")`.
     - With `[{"a":[1]}]` and `columns=[]`, the nested TypeError wins.
5. **Sequence branch.**
   - `columns is None`: `from_records([[1,2]])` gives `ValueError("columns must be provided when records are lists or tuples")`.
   - Then, per row i, in this order: type, then length, then nested values.
     - Non-list/tuple row: `([[1], {"a":1}], columns=["a"])` gives `TypeError("all records must be the same type, but row 1 is 'dict'")`.
     - Length mismatch: `([[1,2],[3]], columns=["a","b"])` gives `ValueError("row 1 has 1 value(s) but 2 column(s) were provided")`.
     - Nested value: `([[1,[2]]], columns=["a","b"])` gives `TypeError("nested values are not supported; column 'b' at row 0 contains a 'list'")`. The column name is `columns[j]`.
6. **Other first element:**
   - Template: `TypeError(f"records must contain dicts, lists, or tuples, got {type(first).__name__!r}")`.
   - `from_records([1,2])` gives `"... got 'int'"`; `from_records(["x"])` gives `"... got 'str'"`.
   - Only `records[0]` is inspected here. With `columns="a"` the columns error from step 3 fires first.
- Exception types are plain `TypeError` and `ValueError`, not ArnioError subclasses.

## 4. Returns and guarantees
- Returns a new `ArFrame` from `from_pandas`.
- Column order (this is pandas behaviour):
  - Dict records with `columns=None`: keys in order of first appearance across the records.
    - Example: `[{"a":1,"b":2},{"c":3,"a":4}]` gives `["a","b","c"]`.
  - Dicts with `columns` given: the order of `columns`. Keys not listed are dropped, and listed names missing from the data become all-null columns. Prefer simple cases.
  - Sequence records: the order of `columns`.
- Missing dict keys become NaN or None, i.e. nulls. Their dtype and value after from_pandas are not visible here, so avoid asserting exact null values.
- `shape == (len(records), n_columns)`.
- Dtypes come from from_pandas and the C++ core, which are not shown.
  - Likely results: `int64` for Python ints, `float64` for floats, `string` for str, `bool` for bools.
  - Assert dtypes only for clean, homogeneous columns.

## 5. Side effects
- None. `records` and `columns` are not mutated, and no global state is touched.

## 6. Branches and edge cases
- `from_records([{"a":1,"b":"x"},{"a":2,"b":"y"}])` gives columns `["a","b"]`, shape `(2,2)`, `f["a"] == [1,2]` and `f["b"] == ["x","y"]`.
- `from_records([(1,"x"),[2,"y"]], columns=("id","name"))` works: tuple columns and mixed row types are both accepted.
- `columns` as a tuple is accepted in both branches.
- A single-record list is accepted: `[{"a":1}]` gives shape `(1,1)`.

## 7. Rules from the docstring
- Records must be non-empty dicts, lists or tuples, and must not be mixed across kinds.
- Columns are required for sequence records and optional for dicts.
- Nested list/dict cells are unsupported.

## 8. Testing hints
- Use `pytest.raises(TypeError, match=re.escape(msg))`.
  - Messages contain `(`, `)`, `[`, `'` and `.`, so escape them or use the full exact string.
- Type names come from `type(x).__name__` with repr quotes, e.g. `'tuple'`, `'int'`, `'list'`, `'dict'`, `'NoneType'`.
- Read values with `frame["col"]` (returns a list) or `frame.to_dict()`. Compare string and int columns exactly.
  - Avoid columns that contain nulls, and mixed int/float columns.
- Good regression targets:
  - The exact messages, including the row index i (0-based) and the column name in the nested message.
  - The check order: columns validation before record type, and the empty-columns check after the dict loop.
  - The length check before the nested check.
  - Tuple acceptance for both `records` rows and `columns`.
- Do not test:
  - Duplicate column names.
  - `columns=[]` with sequence rows.
  - Tuple or set cell values.
  - `attrs` (from_pandas is not shown).

# Notes: `ArFrame.select_columns(self, columns)` (arnio/frame.py)

## 1. Purpose
- Returns a **new** ArFrame that holds only the requested columns. Every row is kept.
- The column projection itself happens in the C++ core (`self._frame.select_columns(columns)`).
- This is the **method**. It is a different function from the module-level `ar.select_columns` in `arnio.cleaning`, whose code is not shown. Call it as `frame.select_columns([...])`.

## 2. Inputs
- `columns`: must be a `list` or `tuple` of `str`.
  - It must be non-empty.
  - It must not contain duplicates.
  - Every name must exist in `self.columns`.
- There is no normalisation: no dedup, no stripping and no case folding. Names must match exactly, including case and whitespace.
- There are no other parameters and no defaults.

## 3. Raises (checks run in this exact order)
1. `columns` is a `str` → `TypeError("columns must be a sequence of column names, not a string.")`
   - Example: `f.select_columns("a")`
2. `columns` is not a list or tuple → `TypeError("columns must be a list or tuple of column names.")`
   - Examples: `f.select_columns({"a"})`, `f.select_columns(None)`, `f.select_columns(b"a")`, `f.select_columns(iter(["a"]))`
3. `columns` is empty → `ValueError("Column selection cannot be empty.")`
   - Examples: `f.select_columns([])`, `f.select_columns(())`
4. Any element is not a `str` → `TypeError("All column names must be strings.")`
   - Examples: `f.select_columns(["a", 1])`, `f.select_columns([1, 1])`
   - `[1, 1]` hits this TypeError, not the duplicate error, because the type check comes first.
5. Duplicates are present → `ValueError("Duplicate column names are not allowed.")`
   - Example: `f.select_columns(["a", "a"])`
   - `["zz", "zz"]` with `zz` missing also gives the **duplicate** error, because this check runs before the existence check.
6. Unknown names → `ValueError(f"Unknown columns: {missing}")`
   - `missing` is a list in **request order** that contains only the absent names.
   - Example: `f.select_columns(["x", "a", "y"])` gives the message `Unknown columns: ['x', 'y']`.
- Nothing is returned partially; the whole call fails.

## 4. Returns and guarantees
- Returns a new `ArFrame`; `result is not f`.
- Row count is unchanged: `len(result) == len(f)`.
- Column count: `result.shape == (len(f), len(columns))`.
- Each selected column keeps its dtype and its values: `result.dtypes[c] == f.dtypes[c]` and `result[c] == f[c]`.
- Column order is decided by the C++ core, whose code is not shown.
  - It is very likely the request order, but that is not verified.
  - It is certain that selecting in the frame's original order gives that order back.
- Metadata: `result._attrs == copy.deepcopy(f._attrs)`.
  - The result's attrs are equal to the original's but are a separate object.
  - Nested containers are not shared either.

## 5. Side effects
- None. The original frame's columns, values, dtypes and `_attrs` are not mutated. There is no global state and no I/O.

## 6. Branches and edge cases
- **Select all columns**, e.g. `f.select_columns(["a", "b", "c"])`: works and gives a frame equal to `f` (`result == f` is True).
- **Tuple input**, e.g. `f.select_columns(("a",))`: works the same as a list.
- **Single column**: shape is `(n, 1)` and `result.columns == ["a"]`.
- **Zero-row frame**: the validation is the same. The result shape should be `(0, k)`, but this relies on the C++ core.
- **Other callers**:
  - `f[["a", "b"]]` (`__getitem__` with a list) calls this method. However, `__getitem__` first raises its own errors: **KeyError** for missing names and a different TypeError for non-strings.
    - Duplicates passed through `f[["a", "a"]]` still reach this method and raise the duplicate ValueError.
  - `select_dtypes` calls it with matched columns in original order.

## 7. Rules from the docstring
- Raises TypeError for an invalid sequence type or non-string names.
- Raises ValueError for an empty selection, duplicates, or unknown columns.
- The returned frame contains only the selected columns.

## 8. Testing hints
- **Build input:**
  ```python
  f = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5], "c": ["x", "y", "z"]}))
  ```
- **Observe the result** through `.columns`, `.shape`, `len()`, `.dtypes`, `res["a"]`, `res.to_dict()` (gives `{'a': [1, 2, 3], 'c': ['x', 'y', 'z']}`), and `==`.
  - Prefer `dtypes` and `columns` comparisons against the original over hard-coded dtype strings, unless you are confident of them.
  - Expected values are `"int64"`, `"float64"` and `"string"`.
- **Messages:** match exactly with `pytest.raises(ValueError, match=re.escape("Unknown columns: ['x', 'y']"))`.
  - Escape the brackets and the trailing periods.
  - Pin the order of the checks with combined-fault inputs such as `[1, 1]`, `["zz", "zz"]` and `[]`.
- **Attrs:** these are only reachable privately. Build `g = ar.ArFrame(f._frame, attrs={"meta": {"k": [1]}})`, then `r = g.select_columns(["a"])`.
  - Assert `r._attrs == {"meta": {"k": [1]}}`.
  - Assert `r._attrs is not g._attrs` and `r._attrs["meta"] is not g._attrs["meta"]`.
  - Mutate `r._attrs["meta"]["k"].append(2)`, then assert that `g._attrs` is unchanged.
- **Non-mutation:** after the call, assert `f.columns == ["a", "b", "c"]` and `f.shape == (3, 3)`.
- **Pitfalls:**
  - For a reversed request such as `["c", "a"]`, assert only `set(res.columns) == {"a", "c"}` and the length, because the order is C++-dependent.
  - Avoid nulls and NaN in value assertions. Their Python representation (`None` vs `nan`) is decided by the C++ core and `from_pandas`, which are not shown.
  - Don't use `pd.DataFrame` as the argument (it is not a list or tuple, so it raises the TypeError from check 2).

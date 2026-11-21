# Notes: `arnio.cleaning.drop_columns(frame, columns)`

## 1. Purpose
- Returns a new ArFrame with the named columns removed.
- The remaining columns keep their original frame order.
- The work is done by a pandas round trip: `to_pandas` → `df.loc[:, remaining]` → `from_pandas`.

## 2. Inputs
- **`frame`**: must be an `ArFrame`.
  - It is checked by `_validate_arframe` from `arnio.frame`, whose code is not visible.
  - Assume it raises `TypeError` for non-ArFrames such as a `pd.DataFrame`, but the exact message is unknown, so do not match on it.
- **`columns`**: no default. Validated by `_validate_existing_column_sequence(argument_name="columns", missing_error=ValueError)`:
  - `str` or `bytes` is rejected.
  - It must be a `collections.abc.Sequence` (list, tuple) or a `pd.Index`.
    - `set`, `dict`, generators and `dict_keys` are rejected with `TypeError`.
  - Every item must be a `str`.
  - It is normalised with `list(columns)`. Duplicates are not rejected.
  - An empty sequence is allowed.

## 3. Raises
Checks run in this order:

| # | Check | Exception and message | Example |
|---|---|---|---|
| 1 | frame is not an ArFrame | `TypeError` (message comes from frame.py, not visible) | `drop_columns(pd.DataFrame({"a":[1]}), ["a"])` |
| 2 | `columns` is a str/bytes | `TypeError("columns must be a sequence of column names, not a string")` | `drop_columns(f, "a")` |
| 3 | not a Sequence or `pd.Index` | `TypeError("columns must be a sequence of column names")` | `drop_columns(f, {"a"})` |
| 4 | a non-str item | `TypeError("columns must contain only string column names")` | `drop_columns(f, ["a", 1])` |
| 5 | a column is missing | `ValueError(f"Columns not found in frame: {missing}")` | see below |
| 6 | would drop all columns | `ValueError("drop_columns cannot remove all columns from the frame")` | `drop_columns(f, ["a","b"])` on a frame with columns a, b |

- **Check 5 (missing columns):**
  - `missing` is a Python list in request order, e.g. `"Columns not found in frame: ['zz']"`.
  - This is a `ValueError`, **not** `KeyError`. That differs from `select_columns` and `validate_columns_exist`.
  - The missing check runs before the all-columns check. With columns `[a, b]`, `["a", "zz"]` raises the "not found" error.
- **Check 6 (all columns):**
  - It compares `len(requested list)` with `len(frame.columns)`, so duplicates count.
  - With 3 columns a, b, c: `["a","a","a"]` raises, even though only `a` would go.
  - With 2 columns a, b: `["a","a"]` raises.
- **Matching messages:** use `pytest.raises(ValueError, match=re.escape("Columns not found in frame: ['zz']"))`.

## 4. Returns and guarantees
- **Type:** a new `ArFrame`, built by `from_pandas`.
- **Columns:** the remaining columns in **original frame order**, not request order.
  - Columns `[a, b, c]` with `["c", "a"]` gives `["b"]`.
  - Columns `[a, b, c]` with `["c"]` gives `["a", "b"]`.
- **Rows:** row count and row order are unchanged. Values and nulls are carried through the pandas round trip.
- **Dtypes:** for normal int64/float64/string/bool columns, expect the same dtypes as the input. The exact rules live in `arnio.convert`, which is not visible.
- **Attrs:** not explicitly copied, because there is no `_wrap` call. Do not assert anything about attrs.
- **Duplicates in the request** (when check 6 does not fire) are harmless: `["a","a"]` on columns `[a, b, c]` gives `["b","c"]`.

## 5. Side effects
- The input frame is not mutated: its `.columns` and `.shape` are unchanged afterwards.
- There is no global state and no file I/O.

## 6. Branches and edge cases
- **Empty request** (`[]` or `()`):
  - Returns `frame.drop_columns([])`, the ArFrame method (not visible).
  - Expect the same columns and shape as the input.
  - It is reached before the all-columns check, so `[]` works even on a zero-column frame.
  - Do not assert `result is frame`.
- **Normal path:** `drop_columns(f, ["b"])` on `{"a":[1,2], "b":["x","y"], "c":[1.5, None]}` gives:
  - columns `["a","c"]`, shape `(2, 2)`;
  - `c` still holds a null at row 1.
- **`pd.Index` or tuple input** works the same as a list, e.g. `drop_columns(f, pd.Index(["b"]))`.
- **Duplicates beyond the column count** (e.g. `["a","b","c","a"]` on 3 columns) slip past check 6 and produce a zero-column frame. The behaviour of `from_pandas` there is unknown, so avoid this case.

## 7. Rules from the docstring and comments
- Raises `TypeError` for a str/bytes or a non-str item, and `ValueError` if any requested column is missing.
- Returns a "New frame with the requested columns removed".

## 8. Testing hints
- Build inputs with `ar.from_pandas(pd.DataFrame({...}))`. Check `result.columns` (a list), `result.shape`, and `ar.to_pandas(result)["a"].tolist()`.
- Good pinning tests:
  - remaining columns stay in original order regardless of request order;
  - the exact error messages and types (`ValueError`, not `KeyError`, for missing columns);
  - missing-before-all-columns ordering;
  - the duplicate-counting all-columns error (`["a","a"]` on a 2-column frame raises);
  - the input stays unchanged;
  - values are preserved, e.g. `[1, 2]` stays `[1, 2]` for an int column.
- Pitfalls:
  - Do not test with set or generator inputs expecting success.
  - Do not assert attrs or object identity on the empty path.
  - Use `re.escape` for messages that contain brackets and quotes.
  - When comparing float values, a null comes back as NaN; use `pd.isna`.

# Notes: `ArFrame.drop_columns(self, cols)` (arnio/frame.py)

## 1. Purpose
- Returns a new `ArFrame` without the named columns.
- The remaining columns keep their original order, and every row is kept.
- This is the **method** on `ArFrame`. It is different from the module function `ar.drop_columns` in `arnio.cleaning`, so call `frame.drop_columns(...)`.

## 2. Inputs
- `cols` must be a `list` or a `tuple`. A `str`, `set`, `None`, generator or dict raises TypeError (no special case for `str`).
- Every element must be a `str` (checked with `isinstance`). Ints, bools and None each raise TypeError.
- Duplicates are removed, keeping the order of first appearance. `["a", "a"]` behaves like `["a"]`.
- An empty `[]` or `()` is allowed and returns an unchanged copy.

## 3. Raises (checks run in this order)
1. **Not a list or tuple** → `TypeError(f"cols must be a list or tuple of column names, got {type(cols).__name__!r}")`
   - `f.drop_columns("a")` → `"cols must be a list or tuple of column names, got 'str'"`
   - `f.drop_columns({"a"})` → `... got 'set'`
   - `f.drop_columns(None)` → `... got 'NoneType'`
2. **A non-string element** → `TypeError("All column names in cols must be strings.")`
   - `f.drop_columns(["a", 1])`
   - This check runs before the existence check, so `["zz", 1]` also gives this TypeError.
3. **Unknown names** → `ValueError(f"Unknown column(s): {missing}. Available columns: {self.columns}")`
   - `missing` is a list of the deduplicated unknown names, in input order.
   - With columns `a, b, c`: `f.drop_columns(["z", "a", "z"])` → `"Unknown column(s): ['z']. Available columns: ['a', 'b', 'c']"`
   - This runs before the all-columns check, so `["a", "b", "c", "zz"]` gives this error and not error 4.
4. **Every column would be removed** → `ValueError("drop_columns cannot remove all columns from the frame")`
   - `f.drop_columns(["a", "b", "c"])`
   - Duplicates do not matter: `["a", "a"]` on a one-column frame `a` also raises this.

## 4. Returns and guarantees
- Always a **new** `ArFrame`, so `result is not frame`, even for an empty `cols`.
- `result.columns` equals `[c for c in frame.columns if c not in cols]`. The order of `cols` has no effect: dropping `["c", "a"]` from `a, b, c, d` gives `["b", "d"]`.
- The row count is unchanged.
- Values and nulls in the kept columns are carried over by the C++ `select_columns`. That code is not visible, but it is expected to preserve `dtypes`.
- `result._attrs` is a **`copy.deepcopy`** of `frame._attrs`, so it is equal to the original but a separate object, including nested dicts.

## 5. Side effects
- None. The input frame is not changed: its columns, data and `_attrs` stay the same.

## 6. Branches and edge cases
| Case | Input (frame columns `a, b, c`) | Result |
|---|---|---|
| Empty list or tuple | `[]` or `()` | Copy with columns `['a', 'b', 'c']`, `result == frame` is True, `result is not frame` |
| Normal drop | `["b"]` | `['a', 'c']` |
| Tuple input | `("a", "c")` | `['b']` |
| Duplicates | `["b", "b"]` | `['a', 'c']` |
| Drop all | `["a", "b", "c"]` | ValueError (error 4 above) |

- Avoid calling `[]` on a zero-column frame. That path goes into C++ `select_columns([])`, whose behaviour is not visible.

## 7. Rules from the docstring and comments
- "Duplicates are silently ignored."
- "An empty list returns a copy of the frame unchanged."
- "Original column order is preserved."
- "Dropping all columns is not supported."

## 8. Testing hints
- Build the frame:
  ```python
  f = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": ["x", None, "z"], "c": [1.5, 2.5, 3.5]}))
  ```
- Check columns with `r.columns`, rows with `len(r)` or `r.shape`, and values with `r["a"]` (returns a list) or `r.to_dict()`.
- Dtypes: compare `r.dtypes` with `{k: v for k, v in f.dtypes.items() if k in r.columns}`. Do not hard-code dtype strings for columns that contain None.
- Testing attrs: whether `from_pandas` sets attrs depends on the convert module, which is not shown. Use explicit attrs instead:
  ```python
  g = ar.ArFrame(f._frame, attrs={"meta": {"k": 1}})
  r = g.drop_columns(["a"])
  assert r._attrs == {"meta": {"k": 1}}
  assert r._attrs is not g._attrs
  assert r._attrs["meta"] is not g._attrs["meta"]
  ```
  - Mutate `r._attrs["meta"]["k"]` and assert that `g` is unchanged. This works for the empty-list branch too.
- Matching messages: they contain `(`, `)`, `[` and `'`. Use `pytest.raises(..., match=re.escape(full_msg))`, or compare `str(exc.value)` exactly.
- Non-mutation: after the call, assert that `f.columns == ['a', 'b', 'c']` and `f.shape == (3, 3)`.
- Confirm check order with inputs that break two rules at once, such as `["zz", 1]` (TypeError) and `["a", "b", "c", "zz"]` (Unknown, not "cannot remove all").

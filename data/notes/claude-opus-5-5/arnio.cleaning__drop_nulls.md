# Notes: `arnio.cleaning.drop_nulls(frame, *, subset=None)`

## 1. Purpose
- Removes every row that has a null in any checked column. With `subset=None` it checks all columns; otherwise it checks only the `subset` columns.
- The row filtering itself is done by the C++ core (`_drop_nulls`). Python only validates the arguments and wraps the result.

## 2. Inputs
- `frame`: must be an `ArFrame`.
  - It is checked by `_validate_arframe`, which lives in `arnio.frame` and is not shown here.
  - A non-ArFrame (for example a `pd.DataFrame`) presumably raises `TypeError`, but the exact message is not visible.
- `subset` (keyword-only), default `None`:
  - `None` means all columns are checked. It is passed to C++ as `subset=None`.
  - Otherwise it is normalised by `_validate_column_sequence(subset, argument_name="subset")`:
    - `str` or `bytes` is rejected.
    - The value must be a `collections.abc.Sequence` (list, tuple) or a `pd.Index`.
    - A `set`, `dict` or generator is rejected.
    - All items must be `str`.
    - It is converted with `list(subset)`.
  - Duplicate names in `subset` are **not** rejected.
  - Column-name matching is exact and case-sensitive. It uses `in frame.columns`.

## 3. Raises (checks happen in this order)
1. Frame check (non-ArFrame): exception type and message come from `arnio.frame`, so they are unverified. Use only `pytest.raises(TypeError)` if you test this at all.
2. `subset` is a string or bytes:
   - Raises `TypeError("subset must be a sequence of column names, not a string")`.
   - Trigger: `ar.drop_nulls(f, subset="a")`.
3. `subset` is not a Sequence or Index:
   - Raises `TypeError("subset must be a sequence of column names")`.
   - Trigger: `subset={"a"}`.
4. `subset` contains a non-str item:
   - Raises `TypeError("subset must contain only string column names")`.
   - Trigger: `subset=["a", 1]`.
   - This check runs before the missing-column check, so `subset=[1]` gives this TypeError, not a KeyError.
5. Empty `subset`:
   - Raises `ValueError("drop_nulls: subset cannot be empty; pass subset=None to check all columns")`.
   - Trigger: `subset=[]` or `subset=()`.
6. A column in `subset` does not exist:
   - Raises `KeyError(f"Missing columns for drop_nulls: {missing}. Available columns: {available}")`.
   - `missing` is a Python list repr of the missing names, in the order they appear in `subset`. Example: `['zz']`.
   - `available` is `", ".join(frame.columns)`, or `"<none>"` if the frame has no columns.
   - Example: for a frame with columns `a,b`, `subset=["zz"]` gives the message `Missing columns for drop_nulls: ['zz']. Available columns: a, b`.
   - Because `str(KeyError)` adds quotes, assert on `exc.value.args[0] == ...`, or use `match=re.escape("Missing columns for drop_nulls: ['zz']")`.
7. Errors raised inside the C++ `_drop_nulls` are unknown and not visible.

## 4. Returns and guarantees
- Always returns a **new** `ArFrame` built by `_wrap(result, frame)`. It is never the input object itself: `result is not frame`.
- Metadata: `result._attrs` is `copy.deepcopy(frame._attrs)`.
  - It is equal to the original but is not the same object. Nested objects are also not shared.
- Rows: kept rows are presumably in their original relative order. This is C++ behaviour, but it is the expected semantics.
- Columns and dtypes: the C++ layer is expected to keep the column set, column order and dtypes unchanged. This is not verifiable from the Python code. Asserting that `columns` is unchanged is reasonable.
- "null/empty": the docstring says empty values also count. Whether `""` counts as null is decided in C++ (and by how `from_pandas` stores `""`), so it is unverified. **Do not write tests that depend on `""` being dropped or being kept.**

## 5. Side effects
- No module or global state is touched and no files are written.
- The input frame should be unchanged. A test asserting that `frame.shape` and `to_pandas(frame)` are the same before and after the call is safe.

## 6. Branches and edge cases
- **`subset=None`.** Any null in any column drops the row.
  - Frame: `{"a": [1.0, None, 3.0], "b": ["x", "y", None]}`.
  - Result: 1 row, `a=1.0`, `b="x"`.
- **`subset=["a"]` on the same frame.** Only nulls in `a` count.
  - Rows 0 and 2 are kept: `a=[1.0, 3.0]`, `b=["x", None]`.
  - The null in `b` survives.
- **`subset=["b"]`.** Rows 0 and 1 are kept: `a=[1.0, nan]`, `b=["x", "y"]`.
- **Tuple subset** `("a",)`: gives the same result as `["a"]`.
- **Duplicate names** such as `["a", "a"]`: accepted; the result is presumably the same as `["a"]`.
- **No nulls at all:** the result has the same shape and values as the input, but it is a new object.
- **All rows contain a null:** the result has 0 rows and the same columns.
- **Zero-row input:** it goes straight to C++, and the result presumably has 0 rows with the same columns.

## 7. Rules from docstring and comments
- "A row is dropped if ANY column in the subset contains a null."
- `subset=None` means check all columns.
- An empty subset is an explicit error. Do not treat it as "check all columns".

## 8. Testing hints
- **Building a frame:**
  ```python
  f = ar.from_pandas(pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", None]}))
  ```
  - Use `float` columns with `None`/NaN and object columns with `None` to create nulls.
  - Avoid nullable int columns and `""`, because how they map to nulls depends on `from_pandas` and the C++ core.
- **Observing the result:**
  - `r.shape`, `r.columns`, `r.to_dict()` (orient `"list"`), or `ar.to_pandas(r)`.
  - When comparing a pandas result, call `.reset_index(drop=True)` on it and check values with `pd.isna(...)` rather than `== None`.
  - A float `a` comes back as `1.0`, not `1`.
- **Checking attrs:**
  ```python
  f._attrs = {"src": {"k": [1]}}
  r = ar.drop_nulls(f)
  assert r._attrs == f._attrs and r._attrs is not f._attrs and r._attrs["src"] is not f._attrs["src"]
  ```
- **Precise assertions:**
  - The subset path is easy to pin down: `subset=["a"]` must keep the row that is null only in `b`. A regression that ignores `subset` would drop it.
  - Also check `subset=[]` → ValueError and `subset="a"` → TypeError, each with its exact message.
- **Pitfall:** a set subset raises TypeError (item 3 above); do not expect it to work.

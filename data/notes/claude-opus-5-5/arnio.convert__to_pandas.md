# Notes: `arnio.convert.to_pandas(frame, *, copy=False)`

## 1. Purpose
- Converts an `ArFrame` (backed by C++) into a new `pandas.DataFrame`.
- Each column's arnio dtype becomes a fixed pandas dtype, and null masks become pandas missing values.
- If the frame has `attrs` metadata, a deep copy of it is placed on the result.

## 2. Inputs
- **`frame`** must be an `ArFrame` instance, checked with `isinstance`. Build one with `ar.from_pandas(df)`.
- **`copy`** is keyword-only and defaults to `False`.
  - It must be exactly a Python `bool`.
  - `isinstance(copy, bool)` rejects `1`, `0`, `None`, `"yes"` and `np.bool_(True)`.
  - `copy` only controls whether numpy buffers are copied. The output values and dtypes are the same either way.

## 3. Raises (in this order)
1. **`copy` check runs first:** `TypeError("copy must be a bool")`.
   - Example: `ar.to_pandas(frame, copy=1)`.
   - It also wins when both arguments are bad: `ar.to_pandas("x", copy=None)` gives this message.
2. **Frame check runs second:** `TypeError` with the template
   `f"to_pandas() expects an ArFrame, got {type(frame).__name__}. Use arnio.from_pandas() first."`
   - `ar.to_pandas(pd.DataFrame({"a":[1]}))` gives `"to_pandas() expects an ArFrame, got DataFrame. Use arnio.from_pandas() first."`
   - `ar.to_pandas(None)` gives `"... got NoneType. ..."`
   - `ar.to_pandas({"a":[1]})` gives `"... got dict. ..."`
- No other Python-level raises. The C++ accessors are not visible.

## 4. Returns and guarantees
- The result is always a new `pd.DataFrame`.
- **Columns**
  - Column order equals the C++ column index order. For `from_pandas` input this is presumably the original column order, since the dict is built in that order; the C++ side is not visible.
  - Column names are strings, because `from_pandas` applies `str(label)`. A label of `1` comes back as `"1"`.
- **Index:** default `RangeIndex(0..n-1)`.
- **Dtype mapping (exact):**

| arnio dtype | pandas dtype | null representation |
|---|---|---|
| INT64 | `Int64` (nullable `pd.Int64Dtype()`) | `pd.NA` |
| FLOAT64 | numpy `float64` | `np.nan` |
| BOOL | `boolean` (`pd.BooleanDtype()`) | `pd.NA` |
| anything else (string) | `pd.StringDtype()`, so `str(dtype) == "string"` | `pd.NA` |

- **Round-trip pitfall:** a numpy `int64` column comes back as `Int64`, and a numpy `bool` column comes back as `boolean`. Only `float64` stays a numpy dtype.
- **attrs**
  - If `frame._attrs` is truthy, `result.attrs == original attrs` as a deep copy (equal, not the same object).
  - If it is empty, `result.attrs == {}`.

## 5. Side effects
- The frame is not mutated, and no global state is touched.
- Because attrs are deep-copied:
  - mutating `result.attrs["k"]["nested"]` does not affect a later `to_pandas(frame).attrs`;
  - mutating `df.attrs` after `from_pandas` does not affect the frame, since `from_pandas` also deep-copies.

## 6. Branches and edge cases
- **Zero columns** returns `pd.DataFrame(index=pd.RangeIndex(num_rows))`.
  - `ar.to_pandas(ar.from_pandas(pd.DataFrame()))` has `shape == (0, 0)` and an index that is a `RangeIndex`.
  - For `pd.DataFrame(index=range(3))` the shape is likely `(3, 0)`, but this depends on C++ keeping the row count. Prefer the `(0, 0)` case.
- **Zero rows with columns:** dtypes still follow the table above.
  - Example: `pd.DataFrame({"a": pd.Series([], dtype="float64")})` comes back as a `float64` column of length 0. This assumes C++ honours the dtype hint for empty input, which is likely because a hint is passed.
- **FLOAT64 nulls:** when any null exists, the array is copied and NaN written at the masked positions.
- **INT64, BOOL and STRING nulls:** `series[mask] = pd.NA`.
- **`copy=True`:** the int, float and bool arrays are copied. String columns are always built from Python lists.

## 7. Rules from the docstring
- `copy=False` keeps the zero-copy path "where supported", though some columns still copy.
- `copy=True` returns pandas-owned defensive copies.
- `attrs` from a `from_pandas()` source frame are restored on the result.

## 8. Testing hints
- **Build inputs with explicit pandas dtypes** so the arnio dtype is certain. These dtypes send a hint to C++:
  - `pd.Series([1, None, 3], dtype="Int64")`
  - `pd.Series([1.5, np.nan], dtype="float64")`
  - `pd.Series([True, None], dtype="boolean")`
  - `np.array([1, 2], dtype="int64")`
  - `[True, False]` (numpy `bool`)
- **String columns:** use `["a", None, "b"]` (object dtype). C++ infers string here since no hint is given. Avoid object columns of plain ints or all-`None` columns, because their inferred type is decided in C++ and not visible.
- **Check dtypes with:**
  - `str(df["i"].dtype) == "Int64"`
  - `df["f"].dtype == np.float64`
  - `str(df["b"].dtype) == "boolean"`
  - `isinstance(df["s"].dtype, pd.StringDtype)`
- **Check nulls with:**
  - `df["i"].isna().tolist() == [False, True, False]`
  - `df["i"][1] is pd.NA`
  - `np.isnan(df["f"][1])` for floats
  - Never use `== None` or `== pd.NA`.
- **Values:**
  - `df["i"].tolist()` gives `[1, <NA>, 3]`; compare non-null entries.
  - `df["f"].iloc[0] == 1.5`; compare floats with `==` only for exactly representable values.
- **Full-frame comparison:** build the expected frame with matching dtypes, e.g. `pd.testing.assert_frame_equal(out, pd.DataFrame({"i": pd.array([1,None], dtype="Int64")}))`.
- **Mixed columns are coerced by `from_pandas`, not by `to_pandas`:**
  - an object column with ints and floats becomes float64;
  - an object column with str and int becomes all strings, e.g. `["a", 1]` comes back as `"a"`, `"1"`.
- **attrs test:** set `df.attrs = {"src": {"k": 1}}`, then check:
  - `ar.to_pandas(ar.from_pandas(df)).attrs == {"src": {"k": 1}}`
  - `out.attrs is not df.attrs`
- **Equivalence of `copy`:** compare `to_pandas(frame, copy=True)` against `to_pandas(frame)` with `assert_frame_equal`; they must be equal.
- **Do not assert:**
  - memory sharing or writability under `copy=False`;
  - that mutating the result leaves the frame unchanged under `copy=False`. A float column with no nulls may alias the C++ buffer.
- **Safe mutation test:** under `copy=True`, mutate `out["f"].iloc[0] = 99.0`, then check that `ar.to_pandas(frame)["f"].iloc[0]` still holds the original value.

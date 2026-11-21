# Notes: `arnio.cleaning.normalize_whitespace(frame, columns=None)`

## 1. Purpose
- In string columns, replaces each run of whitespace with a single space `" "`, then strips leading and trailing whitespace.
- Accepts an ArFrame or a pandas DataFrame and returns the same kind of object.

## 2. Inputs
- `frame`: an `ArFrame` or a `pd.DataFrame`.
  - It is checked by `_validate_frame(frame, allow_pandas=True)`.
  - ArFrame input: `df = to_pandas(frame)`.
  - DataFrame input: `df = frame.copy(deep=False)`.
- `columns`: positional or keyword, default `None`. It is **not** keyword-only.
  - `None`: process every column picked by `df.select_dtypes(include=["object", "string"])`.
  - Otherwise it is validated by `_validate_existing_column_sequence(argument_name="columns", missing_error=ValueError)`:
    - A list, tuple or `pd.Index` is accepted.
    - Every item must be a `str`.
    - An empty list is allowed and processes nothing.
    - Duplicates are allowed (the operation is idempotent).
  - After validation the names are **filtered**: only columns with `df[c].dtype in ("object", "string")` are kept. Requested numeric, bool, category or datetime columns are skipped silently, with no error.

## 3. Raises (checks run in this order)
1. Bad frame type: `TypeError("frame must be an ArFrame or a pandas DataFrame")`.
   - Triggers: `normalize_whitespace({"a": [1]})` or `normalize_whitespace(None)`.
2. `columns` is a `str` or `bytes`: `TypeError("columns must be a sequence of column names, not a string")`.
   - Trigger: `normalize_whitespace(fr, "a")` or `columns="a"`.
3. `columns` is not a Sequence and not a `pd.Index`: `TypeError("columns must be a sequence of column names")`.
   - Triggers: `columns={"a"}` (a set), or a generator.
4. `columns` contains a non-str item: `TypeError("columns must contain only string column names")`.
   - Trigger: `columns=["a", 1]`.
5. Missing column: `ValueError(f"Missing columns for normalize_whitespace: {missing}. Available columns: {available}")`.
   - `missing` is a list repr in request order, e.g. `['zz']`.
   - `available` is `", ".join(df.columns)`, or `"<none>"` if the frame has no columns.
   - Example: a frame with columns `a,b` and `columns=["zz"]` gives exactly
     `"Missing columns for normalize_whitespace: ['zz']. Available columns: a, b"`.
   - Note this is a ValueError, **not** a KeyError.
- The frame-type check comes before any `columns` check.

## 4. Returns and guarantees
- ArFrame in gives `from_pandas(df)`: a new ArFrame.
  - Column order and the names of untouched columns are kept.
  - `_wrap` is not used, so copying of attrs is not guaranteed. Whether they survive depends on `convert` (not visible here), so do not test it.
- DataFrame in gives a new DataFrame object (`result is not df_in`).
  - This holds even when nothing changes.
  - Index, column order and dtypes of untouched columns are kept.
- Per-value transform (`_fix_whitespace`):
  - A `str` value becomes `re.sub(r"\s+", " ", v).strip()`.
  - Any non-str value (None, NaN, `pd.NA`, int, float, bool) is returned unchanged.
- Examples:
  - `"  a \t\n b  "` → `"a b"`
  - `"x    y"` → `"x y"`
  - `"   "` → `""` (an empty string, **not** null)
  - `""` → `""`
  - `"ok"` → `"ok"`
  - `"a\u00a0\u00a0b"` → `"a b"` (Unicode `\s` matches the non-breaking space)
- Nulls stay null and are never created.

## 5. Side effects
- The input DataFrame is not mutated: column assignment goes to a shallow copy.
- The input ArFrame is not mutated.
- No global state is touched and no files are written.

## 6. Branches and edge cases
- `columns=None` with a mix of string and int columns: only string columns change; ints are identical.
- `columns=["num"]` where `num` is int64: no error, the column is unchanged.
- `columns=[]`: nothing is processed, the data equals the input, still a new object.
- Pandas object column with mixed values `[" a  b ", 5, None]` → `["a b", 5, None]`. The int passes through.
- Pandas `"string"` dtype column (`pd.Series([...], dtype="string")`) is processed; `pd.NA` is kept.
- Zero-row frame: no error, columns are kept.
- Internal newlines and tabs become a single space; they are not removed: `"a\n\nb"` → `"a b"`.

## 7. Rules from comments and docstring
- "Only process actual str values; pass through int, bool, float, None, etc."
- When `columns` is omitted the default is all string (object) columns.
- Leading and trailing whitespace is also stripped.

## 8. Testing hints
- Build an ArFrame with `fr = ar.from_pandas(pd.DataFrame({"s": ["  a   b ", "c\td"], "n": [1, 2]}))`.
- Inspect results with `ar.to_pandas(out)["s"].tolist()`, which should equal `["a b", "c d"]`.
- Check the `"n"` column with `.tolist() == [1, 2]` and `out.dtypes["n"] == "int64"`.
- The pandas path is the easiest place to pin pass-through of non-str values and to check no mutation:
  - Use `df = pd.DataFrame({"s": [" x  y ", 3, None]}, dtype=object)`.
  - Then check `df["s"].tolist() == [" x  y ", 3, None]` after the call.
  - Check `isinstance(result, pd.DataFrame)` and `result is not df`.
- When the input is an ArFrame, assert `isinstance(out, ar.ArFrame)`.
- Assert that whitespace-only strings become `""`, not null: `out_list[i] == ""`.
- To check nulls in ArFrame output, use `pd.isna(x)`. Whether a null comes back as None or NaN depends on `to_pandas`, so do not compare with `is None`.
- Use `pytest.raises(ValueError, match=...)` for missing columns.
  - Escape the regex or match a prefix, e.g. `match=r"Missing columns for normalize_whitespace: \['zz'\]"`.
  - Do not expect a KeyError.
- Pitfalls to avoid:
  - Do not expect collapse in non-string columns, or an error for them.
  - Do not expect `"   "` to become null.
  - Do not use a set for `columns`: it raises TypeError.

# Notes: `arnio.cleaning.normalize_case(frame, *, subset=None, case_type="lower")`

## 1. Purpose
- Returns a new ArFrame in which the ASCII letters of string columns are converted to lower, upper or title case.
- The case change itself runs in the C++ core (`_core._normalize_case`). The Python side only validates arguments and wraps the result.

## 2. Inputs
- `frame`: must be an `ArFrame`. It is checked by `_validate_arframe` from `arnio.frame`. That helper is not visible here; it presumably raises `TypeError` for a non-ArFrame, but the exact message is unknown.
- `case_type`: a `str`, default `"lower"`. It must be exactly one of `"lower"`, `"upper"`, `"title"`.
  - The check is case-sensitive, so `"Lower"` is rejected.
  - Whitespace is not stripped, so `" lower"` is rejected.
- `subset`: keyword-only, default `None`.
  - `None` means all string columns (per the docstring; the column choice is made in C++).
  - Otherwise it goes through `_validate_existing_column_sequence`:
    - A `str` or `bytes` value is rejected.
    - The value must be a `collections.abc.Sequence` (list or tuple) or a `pd.Index`. A `set`, a generator or a dict view is rejected.
    - Every item must be a `str`.
    - Every name must exist in `frame.columns`.
    - It is converted to a `list` before being passed to C++.
  - An empty list `[]` is allowed (no "cannot be empty" check). It is passed to C++ as `[]`; what happens then is decided in C++ and is not visible.
  - Duplicate names are not rejected.

## 3. Raises (in this order)
1. Frame validation from `_validate_arframe`. Message unknown; for example `ar.normalize_case(pd.DataFrame())` probably raises `TypeError`.
2. `TypeError("case_type must be a string")`. Example: `ar.normalize_case(f, case_type=1)` or `case_type=None`.
3. `ValueError(f"case_type must be one of {valid_cases}, got {case_type!r}")`, where `valid_cases` is the set `{"lower","upper","title"}`.
   - Example: `case_type="snake"` gives `... got 'snake'`.
   - The order of the set inside the message varies between runs (hash randomisation). Do not assert the full message. Use `match="case_type must be one of"` and `match="got 'snake'"`.
4. Subset errors (only checked after `case_type` is valid):
   - `subset="name"` raises `TypeError("subset must be a sequence of column names, not a string")`.
   - `subset={"name"}` (a set) raises `TypeError("subset must be a sequence of column names")`.
   - `subset=["name", 3]` raises `TypeError("subset must contain only string column names")`.
   - `subset=["zzz"]` raises `KeyError`. The message is `"Missing columns for normalize_case: ['zzz']. Available columns: name, n"`.
     - Available columns are joined with `", "`, or the string `"<none>"` if the frame has no columns.
     - This is a `KeyError`, not a `ValueError`.
     - `str(exc)` adds surrounding quotes. Use `match=re.escape("Missing columns for normalize_case: ['zzz']")`.
- Precedence: a bad `case_type` wins over a bad subset. `normalize_case(f, subset=["zzz"], case_type="bad")` raises `ValueError`, not `KeyError`.
- Raise-tests do not depend on C++ behaviour, so they are the most reliable kind.

## 4. Returns and guarantees
- Returns a new `ArFrame` created by `_wrap(result, frame)`. It is never the input object (`result is not frame`).
- `attrs` (`frame._attrs`) is deep-copied:
  - The result's `_attrs` equals the source's `_attrs`.
  - It is a different object (`is not`), so mutating it does not affect the source.
- The following come from C++ and cannot be confirmed from this source:
  - column order and dtypes (expected to be unchanged);
  - preservation of nulls (expected);
  - leaving non-string columns untouched (expected).
- Docstring rule: only ASCII letters change, and non-ASCII UTF-8 is preserved byte-for-byte. For example, `"ÉCOLE"` with `"lower"` should give `"École"`, not `"école"` as Python's `str.lower` would.

## 5. Side effects
- The input frame is not mutated, because C++ returns a new frame.
- No global state is changed and no files are written.

## 6. Branches / edge cases
- With `subset=None`, C++ processes all string columns.
- With `subset=[...]`, only the listed columns should change. Other string columns should keep their original case.
- Things decided in C++ that should not be asserted strictly:
  - a non-string column inside `subset` (for example an int column);
  - `subset=[]`;
  - a zero-row frame (expected to keep the columns with 0 rows).
- Title-case semantics are defined in C++. Only assert simple words, e.g. `"hello world"` → `"Hello World"` and `"hELLO"` → `"Hello"`. Avoid apostrophes, digits and non-ASCII characters when testing title case.

## 7. Rules from docstring/comments
- "Normalize ASCII letters … Non-ASCII UTF-8 bytes are preserved unchanged" (accents, CJK, emoji).
- The docstring lists the raised exceptions as: `TypeError` for a non-str `case_type`, `ValueError` for an unsupported option, and `KeyError` for a missing subset column.

## 8. Testing hints
```python
df = pd.DataFrame({"name": ["Alice", "BOB", None], "city": ["New York", "PARIS", "rome"], "n": [1, 2, 3]})
f = ar.from_pandas(df)
out = ar.to_pandas(ar.normalize_case(f, subset=["name"], case_type="upper"))
assert out["name"].tolist()[:2] == ["ALICE", "BOB"] and pd.isna(out["name"][2])
assert out["city"].tolist() == ["New York", "PARIS", "rome"]   # untouched
assert out["n"].tolist() == [1, 2, 3]
```
- Check nulls with `pd.isna(...)`. The null representation after `to_pandas` (None vs NaN) depends on C++.
- Check that the default really is `"lower"`: `["Alice", "BOB"]` should become `["alice", "bob"]` when `case_type` is omitted.
- Run all three modes on the same input so that swapped modes are detected (lower, upper and title outputs all differ).
- A tuple subset is accepted: `subset=("name",)` works the same as `["name"]`.
- The input is unchanged: `ar.to_pandas(f)` is still the original data after the call.
- To test attrs: set `f._attrs = {"k": [1]}`. Check that `res._attrs == {"k": [1]}` and that `res._attrs is not f._attrs`.
- Pitfalls:
  - Do not compare the full `case_type` error message (set ordering varies).
  - Do not expect a `ValueError` for a missing subset column; it is a `KeyError`.
  - Do not expect Python Unicode casing on non-ASCII text.

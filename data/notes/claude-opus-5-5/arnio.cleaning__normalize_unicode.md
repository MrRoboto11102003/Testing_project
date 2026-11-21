# Notes: `arnio.cleaning.normalize_unicode(frame, *, subset=None, form="NFC")`

## 1. Purpose
- Applies `unicodedata.normalize(form, value)` to every non-null value in STRING columns.
- Works directly on the C++ columns and does not convert to pandas.
- Every other column is copied unchanged.
- Returns a new `ArFrame`.

## 2. Inputs
- `frame`: must be an `ArFrame`. It is checked first by `_validate_arframe`, which lives in `arnio.frame` and is not shown. It presumably raises `TypeError` for non-ArFrames, but the message is unknown, so only assert the type, e.g. `pytest.raises(TypeError)`.
- `form`: must be a `str` and exactly one of `{"NFC", "NFD", "NFKC", "NFKD"}`.
  - The check is case-sensitive, so `"nfc"` is rejected.
  - The default is `"NFC"`.
- `subset`: either `None` (meaning all STRING columns) or a sequence of column names.
  - Validated by `_validate_existing_column_sequence`.
  - It must be a `Sequence` (list or tuple) or a `pd.Index`. It must not be a str or bytes. Every item must be a `str`. Every name must exist in `frame.columns`.
  - It is converted to a `set`, so duplicates and ordering do not matter.
  - An empty list `[]` is allowed: there is no empty check.
  - Columns in `subset` that are not STRING dtype are silently skipped (no error).

## 3. Raises (checks run in this order)
1. Non-ArFrame `frame`: `TypeError` from `_validate_arframe`; the message is not visible here.
2. Non-str `form`: `TypeError("form must be a string")`.
   - Trigger: `ar.normalize_unicode(f, form=1)` or `form=None`.
3. Bad `form` value: `ValueError(f"Unsupported Unicode normalization form: {form}")`.
   - Trigger: `form="nfc"` gives `"Unsupported Unicode normalization form: nfc"`.
   - `form="NFX"` and `form=""` also raise.
4. `subset` is a str or bytes: `TypeError("subset must be a sequence of column names, not a string")`.
   - Trigger: `subset="s"`.
5. `subset` is not a Sequence and not a `pd.Index`: `TypeError("subset must be a sequence of column names")`.
   - Trigger: `subset={"s"}` (a set), or `subset=5`.
6. A non-str item in `subset`: `TypeError("subset must contain only string column names")`.
   - Trigger: `subset=["s", 1]`.
7. A missing column: `KeyError(f"Missing columns for normalize_unicode: {missing}. Available columns: {available}")`.
   - `missing` is a list repr, e.g. `['zz']`.
   - `available` is `", ".join(frame.columns)`, e.g. `s, n`.
   - Trigger: `subset=["zz"]` on columns `["s", "n"]` gives `"Missing columns for normalize_unicode: ['zz']. Available columns: s, n"`.
   - `str(KeyError)` adds surrounding quotes, so use `match="Missing columns for normalize_unicode"` or `re.escape(...)` rather than an exact `==`.
- Because `form` is checked before `subset`, `form="bad", subset=["zz"]` raises `ValueError`, not `KeyError`.

## 4. Returns and guarantees
- The result is always a new `ArFrame`, never the input object, even when nothing changes.
- It is built with `_Frame.from_dict(new_columns, dtype_hints, frame.shape[0])`:
  - The column order matches the input order.
  - Every column keeps its original dtype. Target columns stay `"string"`.
  - The row count is the same as the input.
- Null cells (`None` from `to_python_list`) stay null.
- Only Unicode normalization happens:
  - Whitespace is not stripped.
  - Case is not changed.
  - Values that are already normalized are unchanged.
- `attrs`: `_wrap` sets `result._attrs = copy.deepcopy(frame._attrs)`. The result is equal to the source attrs but is a different object.

## 5. Side effects
- None. The input frame is not mutated, there is no global state, and no files are written.

## 6. Branches and edge cases
- **`subset=None`**: every column whose dtype is STRING is normalized. int, float and bool columns are copied as-is.
- **`subset=["a"]`**: only `a` is normalized. Other string columns keep un-normalized text.
  - Example: `b=["e\u0301"]` stays a 2-character string.
- **`subset=["n"]` where `n` is int64**: no error, and nothing changes.
- **`subset=[]`**: no column is targeted. The result is a new frame equal to the input.
- **Zero-row frame**: the result has 0 rows with the same columns and dtypes.
- **Value behaviour per form**:
  - NFC: `"e\u0301"` → `"\u00e9"` (length 2 → 1). `"\ufb01"` (the ﬁ ligature) is unchanged.
  - NFD: `"\u00e9"` → `"e\u0301"` (length 1 → 2).
  - NFKC: `"\ufb01"` → `"fi"`; `"\uff21"` (fullwidth A) → `"A"`; `"x\u00b2"` → `"x2"`; `"\u2460"` → `"1"`.
  - NFKD: `"\u00e9"` → `"e\u0301"`; `"\ufb01"` → `"fi"`.
  - Plain ASCII such as `"abc"` is unchanged under every form.
- **Zero-column frames**: avoid them. It depends on the C++ `from_dict({}, {}, n)`, which is not visible here.

## 7. Rules from the docstring
- "Only STRING columns are processed; all other column types are cloned unchanged."
- The function avoids a full pandas roundtrip.
- `subset=None` means all string columns.

## 8. Testing hints
```python
import pandas as pd, arnio as ar
df = pd.DataFrame({"s": ["e\u0301", None, "\ufb01"], "t": ["e\u0301", "x", "y"], "n": [1, 2, 3]})
f = ar.from_pandas(df)
r = ar.normalize_unicode(f, form="NFKC", subset=["s"])
out = ar.to_pandas(r)
assert out["s"][0] == "\u00e9" and out["s"][2] == "fi" and pd.isna(out["s"][1])
assert out["t"][0] == "e\u0301"          # not in subset -> untouched
assert r.columns == ["s", "t", "n"] and r.dtypes == f.dtypes
assert r is not f
```
- Compare strings using explicit escapes and `len()`. Composed and decomposed forms look identical when printed.
- Check nulls with `pd.isna(...)`. Do not assert whether a null comes back as `None` or `NaN`, because that depends on `to_pandas`.
- To check attrs, set `f._attrs = {"k": [1]}` and then assert both:
  - `r._attrs == {"k": [1]}`
  - `r._attrs is not f._attrs`
- Do not assume the default form is NFKC. The default NFC leaves `"\ufb01"` unchanged. That makes it a good test that the default is really `"NFC"`.
- Do not expect an error for a non-string column in `subset`, or for an empty `subset`.

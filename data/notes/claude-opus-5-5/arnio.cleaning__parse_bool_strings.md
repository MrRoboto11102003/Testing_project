# Notes: `arnio.cleaning.parse_bool_strings(frame, *, subset=None, true_values=None, false_values=None)`

## 1. Purpose
- Converts boolean-like strings to real Python `True`/`False`. Matching is case-insensitive and ignores surrounding whitespace.
- Works by converting the frame to pandas, editing it, and calling `from_pandas`.
- Values that do not match are left exactly as they were, including their whitespace and any nulls.

## 2. Inputs
- `frame`: must be an `ArFrame`. It is checked by `_validate_arframe`, which lives in `arnio.frame` and is not shown here. It raises `TypeError` for non-ArFrames, but the exact message is not visible here.
- `true_values`
  - Accepted: `None`, or a `set`, `list` or `tuple` of `str`.
  - Default: `{"true", "yes", "y", "1"}`.
  - A custom collection **replaces** the defaults. It is not merged with them.
  - Each value is normalised with `v.strip().lower()`, so `{" SI "}` matches `"si"` and `"Si "`.
- `false_values`
  - Same rules as `true_values`.
  - Default: `{"false", "no", "n", "0"}`.
  - The two collections are independent. A custom `true_values` still uses the default `false_values`.
- Empty collections such as `set()` or `[]` are allowed. Then nothing maps to that boolean.
- `frozenset` is **rejected**, because it is not a `set`/`list`/`tuple` instance.
- `subset`
  - `None` means every column whose pandas dtype is `object` or `string`.
  - Otherwise it must be a non-empty sequence (list/tuple) of existing column names.
  - Any listed column whose pandas dtype is bool is silently skipped.
  - Numeric columns may be listed. Their values are left unchanged, because only `str` values are converted.

## 3. Raises (in this order)
1. Bad frame: `_validate_arframe` raises `TypeError`. The message is defined in `frame.py`, so do not assert on it.
2. `true_values="yes"` (a str or bytes) raises `TypeError("true_values must be a set/list/tuple of strings, not a bare string")`.
3. `false_values="no"` raises `TypeError("false_values must be a set/list/tuple of strings, not a bare string")`.
4. `true_values={"a": 1}`, `frozenset({"a"})` or `5` raises `TypeError("true_values must be a set, list, or tuple of strings")`.
5. The same for `false_values`: `TypeError("false_values must be a set, list, or tuple of strings")`.
6. A non-string element, e.g. `true_values=["yes", 1]`, raises `TypeError(f"true_values must contain only strings, got {type(invalid[0]).__name__}")`, i.e. `"true_values must contain only strings, got int"`.
   - The type name is not quoted.
   - Use a list so that "first invalid" is deterministic.
7. The same for `false_values`: `"false_values must contain only strings, got int"`.
   - The `true_values` element check runs before the `false_values` element check.
   - Both bare-string checks run before both container checks.
8. Overlap after normalisation raises `ValueError(f"true_values and false_values overlap after normalization: {overlap}")`, where `overlap` is a Python set.
   - `true_values={"Yes"}, false_values={" yes"}` gives the message `"... normalization: {'yes'}"`.
   - Defaults are included in the check: `true_values={"no"}` (with false defaults) gives `{'no'}`.
9. `subset` errors are checked only after all of the above:
   - `subset="a"` raises `TypeError("subset must be a sequence of column names, not a string")`.
   - `subset={"a"}` (a set is not a Sequence) raises `TypeError("subset must be a sequence of column names")`.
   - `subset=["a", 1]` raises `TypeError("subset must contain only string column names")`.
   - `subset=[]` raises `ValueError("subset cannot be empty")`.
   - `subset=["zz"]` raises `ValueError("Columns not found in frame: ['zz']")`. This is a **ValueError, not KeyError**, and the message has no "Available columns" part.

## 4. Returns and guarantees
- Returns a **new** `ArFrame` built with `from_pandas(df)`.
- Column names and their order are unchanged. The row count is unchanged.
- Per-cell rule, applied to each targeted column:
  - The cell is a `str` and `x.strip().lower()` is in `true_values`: it becomes `True`.
  - Otherwise, if that normalised string is in `false_values`: it becomes `False`.
  - Otherwise the cell is unchanged. This covers `None`/NaN, non-matching strings (original spacing and case kept) and non-str values.
- Result dtype: a column whose cells are all parsed to booleans should come back as bool. Dtype inference happens in `from_pandas`, which is not shown here.
- Mixed columns (booleans plus unparsed strings) "may round-trip as strings", according to the docstring. Their dtype and value representation are not pinned down, so do not assert on them.
- Attrs: the source `frame._attrs` is **not** explicitly copied (no `_wrap`). Do not rely on attrs being preserved.

## 5. Side effects
- None. The input frame is not mutated: the work happens on a shallow pandas copy whose columns are reassigned.
- No global state is changed.

## 6. Branches and edge cases
- Defaults: `{"c": ["Yes", " TRUE ", "y", "1", "no", "FALSE", "N", "0"]}` becomes `[True, True, True, True, False, False, False, False]`.
- Unsupported tokens: `"t"`, `"on"`, `"off"`, `"2"` and `"yes please"` are left unchanged with defaults.
- Custom `true_values={"si"}`: `"yes"` is no longer converted, but `"no"` still becomes `False` through the default false set.
- `subset=["a"]`: column `b` is untouched even if it holds `"yes"`.
- A bool column named in `subset` is skipped and causes no error.
- `subset=None` on a frame with only int/float columns: values are unchanged.
- A zero-row frame returns a zero-row frame with the same columns.
- Strings are checked against `true_values` first. Overlap is impossible anyway, because it raises.

## 7. Rules from comments and docstring
- "Unsupported values are preserved unchanged."
- Mixed bool and string columns may round-trip as strings, because of ArFrame column typing.

## 8. Testing hints
- Build inputs with `ar.from_pandas(pd.DataFrame({"c": ["yes", "no"]}))`.
  - Use explicit strings such as `"1"` and `"0"`.
  - `read_csv` would infer those as int64, and then nothing would be parsed in the default `subset=None` path.
- Observe results with `ar.to_pandas(res)["c"].tolist() == [True, False]`.
  - Compare with `==`, not `is`, because the values may be numpy bools.
  - Use fully parseable columns when asserting values or `res.dtypes["c"] == "bool"`.
- Check that untouched columns are unchanged: `ar.to_pandas(res)["b"].tolist() == ["yes"]`.
- For non-mutation, compare `ar.to_pandas(frame)` before and after the call.
- Use `pytest.raises(ValueError, match=...)` with `re.escape` for messages containing `[`, `{`, `(` or `/`.
- The overlap message embeds a set. Use a single overlapping value so the text is deterministic, e.g. `"{'yes'}"`.
- Do not assert the exact `TypeError` message for a non-ArFrame `frame` argument; it is defined outside this module.

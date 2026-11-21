# Notes: `arnio.cleaning.standardize_missing_tokens(frame, tokens=None, subset=None)`

## 1. Purpose
- Replaces string cells that look like missing-value tokens with `float("nan")`.
- Matching ignores case and surrounding whitespace.
- Accepts an ArFrame or a pandas DataFrame and returns the same kind of object.

## 2. Inputs
- **`frame`**: an `ArFrame` or a `pd.DataFrame`. Anything else raises TypeError.
  - An ArFrame is converted with `to_pandas(frame)`. A DataFrame is used directly.
  - In both cases the function works on `df.copy(deep=False)`.
- **`tokens`**: `None` or a `list` of `str`. A tuple or set is rejected.
  - `None` uses the defaults: `["N/A","NA","n/a","na","nan","-","none","nil","null","","?"]`.
  - A custom list replaces the defaults completely. It is not added to them.
  - Every token is normalised with `.strip().lower()`. The effective default set is `{"n/a","na","nan","-","none","nil","null","","?"}`.
  - `tokens=[]` means nothing is ever replaced.
- **`subset`**: `None` (all columns) or a sequence of column names.
  - A `list`, `tuple` or `pd.Index` is accepted.
  - An empty `[]` is allowed and replaces nothing.
  - Duplicate names are allowed.
  - Validation is done by `_validate_existing_column_sequence` with `missing_error=ValueError`.

## 3. Raises (checks run in this order)
1. Bad frame type: `TypeError("frame must be an ArFrame or a pandas DataFrame")`.
   - Trigger: `standardize_missing_tokens([1,2])`.
2. `subset` is a `str`:
   - Template: `TypeError(f"subset must be a list of column names, not a string. Did you mean subset=['{subset}']?")`.
   - `subset="a"` gives `"...Did you mean subset=['a']?"`.
3. `tokens` is a `str`:
   - Template: `TypeError(f"tokens must be a list of strings, not a bare string. Did you mean tokens=['{tokens}']?")`.
4. `tokens` is not a list:
   - Template: `TypeError(f"tokens must be a list of strings, got {type(tokens).__name__!r}")`.
   - `tokens=("x",)` gives `"tokens must be a list of strings, got 'tuple'"`.
5. A token item is not a `str`:
   - Template: `TypeError(f"tokens must be a list of strings, got {type(item).__name__!r} item")`.
   - `tokens=["x", 1]` gives `"... got 'int' item"`.
6. Only when `subset` is not None:
   - `subset` is bytes: `TypeError("subset must be a sequence of column names, not a string")`.
   - `subset` is a set or other non-Sequence: `TypeError("subset must be a sequence of column names")`.
   - A non-str name, e.g. `subset=[1]`: `TypeError("subset must contain only string column names")`.
   - A missing column: `ValueError(f"Unknown columns in subset: {missing}")`, where `missing` is a list repr.
   - `subset=["zz"]` gives `"Unknown columns in subset: ['zz']"`.
- Ordering pitfalls:
  - The `subset`-is-str check runs before any tokens check.
  - The column-existence check for `subset` runs last, after tokens validation.

## 4. Returns and guarantees
- **ArFrame in, ArFrame out.** The result is a new object built with `from_pandas(df)`.
  - Attrs are not explicitly copied, so do not assert on attrs.
- **DataFrame in, DataFrame out.** The result is the shallow copy, so it is not the input object (`result is not df`).
  - The index and column order are preserved.
- A cell is replaced only if `isinstance(value, str)` and `value.strip().lower()` is in the token set.
  - Replaced cells become `float("nan")`.
- All other cells are left unchanged, including ints, floats, bools, `None`, `pd.NA` and non-matching strings.
  - Non-matching strings keep their whitespace: `" kept "` stays `" kept "`.
- Whitespace-only strings (`"   "`, `"\t"`) normalise to `""`, which is a default token, so they become NaN.
- These values match by default:
  - `"NULL"`, `" None "`, `"\tNaN\t"`, `"N/A"`, `"NIL"`, `"-"`, `"?"`, `""`.
- These values do not match by default:
  - `"missing"`, `"#N/A"`, `"N.A."`, `"undefined"`, `"--"`, `"0"`.
- The dtype that ArFrame infers after the round-trip depends on `from_pandas` (convert module / C++), which this code does not show.

## 5. Side effects
- The input is not mutated.
  - For pandas input, `df[col] = ...map(...)` assigns a new Series on the shallow copy only.
  - The caller's DataFrame keeps its original values.
- There is no global state and no file I/O.

## 6. Branches and edge cases
- **`subset=None`**: every column is mapped, including numeric ones. Non-str values pass through.
- **`subset=["a"]`**: only column `a` changes. Tokens in other columns stay as strings.
  - Example: `{"a":["NA"],"b":["NA"]}` gives `a=[NaN]`, `b=["NA"]`.
- **`subset=[]`**: no columns are processed, so the output equals the input.
- **Custom tokens**: `tokens=["  MISSING "]` makes `"missing"` and `"Missing"` become NaN.
  - With custom tokens, `"N/A"` is no longer replaced.
- **`tokens=[]`**: nothing is replaced, not even `""`.
- **Zero rows**: works and returns an empty frame with the same columns.

## 7. Rules from the docstring
- "Matching is case-insensitive and trims surrounding whitespace before checking token membership."
- "Values that do not match a missing token preserve their original whitespace."
- The default token list is used when `tokens` is None.

## 8. Testing hints
- The pandas path is the most deterministic. Use object-dtype columns:
  ```python
  df = pd.DataFrame({"v": [" NULL ", "\tNaN\t", "kept ", "x"]})
  out = ar.standardize_missing_tokens(df)
  assert pd.isna(out["v"][0]) and pd.isna(out["v"][1])
  assert out["v"][2] == "kept " and out["v"][3] == "x"
  assert df["v"][0] == " NULL "   # input not mutated
  ```
- ArFrame path:
  - Build with `ar.from_pandas(pd.DataFrame({...}))`.
  - Check `isinstance(result, ar.ArFrame)`.
  - Inspect with `ar.to_pandas(result)` and use `pd.isna(...)`. Nulls may come back as `None` or `NaN`, so never compare with `==`.
  - Keep at least one real string in the column so it stays a string column.
- Pitfalls:
  - Do not assert the ArFrame dtype of mixed columns such as `[1, 2, "N/A"]`, or of columns that become all-null. That inference is in convert/C++ and is not shown here.
  - Avoid pandas `"string"` (StringDtype) inputs, because `.map` returning NaN has dtype-specific behaviour. Use object dtype instead.
  - For error messages, use `pytest.raises(..., match=re.escape(msg))`, since messages contain `[`, `'` and `?`.

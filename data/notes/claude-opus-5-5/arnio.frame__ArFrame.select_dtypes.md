# Notes: `ArFrame.select_dtypes(include=None, exclude=None)`

## 1. Purpose
- Returns a new ArFrame that keeps only the columns whose dtype string (from `self.dtypes`) passes an include filter and/or an exclude filter.
- Columns keep their original order.

## 2. Inputs
- `include` and `exclude` each accept `None`, a single `str`, or a `list`/`tuple` of `str`.
  - A single string `s` is treated as `[s]`.
  - Lists and tuples are converted to a `frozenset`, so duplicates such as `["int64","int64"]` are allowed.
  - Any other type is rejected: `set`, `bytes`, `dict`, `int`, a generator, and so on.
- Valid dtype strings come from the constant `_VALID_DTYPES = {"int64","float64","string","bool","null"}`.
  - Matching is exact and case-sensitive. `"Int64"`, `"object"`, `"str"` and `"float"` are all invalid.
- An empty list or tuple counts as provided (it is not `None`):
  - `include=[]` gives an empty set, so no column matches.
  - `exclude=[]` excludes nothing.
- Parsing order: `include` is parsed fully, then `exclude`, then the overlap check runs.
- Validation does not depend on frame contents. It runs even on a frame with zero rows or zero columns.

## 3. Raises
All checks run in this order. The first failure wins.

1. Both arguments are `None`: `ValueError("select_dtypes() requires at least one of 'include' or 'exclude'.")`
   - Trigger: `f.select_dtypes()`
2. A list or tuple contains a non-string: `TypeError(f"'{name}' must contain only strings, got {[type(v).__name__ for v in non_strings]}.")`
   - `f.select_dtypes(include=["int64", 1])` gives `'include' must contain only strings, got ['int'].`
   - `exclude=(True,)` gives `... got ['bool'].`
3. Wrong container type: `TypeError(f"'{name}' must be a string, list, or tuple of strings, got {type(arg).__name__!r}.")`
   - `include={"int64"}` gives `'include' must be a string, list, or tuple of strings, got 'set'.`
   - `include=b"int64"` gives `got 'bytes'.`
4. Unknown dtype: `ValueError(f"Unrecognised dtype(s) in '{name}': {unknown}. Valid dtypes are: {sorted(_VALID_DTYPES)}.")`
   - `include="object"` gives `Unrecognised dtype(s) in 'include': ['object']. Valid dtypes are: ['bool', 'float64', 'int64', 'null', 'string'].`
   - `unknown` keeps input order and any duplicates.
5. Overlap, checked only when both are non-None: `ValueError(f"'include' and 'exclude' overlap: {sorted(overlap)}. A dtype cannot be both included and excluded.")`
   - `include=["int64","string"], exclude=["string","int64"]` gives `'include' and 'exclude' overlap: ['int64', 'string']. A dtype cannot be both ...`

Ordering consequences:
- `include` errors win over `exclude` errors. For example, `include="bad", exclude=5` raises the ValueError about `'include'`.
- A type error inside one argument beats an unknown dtype in the same argument. For example, `["bad", 1]` raises TypeError.

## 4. Returns and guarantees
- A column is kept if (`include` is None or its dtype is in `include`) **and** (`exclude` is None or its dtype is not in `exclude`).
- Non-empty match: returns `self.select_columns(matched)`.
  - The result is a new ArFrame with all rows, the matched columns in original order, and unchanged values and dtypes.
  - `attrs` is deep-copied.
  - The result is a new object even when every column matches. It compares `==` to the original but `is not` it.
- No match: returns `ArFrame(_Frame(len(self)), attrs=self._attrs.copy())`.
  - `columns == []`.
  - `attrs` is a **shallow** copy, so nested values are shared with the source.
  - The row count is presumably `len(self)`, but this depends on the C++ `_Frame(n)` constructor, which is not visible here. Only assert `columns == []` and `shape[1] == 0`.

## 5. Side effects
- None. The source frame's columns and `_attrs` are not mutated.

## 6. Branches and edge cases
Example frame: `f = ar.from_pandas(pd.DataFrame({"a":[1,2], "b":[1.5,2.5], "c":["x","y"], "d":[True,False]}))`. Expected dtypes are `a:int64, b:float64, c:string, d:bool`; verify this with `f.dtypes` in the test.

| Call | Result `columns` |
|---|---|
| `include="int64"` | `["a"]` |
| `include=["string","int64"]` | `["a","c"]` (frame order, not argument order) |
| `exclude="string"` | `["a","b","d"]` |
| `include=("int64","float64","string"), exclude=["bool"]` | `["a","b","c"]` (disjoint, so allowed) |
| `include=[]` | `[]` (no-match path) |
| `exclude=[]` | `["a","b","c","d"]` |
| `include="null"` (no null columns) | `[]`, no error |
| `exclude=["int64","float64","string","bool"]` | `[]` |

- A frame with zero columns and any valid filter goes to the no-match path.
- On a zero-row frame, validation still raises as usual.

## 7. Rules from the docstring
- At least one of `include` or `exclude` is required.
- `exclude` is "applied after include".
- The result keeps original column order.
- The docstring lists the accepted dtype values above.

## 8. Testing hints
- Build frames with `ar.from_pandas(df)`. Inspect results with `.columns`, `.shape`, `.dtypes`, `res["a"]` (returns a list), and `res == other`.
- Do not hardcode inferred dtypes:
  - The `int64`/`string` mapping is produced by convert and the C++ core.
  - Whether an all-None column becomes `"null"` is not visible here.
  - Prefer `res.dtypes` checks or only use int, float, str and bool columns.
- For attrs, set `f._attrs = {"k": {"n": 1}}` directly. There is no public setter here, but `_attrs` is a slot.
  - Match path: `res._attrs == f._attrs` and `res._attrs["k"] is not f._attrs["k"]` (deep copy).
  - No-match path: `res._attrs["k"] is f._attrs["k"]` (shallow copy), and `res._attrs is not f._attrs`.
- Use `pytest.raises(..., match=re.escape(msg))` with the exact messages above. They contain `(`, `[`, `'` and `.`.
- Check ordering by passing `include` in reverse order and asserting the result follows frame order.
- Do not assert the row count of the no-match result (it depends on C++).

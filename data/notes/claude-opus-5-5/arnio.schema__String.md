# Notes: `arnio.schema.String`

## 1. Purpose
- Factory that validates its arguments and returns a frozen `Field` with `dtype="string"`.
- The `Field` is used with `ar.Schema` / `ar.validate` to check string columns for nullability, uniqueness, an allowed set, a regex pattern and min/max length.

## 2. Inputs
All parameters are keyword-only. `String("x")` raises a built-in `TypeError` because positional arguments are not accepted.

- `nullable=True`: must be a real `bool`. `1` and `0` are rejected by `Field.__post_init__`.
- `pattern=None`: a `str` or `None`. It is compiled with `re.compile` and then checked for nested quantifiers.
- `allowed=None`: any iterable except `str`/`bytes`. It is converted with `set(allowed)`:
  - a list or tuple becomes a set, and duplicates collapse;
  - a dict becomes the set of its keys;
  - `[]` becomes `set()`, which is not `None`.
- `case_sensitive=True`: must be a `bool`.
- `unique=False`: must be a `bool`.
- `severity="error"`: must be `"error"` or `"warning"`.
- `min_length` / `max_length=None`: normalised by `_normalize_length`:
  - `None` stays `None`;
  - `bool` and `numpy.bool_` are rejected;
  - the value must be a `numbers.Integral` (so `np.int64(3)` is accepted);
  - it is converted with `int()`, so `np.int64(3)` is stored as the Python int `3`;
  - it must be `>= 0`; `0` is allowed.
- `required_if=None`: a 2-tuple `(str, scalar)` or `None`, checked in `Field`.

## 3. Raises (in check order; the first failing check wins)
1. `min_length` (via `_normalize_length`):
   - bool, `np.bool_`, `1.5`, or `"3"` → `TypeError("min_length must be an integer or None")`
   - `-1` → `ValueError("min_length must be greater than or equal to 0")`
2. `max_length`: the same two messages, with `max_length` in place of `min_length`.
3. `String(min_length=5, max_length=2)` → `ValueError("min_length must be less than or equal to max_length")`.
4. `allowed="abc"` or `b"x"` → `TypeError("allowed must be a sequence of allowed values, not a bare string")`.
5. `pattern` checks:
   - `pattern=123` → `TypeError("pattern must be a string or None")`
   - `pattern="("` → `ValueError("Invalid regex pattern: '('")`, i.e. the template is `f"Invalid regex pattern: {pattern!r}"`, chained from `re.error`.
   - `pattern="(a+)+"` → `ValueError("Unsafe regex pattern rejected: nested quantifiers can cause catastrophic backtracking during schema validation")`
6. `set(allowed)` failures, which are Python's own `TypeError`:
   - `allowed=5` → not iterable
   - `allowed=[[1]]` → unhashable type
7. `Field.__post_init__`, checked in this order:
   - `nullable=1` → `TypeError("nullable must be a bool")`
   - `unique="y"` → `TypeError("unique must be a bool")`
   - `case_sensitive=None` → `TypeError("case_sensitive must be a bool")`
   - `required_if=["a", 1]` (not a tuple) → `TypeError("required_if must be a tuple or None")`
   - `required_if=("a",)` → `TypeError("required_if must be a (column_name, expected_value) tuple")`
   - `required_if=(1, "x")` → `TypeError("required_if column name must be a string")`
   - `required_if=("a", [1])` → `TypeError("required_if expected value must be a scalar")`. `None` counts as a scalar, so it is OK.
   - `severity=1` → `TypeError("severity must be a string")`
   - `severity="info"` → `ValueError("severity must be 'error' or 'warning'")`

Ordering examples:
- `String(min_length=-1, pattern="(")` raises the `min_length` ValueError.
- `String(pattern="(a+)+", nullable="x")` raises the unsafe-regex ValueError.

Unsafe-pattern detector: a `(...)` group followed by `*`, `+`, `?` or `{n}` / `{n,m}` whose body contains a variable quantifier. Inside a group body, `*`, `+`, `?` and `{n,}` / `{n,m}` count as variable; a plain `{n}` does not.
- Rejected: `(a+)+`, `(a*)*`, `(?:a+)+`, `(a?)+`, `(a+){3}`, `(\d{2,4})+`
- Accepted: `(ab)+`, `a+b+`, `(\d{3})+`, `([a+])+` (the `+` sits inside a character class), `^\d{5}(-\d{4})?$`

## 4. Returns and guarantees
- Returns a new `Field` (a frozen dataclass) with `dtype="string"` and the given `nullable`, `pattern`, `allowed` (a `set` or `None`), `case_sensitive`, `unique`, `min_length`, `max_length`, `required_if` and `severity`.
- These fields are always left at their defaults: `min=None`, `max=None`, `semantic=None`, `format=None`, `_datetime_min/_datetime_max/_date_min/_date_max=None`.
- Equality: `ar.String() == ar.Field(dtype="string")` is `True`.
- `String(allowed=["a","a","b"]).allowed == {"a","b"}`
- Setting any attribute raises `dataclasses.FrozenInstanceError`.

## 5. Side effects
- None. Nothing is registered and no input is mutated, because `allowed` is copied into a new set.

## 6. Validation behaviour of the returned Field (observed via `ar.validate`)
- **Issue order within one column:**
  1. `dtype`
  2. `nullable`
  3. `required_if`
  4. `unique`
  5. `allowed`
  6. `pattern`
  7. `min_length`
  8. `max_length`
- **`row_index` is 1-based:** it is the pandas index + 1. `bad_rows` is a sorted, de-duplicated list of row indices.
- **Nulls:** a value counts as null if it is NaN/None. When the column dtype is `"string"`/`"object"`, empty or whitespace-only strings also count as null. Null values are skipped by the `unique`, `allowed`, `pattern` and length checks.
- **Rule names and messages** (here `name!r` means the column name in repr form, e.g. `'s'`):
  - `"nullable"`: `Column 's' contains null or empty values`
  - `"unique"`: every duplicate row is flagged (`keep=False`). Message: `Column 's' contains duplicate values`
  - `"allowed"`: `Column 's' contains values outside the allowed set`. With `case_sensitive=False`, both sides are compared after `str.casefold()`.
  - `"pattern"`: uses `str.fullmatch`, so `pattern="a"` rejects `"ab"`. Message: `Column 's' has values that do not match the pattern`
  - `"min_length"`: `Column 's' has values shorter than 3`
  - `"max_length"`: `Column 's' has values longer than 3`
  - `"required_if"`: `Column 's' is required when 'flag' == 'Y'`. If the condition column is absent, the issue has `rule="missing_column"`, message `Column 'flag' not found`, `column="flag"`, and severity `"error"`.
- **Length** is measured on the raw value with no stripping, so `" ab "` has length 4.
- **`allowed=[]`:** every non-null value is flagged.
- **`severity="warning"`:** the issues are still reported, but `result.passed` is `True`.
- **dtype check:** when the frame's dtype for the column is not `"string"`, the issue is `rule="dtype"` with message `Column 'n' has dtype 'int64'; expected 'string'`. The exact dtype names come from `ArFrame.dtypes`, which is the C++ core. Probe them first (`frame.dtypes`) instead of hard-coding.

## 7. Rules from comments and docstrings
- `nullable`: whether null values are allowed.
- `pattern`: a regex that non-null values must match.
- `unique`: non-null values must be unique.
- `case_sensitive`: applies to `allowed` matching only.
- bool and `numpy.bool_` are explicitly rejected for the length arguments.

## 8. Testing hints
- **Build frames from pandas:**
  `ar.from_pandas(pd.DataFrame({"s": ["ab", "abc", None]}))`
  then `res = ar.validate(f, ar.Schema({"s": ar.String(max_length=2)}))`.
  - Expect `[i.rule for i in res.issues] == ["max_length"]`.
  - Expect `res.issues[0].row_index == 2` and `res.issues[0].value == "abc"`.
- **Exact exception messages:** use `pytest.raises(..., match=re.escape(msg))`. Do this especially for messages containing `(`, `'` or `|`.
- **Assert stored attributes directly:**
  - `String(min_length=np.int64(2)).min_length` has type `int`.
  - `String(allowed=("x",)).allowed == {"x"}`.
  - `String().allowed is None`.
  - `String().semantic is None`.
- **Message-template pitfall:** do not confuse `String`'s messages with `Field`'s own messages for the same mistakes. `Field` says "pattern must be a str or None" and "pattern is not a valid regular expression"; `String` reaches its own messages first.
- **Value pitfall:** `issue.value` is the raw pandas value. Compare string values directly.

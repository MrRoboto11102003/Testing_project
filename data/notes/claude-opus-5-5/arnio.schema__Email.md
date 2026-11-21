# Notes: `arnio.schema.Email`

## 1. Purpose
- Factory that returns a frozen `Field` for an email-address string column.
- It does not validate any data itself. The email check runs later, in `ar.validate(frame, schema)` or `Schema.validate`.

## 2. Inputs (all keyword-only)
- `nullable: bool = True`: must be a real `bool`. `Field.__post_init__` rejects `1`/`0`.
- `unique: bool = False`: must be a real `bool`.
- `severity: str = "error"`: must be `"error"` or `"warning"`.
- `validation: str = "light"`: must be exactly `"light"` or `"strict"`. The check is case-sensitive, so `"LIGHT"` is rejected.
- `required_if: tuple[str, Any] | None = None`: must be a 2-tuple `(column_name: str, scalar)`.
- Positional arguments are not accepted: `ar.Email(False)` raises `TypeError` from Python itself.

## 3. Raises (checks run in this order)
1. `validation` is not a `str` → `TypeError("Email validation must be a string: 'light' or 'strict'")`, e.g. `ar.Email(validation=None)`.
2. `validation` is not in `{"light","strict"}` → `ValueError("Email validation must be 'light' or 'strict'")`, e.g. `ar.Email(validation="LIGHT")`.

Then the checks in `Field.__post_init__`, in this order:

3. `nullable` is not a bool → `TypeError("nullable must be a bool")`, e.g. `ar.Email(nullable=1)`.
4. `unique` is not a bool → `TypeError("unique must be a bool")`, e.g. `ar.Email(unique="yes")`.
5. `required_if` problems:
   - Not a tuple → `TypeError("required_if must be a tuple or None")`, e.g. `required_if=["a", 1]`.
   - Length is not 2 → `TypeError("required_if must be a (column_name, expected_value) tuple")`, e.g. `("a",)`.
   - First item is not a str → `TypeError("required_if column name must be a string")`, e.g. `(1, "x")`.
   - Second item is not a scalar → `TypeError("required_if expected value must be a scalar")`, e.g. `("a", [1])`.
6. `severity` problems:
   - Not a str → `TypeError("severity must be a string")`, e.g. `severity=1`.
   - Not one of the two allowed values → `ValueError("severity must be 'error' or 'warning'")`, e.g. `severity="info"`.

- Order matters: `ar.Email(validation="x", nullable=1)` raises the `ValueError` about validation, not the nullable `TypeError`.

## 4. Returns and guarantees
- Returns a new `ar.Field` (a frozen dataclass) with:
  - `dtype="string"`
  - `semantic="email"` for light, or `"email:strict"` for strict
  - `nullable`, `unique`, `required_if` and `severity` as passed in
- All other attributes keep their defaults:
  - `min`, `max`, `pattern`, `allowed`, `min_length`, `max_length`, `format`: `None`
  - `case_sensitive`: `True`
- Because `Field` is a dataclass, equality works by value:
  - `ar.Email() == ar.Field(dtype="string", semantic="email")` is `True`.
  - `ar.Email(validation="strict") == ar.Field(dtype="string", semantic="email:strict")` is `True`.
- Setting an attribute raises `dataclasses.FrozenInstanceError`.

## 5. Side effects
- None. No registry is touched and the inputs are not mutated.

## 6. Behaviour during `ar.validate`
All `row_index` values are **1-based** (`pandas index + 1`).

**Missing column.** If the column is absent from the frame, you get one issue:
- `rule="required_column"`
- `message="Missing required column: email"`
- `row_index=None`
- `severity` = the field's severity

**Dtype check.** If the actual dtype (from `frame.dtypes`, computed in C++/`frame` code I cannot see) is not `"string"`, you get one issue:
- `rule="dtype"`
- message = `f"Column {name!r} has dtype {actual!r}; expected 'string'"`
- Prefer asserting on `rule` over the exact actual-dtype text.

**Null mask.** A value counts as null if it is NaN/None, or (for object/string dtype) if `str(v).strip() == ""`. So `""` and `"   "` count as null.

**`nullable=False`.** Each null row gives:
- `rule="nullable"`
- `message=f"Column {name!r} contains null or empty values"`

**Nulls are skipped by the email check**, even when `nullable=True`, so they produce no issues.

**`required_if=(c, v)`:**
- If column `c` is missing: one issue with `column=c`, `rule="missing_column"`, `message=f"Column {c!r} not found"`. Its severity is always `"error"`; the field's severity is ignored here.
- Otherwise, each row where `df[c] == v` and the value is null gives `rule="required_if"`, `message=f"Column {name!r} is required when {c!r} == {v!r}"`.

**`unique=True`.** Every occurrence of a duplicated non-null value is reported (`keep=False`):
- `rule="unique"`
- `message=f"Column {name!r} contains duplicate values"`
- The comparison is case-sensitive: `"A@b.co"` and `"a@b.co"` are not duplicates.

**Email check.** Applied to each non-null value with `str.fullmatch` (the whole string must match):
- light: `[^@\s]+@[^@\s]+\.[^@\s]+`
- strict: `[a-zA-Z0-9.!#$%&'*+/=?^_\`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+`
- `rule` = the semantic string, i.e. `"email"` or `"email:strict"`.
- message = `f"Column {name!r} contains invalid {semantic} values"`, e.g. `"Column 'email' contains invalid email:strict values"`.
- `value` = the original value.

| value | light | strict |
|---|---|---|
| `"a@b.co"`, `"user@sub.example.com"` | valid | valid |
| `"a@b"` (no dot) | invalid | invalid |
| `"a b@c.de"`, `" a@b.co"` (space) | invalid | invalid |
| `"a@@b.co"` | invalid | invalid |
| `"user@exa_mple.com"` | valid | invalid |
| `"user@example.com."` | valid | invalid |
| `"üser@example.com"` | valid | invalid |

**Order of issues within the column:** dtype, then nullable, then required_if, then unique, then email. Within each rule, rows appear in row order.

**`severity="warning"`.** Issues carry `severity="warning"` and `result.passed` is `True`. With `"error"`, any issue makes `passed` `False`.

## 7. Rules from docstrings
- `validation` selects the mode: `"light"` or `"strict"`.
- `unique` means non-null values must be unique.

## 8. Testing hints
- Build a frame like this:
  ```python
  frame = ar.from_pandas(pd.DataFrame({"email": ["a@b.co", "bad"]}))
  r = ar.validate(frame, ar.Schema({"email": ar.Email()}))
  ```
  Expected: `[(i.rule, i.row_index, i.value) for i in r.issues] == [("email", 2, "bad")]` and `r.bad_rows == [2]`.
- Use the light/strict table above to tell the two modes apart. `"user@exa_mple.com"` gives zero issues in light mode and one `"email:strict"` issue in strict mode.
- Keep test columns all-string. Mixing in non-strings may change the dtype that the C++ code reports.
- Do not assert on the `value` of null rows; how nulls are represented after `to_pandas` is not visible.
- Check `rule` and `row_index` rather than full message text whenever the message contains an actual dtype.
- Schema round-trips are observable: `diff_schema({"e": ar.Email()}, {"e": ar.Email(validation="strict")})` gives exactly one difference, with `change="changed_field"`, `attribute="semantic"`, `expected="email"` and `observed="email:strict"`.

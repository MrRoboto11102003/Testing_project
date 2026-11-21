# Notes: `arnio.schema.PhoneNumber`

## 1. Purpose
- A factory that returns a frozen `Field` for phone-number string columns: `dtype="string"` and `semantic="phone"`.
- The phone check itself runs only later, inside `ar.validate(frame, schema)` or `Schema.validate`.

## 2. Inputs
All parameters are keyword-only. Nothing is normalised; values are passed straight into `Field(...)`.
- `nullable: bool = True` must be an actual `bool`. `1` and `None` are rejected.
- `unique: bool = False` must be an actual `bool`.
- `severity: str = "error"` must be `"error"` or `"warning"`.
- `required_if: tuple[str, Any] | None = None` must be a 2-tuple of `(str, scalar)`. The scalar check is `pd.api.types.is_scalar`.
- The resulting `Field` has these fixed values:
  - `dtype="string"`, `semantic="phone"`
  - `pattern=None`, `allowed=None`, `case_sensitive=True`
  - `min=max=None`, `min_length=max_length=None`, `format=None`

## 3. Raises
All checks happen in `Field.__post_init__`, in this order: nullable → unique → required_if → severity.

| Exception | Exact message | Minimal call |
|---|---|---|
| `TypeError` | `"nullable must be a bool"` | `ar.PhoneNumber(nullable=1)` |
| `TypeError` | `"unique must be a bool"` | `ar.PhoneNumber(unique="yes")` |
| `TypeError` | `"required_if must be a tuple or None"` | `ar.PhoneNumber(required_if=["a", 1])` |
| `TypeError` | `"required_if must be a (column_name, expected_value) tuple"` | `ar.PhoneNumber(required_if=("a",))` |
| `TypeError` | `"required_if column name must be a string"` | `ar.PhoneNumber(required_if=(1, "x"))` |
| `TypeError` | `"required_if expected value must be a scalar"` | `ar.PhoneNumber(required_if=("a", [1]))` |
| `TypeError` | `"severity must be a string"` | `ar.PhoneNumber(severity=1)` |
| `ValueError` | `"severity must be 'error' or 'warning'"` | `ar.PhoneNumber(severity="info")` |

- Ordering example: `PhoneNumber(nullable=1, severity="bad")` raises the nullable `TypeError`, because nullable is checked first.
- Use `pytest.raises(..., match=re.escape(msg))`. The messages contain `(`, `)` and `'`.

## 4. Returns and guarantees
- Returns a new frozen `Field`. It is a dataclass with equality, so:
  - `ar.PhoneNumber() == ar.Field(dtype="string", semantic="phone")` is True.
  - `ar.PhoneNumber(nullable=False).nullable is False`.
- Assigning to an attribute raises `dataclasses.FrozenInstanceError`.

During `validate`, a phone column gives these issues, in this order within the column:
1. **dtype mismatch.** Emitted if `frame.dtypes[col] != "string"`.
   - Message: `f"Column {name!r} has dtype {actual!r}; expected 'string'"`.
   - `rule="dtype"`, `row_index=None`.
   - The dtype names come from the C++ core, which I cannot see. Check `frame.dtypes` before asserting on this.
2. **nullable.** Only when `nullable=False`. One issue per null/empty/whitespace-only row.
   - `rule="nullable"`, message `f"Column {name!r} contains null or empty values"`.
3. **required_if** (see section 6).
4. **unique.** Uses `duplicated(keep=False)`, so every copy of a duplicate is flagged.
   - `rule="unique"`, message `f"Column {name!r} contains duplicate values"`.
5. **phone check.**
   - Each non-null value is cast to string and must **fullmatch** `r"\+?[0-9][0-9 .()\-]{6,}[0-9]"`.
   - Failures give `rule="phone"`, message `f"Column {name!r} contains invalid phone values"`.
   - `value` is the original cell value.
- For every row issue:
  - `row_index = pandas index + 1`, so it is **1-based**.
  - `severity` is the field's severity.
- `result.bad_rows` is the sorted set of row indexes.

## 5. Side effects
- None. No registries are touched; `_CUSTOM_VALIDATORS` is unused.

## 6. Branches and edge cases for the phone regex
The regex requires: an optional leading `+`, then a digit, then at least 6 characters from `[0-9 .()-]`, then a final digit. That means at least 8 characters, not counting the `+`.

- **Valid:**
  - `"12345678"`, `"555-1234"` (8 chars)
  - `"555-123-4567"`, `"555.123.4567"`
  - `"+1 (555) 123-4567"`, `"+44 20 7946 0958"`
- **Invalid:**
  - `"5551234"`: only 7 chars.
  - `"(555) 123-4567"`: the first character must be a digit.
  - `"555-123-4567-"` and `"555-123-456)"`: must end with a digit.
  - `"++15551234567"`: more than one `+`.
  - `"1-800-FLOWERS"` and `"abc"`: letters are not allowed.
  - `" 555-123-4567"`: the leading space is kept in the text, and the space is not a digit.
- **Nulls and blanks:**
  - Applies when the actual dtype is `"string"` or `"object"`.
  - `None`, NaN, `""` and `"   "` are treated as null and are **skipped** by the phone check.
  - They are only reported if `nullable=False`, and then only as `rule="nullable"`.
- **`required_if=("country", "US")`:**
  - If `country` is missing from the frame, there is one issue:
    - `column="country"`, `rule="missing_column"`, message `"Column 'country' not found"`.
    - Its severity is always `"error"`, even when `severity="warning"`.
  - Otherwise, each row where `country == "US"` and the phone is null/blank gives:
    - `rule="required_if"`, message `"Column 'phone' is required when 'country' == 'US'"`.
- **`severity="warning"`:** issues are still produced, but `result.passed` is True.
- **Missing phone column in the frame:** one issue with `rule="required_column"`, message `"Missing required column: phone"`, `row_index=None`.

## 7. Rules from comments/docstrings
- "nullable: Whether null values are allowed."
- "unique: Whether non-null values must be unique."
- "required_if: Conditional requirement as a column/value pair."

## 8. Testing hints
- Build a frame with `ar.from_pandas(pd.DataFrame({"phone": ["555-123-4567", "abc", None]}))`.
- Then run `res = ar.validate(frame, {"phone": ar.PhoneNumber()})`.
- Expected issues for that frame:
  - `[(i.rule, i.row_index, i.value) for i in res.issues] == [("phone", 2, "abc")]`.
  - `res.bad_rows == [2]`.
- Assert exact `rule`, `message`, `row_index` and `severity`. Do not stop at checking `passed`.
- Boundary pair worth testing: `"555-1234"` is valid and `"5551234"` is invalid.
- Also test that `"(555) 123-4567"` fails.
- Use all-string columns so no `dtype` issue appears; with all-string columns, the actual dtype is `"string"`.
- How nulls round-trip through `from_pandas`/`to_pandas` depends on the C++ core, which I cannot see. To be safe, use `""` for blanks, since it is treated as null either way when the dtype is `"string"`.
- Each duplicate value produces one `"unique"` issue per occurrence. The `"phone"` issues come after the `"unique"` issues.
- `ValidationIssue` equality includes `value`. When comparing whole issues, prefer comparing tuples of fields instead.

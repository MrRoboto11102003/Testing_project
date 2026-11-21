# Notes: `arnio.schema.Schema.validate(self, frame, max_errors=None)`

## 1. Purpose
- Validates an `ArFrame` against this schema and returns a `ValidationResult`.
- It is a one-line wrapper: `return validate(frame, self, max_errors=max_errors)`. Every behaviour below is the module-level `ar.validate`.

## 2. Inputs
- `frame`: must be an `ArFrame` instance. A `pd.DataFrame` or a chunk iterator is rejected.
- `max_errors`: `None` (no limit) or an `int >= 1`. A `bool` is rejected.
- Internally:
  - `df = to_pandas(frame)` and `dtypes = frame.dtypes`. The dtype strings come from the C++ core; they are expected to be `"int64"`, `"float64"`, `"string"`, `"bool"`, but I cannot see that code.
  - Fields are checked in the dict insertion order of `self.fields`.

## 3. Raises (checked in this order)
1. Non-ArFrame `frame` raises `TypeError`. Message: `"validate() expects a single ArFrame. Chunked validation is not currently supported; validate each chunk explicitly or materialize the data before validation."` Example: `schema.validate(pd.DataFrame())`.
2. `max_errors` is a bool or not an int: `TypeError("max_errors must be an int or None")`. Examples: `max_errors=True`, `max_errors=1.5`.
3. `max_errors <= 0`: `ValueError("max_errors must be >= 1")`. Examples: `0`, `-1`.
4. A rule in `rules` returns a non-list: `TypeError(f"Rule {rule_name!r} must return a list of ValidationIssue, got {type(result).__name__!r}")`.
   - `rule_name` is `fn.__name__`.
   - Example: `def r(df): return None` gives `"Rule 'r' must return a list of ValidationIssue, got 'NoneType'"`.
5. A rule returns a list containing a non-issue: `TypeError(f"Rule {rule_name!r} returned a non-ValidationIssue item: {type(item).__name__!r}")`.
   - Only `KeyError` from a rule is caught. Any other exception propagates.
6. A Custom validator raises: `ArnioError(f"Custom validator {name!r} failed for column {col!r} with value {v!r}: {exc}")`.
7. A Custom validator returns something other than True/False/None/pd.NA: `TypeError`.

## 4. Returns and guarantees
`ValidationResult` fields:
- `row_count = len(df)`.
- `issues`: a list in check order.
- `issue_count == len(issues)`.
- `bad_rows`: sorted, de-duplicated `row_index` values (`None` excluded).

Indexing and status:
- `row_index` is **1-based**: pandas index + 1. Row 0 gives `row_index == 1`.
- `issue.value` is the raw offending cell value. Column-level issues have `value=None` and `row_index=None`.
- `result.passed` is True iff no issue has `severity == "error"`. Warning-only issues still pass.

Issue order:
1. Per field, in field order. A missing column gives `rule="required_column"`, `message=f"Missing required column: {name}"`, `severity=field.severity`.
2. Strict check: `rule="unexpected_column"`, `message=f"Unexpected column: {name}"`, severity `"error"`, in frame column order.
3. Composite `unique`.
4. `rules`.

Within one column, `_validate_column` checks in this order:

| Order | Rule | Fires when / message |
|---|---|---|
| 1 | `dtype` | Fires if `field.dtype` is set and differs from the actual dtype. Skipped when the field is `"datetime"` and the column is `"string"`. Message: `f"Column {name!r} has dtype {actual!r}; expected {exp!r}"`. If the column is string, the field is int64/float64, and the values look convertible, it gains the suffix `". Values appear safely convertible to 'int64'"`. "Convertible" means all non-null values match `^-?\d+$` for int. For id-like names (`id`, `*_id`, `uuid`, `zip*`), leading-zero values like `"007"` block the suffix. |
| 2 | `nullable` | Only if `nullable=False`. Message: `"Column 'x' contains null or empty values"`. For string/object columns, `""` and whitespace-only values also count as null. |
| 3 | `required_if` | If the condition column is absent: `rule="missing_column"`, `message=f"Column {c!r} not found"`, severity `"error"`. Otherwise one issue per row where `df[c] == v` and the cell is null. Message: `f"Column {name!r} is required when {c!r} == {v!r}"`. |
| 4 | `unique` | Duplicates among non-null values, with `keep=False`, so **every** occurrence is reported. Message: `"Column 'x' contains duplicate values"`. |
| 5 | `allowed` | Message: `"Column 'x' contains values outside the allowed set"`. Case folding applies only when `dtype=="string"` and `case_sensitive=False`. |
| 6 | datetime: `format`, `min`, `max` | Bound messages: `"Column 'x' has values below <Timestamp>"` / `"above"`. |
| 6 (else) | numeric: `numeric`, `min`, `max` | Applies when min or max is set. Messages: `"Column 'x' contains non-numeric values"`, `f"Column {name!r} has values below {min}"`, `f"... above {max}"`. Bounds are strict: a value equal to a bound passes. |
| 7 | `pattern` | Uses `fullmatch`. Message: `"Column 'x' has values that do not match the pattern"`. |
| 8 | semantic | The rule name equals the semantic (e.g. `"email"`, `"url"`, `"ipv4"`, `"date"`). Message: `f"Column {name!r} contains invalid {semantic} values"`. Date min/max use rule `min`/`max` with message `"has date values before 2020-01-01"` / `"after"`. Custom validators use `rule="custom"`. |
| 9 | `min_length` / `max_length` | Messages: `"has values shorter than N"` / `"longer than N"`. |

- Null and empty cells are skipped by every check after `nullable`.
- Severity: column checks use `field.severity`. Strict, composite-unique and rule-`KeyError` issues use `"error"`.

## 5. Side effects
- None. The frame and schema are not mutated. Rules receive the `df` produced by `to_pandas`.

## 6. Branches and edge cases
- **`max_errors=N`**: stops as soon as N issues have accumulated and returns exactly N.
  - Example: three bad rows plus `max_errors=1` gives `issue_count==1`.
  - Later columns, strict, unique and rules are then not evaluated.
- **Composite `unique=[]`**: one issue with `column=None`, `rule="composite_unique"`, `message="Composite unique columns cannot be empty"`.
- **Composite unique column absent**: `rule="missing_column"`, `message=f"Column {c!r} not found"`.
- **Composite duplicates**: one issue per duplicated row (all occurrences), with `message=f"Duplicate rows found for columns {list(unique)}"`. Example: `"Duplicate rows found for columns ['a', 'b']"`.
- **Rule raises `KeyError('z')`**: an issue with `column="z"`, `rule="missing_column"`, `message="Cross-field rule referenced a missing column: 'z'"`.
- **Zero-row frame with the column present**: `row_count==0` and no row issues. A dtype issue can still appear.
- **Empty schema `Schema({})`**: passes. With `strict=True`, every frame column is reported.

## 7. Rules stated in docstrings
- "Validation stops once this limit (`max_errors`) is reached."
- `passed` means "zero error-level issues".

## 8. Testing hints
- Build frames with `ar.from_pandas(pd.DataFrame({...}))`.
- For nulls in string columns, use `""` or `"  "`. These count as null regardless of how the C++ core stores None.
- Prefer substring or `rule` assertions for dtype messages, because the actual-dtype string comes from the C++ core.
- `schema.validate(f)` and `ar.validate(f, schema)` return equal results; you can compare them via `.to_dict()`.
- Example of a precise test:
  ```python
  f = ar.from_pandas(pd.DataFrame({"a": [1, 5, 10]}))
  r = ar.Schema({"a": ar.Int64(min=2, max=9)}).validate(f)
  assert [(i.rule, i.row_index) for i in r.issues] == [("min", 1), ("max", 3)]
  assert r.bad_rows == [1, 3] and not r.passed
  ```
- Pitfalls:
  - `row_index` is 1-based. Do not expect 0.
  - `unique` reports all occurrences, not just the second.
  - `Int64(min=5)` with value 5 passes.
  - `issue.value` may be a numpy scalar, so compare with `==`.
  - `severity="warning"` issues still appear in `issues` and `bad_rows`, but `passed` stays True.
  - Do not write tests that rely on `TypeError` raised by rules being caught; it propagates.

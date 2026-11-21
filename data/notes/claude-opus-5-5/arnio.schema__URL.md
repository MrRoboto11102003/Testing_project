# Notes: `arnio.schema.URL`

## 1. Purpose
- Factory that returns a frozen `Field` for a string column whose non-null values must be URLs of the form `<scheme>://<non-whitespace>`.
- Validation happens later, in `ar.validate(...)` or `Schema.validate(...)`.

## 2. Inputs
All parameters are keyword-only.

- `nullable: bool = True`: passed to `Field`. It must be a real `bool`, otherwise `Field` raises `TypeError`.
- `unique: bool = False`: must be a real `bool`.
- `severity: str = "error"`: must be `"error"` or `"warning"`.
- `required_if: tuple[str, Any] | None = None`: must be a 2-tuple of `(str, scalar)`. `Field` checks this.
- `allowed_schemes: list | tuple | set | None = None`:
  - `None` gives `semantic="url"`. The pattern used is `https?://[^\s]+`, so only lowercase `http` and `https`.
  - Otherwise, each item is checked to be a `str`, then `.strip()`ed, then checked against `^[A-Za-z][A-Za-z0-9+.-]*$`.
  - Items are collected into a set, so duplicates collapse.
  - The set is `sorted` (case-sensitive; uppercase sorts before lowercase) and each item is `re.escape`d.
  - Items are joined with `"|"`, giving `semantic = "url:" + joined`.
  - Case is not changed.
  - Examples:
    - `["https"]` gives `"url:https"`.
    - `["https", "http", "http"]` gives `"url:http|https"`.
    - `[" ftp "]` gives `"url:ftp"`.
    - `["svn+ssh"]` gives `"url:svn\\+ssh"` (in Python source; that is the characters `svn\+ssh`). `re.escape` escapes `+`, `.` and `-`.
    - `["HTTPS", "http"]` gives `"url:HTTPS|http"`.

## 3. Raises
Checks run in this order. The `allowed_schemes` checks run first, inside `URL`.

| Order | Exception | Exact message | Example call |
|---|---|---|---|
| 1 | `TypeError` | `allowed_schemes must be a sequence of scheme names, not a bare string` | `URL(allowed_schemes="https")` (also `b"https"`) |
| 2 | `TypeError` | `allowed_schemes must be a sequence of scheme names` | `URL(allowed_schemes={"https": 1})`, or a generator |
| 3 | `ValueError` | `allowed_schemes must contain non-empty strings` | `URL(allowed_schemes=[1])` or `URL(allowed_schemes=["  "])` |
| 4 | `ValueError` | `allowed_schemes must contain URL scheme names such as 'http' or 'https'` | `["1http"]`, `["ht tp"]`, `["http:"]` |
| 5 | `ValueError` | `allowed_schemes must be a non-empty sequence` | `URL(allowed_schemes=[])`, `()`, `set()` |

Notes on the table:
- Row 3 fires on the first bad item. Items are checked in iteration order.
- Row 3 fires before row 4 for the same item: for `[" "]`, the strip makes it empty, so you get row 3's message.

Then `Field.__post_init__` runs its checks, in this order:
1. `nullable`: `TypeError("nullable must be a bool")`, e.g. `URL(nullable=1)`.
2. `unique`: `TypeError("unique must be a bool")`, e.g. `URL(unique="yes")`.
3. `required_if`:
   - `TypeError("required_if must be a tuple or None")` for `URL(required_if=["a", 1])`.
   - `TypeError("required_if must be a (column_name, expected_value) tuple")` for `("a",)`.
   - `TypeError("required_if column name must be a string")` for `(1, "x")`.
   - `TypeError("required_if expected value must be a scalar")` for `("a", [1])`.
4. `severity`:
   - `TypeError("severity must be a string")` for `severity=1`.
   - `ValueError("severity must be 'error' or 'warning'")` for `severity="info"`.

## 4. Returns and guarantees
- Returns a new frozen `Field` with:
  - `dtype="string"`
  - `semantic="url"` or `"url:<schemes>"`
  - `nullable`, `unique`, `required_if`, `severity` as passed in
  - every other attribute at its default (`pattern=None`, `allowed=None`, `min`/`max=None`, `case_sensitive=True`).
- `Field` is a dataclass with `eq=True`, so you can compare whole objects:
  - `ar.URL() == ar.Field(dtype="string", semantic="url")` is `True`.
  - `ar.URL(allowed_schemes=["https"]).semantic == "url:https"`.
- The field is frozen. Assigning to an attribute raises `dataclasses.FrozenInstanceError`.

How validation behaves with this field (from `_validate_column`):
- If the column's frame dtype (`frame.dtypes[name]`, which comes from the C++ core) is not `"string"`, you get an extra `rule="dtype"` issue.
- Null mask: `isna()` OR a value that is empty or whitespace-only after strip (when the dtype is `object` or `string`). Null values are skipped by the URL check.
- The check is `str.fullmatch` on non-null values:
  - default pattern: `https?://[^\s]+`
  - with schemes: `(<schemes>)://[^\s]+`
- It is case-sensitive:
  - `"HTTP://a"` is invalid with the default.
  - `"https://a"` is invalid with `allowed_schemes=["HTTPS"]`.
- Concrete inputs:
  - Invalid with the default: `"https://"`, `"http://a b"`, `"ftp://x"`, `"www.a.com"`, `" http://a"` (leading space; it is not stripped before the match).
  - Valid with the default: `"http://a"`, `"https://example.com/x?y=1"`.
  - With `allowed_schemes=["https"]`: `"http://a.com"` is invalid.
- Each invalid value produces one `ValidationIssue`:
  - `column = name`
  - `rule = field.semantic` (e.g. `"url"` or `"url:http|https"`)
  - `message = f"Column {name!r} contains invalid {semantic} values"`, e.g. `"Column 'site' contains invalid url values"`
  - `row_index = pandas index + 1` (1-based)
  - `value` = the original value
  - `severity` = the field's severity
- Issue order within a column: dtype, nullable, required_if, unique, then the url issues in row order.

## 5. Side effects
- None. No registry changes and no mutation of inputs. Note that `allowed_schemes` is copied into a new set.

## 6. Branches and edge cases
- `allowed_schemes=None` gives `"url"`. Any non-`None` value takes the validation branch: a tuple or set is fine, and an empty one raises (row 5).
- Mixed case is preserved, and sorting is case-sensitive: `["b", "A"]` gives `"url:A|b"`.
- Allowed characters after the first letter are letters, digits, `+`, `.` and `-`: `["git+ssh", "a.b-c"]` gives `"url:a\\.b\\-c|git\\+ssh"` (characters `a\.b\-c|git\+ssh`).
- `severity="warning"`: invalid URLs still produce issues, but `result.passed` is `True`.
- `nullable=False`: an empty string `""` produces a `rule="nullable"` issue and no url issue.

## 7. Rules stated in docs
- The docstring says that if `allowed_schemes` is `None`, both `http` and `https` are accepted.
- The intended example is `["https"]`.

## 8. Testing hints
Build a frame and validate it like this:
```python
df = pd.DataFrame({"site": ["http://a.com", "ftp://b", "https://c.org"]})
res = ar.validate(ar.from_pandas(df), ar.Schema({"site": ar.URL()}))
# expect exactly one issue: rule "url", row_index 2, value "ftp://b"
```

- **Filter issues by rule.** Use `[i for i in res.issues if i.rule == "url"]`. Do not assert the total issue count, because a `dtype` issue appears if the C++ core does not report `"string"`. That mapping cannot be confirmed from this module; `"string"` is expected for object/str columns.
- **Row indices are 1-based.**
- **Use `""` for nulls in string columns.** Prefer it over `None`, because how `None` round-trips through the C++ core is not visible here.
- **Check `semantic` strings for exact values.** This catches regressions in sorting, escaping or deduplication. Remember backslashes in the expected Python string literal.
- **Use `pytest.raises(..., match=...)` for the exception messages.** Pass the message through `re.escape`, because the messages contain `'` and `.`.

# Notes: `arnio.schema.Schema.from_json(value)` (classmethod)

## 1. Purpose
- Turns a JSON string, normally one produced by `Schema.to_json()`, back into a `Schema` that contains only fields.
- Cross-field `rules` are never restored. The result always has `rules=None`.

## 2. Inputs
- `value` is a JSON text (`str`, `bytes` or `bytearray`) and is passed straight to `json.loads`.
- **Top-level object.** Allowed keys are exactly `fields`, `strict`, `unique` and `rules_omitted`.
  - `fields` is required and must be a JSON object of `{column_name: field_object}`. `{}` is allowed.
  - `strict` defaults to `False` when the key is missing. An explicit `null` does **not** fall back to the default and raises.
  - `unique` may be missing, `null` or a list.
  - `rules_omitted` is accepted with any value and ignored.
- **Each field object.** Allowed keys:
  - `dtype`, `nullable`, `min`, `max`, `pattern`, `semantic`, `allowed`, `case_sensitive`, `unique`
  - `min_length`, `max_length`, `format`
  - `datetime_min`, `datetime_max`, `date_min`, `date_max` (no leading underscore)
  - `required_if`, `severity`
- **Field defaults and conversions.**
  - A missing key gives the default: `None`, or `nullable=True`, `case_sensitive=True`, `unique=False`, `severity="error"`.
  - An explicit JSON `null` for `nullable`, `unique`, `case_sensitive` or `severity` passes `None` and raises in `Field`.
  - `allowed` list → `set`.
  - `required_if` list → `tuple`.
  - `datetime_min`/`datetime_max` → `pd.Timestamp` via `pd.to_datetime`.
  - `date_min`/`date_max` → `datetime.date`.

## 3. Raises (in check order)
1. `value` is not str/bytes, e.g. `from_json(None)` → stdlib `TypeError` ("the JSON object must be str, bytes or bytearray, not NoneType"). This is not wrapped.
2. Bad JSON → `ValueError(f"Invalid schema JSON: {exc.msg}")`. `exc.msg` has no position info.
   - `from_json("not json")` → `"Invalid schema JSON: Expecting value"`.
3. Not a JSON object, e.g. `"[]"`, `"null"`, `"1"` → `TypeError("Schema JSON must decode to an object with 'fields', 'strict', and optional 'unique'.")`.
4. Unknown top-level key → `ValueError(f"Schema JSON contains unknown key(s): {sorted(unknown)}. Allowed keys: ['fields', 'rules_omitted', 'strict', 'unique']")`.
   - `'{"fields":{},"x":1}'` → `"... unknown key(s): ['x']. ..."`.
5. `fields` missing or not an object, e.g. `'{}'` or `'{"fields":[]}'` → `TypeError("Schema JSON 'fields' must be an object mapping names to field definitions.")`.
6. Errors from each field, in JSON key order. These come **before** the `strict`/`unique` checks.
   - Field value not an object, `'{"fields":{"a":1}}'` → `TypeError("Schema JSON field for column 'a' must be an object, got int.")`.
   - Unknown field key → `ValueError(f"Schema JSON field 'a' contains unknown key(s): ['_date_min']. Allowed keys: [...sorted 18 keys...]")`.
   - `"allowed": "x"` → `TypeError("Schema JSON field 'a' 'allowed' must be a list or null.")`.
   - `"required_if": ["b"]` or a non-list → `TypeError("Schema JSON field 'a' 'required_if' must be a 2-item list or null.")`.
   - `"datetime_min": "nope"` → `ValueError("DateTime datetime_min must be a parseable datetime scalar")`. The same template applies to `datetime_max`.
   - `"date_min": 5` → `TypeError("Date date_min must be a date string, datetime.date, or pd.Timestamp, got int")`.
   - `"date_min": "nope"` → `ValueError("Date date_min must be a parseable date (YYYY-MM-DD), got 'nope'")`.
   - Bound parsing happens before `Field.__post_init__`, which then checks:
     - `"dtype":"foo"` → `ValueError("dtype must be one of int64, float64, string, bool, datetime, or None, got 'foo'")`.
     - `"dtype":1` → `TypeError("dtype must be a string or None, got int")`.
     - `"nullable":null` → `TypeError("nullable must be a bool")`.
     - `"unique":null` → `TypeError("unique must be a bool")`.
     - `"case_sensitive":null` → `TypeError("case_sensitive must be a bool")`.
     - `"required_if":[1,"x"]` → `TypeError("required_if column name must be a string")`.
     - `"required_if":["b",[1]]` → `TypeError("required_if expected value must be a scalar")`.
     - `"dtype":"int64","min":5,"max":1` → `ValueError("min must be less than or equal to max")`.
     - `"dtype":"int64","min":"x"` → `TypeError("min must be numeric or None")`.
     - The min/max checks run **only** for int64/float64. For example, `"dtype":"string","min":"x"` is accepted.
     - `"pattern":"("` → `ValueError("pattern is not a valid regular expression: ...")`.
     - `"pattern":"(a+)+"` → `ValueError("Unsafe regex pattern rejected: nested quantifiers can cause catastrophic backtracking during schema validation")`.
     - `"min_length":-1` → `ValueError("min_length must be >= 0, got -1")`.
     - `"min_length":5,"max_length":2` → `ValueError("min_length (5) must be <= max_length (2)")`.
     - `"severity":"info"` → `ValueError("severity must be 'error' or 'warning'")`.
     - `"severity":null` → `TypeError("severity must be a string")`.
7. `strict` not a bool, e.g. `"strict":1`, `"strict":"yes"` or `"strict":null` → `TypeError("Schema JSON 'strict' must be a boolean.")`.
8. `unique` not a list or null, e.g. `"unique":"a"` → `TypeError("Schema JSON 'unique' must be a list of strings or null.")`.
9. Then `Schema.__post_init__` runs:
   - `"unique":[1]` → `TypeError("Schema 'unique' members must be strings, got int for element 1.")`.
   - `"unique":["a","a"]` → `ValueError("Schema 'unique' must not contain duplicate column names")`.

## 4. Returns and guarantees
- Returns a new `Schema(fields=..., strict=..., unique=...)` with `rules=None`.
- `fields` is a dict that keeps the JSON key order. `to_json` writes keys sorted.
- `unique` stays a **list** (not a tuple). `"unique": []` is accepted and gives `[]`.
- Field values:
  - `allowed` is a `set`.
  - `required_if` is a `tuple`.
  - `_datetime_min` is a `pd.Timestamp`.
  - `_date_min` is a `datetime.date`.
  - `min`/`max` keep their JSON int/float type.

## 5. Side effects
- No mutation, no registry changes, no warnings, no I/O.
- `to_json` warns when rules are present; `from_json` never warns.

## 6. Branches and edge cases
- `'{"fields":{}}'` → `Schema` with `fields == {}`, `strict is False`, `unique is None`, `rules is None`.
- `'{"fields":{"a":{}}}'` → `fields["a"] == Field()`, which has all defaults (`dtype None`, `nullable True`, `severity "error"`).
- `'{"fields":{},"rules_omitted":true}'` → OK and `rules is None`. `"rules_omitted": "anything"` is also accepted.
- `'{"fields":{"a":{"allowed":["x","y"],"required_if":["b",1]}}}'` → `allowed == {"x","y"}` and `required_if == ("b", 1)`.
- `'{"fields":{"t":{"dtype":"datetime","datetime_min":"2024-01-01"}}}'` → `_datetime_min == pd.Timestamp("2024-01-01")`.
- `'{"fields":{"d":{"date_max":"2024-12-31"}}}'` → `_date_max == datetime.date(2024,12,31)`.
- If both a field error and a bad `strict` are present, the field error is raised.

## 7. Rules stated in comments or docstrings
- The marker `rules_omitted` "is accepted and silently ignored — rules cannot be round-tripped through JSON and must be re-attached manually".

## 8. Testing hints
- Round trip: for a schema such as the one below, `ar.Schema.from_json(s.to_json()) == s` holds. Field and Schema are dataclasses with `__eq__`.
  ```python
  s = ar.Schema({"id": ar.Int64(nullable=False, min=1, max=9),
                 "c": ar.String(allowed=["x","y"]),
                 "d": ar.Date(min="2024-01-01")},
                strict=True, unique=["id"])
  ```
  - Equality breaks if the original used `unique=("id",)`, because `from_json` returns a list.
- Build field objects directly when needed: `from arnio.schema import Field`.
- The restored schema should behave the same when used. Use `ar.validate(ar.from_pandas(df), restored)` to check behaviour.
- Use `pytest.raises(..., match=re.escape(msg))`. The messages contain `[ ] ( ) . '`.
- Don't hash `Schema` or `Field` objects that contain a set or dict; they are unhashable. Compare them with `==` only.
- Don't assert the full `JSONDecodeError` text beyond `exc.msg`. For example, `"Invalid schema JSON: Expecting value"` is fine to assert exactly.

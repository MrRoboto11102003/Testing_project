# Notes: `arnio.schema.Schema.to_json`

## 1. Purpose
- Serializes a `Schema` (its fields, `strict` and `unique`) to a deterministic JSON string.
- Cross-field `rules` are not serialized. If any are present, a `UserWarning` is emitted and a `"rules_omitted": true` marker is added.

## 2. Inputs
- `self` only; there are no parameters.
- The `Schema` has already been validated in `__post_init__`:
  - field names are `str`;
  - values are `Field` instances;
  - `strict` is a bool;
  - `unique` is a list/tuple of distinct strings or `None`;
  - `rules` is a list/tuple of callables or `None`.

## 3. Raises
- `TypeError` from `json.dumps` when a field value is not JSON-serializable. The message is the stdlib one: `"Object of type <T> is not JSON serializable"`. Examples:
  - `ar.Schema({"d": ar.String(allowed=[datetime.date(2024,1,1)])}).to_json()` → `Object of type date is not JSON serializable`
  - `ar.Schema({"a": ar.Int64(required_if=("b", np.int64(1)))}).to_json()` → `Object of type int64 is not JSON serializable` (`np.int64` is accepted by `Field` because it passes `pd.api.types.is_scalar`)
- Order of events: the rules warning is emitted first, then the payload is built, then `json.dumps` runs. So a schema with rules and a bad value warns and then raises.

## 4. Returns and guarantees
- Returns a `str` produced by `json.dumps(payload, sort_keys=True)`:
  - no indent;
  - default separators `", "` and `": "`;
  - `allow_nan=True`, so a `NaN` float is written as `NaN`.
- Top-level keys, sorted: `"fields"`, optionally `"rules_omitted"` (only when rules are present, value `true`), `"strict"`, `"unique"`.
- `"unique"` is `list(self.unique)`, keeping the original order. It is not sorted (unlike `diff_schema`). If unset it is `null`. A tuple becomes a JSON array.
- `"fields"` maps each column name to a dict with exactly 18 keys, in this sorted order:
  `allowed, case_sensitive, date_max, date_min, datetime_max, datetime_min, dtype, format, max, max_length, min, min_length, nullable, pattern, required_if, semantic, severity, unique`.
- How each field value is written:
  - **`allowed`**: a set becomes a sorted list. If its types are mixed and not comparable, it is sorted by `(type(x).__name__, repr(x))`, so `{1, "a"}` → `[1, "a"]`. `None` → `null`.
  - **`required_if`**: the tuple becomes a list, e.g. `("x", 1)` → `["x", 1]`.
  - **`datetime_min` / `datetime_max`**: `Timestamp.isoformat()`. `ar.DateTime(min="2024-01-01")` → `"2024-01-01T00:00:00"`.
  - **`date_min` / `date_max`**: `date.isoformat()`. `ar.Date(min="2024-01-01")` → `"2024-01-01"`.
  - **`min` / `max`**: raw values. `ar.Int64(min=1, max=5)` → `1`, `5`. `ar.Float64(min=1)` stays `1`, not `1.0`.
- `semantic` for each field factory:

  | Factory | `semantic` |
  |---|---|
  | `Email()` | `"email"` |
  | `Email(validation="strict")` | `"email:strict"` |
  | `URL()` | `"url"` |
  | `URL(allowed_schemes=["https","http"])` | `"url:http|https"` (sorted, joined by `|`) |
  | `PhoneNumber()` | `"phone"` |
  | `CountryCode()` | `"country_code"` |
  | `UUID()` | `"uuid"` |
  | `IPv4()` | `"ipv4"` |
  | `MACAddress()` | `"mac_address"` |
  | `Date()` | `"date"`; `dtype` is `"string"` |
  | `DateTime()` | `dtype` is `"datetime"`; `semantic` is `null` |
  | `Custom("n")` | `"custom:n"`; `dtype` is `null` |

- Exact output for `ar.Schema({"a": ar.Int64()}).to_json()`:
```
{"fields": {"a": {"allowed": null, "case_sensitive": true, "date_max": null, "date_min": null, "datetime_max": null, "datetime_min": null, "dtype": "int64", "format": null, "max": null, "max_length": null, "min": null, "min_length": null, "nullable": true, "pattern": null, "required_if": null, "semantic": null, "severity": "error", "unique": false}}, "strict": false, "unique": null}
```
- `ar.Schema({}).to_json()` → `'{"fields": {}, "strict": false, "unique": null}'`

## 5. Side effects
- Emits a `UserWarning` only when `bool(self.rules)` is true. The warning uses `stacklevel=2`. Exact message:
  `Schema.to_json(): cross-field rules cannot be serialized and have been omitted. The JSON payload includes "rules_omitted": true. Restore rules manually after from_json().`
- The schema is not mutated (it is a frozen dataclass).

## 6. Branches and edge cases
- `rules=None` or `rules=[]`: no warning and no `rules_omitted` key.
- `rules=[f]` (list or tuple): warns, and the key appears between `fields` and `strict`: `'{"fields": {...}, "rules_omitted": true, "strict": ...}'`.
- `strict=True` → `"strict": true`.
- `unique=("b","a")` → `"unique": ["b", "a"]` (order preserved).
- Multiple fields are emitted in name order, whatever order they were inserted in. For example, `{"b":..., "a":...}` gives `a` first.
- `severity="warning"` → `"severity": "warning"`.
- `ar.String(min_length=2, max_length=5, pattern=r"^x$", case_sensitive=False)` → those values appear as-is (`"pattern": "^x$"`, `"case_sensitive": false`).

## 7. Rules from the docstring
- The output is stable: field names are sorted and `sort_keys=True` is used.
- Rules are silently omitted apart from the warning.
- `from_json()` restores only the field part of the schema. It accepts and ignores the `rules_omitted` key.

## 8. Testing hints
- Prefer `json.loads(s)` and compare dicts. Exact-string checks are also deterministic (see §4), and checking the exact string is a good way to catch separator or ordering regressions.
- Check the key order with `list(json.loads(s)["fields"]["a"].keys())` (dicts preserve the parsed order) or by checking the raw string.
- Warning: `with pytest.warns(UserWarning, match=r"cross-field rules cannot be serialized"):`
- No warning: use
  ```python
  with warnings.catch_warnings():
      warnings.simplefilter("error")
      schema.to_json()
  ```
- Round trip: `ar.Schema.from_json(s.to_json()).fields == s.fields` holds for factory-built fields, including `allowed` sets, `required_if` tuples, and date/datetime bounds.
- Round-trip pitfalls:
  - `Schema` equality fails if the original `unique` was a tuple (it comes back as a list).
  - `Schema` equality fails if the original had rules (they come back as `None`).
- `Custom` needs `ar.register_validator(...)` first. The registry is global and duplicate names raise, so use a unique name or pass `overwrite=True`.
- Do not expect `NaN`/`Infinity` from `Int64`/`Float64`, which reject non-finite bounds. Only a raw `ar.Field(dtype="float64", min=float("nan"))` produces `"min": NaN`.

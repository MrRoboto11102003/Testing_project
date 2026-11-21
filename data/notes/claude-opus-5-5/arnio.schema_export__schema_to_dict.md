# Notes: `arnio.schema_export.schema_to_dict(schema)`

## 1. Purpose
- Converts a schema-like input into a plain dict of the form `{"fields": {...}, ...metadata}` that is ready to serialise.
- `schema_to_yaml` calls it first.
- Pure Python. It does not call the C++ core except to read `ArFrame.schema_summary`.

## 2. Inputs
`schema` can be any object. The checks run **in this order**, and the first match wins:
1. **Has a `schema_summary` attribute** (an ArFrame). It is replaced by `schema.schema_summary`, which is a list of `ColumnSummary`, and then handled by step 2.
2. **`isinstance(schema, list)`**. Each entry must be one of:
   - an object with `.name` and `.dtype`, or
   - a dict with the keys `"name"` and `"dtype"`.
3. **Has a `fields` attribute.** This covers `ar.Schema` or any object such as `types.SimpleNamespace(fields=...)`.
   - A plain dict does not have a `fields` *attribute*, so it skips this step.
4. **`isinstance(schema, dict)`**. This is either the structured form `{"fields": {...}, ...}` or a flat `scan_csv`-style form `{"col": "int64"}`.
5. Anything else raises `TypeError`.
- Tuples, strings, `None` and ints are not accepted. A tuple of `ColumnSummary` also raises `TypeError`.

## 3. Raises
- **Final fallback** (`TypeError`):
  - Template: `f"Expected a dict, an ArFrame, or an object with a 'fields' attribute, got {type(schema)!r}."`
  - Example: `schema_to_dict(42)` gives `...got <class 'int'>.`
- **List path, bad entry** (`TypeError`):
  - Template: `f"Unsupported list item type: {type(entry)!r}"`
  - Example: `[{"name": "a"}]` (no `"dtype"`) gives `Unsupported list item type: <class 'dict'>`.
  - Example: `[1]` gives `<class 'int'>`.
- **Schema path, truthy `rules`** (`ValueError`), with this exact message:
  `"schema_to_yaml does not support Schema objects with custom rules because callables are not serializable. Remove schema.rules before exporting."`
  - Example: `SimpleNamespace(fields={}, rules=[lambda df: []])`.
  - `rules=[]` or `rules=None` is fine.
  - The rules check happens **before** the field-name check.
- **Non-str field name** (`TypeError`, `"schema field names must be strings"`). Applies to the Schema path and the dict path.
  - Example: `{1: "int64"}`, or `{"fields": {1: "x"}}`.
  - Checked before sorting.
- **Non-str key inside a field dict or nested dict** (`TypeError`, `"schema dict keys must be strings"`).
  - Example: `{"a": {1: "x"}}`.
- **Unsupported value type** (`TypeError`):
  - Template: `f"schema_to_yaml does not support values of type {type(value)!r}. Only str, int, float, bool, None, list, and dict are allowed."`
  - Example: `{"a": {"x": object()}}`.
- **Gotcha: nested tuples raise.** A tuple is converted to a list only when it is the value handed *directly* to the normaliser (see the table in section 6). Validation runs over the whole dict first and rejects any tuple found inside a dict or list:
  - `{"a": {"x": (1, 2)}}` raises `TypeError ... <class 'tuple'> ...`.
  - `{"a": {"x": [(1,)]}}` also raises.
- **Structured-dict path, non-str metadata key** (`TypeError`):
  - Template: `f"schema metadata keys must be strings, got {type(meta_key)!r}"`
  - Example: `{"fields": {}, 1: "x"}`.
- **Set with mixed incomparable types**, e.g. `{"a": {"s": {1, "x"}}}`: Python's own `TypeError` from `sorted()`.

## 4. Returns and guarantees
- The result is always a new dict, and its first key is `"fields"`.
- **List / ArFrame path** returns `{"fields": {name: {"type": str(dtype).upper(), "nullable": nullable}}}`.
  - `"nullable"` is included only if the entry has a `nullable` attr/key.
  - **Field order is input order (NOT sorted).**
  - Duplicate names: the later entry overwrites the earlier one.
  - There are no `strict`/`unique` keys.
  - Only this path upper-cases the type.
- **Schema path** returns `{"fields": <sorted by name>}`, plus `"strict": schema.strict` if that attr exists, plus `"unique": schema.unique` if that attr exists.
  - `unique` is passed through **unchanged**: no normalisation, and a tuple stays a tuple.
  - `ar.Schema` always has both attrs. A `SimpleNamespace(fields=...)` gets only `"fields"`.
  - How each field value is converted:
    - dict: normalised.
    - has `.dtype`: `_field_to_dict(field)` from `arnio.schema`. **Its exact output cannot be seen here, so avoid asserting it.**
    - has `__dict__`: `vars()` minus keys starting with `_`, then normalised.
    - otherwise: `{"type": str(field)}`. So `"int64"` gives `{"type": "int64"}`, `5` gives `{"type": "5"}`, and `None` gives `{"type": "None"}`.
- **Dict path** returns `{"fields": <sorted by name>}`, followed by the metadata keys in input order. Metadata values are normalised, so the tuple `("id",)` becomes `["id"]`.
  - Each field value is converted like this:
    - str `v`: `{"type": v}`. The case is kept, so `"int64"` stays `"int64"`.
    - anything else: normalised.
  - Normalisation rules:
    - Dict keys are sorted.
    - A set becomes a sorted list.
    - A top-level tuple becomes a list.
    - Scalars (`str`, `int`, `float`, `bool`, `None`) are returned unchanged.
    - NaN is allowed.

## 5. Side effects
- None. The input is not mutated and no files are written.
- Schema-path `unique` is the very same object as `schema.unique`.

## 6. Branches and edge cases
| Input | Result |
|---|---|
| `[]` | `{"fields": {}}` |
| `{}` | `{"fields": {}}` (flat path) |
| `{"fields": {}}` | `{"fields": {}}` |
| `{"b": "int64", "a": "string"}` | `{"fields": {"a": {"type": "string"}, "b": {"type": "int64"}}}`; `list(result["fields"]) == ["a", "b"]` |
| `{"fields": {"z": "int64"}, "strict": True, "unique": ("id",)}` | `{"fields": {"z": {"type": "int64"}}, "strict": True, "unique": ["id"]}` |
| `{"fields": "int64"}` (fields not a dict) | flat path: `{"fields": {"fields": {"type": "int64"}}}` |
| `{"a": 5}` | `{"fields": {"a": 5}}` (non-str scalar is not wrapped) |
| `{"a": (1, 2)}` | `{"fields": {"a": [1, 2]}}` |
| `{"a": {"allowed": {"b", "a"}, "z": 1, "m": None}}` | `allowed` becomes `["a", "b"]`; inner keys are ordered `allowed`, `m`, `z` |
| `[ColumnSummary("b", "int64", False), ColumnSummary("a", "string", True)]` | `{"fields": {"b": {"type": "INT64", "nullable": False}, "a": {"type": "STRING", "nullable": True}}}`, order `b, a` |
| `[{"name": "x", "dtype": "float64"}]` | `{"fields": {"x": {"type": "FLOAT64"}}}` (no `nullable`) |
| `SimpleNamespace(fields={"b": "int64", "a": {"k": 1}})` | `{"fields": {"a": {"k": 1}, "b": {"type": "int64"}}}`, with no `strict`/`unique` |
| `SimpleNamespace(fields={"a": SimpleNamespace(min=1, _p=2)}, strict=True, unique=("id",))` | `{"fields": {"a": {"min": 1}}, "strict": True, "unique": ("id",)}` (a tuple!) |

## 7. Rules from comments/docstrings
- Field order is meant to be deterministic, through sorted keys and sorted sets.
- Metadata must "never touch the fields namespace".
- The comment says tuples like `required_if=("col", "val")` should "survive export". In practice this only works for a top-level value; nested tuples raise (see section 3).

## 8. Testing hints
- `from types import SimpleNamespace` and `from arnio.frame import ColumnSummary`.
- Use these to drive the Schema path and list path deterministically, without relying on `_field_to_dict`.
- **Check order** with `list(d["fields"])`. Plain `==` on dicts ignores order.
- **ArFrame path:** `ar.from_pandas(pd.DataFrame({"b": [1], "a": ["x"]}))`.
  - Assert only `list(d["fields"]) == ["b", "a"]` (column order, not sorted).
  - Assert that each `type` equals `type.upper()` and that it has a `"nullable"` key.
  - The exact dtype strings and nullable values come from the C++ core, so do not hardcode them.
- **Real `ar.Schema(fields={})`:** you can assert `"strict"` and `"unique"` keys exist and equal `schema.strict` and `schema.unique`.
- **Error messages:** use `pytest.raises(TypeError, match=...)` and wrap the message in `re.escape`, because the messages contain `(`, `'`, `.`, `<`.
- The type is upper-cased **only** on the list/ArFrame path. The dict path keeps `"int64"` lower-case.

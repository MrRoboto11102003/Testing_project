# Notes: `arnio.schema_export.schema_to_yaml(schema, path=None) -> str`

## 1. Purpose
- Turns a schema into a deterministic YAML string with sorted keys. It is built by `schema_to_dict(schema)` and then the private emitter.
- If `path` is given, it also writes that string to a file (UTF-8). The string is returned either way.

## 2. Inputs
- `schema`: the type decides which `schema_to_dict` branch runs. Checks happen in this order:
  1. **Has a `schema_summary` attribute (e.g. ArFrame)**: it is replaced by `schema.schema_summary` and then goes to the list branch.
  2. **`list`**: each item must either
     - have `.name` and `.dtype` (ColumnSummary-like), or
     - be a dict with `"name"` and `"dtype"` keys.
     - Result is `{"fields": {name: {"type": str(dtype).upper(), "nullable": ...}}}`. `nullable` is included only if the item has it.
  3. **Has a `fields` attribute (Schema-like)**:
     - Each field value is converted as follows:
       - dict → used as-is.
       - Has `.dtype` → `arnio.schema._field_to_dict(field)`. That helper is not visible here, so its exact keys are unknown.
       - Other object with `__dict__` → its public attributes.
       - Anything else (e.g. a str) → `{"type": str(field)}`.
     - Dict field values go through the normaliser.
     - The output also gets `strict` and `unique` copied from the object as-is. Note that `unique` is **not** normalised.
  4. **`dict`**:
     - **Structured form**: if `"fields"` is a key and its value is a dict, the other keys are treated as metadata.
     - **Flat form** (scan_csv style): otherwise, every key is a field name.
       - `{"fields": "x"}` is flat: it produces a field literally named `fields`.
     - A str field value becomes `{"type": value}`. Any other value goes through the normaliser.
     - Metadata values are normalised too.
- Normaliser (`_normalize_serializable`):
  - Dict keys are sorted. Sets become sorted lists.
  - A tuple becomes a list **only when the tuple is the value passed in directly**: a flat field value, or a metadata value.
  - It validates the whole container recursively *before* converting. So a tuple nested inside a dict or list raises TypeError, e.g. `{"a": {"x": ("p", "q")}}`.
- `path`: `None`, `str`, or `os.PathLike`. `bytes` and `int` are rejected.
  - Parent directories are created (`mkdir(parents=True, exist_ok=True)`).
  - An existing file is overwritten.

## 3. Raises
Schema conversion happens **before** any path check. A bad schema raises even when `path` is also bad, and no file is written.
- TypeError `f"Expected a dict, an ArFrame, or an object with a 'fields' attribute, got {type(schema)!r}."`
  - Trigger: `schema_to_yaml(42)` → `"...got <class 'int'>."`
- TypeError `f"Unsupported list item type: {type(entry)!r}"`
  - Trigger: `schema_to_yaml(["a"])`
- ValueError `"schema_to_yaml does not support Schema objects with custom rules because callables are not serializable. Remove schema.rules before exporting."`
  - Trigger: an object with truthy `.rules`.
- TypeError `"schema field names must be strings"`
  - Trigger: `schema_to_yaml({1: "INT64"})`
  - Also triggered by `{"fields": {1: "x"}}`.
- TypeError `"schema dict keys must be strings"`
  - Trigger: a non-str key nested inside a field value, e.g. `{"a": {1: "x"}}`.
- TypeError `f"schema metadata keys must be strings, got {type(meta_key)!r}"`
  - Trigger: `{"fields": {}, 1: "x"}` → `"...got <class 'int'>"`
- TypeError `f"schema_to_yaml does not support values of type {type(v)!r}. Only str, int, float, bool, None, list, and dict are allowed."`
  - Trigger: `{"a": {"x": object()}}`
  - Also: `{"a": {"x": (1, 2)}}` → type `<class 'tuple'>`.
  - Also raised by the emitter for an unnormalised value, e.g. a Schema-like object with `unique=("id",)`.
- Path errors, checked in this order after serialization:
  1. TypeError `"path must be a string or os.PathLike"` (e.g. `path=123` or `path=b"x"`).
  2. ValueError `"path must not be empty"` for a str that is empty or whitespace-only (`""` or `"   "`).
  3. ValueError `"path must point to a file, not a directory"` (e.g. `path=tmp_path`).
     - Also raised by `pathlib.Path("")`, because it resolves to `.`.

## 4. Returns and guarantees
- Returns a `str` that always ends with exactly one `"\n"`. The body never ends in a newline, because newlines inside strings are escaped.
- Keys are sorted at every dict level, so top-level `fields` comes before `strict`, which comes before `unique`.
- Indentation is 2 spaces per level.
  - A non-empty dict or list value is written as `key:` and then nested lines.
  - An empty dict prints as `{}` and an empty list as `[]`.
  - List items under a key at depth d use (d+1)*2 spaces of indent plus `- `.
- Scalars:
  - `None` → `null`; booleans → `true` / `false`; int → `str()`.
  - float → `repr()`, e.g. `1.5`, `1e-05`.
  - NaN → `.nan`, inf → `.inf`, -inf → `-.inf`.
- A string is double-quoted if any of these hold:
  - It is empty.
  - `float(s)` succeeds: `"1"`, `"1.5"`, `"nan"`, `"inf"`, `"1_0"`, `" 1"`.
  - Its lowercase form is one of true/false/null/yes/no/on/off.
  - It matches the date pattern `^\d{4}-\d{2}-\d{2}...`.
  - It contains `\n` or `\r`.
  - Its first char is one of `" ' { [ | > ! & * # ? : - , @ \` % ~`.
  - It contains `:` or `#` anywhere.
  - It has leading or trailing whitespace.
- Escaping when quoted: `\` → `\\`, `"` → `\"`, newline → `\n`, CR → `\r`.
  - Unquoted strings are emitted raw: `a\b` stays `a\b`. `INT64`, `a b`, `y` are not quoted.
- Keys use the same quoting rules: key `"1"` → `"1": ...`, key `"true"` → `"true": ...`.

Exact examples:
```python
ar.schema_to_yaml({"id": "INT64", "name": "STRING"}) == \
  "fields:\n  id:\n    type: INT64\n  name:\n    type: STRING\n"
ar.schema_to_yaml({"fields": {"b": "x"}, "strict": True, "unique": None}) == \
  "fields:\n  b:\n    type: x\nstrict: true\nunique: null\n"
ar.schema_to_yaml({}) == "fields: {}\n"
ar.schema_to_yaml([]) == "fields: {}\n"
ar.schema_to_yaml({"a": {"allowed": {"y", "x"}}}) == \
  "fields:\n  a:\n    allowed:\n      - x\n      - y\n"
ar.schema_to_yaml({"a": ("x", "y")}) == "fields:\n  a:\n    - x\n    - y\n"
ar.schema_to_yaml({"a": {"type": "1", "n": 2, "f": 0.5}}) == \
  "fields:\n  a:\n    f: 0.5\n    n: 2\n    type: \"1\"\n"
```

## 5. Side effects
- The input schema is not mutated; new dicts are built.
- With `path`, the function creates parent directories and writes the file with `pathlib.Path.write_text(encoding="utf-8")`.
  - On Windows, `"\n"` is translated to `"\r\n"` on disk.

## 6. Branches and edge cases
- ArFrame input: `ar.from_pandas(pd.DataFrame({"a": [1]}))` produces `fields: a: type: <DTYPE upper>, nullable: <bool>`.
  - The dtype string and nullable flag come from the C++ core. Compute the expected values from `frame.schema_summary`.
- List of dicts: `[{"name": "a", "dtype": "int64"}]` → `"fields:\n  a:\n    type: INT64\n"`. There is no nullable line because the key is absent.
- Schema-like stub with no Field objects:
  - `types.SimpleNamespace(fields={"a": "int64"}, rules=None, strict=False, unique=None)`
  - → `"fields:\n  a:\n    type: int64\nstrict: false\nunique: null\n"`
  - With `unique=["id"]` → the output ends `unique:\n  - id\n`.
- The structured dict's metadata keys are emitted at top level, sorted together with `fields`.

## 7. Rules stated in comments/docstrings
- The output is deterministic (sorted keys); the docstring says "always ending with `\n`".
- An existing file is overwritten, and the returned string is the same whether or not `path` is given.
- Tuples such as `required_if` or `unique` are meant to "survive export" as lists. In the current code this holds only at the direct-value level (see §2).

## 8. Testing hints
- Compare full strings exactly. Prefer flat or structured dicts, or SimpleNamespace stubs, over real `ar.Schema`/`Field` objects, because the output of `_field_to_dict` is not visible here.
- When checking a written file, compare `Path(p).read_text(encoding="utf-8") == returned`. Do not compare bytes, because of the Windows CRLF translation.
- Use pytest `tmp_path`:
  - `tmp_path / "sub" / "s.yaml"` checks that parent directories are created.
  - Pre-writing junk to the file checks that it is overwritten.
- Round-tripping with `yaml.safe_load(out)` is a good structural check, e.g. that the quoted `"1"` loads back as a str.
  - Avoid `schema_from_yaml` for this, because it requires keys that `Schema.from_json` accepts.
- Use `pytest.raises(..., match=re.escape(msg))` for messages containing `(` `)` `.` or `'`.
- Do not assert the formatting of a dict inside a list. The spacing is odd (e.g. `- ` followed by extra indent).

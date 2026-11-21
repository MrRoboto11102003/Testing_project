# Notes: `arnio.schema.Schema.bootstrap_from_report(report)`

## 1. Purpose
- Classmethod that builds a `Schema` from a `DataQualityReport` (normally from `ar.profile(frame)`).
- It creates one `Field(dtype=..., nullable=...)` per profiled column. Nothing else is inferred.

## 2. Inputs
- `report`: must be an instance of `arnio.quality.DataQualityReport`. The class is imported lazily inside the method.
- Only two parts of the report are read:
  - `report.columns`: a dict mapping column name to profile. It is iterated with `.items()`, in insertion order.
  - For each profile:
    - `getattr(profile, "dtype", None)`
    - `getattr(profile, "null_count", 0)`. A missing attribute counts as 0 nulls.
- Every other report or profile attribute is ignored: `null_ratio`, `empty_string_count`, `min`/`max`, `semantic_type`, `unique_ratio`, and so on.
- dtype mapping uses `_DTYPE_MAP.get(str(dtype).lower(), "string")`. Matching is case-insensitive.

| Profile dtype (lowercased) | Field dtype |
|---|---|
| `int`, `int64`, `int32` | `int64` |
| `float`, `float64`, `float32` | `float64` |
| `bool` | `bool` |
| `object`, `string`, `str` | `string` |
| `null` | `"null"` → **Field raises** (see Raises) |
| anything else (`datetime`, `date`, `int8`, `category`, …) | `string` (fallback) |

- Note: `"datetime"` maps to `"string"`, not to `"datetime"`.
- `"INT64"` maps to `int64`, and `"Float32"` maps to `float64`.
- nullable rule: `nullable = null_count > 0`. So 0 nulls gives `False`, and 1 or more gives `True`.

## 3. Raises (checks happen in this order)
1. **Wrong report type**
   - `TypeError(f"Expected DataQualityReport, got {type(report).__name__}")`
   - Example: `ar.Schema.bootstrap_from_report({})` gives `"Expected DataQualityReport, got dict"`.
   - Example: `None` gives `"... got NoneType"`.
   - A duck-typed object with `.columns` is still rejected.
2. **Empty report**
   - Condition: `report.columns` is falsy (`{}`).
   - `ValueError("Cannot bootstrap schema from an empty report (no columns).")`
3. **Missing dtype** (checked per column, in dict order, during the loop)
   - Condition: a profile's `dtype` is falsy (`None`, `""`) or the attribute is missing.
   - `ValueError(f"Column profile for {col_name!r} is missing 'dtype' key.")`
   - Example: column `"a"` with `dtype=""` gives `"Column profile for 'a' is missing 'dtype' key."`
   - No partial schema is returned.
4. **Errors raised by `Field.__post_init__`**
   - dtype `"null"` is mapped to `"null"`, which is not a valid Field dtype:
     - `ValueError("dtype must be one of int64, float64, string, bool, datetime, or None, got 'null'")`
   - `null_count` is a numpy integer (e.g. `np.int64(2)`): `np.int64(2) > 0` gives a `np.bool_`, which is not a Python `bool`.
     - `TypeError("nullable must be a bool")`
     - Use plain Python ints when you build profiles by hand.
   - `null_count=None`: `None > 0` raises a Python `TypeError` (comparison not supported).
5. **Errors raised by `Schema.__post_init__`**
   - A non-str column key (e.g. `columns={1: prof}`):
     - `TypeError("Schema field names must be strings, got 'int': 1")`

## 4. Returns and guarantees
- Returns a new `cls(fields=fields)` object. `type(result) is ar.Schema` (or the subclass it was called on).
- Schema-level settings: `strict=False`, `unique=None`, `rules=None`.
- `list(result.fields)` equals `list(report.columns)`: same names, same order.
- Each Field has only `dtype` and `nullable` set. Every other attribute keeps its default:
  - `min/max/pattern/semantic/allowed/min_length/max_length/format/required_if` are `None`
  - `unique=False`, `case_sensitive=True`, `severity="error"`
- Field equality works: `result.fields["a"] == ar.Schema({...}).fields["a"]` for `Field(dtype="int64", nullable=False)`. Field is a frozen dataclass.

## 5. Side effects
- None. The report is not mutated and no global state changes.

## 6. Branches and edge cases
- Non-report input gives `TypeError` (check 1).
- `columns == {}` gives `ValueError` (check 2).
- A profile with `dtype=None` or `""` gives `ValueError` (check 3). The first failing column in dict order is the one named in the message.
- Unknown dtype string such as `"datetime64[ns]"` or `"weird"` gives a Field with dtype `"string"`.
- `null_count=0` gives `nullable=False`. `null_count=3` gives `nullable=True`.
- A profile object without a `null_count` attribute (e.g. `types.SimpleNamespace(dtype="int64")`) gives `nullable=False`.
- The input type is checked strictly, so `columns` itself must sit inside a real DataQualityReport.

## 7. Rules from the docstring
- Raises `TypeError` for non-report input.
- Raises `ValueError` for an empty report or a missing dtype.
- Intended flow: `ar.profile(frame)` → `bootstrap_from_report` → `schema.validate(frame)`.

## 8. Testing hints
- Build reports by hand for exact control:
  ```python
  from arnio.quality import ColumnProfile, DataQualityReport
  def prof(name, dtype, nulls=0):
      return ColumnProfile(name=name, dtype=dtype, semantic_type="x", row_count=3,
                           null_count=nulls, null_ratio=nulls/3, unique_count=3, unique_ratio=1.0)
  rep = DataQualityReport(row_count=3, column_count=2, memory_usage=0, duplicate_rows=0,
                          duplicate_ratio=0.0, columns={"a": prof("a","int32"), "b": prof("b","WEIRD",1)})
  s = ar.Schema.bootstrap_from_report(rep)
  # s.fields["a"] == Field(dtype="int64", nullable=False); s.fields["b"] == Field(dtype="string", nullable=True)
  ```
  - I assume both classes are plain dataclasses with no extra validation, based on the field lists in the index. I cannot see `quality.py`, so this is unverified.
  - `Field` can be imported with `from arnio.schema import Field`.
- Test every row of the mapping table. In particular, check that `"float32"` maps to `float64`, `"str"` and `"object"` map to `string`, and `"null"` raises.
- Use `pytest.raises(..., match=re.escape(msg))` with the exact messages listed in Raises.
- Real-profile path: `ar.profile(ar.from_pandas(df))`.
  - The exact dtype strings come from the C++ core / `quality.py`, which I cannot see.
  - Assert relative to the report rather than hard-coding: `s.fields[c].dtype == expected_map(rep.columns[c].dtype)` and `s.fields[c].nullable == (rep.columns[c].null_count > 0)`.
  - Do not assume what dtype an all-null column gets. If it is `"null"`, the call raises.
- Pitfalls:
  - Do not assume that `schema.validate(frame).passed` is True after bootstrapping. Validation treats empty or whitespace strings as null, but `null_count` may not count them, so a column could get `nullable=False` and still fail validation.
  - Dtype mismatches caused by the `"string"` fallback can also make validation fail.
  - Do not expect `unique`, `min`, or `max` to be inferred.

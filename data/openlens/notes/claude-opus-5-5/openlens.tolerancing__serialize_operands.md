# Notes: `openlens.tolerancing.serialize_operands(operands)`

## 1. Purpose
- Turns a list of `ToleranceOperand` objects into a list of plain dicts that can be stored as JSON.
- It is the inverse of `deserialize_operands`, so a saved tolerance set loads back the same.

## 2. Inputs
- `operands`: any iterable of operand-like objects. A list comprehension walks it, so a list, tuple or generator all work.
- Each item must have these attributes, read directly with no default:
  - `element_index`
  - `param_type`, which must have a `.value` (normally a `ToleranceType` member)
  - `min_val`
  - `max_val`
- These three are read with `getattr` and a default, so they may be missing:
  - `distribution` → `"uniform"` if missing
  - `std_dev` → `0.0` if missing
  - `surface` → `1` if missing
- A real `ToleranceOperand` dataclass always has all three. The defaults only apply to duck-typed objects such as `types.SimpleNamespace`.
- **No validation or conversion at all.**
  - Values are copied exactly as they are: an int `min_val=1` stays `1`.
  - `distribution="foo"` comes out as `"foo"`.
  - `surface=7` comes out as `7`.
  - A negative `element_index` passes through unchanged.

## 3. Raises
- No explicit `raise` in the function.
- `AttributeError` if an item is missing a required attribute:
  - `serialize_operands([object()])` → `AttributeError` (no `element_index`).
  - `serialize_operands([SimpleNamespace(element_index=0, param_type="x", min_val=0, max_val=1)])` → `AttributeError`, because `str` has no `.value`.
- `TypeError` if `operands` is not iterable, e.g. `serialize_operands(None)`.
  - This differs from `deserialize_operands(None)`, which returns `[]`.

## 4. Returns and guarantees
- Returns a new `list`, one dict per operand, in the same order as the input.
- Each dict has exactly 7 keys, in this insertion order:
  ```python
  {"element_index": op.element_index,
   "type": op.param_type.value,      # the enum VALUE string, not the name
   "min_val": op.min_val, "max_val": op.max_val,
   "distribution": op.distribution, "std_dev": op.std_dev, "surface": op.surface}
  ```
- The key is `"type"`, not `"param_type"`.
- Concrete example:
  - Input: `ToleranceOperand(0, ToleranceType.RADIUS_1, -0.1, 0.1)`
  - Output: `{"element_index": 0, "type": "Radius 1", "min_val": -0.1, "max_val": 0.1, "distribution": "uniform", "std_dev": 0.0, "surface": 1}`
- Enum values that go into `"type"`:

  | Member | `"type"` value |
  |---|---|
  | `RADIUS_1` | `"Radius 1"` |
  | `RADIUS_2` | `"Radius 2"` |
  | `THICKNESS` | `"Thickness"` |
  | `REFRACTIVE_INDEX` | `"Refractive Index"` |
  | `DECENTER_X` | `"Decenter X"` |
  | `DECENTER_Y` | `"Decenter Y"` |
  | `TILT_X` | `"Tilt X"` |
  | `TILT_Y` | `"Tilt Y"` |
  | `DECENTER_Z` | `"Decenter Z"` |
  | `AIR_GAP` | `"Air Gap"` |
  | `ABBE_NUMBER` | `"Abbe Number"` |
  | `IRREGULARITY` | `"Surface Irregularity"` |
  | `WEDGE` | `"Wedge"` |
  | `FOCUS` | `"Focus"` |

- Every call builds fresh dicts and a fresh list. Changing the output does not change the operands.

## 5. Side effects
- None: the operands are not changed, nothing is logged, and nothing is written to disk.

## 6. Branches and edge cases
- `serialize_operands([])` → `[]`.
- Gaussian operand:
  - Input: `ToleranceOperand(2, ToleranceType.IRREGULARITY, -0.25, 0.25, "gaussian", 0.05, surface=2)`
  - Output: `{"element_index": 2, "type": "Surface Irregularity", "min_val": -0.25, "max_val": 0.25, "distribution": "gaussian", "std_dev": 0.05, "surface": 2}`
- Duck-typed item with no optional attributes:
  - Input: `SimpleNamespace(element_index=1, param_type=ToleranceType.TILT_X, min_val=-1, max_val=1)`
  - Output: `distribution "uniform"`, `std_dev 0.0`, `surface 1`, and `min_val -1` stays an int.
- Duck-typed item with explicit optional values: `SimpleNamespace(..., distribution="gaussian", std_dev=0.3, surface=2)` keeps those exact values.
- Duplicate operands are all kept; there is no de-duplication.
- `FOCUS` (compensator-only) is serialised like any other type: `"type": "Focus"`.

## 7. Rules from the docstring
- The output must round-trip through `deserialize_operands`, so a saved set comes back unchanged. The docstring says it "round-trips exactly instead of being written with no read path".
- The output must be "safe to store in a model's metadata": `json.dumps(result)` succeeds for normal operands.

## 8. Testing hints
- Imports: `from openlens.tolerancing import serialize_operands, deserialize_operands, ToleranceOperand, ToleranceType, tolerances_for_system`.
- Check whole dicts with `==` against literal expected dicts. This catches renamed keys, `.name` used instead of `.value`, and dropped fields.
- To pin the key order, use `list(d.keys()) == ["element_index", "type", "min_val", "max_val", "distribution", "std_dev", "surface"]`.
- Round-trip test: `deserialize_operands(serialize_operands(ops)) == ops`. The dataclass compares fields, and an int bound `1` equals `1.0` after `float()`. Use valid distributions and `surface` ≥ 1 for this, because:
  - an unknown distribution comes back as `"uniform"`
  - `surface=0` comes back as `1`
- Test with a non-default `surface` and `std_dev`, so that hard-coding the defaults would be caught.
- Larger fixture: `tolerances_for_system(system)` builds 12 operands per element plus 1 per gap. The output length must match, and the order must be preserved.
- Pitfalls:
  - Do not expect float conversion: `min_val=1` stays `1`. Python's `1 == 1.0` hides this anyway.
  - Do not expect `None` input to give `[]`; it raises `TypeError`.
  - Do not assert any logging.

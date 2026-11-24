# Notes: `openlens.tolerancing.ToleranceType`

## 1. Purpose
- A plain `enum.Enum` (not `IntEnum`/`StrEnum`) that names each toleranced parameter.
- Units are native per type. The docstring says they are "used as-is, no conversion".
- Several functions use the enum members:
  - as lookup keys: `.value` in (de)serialisation;
  - as labels: `.name` in perturbation keys and sensitivity dicts.

## 2. Members (exact definition order, name → value)
| # | name | value |
|---|------|-------|
| 0 | RADIUS_1 | `"Radius 1"` |
| 1 | RADIUS_2 | `"Radius 2"` |
| 2 | THICKNESS | `"Thickness"` |
| 3 | REFRACTIVE_INDEX | `"Refractive Index"` |
| 4 | DECENTER_X | `"Decenter X"` |
| 5 | DECENTER_Y | `"Decenter Y"` |
| 6 | TILT_X | `"Tilt X"` |
| 7 | TILT_Y | `"Tilt Y"` |
| 8 | DECENTER_Z | `"Decenter Z"` |
| 9 | AIR_GAP | `"Air Gap"` |
| 10 | ABBE_NUMBER | `"Abbe Number"` |
| 11 | IRREGULARITY | `"Surface Irregularity"` |
| 12 | WEDGE | `"Wedge"` |
| 13 | FOCUS | `"Focus"` |

Facts to pin:
- There are exactly **14 members**, and all values are unique, so there are no aliases.
- **DECENTER_Z comes after TILT_Y**, not after DECENTER_Y.
- The IRREGULARITY value is `"Surface Irregularity"`, not `"Irregularity"`.

Related module constant:
- `COMPENSATOR_ONLY_TYPES = frozenset({ToleranceType.FOCUS})`.

## 3. Raises (standard Enum behaviour)
- Unknown value: `ToleranceType("bogus")` raises `ValueError` with message `'bogus' is not a valid ToleranceType`.
- Value lookup is case- and space-exact: `ToleranceType("radius 1")` raises `ValueError`.
- Unknown name: `ToleranceType["BOGUS"]` raises `KeyError`.
- Lookup by name string as a value fails: `ToleranceType("RADIUS_1")` raises `ValueError`, because the value is `"Radius 1"`.
- Members are not strings: `ToleranceType.RADIUS_1 == "Radius 1"` is `False`.

## 4. Guarantees / how it is consumed (all in this module)
- **Round trip by value:** `ToleranceType("Air Gap") is ToleranceType.AIR_GAP`.
- **Round trip by name:** `ToleranceType["WEDGE"] is ToleranceType.WEDGE`.
- **Iteration:** `list(ToleranceType)` returns the 14 members in the table order above.
- **`serialize_operands`:** stores `"type": op.param_type.value`.
  - Example: `"Surface Irregularity"`.
- **`deserialize_operands`:** looks the type up via `{t.value: t for t in ToleranceType}`.
  - A name string such as `"RADIUS_1"` is an *unknown* type, so that entry is skipped.
- **`randomisable_tolerance_types(i, n)`:** returns the members in enum order, minus FOCUS.
  - DECENTER_X is also removed if `i <= 0`.
  - AIR_GAP is also removed if `i >= max(n-1, 0)`.
- **`MonteCarloAnalyzer` perturbation keys:** use `.name`, in the form `f"El_{idx}_{param_type.name}"`.
  - Example: `"El_0_RADIUS_1"`.
- **`MonteCarloAnalyzer` skipped labels:** use `.value`, in the form `f"Element {idx+1} {param_type.value}"`.
  - Example: `"Element 1 Focus"`.
- **`InverseSensitivityAnalyzer.calculate_sensitivities`:** sets `"type": tol.param_type.name`.

## 5. Side effects
None. The members are immutable singletons.

## 6. Branches / edge cases (observable through public functions)
**`randomisable_tolerance_types`:**
- `randomisable_tolerance_types(0, 1)` returns 11 types:
  `[RADIUS_1, RADIUS_2, THICKNESS, REFRACTIVE_INDEX, DECENTER_Y, TILT_X, TILT_Y, DECENTER_Z, ABBE_NUMBER, IRREGULARITY, WEDGE]`.
- `randomisable_tolerance_types(1, 3)` returns 13 types: everything except FOCUS.
- `randomisable_tolerance_types(2, 3)` returns 12 types: AIR_GAP is gone, DECENTER_X is kept.
- `randomisable_tolerance_types(0, 0)`: `max(-1, 0)` is 0, so both DECENTER_X and AIR_GAP are removed. The result has 11 types.
- FOCUS is never in the result.

**`deserialize_operands`:**
- `[{"type": "Focus", "element_index": 0, "min_val": -1, "max_val": 1}]` gives one operand with `param_type is ToleranceType.FOCUS`.
- `{"type": "Irregularity", ...}` is skipped, so the result is `[]`.

**`tolerances_for_system`:**
- Per element, it emits these types in this order:
  `RADIUS_1, RADIUS_2, THICKNESS, REFRACTIVE_INDEX, ABBE_NUMBER, DECENTER_Y, DECENTER_Z, TILT_X, TILT_Y, IRREGULARITY (surface=1), IRREGULARITY (surface=2), WEDGE`.
  That is 12 operands per element.
- It then adds one AIR_GAP operand per gap.
- It never emits DECENTER_X or FOCUS.

## 7. Rules from docstrings/comments
- Units per member:

  | Member(s) | Unit / meaning |
  |---|---|
  | RADIUS_1/2, THICKNESS | mm |
  | REFRACTIVE_INDEX, ABBE_NUMBER | absolute |
  | DECENTER_X | mm, axial despace realised as the gap *before* the element |
  | DECENTER_Y/Z | mm |
  | TILT_X/Y | degrees |
  | AIR_GAP | mm, gap *after* the element |
  | IRREGULARITY | fringes at 632.8 nm |
  | WEDGE | arcmin; applied as tilt X of `value/120` deg |
  | FOCUS | mm image shift; compensator-only, never randomised |

- Comment on `COMPENSATOR_ONLY_TYPES`: the "Add Tolerance dialog must not offer them".

## 8. Testing hints
- Pin the exact `[t.name for t in ToleranceType]` list and the exact `[t.value for t in ToleranceType]` list. A reorder, rename or value change then fails the test.
- Pin `len(ToleranceType) == 14` and `ToleranceType.FOCUS in COMPENSATOR_ONLY_TYPES`. Import the constant: `from openlens.tolerancing import COMPENSATOR_ONLY_TYPES`.
- Check the serialise/deserialise round trip for **every** member:

  ```python
  ops = [ToleranceOperand(0, t, -1.0, 1.0) for t in ToleranceType]
  back = deserialize_operands(serialize_operands(ops))
  assert [o.param_type for o in back] == list(ToleranceType)
  assert serialize_operands(ops)[11]["type"] == "Surface Irregularity"
  ```

- Pin `randomisable_tolerance_types` outputs as exact lists. They depend on enum order.
- For `tolerances_for_system`, pass a real system, e.g. `OpticalSystem()` plus `add_lens(Lens())`. Read `system.elements` and `system.air_gaps` from it; they are defined in `optical_system`, which is not shown here.
- Pitfalls:
  - Do not compare members to strings.
  - Do not assume alphabetical or name-grouped order.
  - Do not use `auto()`/int values.
  - `ToleranceType.FOCUS.value` is `"Focus"`, not `"FOCUS"`.

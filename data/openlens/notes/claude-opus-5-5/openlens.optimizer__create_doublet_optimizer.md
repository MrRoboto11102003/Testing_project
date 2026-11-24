# Notes: `openlens.optimizer.create_doublet_optimizer(system, target_focal_length=100.0)`

## 1. Purpose
- A factory that builds a `LensOptimizer` for a two-element (crown + flint) doublet.
- It uses 3 radius variables and 2 targets: minimize chromatic aberration, and hit a target focal length.
- It runs no optimization. It only constructs the object.

## 2. Inputs
- `system: OpticalSystem`. Must have at least 2 elements.
  - The function reads `system.elements[0].lens.radius_of_curvature_1`, then `elements[0].lens.radius_of_curvature_2`, then `elements[1].lens.radius_of_curvature_2`, in that order.
  - These are the `Lens` property getters (defined in `openlens.lens`, not shown here). Their returned values become the `current_value`s unchanged.
  - Nothing is validated, clamped or copied.
- `target_focal_length: float = 100.0`
  - Stored unchanged as the focal-length target's `target_value`.
  - It is not validated: an int stays an int, and negative values are accepted.

## 3. Raises
- **Empty system:** `IndexError` from `system.elements[0]`.
  - Trigger: `create_doublet_optimizer(OpticalSystem())`.
- **One element:** `IndexError` from `system.elements[1]`.
  - Trigger: an `OpticalSystem` with one `add_lens(Lens())` call.
- The exact message depends on `OpticalSystem.elements`, which is not shown. It is presumably a list, giving "list index out of range".
- Use `pytest.raises(IndexError)` without `match`.
- No `ValueError` is raised here. The target names are valid dispatch keys, so a later `merit_function.evaluate` will not raise "Unknown optimization target".

## 4. Returns and guarantees
- The result is a new `LensOptimizer`:
  - `opt.system is system`: the same object, not a copy.
  - `opt.targets` is the list of 2 targets, and `opt.merit_function.targets is opt.targets`.
  - `opt.merit_function.system is system`.
  - `opt._merit_cache == {}`.
- `opt.merit_function.constraints` is the default dict, because `constraints=None`:
  `{"min_center_thickness": 1.0, "max_center_thickness": 100.0, "min_edge_thickness": MIN_EDGE_THICKNESS, "min_air_gap": 0.1, "min_edge_clearance": 0.1}`.
  - `MIN_EDGE_THICKNESS` comes from `openlens.constants`. Its value is not shown, so compare the other keys only, or import the constant.

`opt.variables` contains exactly 3 `OptimizationVariable`s, in this order (fields: name, element_index, parameter, current_value, min, max, step, linked_targets):

| # | name | elem | parameter | current_value | min | max | step | linked_targets |
|---|---|---|---|---|---|---|---|---|
| 0 | `"R1 Crown"` | 0 | `"radius_of_curvature_1"` | elements[0].lens.R1 | 20.0 | 500.0 | 10.0 | `[]` |
| 1 | `"R2 Crown"` | 0 | `"radius_of_curvature_2"` | elements[0].lens.R2 | -500.0 | -20.0 | 10.0 | `[(1, "radius_of_curvature_1")]` |
| 2 | `"R2 Flint"` | 1 | `"radius_of_curvature_2"` | elements[1].lens.R2 | -500.0 | -20.0 | 10.0 | `[]` |

- There is deliberately no "R1 Flint" variable. The flint's front surface is driven by the link on "R2 Crown".

`opt.targets` contains exactly 2 `OptimizationTarget`s, in this order:
1. `OptimizationTarget("chromatic_aberration", 0.0, weight=1.0, target_type="minimize")`
2. `OptimizationTarget("focal_length", target_focal_length, weight=100.0, target_type="target")`

## 5. Side effects
- None. The system and its lenses are not mutated.
- No global state is touched.

## 6. Branches and edge cases
- There are no branches apart from the `IndexError` cases above.
- A system with more than 2 elements is accepted. Only elements 0 and 1 are referenced.
- The bounds are fixed and do not depend on the starting radii. A starting radius outside the box still yields a variable whose `is_valid(current_value)` is False. Example: with a positive R2 on the flint, `opt.variables[2].is_valid(131.0)` is False.
- Boundary checks:
  - `variables[0].is_valid(20.0)` and `is_valid(500.0)` are True; `is_valid(19.99)` is False.
  - `variables[1].clamp(0.0) == -20.0` and `variables[1].clamp(-1000) == -500.0`.

## 7. Rules from comments
- The crown's back surface is the cemented interface. "R2 Crown" must therefore also write the flint's front surface, through `linked_targets`.
- The flint's R2 bounds are negative because its rear surface is concave. The doublet starts at about -131 mm. This figure comes from a comment and is not verifiable here.

## 8. Testing hints
Build the system with:
```python
from openlens.optical_system import OpticalSystem
from openlens.lens import Lens

s = OpticalSystem()
s.add_lens(Lens(radius_of_curvature_1=60.0, radius_of_curvature_2=-50.0))
s.add_lens(Lens(radius_of_curvature_1=-50.0, radius_of_curvature_2=-150.0), air_gap_before=0.0)

opt = create_doublet_optimizer(s, 80.0)
```
- `openlens.optical_system.create_doublet()` also works as a ready-made input.
- Compute expected `current_value`s from the getters, e.g. `s.elements[0].lens.radius_of_curvature_1`. Do not hard-code the constructor arguments: the `Lens` setter or getter may transform values (that code is not shown).
- Assert every field exactly: names, element indices, parameter strings, min/max/step (note the sign of each), `linked_targets` (a list of one tuple), target names, values, weights, and `target_type` strings. This is where regressions are most likely to show.
- Dataclass equality works, e.g. `opt.targets[1] == OptimizationTarget("focal_length", 80.0, 100.0, "target")`.
- The default `target_focal_length` is 100.0, so `create_doublet_optimizer(s).targets[1].target_value == 100.0`.
- To observe the link without running the optimizer, call the private method:
  - `sysb = opt._apply_variables([60.0, -70.0, -150.0])`.
  - Then `sysb.elements[1].lens.radius_of_curvature_1 == -70.0` and `sysb.elements[0].lens.radius_of_curvature_2 == -70.0`.
  - The original `s` is unchanged, because the method works on a deep copy.
  - Compare with `pytest.approx` in case the getters transform values.
- Avoid calling `opt.optimize()` in tests. It is slow because it traces rays, and its numeric results depend on other modules.
- Do not assume `opt.merit_function.evaluate(s)` returns a specific value. It only returns a float.

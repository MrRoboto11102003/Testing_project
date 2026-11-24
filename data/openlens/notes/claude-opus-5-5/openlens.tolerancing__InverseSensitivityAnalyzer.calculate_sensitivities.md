# Notes: `InverseSensitivityAnalyzer.calculate_sensitivities(criterion="rms_spot_radius")`

## 1. Purpose
- Applies each operand in `self.tolerances` on its own, at its `max_val`, to `self.system`.
- Measures the RMS spot radius for each one and reports how far it moved from the nominal value, along with a sensitivity (change per unit of perturbation).

## 2. Inputs
- `criterion: str`, default `"rms_spot_radius"`.
  - Compared by plain string equality with the class constant `SUPPORTED_CRITERIA == "rms_spot_radius"`. Note this constant is a str, not a set.
- From `__init__(system, tolerances)`:
  - `self.system` is an `OpticalSystem`.
  - `self.tolerances` is a list of `ToleranceOperand`. Nothing in it is validated or copied.
- Only `tol.max_val`, `tol.param_type`, `tol.element_index` and `tol.surface` are used.
  - `min_val`, `distribution` and `std_dev` are **ignored**.

## 3. Raises
- `ValueError` when `criterion != "rms_spot_radius"`. This is checked **first**, before any tracing.
  - Template: `f"Unsupported criterion {criterion!r}; supported: 'rms_spot_radius'"`
  - Example: `an.calculate_sensitivities("mtf")` gives the message `Unsupported criterion 'mtf'; supported: 'rms_spot_radius'`.
  - Use `pytest.raises(ValueError, match="Unsupported criterion 'mtf'")`.
- The nominal trace `SpotDiagram(self.system).trace_spot()` is **not** guarded. Any exception it raises propagates out unchanged.
- Exceptions raised while applying or tracing a single operand are caught. They produce `val = inf` and log the warning `"Inverse sensitivity for tolerance %d failed: %s"`.

## 4. Returns and guarantees
The result is a new `list` with one dict per tolerance, in the same order as `self.tolerances`. Each dict has exactly these 7 keys:

| Key | Value |
|---|---|
| `"operand_index"` | `i` (0-based position in the tolerances list) |
| `"type"` | `tol.param_type.name`, e.g. `"RADIUS_1"`, `"FOCUS"` (the enum *name*, not `.value`) |
| `"element"` | `tol.element_index` |
| `"test_value"` | `tol.max_val` |
| `"result_value"` | `val` (rms_radius after the perturbation; `None`→`inf`; exception→`inf`) |
| `"change"` | `val - nominal` (nominal `None`→`inf`) |
| `"sensitivity"` | `change / max_val`, or the **int `0`** if `max_val == 0` |

- If `change` or `sensitivity` is not finite, **both** are set to `float("nan")`.
  - `result_value` keeps its inf in that case.
- The docstring says the keys are 'operand', 'sensitivity', 'change'. The real keys are the 7 listed above.
- An empty `tolerances` list returns `[]`, but the nominal trace still runs once.

## 5. Side effects
- Each operand mutates `self.system` in place, through the module helper `_apply_value`.
- After each operand the system is restored in a `finally` block via `_restore_state`. The restore covers:
  - every node's position and rotation;
  - each lens's r1, r2, thickness, `model_glass_mode`, `model_nd`, `model_vd` and `refractive_index`;
  - all air-gap thicknesses;
  - then a call to `system._update_positions()`.
- So after the call, for example, `lens.radius_of_curvature_1` equals its original value.
- `REFRACTIVE_INDEX` and `ABBE_NUMBER` operands call `openlens.lens.ensure_model_glass`, which is not visible here. Only the fields listed above are guaranteed to be restored.
- No files are written and no global state is changed.

## 6. Branches and edge cases
What `_apply_value` does with `value = max_val` (its return value is ignored here):

| `param_type` | Effect |
|---|---|
| `RADIUS_1` / `RADIUS_2` | `+= value`; skipped if that radius is infinite (flat) |
| `THICKNESS` | `+= value` |
| `REFRACTIVE_INDEX` | `model_nd += value` (after switching to model glass) |
| `ABBE_NUMBER` | `model_vd += value` (after switching to model glass) |
| `DECENTER_Y` / `DECENTER_Z` | `node.position.y/z += value` |
| `TILT_X` / `TILT_Y` | `node.rotation.x/y += value` |
| `WEDGE` | `rotation.x += value/120` |
| `DECENTER_X` | `air_gaps[idx-1] += value`; skipped when `idx <= 0` |
| `AIR_GAP` | `air_gaps[idx] += value`; skipped when `idx >= len(air_gaps)` |
| `IRREGULARITY` | Radius solved for the fringe error; skipped if the geometry is unrepresentable |
| `FOCUS` | Always skipped |
| Out-of-range `element_index` | Always skipped |

- When a skip happens, the system is unchanged, so `result_value == nominal`, `change == 0.0` and `sensitivity == 0.0`, assuming the trace is deterministic.
  - Example: `ToleranceOperand(0, ToleranceType.FOCUS, -0.1, 0.1)`.
- `max_val == 0`: `sensitivity` is the int `0` and `change` stays finite (≈0).
- Vignetted nominal (`None`): every entry gets `change` and `sensitivity` as nan.
- Vignetted perturbed trace (`None`), or an exception during it: `result_value == inf`, and `change` and `sensitivity` are nan.

## 7. Rules stated in comments
- A non-finite change or sensitivity means "undefined". It is reported as NaN, never as an infinite sensitivity.
- The restore must happen even if the apply or the trace raises, so a design is never left perturbed.

## 8. Testing hints
**Recommended: monkeypatch `openlens.tolerancing.SpotDiagram`** with a fake that returns scripted values. This gives exact, deterministic results. The real `SpotDiagram` outputs depend on `openlens.analysis`, which is not shown here.
```python
import openlens.tolerancing as T
def fake_factory(seq):
    it = iter(seq)
    class Fake:
        def __init__(self, system): pass
        def trace_spot(self, **kw):
            v = next(it)
            if isinstance(v, Exception): raise v
            return {"rms_radius": v}
    return Fake
monkeypatch.setattr(T, "SpotDiagram", fake_factory([1.0, 1.5, None, RuntimeError("x")]))
```
- Each `calculate_sensitivities` call makes 1 nominal trace and then 1 trace per operand, in order.
- With the sequence above and operands with `max_val` 0.5, 0.5, 0.5:
  - entry 0: `change == pytest.approx(0.5)`, `sensitivity == pytest.approx(1.0)`;
  - entries 1 and 2: `result_value == inf`, and `math.isnan` is true for both `change` and `sensitivity`.
- Nominal `None` followed by `[1.2]`: change is nan.
- A nominal that raises: `pytest.raises` with that exception type.
- To show that the criterion check happens first, use a fake whose constructor fails the test. It should never be called when the criterion is unsupported.

**Building a real system:**
- `s = OpticalSystem(); s.add_lens(Lens())`, then `InverseSensitivityAnalyzer(s, ops)`.
- With real tracing, do not hard-code rms numbers. Instead check:
  - key sets;
  - `type`, `element`, `test_value`;
  - `sensitivity == approx(change / test_value)` when the value is finite;
  - that state is restored (radius, thickness, `refractive_index`, `model_glass_mode` all equal to their values before the call).

**Pitfalls:**
- `"type"` is `.name` (`"RADIUS_1"`), not `.value` (`"Radius 1"`).
- `test_value` is `max_val` even when `min_val` is larger in magnitude.
- NaN never compares equal to anything. Use `math.isnan`.
- Do not assert on `nan` versus `inf` for `result_value`: it is inf, and only change and sensitivity become nan.

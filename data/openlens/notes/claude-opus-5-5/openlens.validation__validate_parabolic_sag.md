# Notes: `openlens.validation.validate_parabolic_sag`

## 1. Purpose
- Validates a parabolic-surface sag (vertex-to-rim distance at D/2, in mm) and returns it as a `float`.
- With no `diameter`, it applies a fixed |sag| ≤ 100 cap.
- With a `diameter`, it applies a diameter-scaled bound from `parabolic_sag_limit(diameter, thickness)`.

## 2. Inputs
- `sag`: anything that is a `numbers.Real` and not a `bool` (int, float, `numpy.float32`/`float64`).
  - It is checked by the private `_validate_number(sag, param_name)`.
  - That helper rejects bool, non-Real types, ints too large for a float, NaN and ±inf.
  - It then converts the value with `float()`.
- `diameter`: `Optional`, default `None`.
  - `None` selects the legacy branch.
  - Otherwise it is validated inside `parabolic_sag_limit` under the fixed name `"diameter"` (not `param_name`).
- `thickness`: default `0.0`.
  - Used only when `diameter` is given.
  - Validated inside `parabolic_sag_limit` under the fixed name `"thickness"`.
  - With `diameter=None` it is ignored completely: `thickness=-5` or `thickness="x"` causes no error.
- `param_name`: default `"parabolic sag"`. It appears only in messages about the sag itself.

## 3. Raises
All exceptions are `openlens.validation.ValidationError`. Checks run in this order:

1. **Sag type and finiteness** (always checked first, even if `diameter` is invalid):
   - bool: `"parabolic sag must be a number, got bool"`, e.g. `validate_parabolic_sag(True)`.
   - non-Real: `f"{param_name} must be a number, got {type(value).__name__}"`.
     - `validate_parabolic_sag("1")` gives `"parabolic sag must be a number, got str"`.
     - `None` gives `"... got NoneType"`.
   - too-large int, e.g. `validate_parabolic_sag(10**400)`: `"parabolic sag must be a finite number, got a value too large to represent (int)"`.
   - NaN/inf: `f"{param_name} must be a finite number, got {value}"`.
     - `float('nan')` gives `"... got nan"`.
     - `float('inf')` gives `"... got inf"`.
2. **`diameter is None` branch**: if `abs(sag) > 100`, raises `f"{param_name} magnitude must be at most 100 mm"`.
   - Example: `validate_parabolic_sag(100.5)` gives `"parabolic sag magnitude must be at most 100 mm"`.
3. **`diameter` given**: `parabolic_sag_limit` checks, in this order:
   - `diameter` type (`"diameter must be a number, got bool"` / `"... got str"`, NaN, etc.).
   - `thickness` type, same messages with the name `thickness`.
   - `diameter <= 0`: `f"diameter must be positive, got {diameter}"`, where `diameter` is the float-converted value. `validate_parabolic_sag(1, diameter=0)` gives `"diameter must be positive, got 0.0"`.
   - `thickness < 0`: `f"thickness cannot be negative, got {thickness}"`. `validate_parabolic_sag(1, 10, -1)` gives `"thickness cannot be negative, got -1.0"`.
4. **Out of bound**: if `abs(sag) > limit`, the message is built as:
   ```python
   r = diameter / 2   # uses the ORIGINAL diameter argument
   msg = (f"{param_name} magnitude {abs(sag):.3f}mm exceeds the {limit:.3f}mm limit"
          f" (implies vertex radius {abs(r*r/(2*sag)):.3f}mm)"
          f" for D={diameter}mm")
   ```
   - The "(implies vertex radius …)" part is added when `abs(sag) > 0`. In practice it is always present, because the limit is always > 0.
   - `D=` prints the argument exactly as passed: `diameter=10` gives `"D=10mm"`, while `diameter=10.0` gives `"D=10.0mm"`.

## 4. Returns and guarantees
- Returns `float(sag)` unchanged, never clamped.
  - `validate_parabolic_sag(5)` returns `5.0`, of type `float`.
  - Negative values are returned with their sign: `-3` returns `-3.0`.
- Limit formula, where `MIN_RADIUS_OF_CURVATURE` comes from `openlens.constants` (its value is not visible here):
  ```
  r = D/2
  limit = min(0.5*D, 0.5*thickness + r*r/(2*MIN_RADIUS_OF_CURVATURE))
  ```
- Both comparisons are strict `>`:
  - `|sag| == 100` passes in the legacy branch.
  - `|sag| == limit` passes in the diameter branch.
- The bound is symmetric: `-limit` and `+limit` are both accepted.

## 5. Side effects
- None. No mutation, no logging, no global state.

## 6. Branches and edge cases
- `validate_parabolic_sag(0)` returns `0.0`.
- Legacy branch:
  - `validate_parabolic_sag(100)` returns `100.0`.
  - `validate_parabolic_sag(-100.0001)` raises the "at most 100 mm" error.
- Diameter branch, at the bound:
  - `validate_parabolic_sag(parabolic_sag_limit(10), 10)` returns the limit.
  - `validate_parabolic_sag(parabolic_sag_limit(10) + 0.01, 10)` raises.
- Thickness loosens the bound only while the `0.5*D` term is not the smaller one:
  - `limit(D, t) >= limit(D, 0)`.
  - The limit never exceeds `0.5*D`.
- For large `D`, the limit equals `0.5*D` whenever `r²/(2*MIN_R) >= 0.5*D`.
- A sag above 100 can pass when a large diameter is given, e.g. sag 150 with D=400 if the limit allows it. Verify with `parabolic_sag_limit(400)` before relying on this.
- `numpy.float32(1.5)` is accepted and returned as a Python `float`.

## 7. Rules from docstrings/comments
- Sag without a diameter is "meaningless". The 100 mm cap exists only as a legacy fallback when there is no aperture context.
- The bound keeps the implied vertex radius R = r²/(2·sag) at or above `MIN_RADIUS_OF_CURVATURE`, and caps the rim slope through D/2.

## 8. Testing hints
- Import what you need: `from openlens.validation import validate_parabolic_sag, parabolic_sag_limit, ValidationError`.
- For the expected limit, either call the public `parabolic_sag_limit(D, t)` or compute it from `from openlens.constants import MIN_RADIUS_OF_CURVATURE`. Do not hard-code a numeric limit, because the constant's value is unknown here.
- For a pinpoint regression test, assert the formula directly: `parabolic_sag_limit(D,t) == pytest.approx(min(0.5*D, 0.5*t + (D/2)**2/(2*MIN_RADIUS_OF_CURVATURE)))`.
- Test the boundary with `limit` itself (passes) and `limit*1.001` (raises). Also test the negative sign.
- For error text, prefer `pytest.raises(ValidationError, match=re.escape(...))`.
  - Build the expected message with the same f-string, using `.3f` formatting.
  - Remember the `D=` part shows the raw argument (int vs float).
- To test check order: `validate_parabolic_sag(float('nan'), diameter=-1)` reports the sag NaN error, not the diameter error.
- `ValidationError` is a plain `Exception` subclass, not a `ValueError`. Do not use `pytest.raises(ValueError)`.

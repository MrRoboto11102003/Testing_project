# Notes: `GeometricTraceAnalysis.calculate_field_curvature_distortion`

## 1. Purpose
- Samples field angles from 0 to `max_field_angle_deg` and real-ray-traces each one through `self.system`.
- For each angle it reports:
  - tangential focus shift relative to the image plane,
  - sagittal focus shift relative to the image plane,
  - percent distortion against `efl * tan(angle)`.

## 2. Inputs
- **`max_field_angle_deg`** (float, default `20.0`).
  - The kwarg `max_field_angle=` is accepted as an alias. It is popped from kwargs and **overrides** the positional or default value.
- **`num_points`** (int, default `11`). Angles come from the private `_linspace(0, max, num)`:
  - If `num <= 1` (this includes `0` and negative values), the result is `[0]`. That is the int `0`, not `0.0`.
  - Otherwise `step = max/(num-1)` and each angle is `0 + i*step`. The last value is `(num-1)*step`, which may differ from `max` by float rounding; use `pytest.approx`.
- **`wavelength_nm`** (float, default `WAVELENGTH_GREEN`).
  - Every element's lens gets `lens.update_refractive_index(wavelength_nm=...)` before tracing.
  - The value is also passed to `resolve_image_plane_x`.
- **`**kwargs`**: any key other than `max_field_angle` is rejected.
- **`self.system`**: an `OpticalSystem`. The tracer (`SystemRayTracer3D`) is created in `__init__`.

## 3. Raises
- **`TypeError(f"Unexpected kwargs: {list(kwargs.keys())}")`**
  - Example: `ga.calculate_field_curvature_distortion(foo=1)` gives the message `"Unexpected kwargs: ['foo']"`.
  - With `max_field_angle=5, foo=1`, the message is still `"Unexpected kwargs: ['foo']"`, because the alias is popped first.
  - Use `pytest.raises(TypeError, match=r"Unexpected kwargs: \['foo'\]")`.
  - Order: the wavelength is applied first, then the kwargs check runs. Both happen inside `try`, so lens state is still restored when this error is raised.
- **Empty system** (`OpticalSystem()` with no lenses):
  - `self.system.elements[0].position` raises `IndexError`. This line runs before the loop.
  - `resolve_image_plane_x` (another module, not visible) runs first and might raise something else.
  - Only assert `pytest.raises(Exception)` here, if you test it at all.
- Errors from `Lens.update_refractive_index` for bad wavelengths come from another module and are unknown; do not test them.

## 4. Returns
Returns a new dict with exactly these 7 keys:
```
"field_angles_deg"   -> the angles list
"tan_focus_shift_mm" -> list[float]
"sag_focus_shift_mm" -> list[float]
"tan_focus_shift"    -> SAME list object as tan_focus_shift_mm (alias, `is`)
"sag_focus_shift"    -> SAME list object as sag_focus_shift_mm
"distortion_pct"     -> list[float]
"image_plane_x_mm"   -> float
```
- All five lists have length `len(angles)`. When `num_points >= 2`, that equals `num_points`; otherwise it is 1.
- **Image plane**: `image_plane_x = resolve_image_plane_x(system, wavelength_nm, tracer=self.tracer)`.
  - This returns the configured detector plane if there is one, otherwise the paraxial focus.
  - If it returns `None`, the value is `system.get_total_length() + 10.0`.
- **Per angle `a`**:
  - `tan_theta = tan(radians(a))`.
  - The chief ray starts 50 mm before `elements[0].position` and aims through the point `(elements[0].position, 0, 0)` with direction `(1, tan_theta, 0)` normalised.
  - **Distortion**:
    - `efl = system.get_system_focal_length() or 100.0`, so `None` or `0` both become `100.0`.
    - `paraxial = efl * tan_theta`.
    - If `|paraxial| > 1e-6`: `100*(real_y_at_image_plane - paraxial)/paraxial`.
    - Otherwise: exactly `0.0`.
  - **Tangential focus**:
    - A parallel ray is launched offset by `delta = 0.01 * elements[0].lens.diameter` in y at the first-lens plane.
    - The x-intersection of its 2D (x, y) line with the chief ray's line is computed, and the result is `x_focus - image_plane_x`.
    - If the slopes differ by `<= 1e-9`, the result is `0.0`, not NaN.
    - If the ray terminated, the result is NaN.
  - **Sagittal focus**:
    - The ray is offset by `delta` in z.
    - The result is the x of the chief-ray point at closest approach to the sagittal ray, minus `image_plane_x`.
    - It is NaN if the rays are near-parallel (`|1 - b²| < 1e-12`) or if the ray terminated.

## 5. Side effects
- Each lens's `wavelength` and `refractive_index` are snapshotted before tracing and restored in `finally`, even if an exception is raised.
- After the call, `lens.refractive_index` and `lens.wavelength` equal their pre-call values, including when a non-default `wavelength_nm` was used.
- No files are written. Lens geometry and positions are not mutated.

## 6. Branches and edge cases
- **Chief ray terminated at an angle**: NaN is appended to all three lists for that angle, then the loop continues. Lengths stay equal.
- **`num_points=1`**: `field_angles_deg == [0]`, and `distortion_pct == [0.0]`.
- **`max_field_angle_deg=0, num_points=3`**: angles `[0.0, 0.0, 0.0]`, `distortion_pct == [0.0, 0.0, 0.0]`.
- **Negative max** (e.g. `-10, 3`): angles `[0.0, -5.0, -10.0]`, i.e. descending.
- **Alias**: `calculate_field_curvature_distortion(max_field_angle=10, num_points=3)` gives angles `[0.0, 5.0, 10.0]`.
- **Defaults**: angles are approximately `[0, 2, 4, ..., 20]` (11 values).
- **Angle 0**: distortion is always exactly `0.0`.

## 7. Docstring and comment rules
- "Trace at the requested wavelength (indices restored on exit)."
- `max_field_angle` is a backward-compatibility alias.
- The sagittal focus is measured on the chief ray, not on the z=0 plane.

## 8. Testing hints
Build a system like this:
```python
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.analysis.geometric import GeometricTraceAnalysis
lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
            thickness=5.0, diameter=25.0, material="BK7")
sysm = OpticalSystem(); sysm.add_lens(lens)
ga = GeometricTraceAnalysis(sysm)
```
- The sign convention and the exact focus value come from other modules. **Do not hard-code** focus shifts or `image_plane_x_mm` values.
- Assertions that are safe to make:
  - the key set,
  - list lengths,
  - aliases are identical objects (`res["tan_focus_shift"] is res["tan_focus_shift_mm"]`),
  - the angle values,
  - `distortion_pct[0] == 0.0`,
  - `isinstance(res["image_plane_x_mm"], float)`,
  - index and wavelength are restored after a call with `wavelength_nm=486.1`,
  - index is still restored after the `TypeError`.
- **Image-plane consistency**: for a `Lens` + `OpticalSystem`, call `resolve_image_plane_x(sysm, wl)` yourself and compare with `pytest.approx`. Use the same `wl`, since the default wavelengths match.
- **On-axis symmetry**: the angle-0 tangential and sagittal shifts should agree closely, e.g. `pytest.approx(abs=1e-6)`. This is a reasonable check, but it relies on the rotational symmetry of the tracer.
- Use `math.isnan` for NaN checks, not `==`.
- Avoid huge angles (≥ 90°): `tan` blows up and the outcome is unpredictable.

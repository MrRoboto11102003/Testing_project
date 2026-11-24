# Notes: `SpotDiagram.trace_spot` (openlens/analysis/spot_diagram.py)

## 1. Purpose
- Traces a hexapolar bundle of parallel rays through `self.system` at a given field angle and wavelength.
- Intersects each surviving ray with the plane `X = target_x`.
- Returns the spot points plus centroid, RMS radius and geometric (max) radius.

## 2. Inputs
- `field_angle_x_deg=0.0`: sets the **z** component of the direction (`tan_z = tan(radians(field_angle_x_deg))`). The name is misleading.
- `field_angle_y_deg=0.0`: sets the **y** component (`tan_y`).
- The direction is `vec3(1.0, tan_y, tan_z).normalize()`.
- `wavelength_nm=WAVELENGTH_GREEN`:
  - Converted with `wl_mm = wavelength_nm * NM_TO_MM` and used as the ray wavelength.
  - Every lens gets `lens.update_refractive_index(wavelength_nm=wavelength_nm)` for the duration of the call.
- `image_plane_x_mm=None`:
  - If `None`, calls `resolve_image_plane_x(system, wavelength_nm, tracer=self.tracer)` from `analysis.paraxial` (not shown). It returns the configured detector plane, else the paraxial focus.
  - If that returns `None`, falls back to `system.get_total_length() + 100`.
  - An explicit value is used as-is, with no validation.
- `num_rings=6`: number of hexapolar rings, passed to `generate_hexapolar_grid(rings=num_rings)`.
  - The grid has `1 + 3*n*(n+1)` points: 127 for 6, 7 for 1, 1 for 0.
  - Ring `i` has `6*i` points at radius `(D/2)*(i/n)`.
  - `D` is `system.elements[0].lens.diameter`, the full diameter, not the clear aperture.
- `focus_shift_mm=0.0`: `target_x = image_plane_x_mm + focus_shift_mm`.

## 3. Raises
- **Empty system** raises `IndexError` (from `self.system.elements[0]`, message `list index out of range`).
  - Reliable trigger: `SpotDiagram(OpticalSystem()).trace_spot(image_plane_x_mm=100.0)`.
  - With `image_plane_x_mm=None`, `resolve_image_plane_x` runs first. Its behaviour on an empty system is not visible, so pass an explicit plane.
- Any exception from the tracer or the lens propagates; it is not swallowed. Lens state is still restored (see §5).
- There is no validation of the arguments.

## 4. Returns and guarantees
The result is always a new `dict`.

**Success** (`valid_rays >= 2`) has exactly 6 keys:
- `"rms_radius"`: `sqrt(sum(dy²+dz²)/valid_rays)`, measured from the centroid.
- `"geo_radius"`: `sqrt(max(dy²+dz²))`. Always `geo_radius >= rms_radius >= 0`.
- `"centroid"`: tuple `(mean_y, mean_z)`.
- `"points"`: list of `(spot_y, spot_z)` tuples, in pupil-grid order (centre first, then ring by ring, angle increasing).
- `"valid_rays"`: an int, `== len(points)`.
- `"image_plane_x"`: `target_x`.
- There is no `"error"` key.

Each point is computed as `t = (target_x - ray.origin.x)/ray.direction.x`, then `(origin.y + t*dir.y, origin.z + t*dir.z)`.

**Failure** (`valid_rays < 2`) has 7 keys:
```python
{"rms_radius": None, "geo_radius": None, "centroid": (0.0, 0.0),
 "points": spot_points, "valid_rays": valid_rays,
 "image_plane_x": target_x, "error": "Fewer than 2 rays reached the image plane"}
```
- It also logs a WARNING on logger `openlens.analysis.spot_diagram`.
- The message is `"Only %d ray(s) reached the image plane (blocked or TIR); spot statistics are undefined"`, for example `Only 1 ray(s) ...`.

## 5. Side effects
- During the call, each lens's refractive index is updated for `wavelength_nm`.
- In `finally`, each lens gets `lens.wavelength` and `lens.refractive_index` set back to their original values. This happens even on an exception.
- After the call, `lens.refractive_index` and `lens.wavelength` equal their pre-call values, even when `wavelength_nm=486.1`.
- No files are written and no global state changes.
- `self.tracer` is built once in `__init__`. Build the system completely before constructing `SpotDiagram`.

## 6. Branches and edge cases
- **Ray launch:**
  - Each pupil point `(py, pz)` becomes `origin = (x0, py, pz) - direction*(50/direction.x)`, where `x0 = elements[0].position`.
  - So the origin X is `x0 - 50` and, on axis, `(y, z) = (py, pz)`.
- **Rays not counted:**
  - Rays with `ray.terminated` after `tracer.trace_ray` (vignetted or TIR).
  - Rays with `|direction.x| <= 1e-6`.
  - Rim rays at exactly `D/2` may be clipped by the tracer (not visible). Do not assert `valid_rays == 127`; assert `<= 127` and `== len(points)`.
- **`num_rings=0`:** only the centre ray, so the failure dict is returned. `valid_rays` is ≤ 1 (1 for a normal lens, with `points ≈ [(0,0)]`). `"image_plane_x"` is still `target_x`.
- **Explicit plane:** `image_plane_x_mm=120.0, focus_shift_mm=-2.5` gives `result["image_plane_x"] == 117.5`, on both the success and failure paths.
- **On-axis symmetric lens:** centroid ≈ `(0, 0)`; use `abs < 1e-9`, not `==`.
- **Off-axis `field_angle_y_deg=5` with z field 0:**
  - `centroid[1] ≈ 0`.
  - `centroid[0] != 0`; it is > 0 for a positive lens imaged at or near its focus.
- **Off-axis `field_angle_x_deg`:** moves `centroid[1]`, not `centroid[0]`.

## 7. Rules from comments
- `spot_points` and `valid_rays` are initialised before the `try`, so earlier failures surface as their real cause, not `UnboundLocalError`.
- The `< 2` check sits after the `finally`, never inside it. Tracer exceptions must propagate and must not become `rms_radius == 0.0`.
- Fewer than two rays must report `None` rather than 0.0, so optimisers and Monte Carlo treat it as a failure.

## 8. Testing hints
Build the system:
```python
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.analysis.spot_diagram import SpotDiagram
sys_ = OpticalSystem(); sys_.add_lens(Lens(radius_of_curvature_1=100, radius_of_curvature_2=-100, thickness=5, diameter=25, material='BK7'))
sd = SpotDiagram(sys_)
```

Checks worth asserting:
- Pass an explicit `image_plane_x_mm` for determinism.
  - `paraxial` is another module, and the element position assigned by `add_lens` is not shown.
  - Read the element position as `sys_.elements[0].position`.
- Recompute the centroid, RMS and geo radius from `result["points"]` with the formulas in §4 and compare with `pytest.approx`.
- Check the key sets exactly: 6 keys on success, 7 on failure, with the exact `"error"` string.
- `isinstance(result["centroid"], tuple)`.
- Failure-path checks: `rms_radius is None`, `geo_radius is None`, `centroid == (0.0, 0.0)` (exact).
- Pupil-grid size: `sd.generate_hexapolar_grid(rings=2)` has length 19, using the lens diameter.
- Defocus check: a large `focus_shift_mm` (for example ±20 from the focus) should give a larger `rms_radius` than at the focus. Compare relative values only.
- Use `caplog` at WARNING level for the log message on the `num_rings=0` path.

Pitfalls:
- Do not hard-code `WAVELENGTH_GREEN` or `NM_TO_MM` values, or the default image plane.
- Do not assert exact float spot coordinates.

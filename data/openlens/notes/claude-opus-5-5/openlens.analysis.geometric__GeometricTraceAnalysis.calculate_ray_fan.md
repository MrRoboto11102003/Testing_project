# Notes: `GeometricTraceAnalysis.calculate_ray_fan`

## 1. Purpose
- Computes a transverse ray-aberration fan by real 3D ray tracing through the whole `OpticalSystem` at one wavelength.
- Each error is the ray's image-plane intercept minus the chief ray's intercept, in mm, as a function of normalized pupil coordinate.

## 2. Inputs
- **Constructor:** `GeometricTraceAnalysis(system)` stores `self.system` and builds `self.tracer = SystemRayTracer3D(system)` once, at construction. Configure the system fully before constructing the analyzer.
- **`field_angle_deg`** (float, default `0.0`)
  - Ray direction is `vec3(1, tan(radians(angle)), 0).normalize()`, i.e. the field tilts in the X-Y (tangential) plane.
  - All fan rays share this direction. There is no validation.
- **`wavelength_nm`** (float, default `WAVELENGTH_GREEN`)
  - Each element's lens gets `update_refractive_index(wavelength_nm=...)` before tracing.
  - The trace uses `wavelength_nm * NM_TO_MM`.
  - Validation, if any, happens inside `Lens.update_refractive_index`, which is not visible here.
- **`num_points`** (int, default `21`). Pupil samples come from `_linspace(-1.0, 1.0, num_points)`:
  - If `num <= 1` (including 0 or negative values), the result is `[-1.0]`, a single point.
  - Otherwise `step = 2/(num-1)` and the values are `[-1.0 + i*step for i in range(num)]`.
  - `num_points=3` gives exactly `[-1.0, 0.0, 1.0]`. `num_points=2` gives `[-1.0, 1.0]`.
  - With 21 points, index 10 is exactly `0.0`, but other values such as `-0.9` carry float error. Use `pytest.approx` for them.
- **`pupil_axis`** (str, default `'y'`)
  - `'y'` gives a tangential fan: offsets go along Y and the error is in Y.
  - **Any other value** (`'z'`, `'x'`, `'foo'`) gives a sagittal fan: offsets go along Z and the error is in Z.
  - The value is echoed unchanged in the result.

## 3. Raises
- There is no explicit validation and no `raise` in this module's path.
- **Empty system** (`OpticalSystem()` with no lenses) raises an exception:
  - `elements[0]` would raise `IndexError`.
  - `resolve_image_plane_x` (from another module) runs first and might raise something else.
  - In a test, assert only `pytest.raises(Exception)`.
- **Float `num_points > 1`** (e.g. `2.5`) raises `TypeError` from `range()` in `_linspace`.
- Lens state is restored in a `finally` block, so it is restored even when an exception is raised.

## 4. Returns and guarantees
The result is a new `dict` on every call, with exactly these keys:
- `"pupil_coords"`: the list of pupil coordinates whose ray gave a non-NaN error, in ascending order. It is a subset of the `_linspace` values.
- `"ray_errors_mm"`: list of float errors, the same length as and aligned with `pupil_coords`.
- `"ray_errors"` and `"transverse_aberration"`: the **same list object** as `ray_errors_mm` (`is`-identical).
- `"axis"`: the `pupil_axis` argument as given.
- `"field_angle_deg"`: the argument as given.
- `"wavelength_nm"`: the argument as given, in nm and not converted.

How the geometry is set up:
- `max_r = elements[0].lens.diameter / 2`.
- Each ray passes through `(elements[0].position, p*max_r, 0)` for the y fan, or `(elements[0].position, 0, p*max_r)` for the z fan.
- Rays start at x = `elements[0].position - 50` and are back-propagated along the direction.

How the reference (chief ray) is set up:
- The chief ray passes through `(elements[0].position, 0, 0)`.
- If it is not terminated and `|dir.x| > 1e-6`, its intercept at `image_plane_x` gives `ref_y`/`ref_z`. Otherwise both stay `0.0`.

How `image_plane_x` is chosen:
- `resolve_image_plane_x(system, wavelength_nm, tracer=self.tracer)` returns a configured detector plane if set, else the paraxial focus. That function is in another module.
- If it returns `None`, the fallback is `system.get_total_length() + 10.0`.

Which rays are kept:
- A ray is kept only if it is not terminated and `|ray.direction.x| > 1e-6`.
- Otherwise its error is NaN and the point is **dropped**, not stored as NaN. The result lists therefore never contain NaN.

## 5. Side effects
- The call snapshots `(lens, lens.wavelength, lens.refractive_index)` for every element.
- It then sets the indices for `wavelength_nm`, traces, and in `finally` restores `lens.wavelength` and `lens.refractive_index` to the snapshot values.
- After the call, each lens's `refractive_index` and `wavelength` equal their pre-call values, even when `wavelength_nm` differs from the lens wavelength.
- Ray tracing goes through `self.tracer`, whose internal caching is not visible here. No files or globals are touched.

## 6. Branches and edge cases
- **p = 0 ray equals the chief ray.** It has an identical origin and direction, computed the same way. If it is not terminated, its error is **exactly `0.0`** for any field angle and either axis.
- **Edge rays (p = ±1)** hit exactly the rim (`diameter/2`). Whether the tracer clips them is unseen behaviour. Do not assert `len(pupil_coords) == num_points`; assert `<=`.
- **`num_points=1` or `0`** yields at most one point, `-1.0`.
- **Chief ray terminated** (e.g. a huge field angle): the reference is 0 and errors become absolute intercepts.
- **Any non-`'y'` axis** behaves as the sagittal fan. For example, `pupil_axis='x'` gives the same errors as `'z'`, with `axis == 'x'`.

## 7. Rules from comments/docstrings
- The wavelength argument "controls the physics, not just labels", and the indices are restored on exit.
- `'y'` means tangential (meridional) and `'z'` means sagittal. Pupil coords are normalized to the range −1 to 1.
- The entrance pupil is approximated as the first lens diameter, and the chief ray goes through the first surface centre.

## 8. Testing hints
Build a system like this:
```python
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.analysis.geometric import GeometricTraceAnalysis
sys_ = OpticalSystem(); sys_.add_lens(Lens())   # or create_doublet()
g = GeometricTraceAnalysis(sys_)
r = g.calculate_ray_fan(num_points=11)
```

Robust assertions on current code:
- **Keys and aliases:** the key set is as listed above, and `r["ray_errors"] is r["ray_errors_mm"] is r["transverse_aberration"]`.
- **Echoed metadata:** `r["axis"] == "y"`. `field_angle_deg` and `wavelength_nm` are echoed. Passing `wavelength_nm=486.1` gives back `486.1`.
- **Lengths and order:** `len(pupil_coords) == len(ray_errors_mm) <= num_points`. The coords are a sorted subset of the linspace values. None of the values is NaN.
- **Centre point:** `0.0` is in `pupil_coords` when `num_points` is odd, and its error `== 0.0` exactly. This holds both on-axis and at, e.g., 5°.
- **On-axis symmetry** (rotationally symmetric single lens):
  - `error(p) ≈ -error(-p)`; use `pytest.approx(abs=1e-9)` for any pair that both survive.
  - The y fan equals the z fan elementwise (approx).
- **Off-axis sagittal fan:** it is still odd, `error(p) ≈ -error(-p)`, by mirror symmetry in Z.
- **Different wavelengths:** errors at 486.1 and 656.3 nm differ (not approx-equal) for a singlet. The lens `refractive_index` is unchanged after each call.
- **Inputs not mutated:** lens wavelength and refractive index are unchanged after the call (see section 5).

Pitfalls:
- Do not assert the sign or magnitude of aberrations. They depend on the unseen tracer, sign conventions and focus search.
- Do not rely on edge rays surviving.
- Do not pass `num_points` as a float.
- Do not expect NaN entries in the result.

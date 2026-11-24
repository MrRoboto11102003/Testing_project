# Notes: `SystemRayTracer3D.trace_off_axis_rays(field_angle_deg, num_rays=DEFAULT_NUM_RAYS, wavelength_mm=WAVELENGTH_GREEN*NM_TO_MM)`

## 1. Purpose
- Builds a 1-D meridional fan of parallel rays at a field angle.
- The fan spans the first element's full diameter in the y direction.
- Each ray is traced through the whole system with `self.trace_ray`.
- Returns the traced `Ray3D` objects in fan order.

## 2. Inputs
- `field_angle_deg` (float, required):
  - Converted with `math.radians`.
  - Every ray gets `direction = vec3(cos(a), sin(a), 0.0)`. At 0° this is exactly `(1.0, 0.0, 0.0)`.
  - There is no validation.
- `num_rays` (int):
  - The default is `openlens.constants.DEFAULT_NUM_RAYS`. Its value is not visible here, so import it rather than hard-coding a number.
  - It is used in `range(num_rays)`, so values of 0 or below give an empty list.
- `wavelength_mm` (float):
  - Passed as `Ray3D(..., wavelength=wavelength_mm)`. It is in **mm**, not nm.
  - The default is `WAVELENGTH_GREEN * NM_TO_MM`, from `openlens.constants`.
  - There is no validation.
- Geometry is read from `self.system.elements[0]`:
  - `max_r = elements[0].lens.diameter / 2.0`. Only the **first** lens's diameter matters.
  - `ep_x = elements[0].position`.
- Two more constants come from `openlens.constants`, with values not visible here: `RAY_START_OFFSET_3D_MM` and `RAY_EXIT_PROPAGATION_3D_MM` (used inside `trace_ray`).

## 3. Raises
- Nothing is raised explicitly.
- `field_angle_deg=90` does not raise. `cos` is about 6e-17, so `t` becomes huge but is finite.
- `num_rays` that is not an int (e.g. `2.5`) raises `TypeError` from `range`.

## 4. Returns and guarantees
- Return type is `List[Ray3D]`. Every ray is a new object.
- Length is `max(num_rays, 0)`.
- Fan heights for ray `i`:
  - When `num_rays > 1`: `r_i = -max_r + 2*max_r*i/(num_rays-1)`. This runs from `-max_r` up to `+max_r`, ascending, and the endpoints are included exactly.
  - When `num_rays == 1`: `r = 0`.
- Initial origin, set before tracing:
  - The ray is aimed so its straight line passes through `(ep_x, r_i, 0)`.
  - `t = RAY_START_OFFSET_3D_MM / cos(a)`.
  - `origin = (ep_x, r_i, 0) - direction*t`.
  - That gives `x = ep_x - OFFSET` and `y = r_i - tan(a)*OFFSET` (up to float rounding), with `z = 0`.
- Each ray has already been traced by `self.trace_ray(ray)`, which mutates the ray:
  - `ray.origin` and `ray.direction` are the **final** values, not the start values.
  - Points are appended to `ray.path`.
  - `optical_path_length` is accumulated.
- How a ray ends:
  - A ray that is not terminated gets a final `propagate(RAY_EXIT_PROPAGATION_3D_MM)`.
  - A ray that misses a surface gets `terminated=True` and stays at its last point.
  - In both cases no extra propagation is added after a termination.
- Whether `Ray3D.__init__` seeds `path` with the origin is in `openlens/ray.py`, which is not visible. Don't assume `path[0]`.

## 5. Side effects
- The system is not mutated.
- `self._batch` (the cached scene) may be built or rebuilt on the first trace.
- No files or global state are touched.

## 6. Branches and edge cases
- **Empty system:** `OpticalSystem()` with no lenses gives `[]`. This happens before any trace, and `self._batch` stays `None`.
- **`num_rays=1`:** one ray at `r=0`. On axis at 0° it stays on the axis: final `y == 0`, `z == 0`, direction about `(1,0,0)`.
- **`num_rays=0` or negative:** `[]`. Note this is not the empty-system path; `ep_x` and `max_r` are still read first.
- **Field 0°:**
  - All rays start at the same x.
  - Every ray's z stays exactly `0.0`, because the direction's z and all normal z-components are 0.
  - For a symmetric lens and odd `n`, ray `i` and ray `n-1-i` are mirror images: `y_i ≈ -y_{n-1-i}`, and the middle ray has y about 0.
- **Nonzero angle:**
  - Directions have `y = sin(a)`.
  - Edge rays may exceed the clear aperture because surface sag shifts the hit point, so they can come back `terminated`.
  - Don't assert pass/fail for edge rays at an angle.
- **Edge rays at 0°:** `r = ±max_r` hit the front surface at radius `max_r`. The aperture test is `(ca/2 + 1e-9)^2`, so they pass the front surface, assuming the clear aperture defaults to the diameter (`Lens.get_clear_aperture_1` is not shown).
- **Inside `trace_ray`:**
  - A casing can absorb rays (`ray.absorbed = True`).
  - An aperture stop can vignette rays.
  - Both are optional; without a casing or stop, neither happens.

## 7. Rules from comments / docstring
- "Rays are distributed across the entrance pupil."
- In practice this means the plane `x = elements[0].position` spanning ±(first diameter)/2.
- Stop and pupil size are **not** used for the distribution.

## 8. Testing hints
Build the system:
```python
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.tracer_3d import SystemRayTracer3D
from openlens.ray import Ray3D
from openlens.vector3 import vec3
from openlens import constants as C
lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0, thickness=5.0,
            diameter=25.0, refractive_index=1.5168)
sys_ = OpticalSystem(); sys_.add_lens(lens)
tr = SystemRayTracer3D(sys_)
```

Strong checks:
- **Count:**
  - `len(tr.trace_off_axis_rays(0.0, num_rays=7)) == 7`.
  - The default gives `len == C.DEFAULT_NUM_RAYS`.
  - `num_rays=0` gives `[]`.
  - An empty system gives `[]`.
- **Exact equivalence (most robust):** rebuild each ray by hand and trace it with a second tracer. Then compare `origin`, `direction` and `optical_path_length` with `pytest.approx`.
  ```python
  ep = sys_.elements[0].position
  ray = Ray3D(vec3(ep, r, 0) - d * (C.RAY_START_OFFSET_3D_MM / d.x), d, wavelength=w)
  SystemRayTracer3D(sys_).trace_ray(ray)
  ```
- **Symmetry at 0° with odd n:**
  - The middle ray's final y is about 0 (`abs < 1e-9`).
  - Outer rays mirror each other.
  - Every final `origin.z == 0.0` and `direction.z == 0.0`.
- **Focusing:** a biconvex lens with positive R1 and negative R2 makes the top edge ray's final `direction.y < 0` and the bottom one's `> 0` at 0°.
- **Fan order:** final y or direction changes monotonically with index. The first ray starts at `-max_r`.
- **Wavelength:** `ray.wavelength == wavelength_mm`, if `Ray3D` stores it under that attribute (check `openlens/ray.py`, which is not visible).
- **First diameter only:** adding a second, larger lens doesn't change the start heights. Check this with the equivalence method above.

Pitfalls:
- Read `sys_.elements[0].position` instead of assuming it is 0; how `add_lens` sets the position is in `optical_system.py`.
- Don't hard-code `DEFAULT_NUM_RAYS` or the offset and exit-propagation constants.
- The returned `ray.origin` is the end point after tracing, not the launch point.
- Use `approx` for all floats.

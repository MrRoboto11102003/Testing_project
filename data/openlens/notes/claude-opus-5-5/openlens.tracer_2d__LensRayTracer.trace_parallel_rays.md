# Notes: `LensRayTracer.trace_parallel_rays`

## 1. Purpose
- Builds a fan of `num_rays` parallel (collimated) `Ray`s, all at the same tilt `angle_deg`.
- Traces each ray through the single lens with `self.trace_ray(ray)` and returns the traced rays in fan order.

## 2. Inputs
- `num_rays: int = DEFAULT_NUM_RAYS`
  - The value of `DEFAULT_NUM_RAYS` lives in `openlens.constants`, which is not shown. Always pass `num_rays` explicitly.
  - Used as `range(num_rays)`, so `0` or a negative value gives no rays.
- `ray_height_range: Optional[(min_h, max_h)] = None`
  - `None`: uses `(-D/2*fill, +D/2*fill)`.
  - `D` is `self.D`, the lens's **mechanical** `lens.diameter`. It is *not* the clear aperture.
  - A tuple is unpacked as `min_h, max_h` and is not validated. A reversed range such as `(5, -5)` is allowed and produces a descending fan.
- `wavelength_mm: float = WAVELENGTH_GREEN * NM_TO_MM`
  - Only stored on each `Ray` (`wavelength_mm=` argument).
  - It does **not** change refraction. The tracer always uses `self.n`, fixed in `__init__` from the `wavelength_nm` argument or from `lens.refractive_index`.
- `angle_deg: float = 0.0`
  - Converted with `math.radians`. Every ray starts with `angle = radians(angle_deg)`.
- `fill: float = 1.0`
  - Used **only** when `ray_height_range is None`. It is ignored when a range is given.

## 3. Raises
- The method raises nothing explicitly.
- Unpacking a `ray_height_range` that is not a 2-sequence raises `ValueError`/`TypeError` (Python unpacking error).
- Errors coming from `Ray(...)` or `trace_ray` are defined in other modules, so they are not known here.

## 4. Returns and guarantees
- Returns a new `list` of new `Ray` objects, `len == max(num_rays, 0)`, in index order `i = 0 .. num_rays-1`.
- Height of ray `i`:
  - If `num_rays == 1`: `height = 0`. This holds **even if a custom range is given**: `(2, 4)` gives height 0, not 3.
  - Otherwise: `height = min_h + (max_h - min_h) * i / (num_rays - 1)`. This is linspace including both endpoints.
- Start point of each ray:
  - `start_x = front_vertex_x - RAY_START_OFFSET_MM`, where `front_vertex_x == x_offset`. The constant's value is not shown.
  - `y_start = height - RAY_START_OFFSET_MM * tan(angle_rad)`.
  - So the unrefracted straight line reaches the plane `x = x_offset` at exactly `y = height`. The fan is referenced to the front vertex, not to x = 0. The code comment calls this a bug fix.
- Tracing:
  - Each ray is traced with `trace_ray(ray)` using the default `propagate_distance = DEFAULT_PROPAGATION_DISTANCE` (value in constants, not shown).
- Ray that misses the front surface:
  - Gets `hit=False` and `terminated=True`.
  - It is still propagated by that distance if the distance is > 0.
  - It is still included in the returned list. Nothing is filtered out.
- Ray that passes through:
  - Gets `hit=True`.
  - Its `path` gains the front-surface point and the back-surface point.
  - It is then propagated forward.
- Whether `Ray.__init__` puts the start point into `path` is defined in `openlens.ray`, which is not shown. Do not assert on the exact length of `path` or on `path[0]`.

## 5. Side effects
- The lens and the tracer's state are not mutated.
- Only the newly created rays change.
- No global state is touched.

## 6. Branches and edge cases

| Call | Observable result |
|---|---|
| `num_rays=0` | `[]` |
| `num_rays=-3` | `[]` |
| `num_rays=1`, any range | one ray, starting height 0. At `angle_deg=0` its `y` stays ≈ 0 and its angle stays ≈ 0 through a centred symmetric lens. |
| `num_rays=5`, range `None`, `D=25`, `fill=1` | heights `-12.5, -6.25, 0, 6.25, 12.5` |
| `fill=0.5` with `D=25` | heights span ±6.25 |
| `fill` together with an explicit range | `fill` is ignored |
| range `(-20, 20)` on a `D=25` lens | outer rays miss (`hit is False`, `terminated is True`) and are still returned |

## 7. Rules stated in comments/docstrings
- `fill` defaults to the full aperture so that rim spherical aberration is traced. Callers pass `0.95` explicitly for display fans.
- The fan start plane moves with the front vertex. The comment's example: before the fix, a 0-height 5° ray for a lens at `x_offset=50` arrived at y = 4.3744 instead of on axis.

## 8. Testing hints
Build a symmetric biconvex lens with explicit parameters, and read the index back instead of assuming it:
```python
from openlens.lens import Lens
from openlens.tracer_2d import LensRayTracer
lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0, thickness=5.0,
            diameter=25.0, refractive_index=1.5)
t = LensRayTracer(lens)            # t.D == 25.0; use t.n, not a hard-coded value
rays = t.trace_parallel_rays(num_rays=5, angle_deg=0.0)
```
Sign convention: `front_center = vertex + R1` and `back_center = back_vertex + R2`, so `R1 > 0` with `R2 < 0` is biconvex.

Checks that are robust on the current code:
- **Count and types:** `len(rays) == 5`, and every element is an `openlens.ray.Ray` instance.
- **Symmetry at angle 0:** `rays[i].y ≈ -rays[4-i].y` and `rays[i].angle ≈ -rays[4-i].angle`. The middle ray has `y ≈ 0` and `angle ≈ 0`. Use `pytest.approx(abs=1e-9)`.
- **Convergence:** the top ray (`i=4`, positive start height) ends with `angle < 0`. The bottom ray ends with `angle > 0`.
- **Front-surface heights at angle 0:** a ray stays horizontal until it hits the lens. So for every hit ray, some point in `ray.path` has `y == pytest.approx(expected_height)`, where `expected_height = -12.5 + 6.25*i`.
- **Offset invariance (strongest regression check):**
  - Compare `LensRayTracer(lens, x_offset=0)` and `LensRayTracer(lens, x_offset=50)` with the same arguments, including `angle_deg=5`.
  - The final `y` and `angle` of each ray pair must be equal, and the final `x` values must differ by exactly 50 (`approx(abs=1e-6)`).
  - This only passes because the fan is referenced to the front vertex.
- **Wavelength independence:**
  - Two fans that differ only in `wavelength_mm` (e.g. `0.0005` vs `0.0007`) give identical final `y` and `angle`.
  - Each ray's `wavelength_mm` equals the value passed. Assert with `approx`, because the Ray property may normalise it.
- **Single ray:** `num_rays=1, ray_height_range=(2, 4)` gives a ray whose final `y` matches the `num_rays=1` default-range ray. Both start at height 0.
- **`fill`:**
  - `fill=0.4` gives the same results as an explicit `ray_height_range=(-5, 5)` when D=25.
  - `fill=0.4` combined with `ray_height_range=(-10, 10)` gives the same results as the range alone.
- **Misses:** range `(-20, 20)` with `num_rays=3` gives `rays[0].hit is False`, `rays[2].hit is False`, `rays[1].hit is True`, and `len == 3`.

Pitfalls:
- Do not hard-code `RAY_START_OFFSET_MM`, `DEFAULT_NUM_RAYS` or `DEFAULT_PROPAGATION_DISTANCE`. Their values are not shown. Import them from `openlens.constants` if you need them.
- Do not assert exact final `x` values or exact path lengths. Use relative comparisons such as symmetry, offset differences and equality between two runs.

# Notes: `openlens.ray.Ray3D.propagate(self, distance: float) -> None`

## 1. Purpose
- Moves the ray's origin forward by `distance` along its current unit `direction`.
- Records the new point in `path` and adds `distance * n` to `optical_path_length`.
- A zero or negligible distance does nothing.

## 2. Inputs
- `distance`: a float or int, in mm. There is no validation or type check.
  - Negative values are accepted. The ray moves backwards and the optical path length decreases.
- The method also reads this ray state:
  - `self.direction`: a `Vector3`. `Ray3D.__init__` stores `direction.normalize()`, so it is a unit vector unless reassigned.
  - `self.n`: the refractive index at call time. The default is `REFRACTIVE_INDEX_AIR`, whose value is in `openlens.constants` and not shown here (presumably 1.0).
- `EPSILON` comes from `openlens.constants`. Its numeric value is not visible here. Do not write tests that depend on its exact size.

## 3. Raises
- No explicit raises.
- A non-numeric `distance` (e.g. `"a"`) fails inside `abs()` with `TypeError`. Do not test this.

## 4. Returns and guarantees
- Returns `None`.
- When `abs(distance) > EPSILON`:
  - `self.origin` is replaced by `self.origin + self.direction * distance`. This is a new `Vector3`, assuming `Vector3` arithmetic returns new objects (that module is not shown).
  - The new origin object is appended to `self.path`, so `ray.path[-1] is ray.origin`.
  - `self.optical_path_length += distance * self.n`.
- These are untouched: `direction`, `n`, `intensity`, `wavelength`, `terminated`, `absorbed`, `polarization_vector`.

## 5. Side effects
- Only mutates the ray's `origin`, `path` and `optical_path_length`.
- The `Vector3` originally passed to the constructor is not modified. It stays as `path[0]`, because `path` starts as `[origin]` with the same object.
- No global state or file I/O.

## 6. Branches and edge cases
- **No-op branch** (`abs(distance) <= EPSILON`): `propagate(0)`, `propagate(0.0)` and `propagate(-0.0)` all return immediately.
  - `origin` is unchanged (same object).
  - `len(path)` is unchanged (still 1 on a fresh ray).
  - `optical_path_length` stays `0.0`.
- **Normal branch**, positive distance. For `r = Ray3D(vec3(0,0,0), vec3(1,0,0))`, calling `r.propagate(10.0)` gives:
  - `r.origin` = (10, 0, 0)
  - `len(r.path) == 2`
  - `r.optical_path_length == 10.0`
- **Direction is unit length.** For `Ray3D(vec3(0,0,0), vec3(2,0,0))`, calling `propagate(5)` moves the origin to x = 5 (not 10).
  - For direction `vec3(3,4,0)`, calling `propagate(5)` gives origin (3, 4, 0).
- **Uses the current n.** For `Ray3D(vec3(0,0,0), vec3(0,0,1), n=1.5)`, calling `propagate(2.0)` gives:
  - origin (0, 0, 2)
  - `optical_path_length == 3.0`
- **Negative distance.** For a fresh x-directed ray, calling `propagate(-4.0)` gives:
  - origin (-4, 0, 0)
  - `optical_path_length == -4.0` (times n)
  - the path gains one point
- **Accumulation.** Calling `propagate(1)` then `propagate(2)` on an x ray gives:
  - origin x = 3
  - `path` x-values `[0, 1, 3]`
  - `optical_path_length == 3 * n`
  - A `propagate(0)` call in between adds no path point.
- **Changing n between calls.** Set `r.n = 2.0` after the first `propagate(1)` (with n = 1), then call `propagate(1)`. The total optical path length is 3.0.

## 7. Rules from the docstring
- "A zero (or negligible) distance is a no-op rather than a recorded step."
- The reason given: consecutive identical path points break consumers that derive a direction or segment length from consecutive points.
- So `path` must never gain a duplicate point from a zero-length move.

## 8. Testing hints
- Build rays like this: `from openlens.ray import Ray3D` and `from openlens.vector3 import vec3, Vector3`. Call it as `Ray3D(vec3(x,y,z), vec3(dx,dy,dz), n=...)`.
- Read coordinates via `.x`, `.y`, `.z` or `.to_tuple()`.
  - Compare with `pytest.approx`, because normalisation introduces float error (e.g. direction (3,4,0)/5).
- Choose distances well clear of `EPSILON`, whose value is unknown.
  - Use `0` / `0.0` for the no-op case.
  - Use values of 1e-3 or more for real moves.
  - Avoid tiny values like 1e-12.
- Assert the exact `len(ray.path)` both after no-op calls and after real calls. This pins down the no-op rule.
- Assert `ray.path[-1]` coordinates equal `ray.origin`, and `ray.path[0]` equals the start point.
- Assert `optical_path_length` is exactly `distance * n` for a fresh ray. Use n values like 1.5 or 2.0, so a dropped or replaced factor of n is detected.
- Assert that `direction`, `n` and `intensity` are unchanged afterwards.
- Do not rely on the default `n` being exactly 1.0. Pass `n=` explicitly.

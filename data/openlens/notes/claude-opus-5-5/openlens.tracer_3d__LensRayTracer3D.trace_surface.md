# Notes: `LensRayTracer3D.trace_surface(ray, surface_type, interaction="refract")`

## 1. Purpose
- Finds where a `Ray3D` meets one surface ("front" or "back") of a lens.
- Moves the ray to that point and adds the optical path length for the segment.
- Then refracts or reflects the ray in place.
- Returns a `RefractionResult` (`openlens.ray`).

## 2. Inputs
- `self`: a `LensRayTracer3D(lens, transform=None, x_offset=0.0)`.
  - Lens values are **snapshotted at construction**: R1/R2 (`get_effective_radius_*`), thickness `d`, diameter `D`, `n=lens.refractive_index`, parabolic/conic flags, clear apertures, Fresnel facets.
  - Editing the lens afterwards does not affect `trace_surface`.
  - With no transform, the front vertex is at world `(x_offset,0,0)` and the back vertex at `(x_offset+d,0,0)`. The optical axis is +x.
- `ray`: a `Ray3D(origin: Vector3, direction: Vector3, ..., n=REFRACTIVE_INDEX_AIR)`. It is mutated.
- `surface_type`: only `"front"` or `"back"`. Any other string returns `MISSED` immediately.
- `interaction`:
  - `"refract"` (default) or `"reflect"`.
  - Any other value is accepted but returns `MISSED` *after* the ray has already been moved (see §6).

**How the surface kind is chosen** (in `_calculate_geometry`; the order matters):
1. Parabolic: `is_parabolic_k` and `|parabolic_sag_k| > EPSILON`.
2. Flat: `_is_flat(R)` (defined in `openlens.lens`, not shown; presumably an infinite or huge radius).
3. Conic: `|conic_constant_k| > EPSILON`.
4. Otherwise a sphere.
   - Front centre is at local `x=R1`; back centre is at local `x=d+R2`.
   - The cap nearest the vertex is used (`abs(R)`, convex flag `R>0`).

## 3. Raises
- No explicit `raise`.
- Errors could only come from helpers in other modules (`Ray3D.reflect`, `refract_or_reflect`, `OpticalIntersector`). Do not test for exceptions.

## 4. Returns and guarantees
- `RefractionResult.MISSED` in the following cases. In all of them the ray is **unchanged** (origin, direction, n, OPL, path):
  - bad `surface_type`;
  - no intersection;
  - the hit lies outside the aperture;
  - the Fresnel facet intersection is None.
- Aperture test:
  - The radial distance squared, measured perpendicular to the axis from the vertex, must be `<= (CA/2 + 1e-9)**2`.
  - `CA` is `lens.get_clear_aperture_k()`, falling back to `D` when it returns falsy.
  - Hit exactly at `CA/2` passes. A hit at radius `> CA/2` is a miss.
- On a hit, in this order:
  1. `ray.optical_path_length += |hit - old_origin| * ray.n`, using the `n` from *before* refraction.
  2. `ray.origin = hit`.
  3. `hit` is appended to `ray.path` only if the path is empty or the ray moved by more than `EPSILON`. A zero-length hit is not recorded again, but it is still refracted.
- Choice of `(n1, n2)`. Defaults: front is `(air, lens.n)`, back is `(lens.n, air)`.
  - If `|ray.n - default_n1| < 1e-3`, use the defaults.
  - Else if `|ray.n - default_n2| < 1e-3`, swap them: `(default_n2, default_n1)`.
  - Otherwise use `(ray.n, default_n2)`.
- `"reflect"`: calls `ray.reflect(normal, n1, n2)` and always returns `REFLECTED`.
- `"refract"`: returns whatever `ray.refract_or_reflect(n1, n2, normal)` returns. That is in `openlens.ray`; expect `REFRACTED`, or `REFLECTED` on total internal reflection.
- Normals are always oriented outward: the front normal is about `-axis`, the back normal about `+axis`.
- Never sets `ray.terminated`.

## 5. Side effects
- Mutates `ray`: origin, OPL, path, and direction/n through the refraction or reflection call.
- The tracer and the lens are not changed.

## 6. Branches and edge cases
Shared test lens:
`lens = Lens(radius_of_curvature_1=50, radius_of_curvature_2=-50, thickness=5, diameter=25, refractive_index=1.5)`
`t = LensRayTracer3D(lens)`

- **On-axis front hit.**
  - Call: `ray = Ray3D(vec3(-10,0,0), vec3(1,0,0))`, then `t.trace_surface(ray,"front")`.
  - Returns `REFRACTED`; origin is `(0,0,0)`; OPL increases by `10.0` (n=1).
  - Direction stays about `(1,0,0)` (normal incidence).
- **Back hit after that.**
  - Origin `(0,0,0)` going +x hits `(5,0,0)` (back sphere centre `-45`, radius `50`).
  - OPL increases by `5*ray.n`. If `refract_or_reflect` set `n=1.5`, that is `+7.5`.
  - Since `ray.n≈1.5`, `(n1,n2)=(1.5,1.0)`.
- **Off-axis sphere hit.**
  - Ray from `(-10,5,0)` going +x hits the front at `x = 50 - sqrt(2475) ≈ 0.2506281`, `y=5`.
  - OPL increases by about `10.2506281`.
- **Aperture miss.**
  - Ray from `(-10,13,0)` going +x: the sphere is hit (|y|<50), but `169 > 12.5²`, so `MISSED`.
  - Origin stays `(-10,13,0)` and OPL is unchanged.
- **Sphere missed entirely.**
  - Ray at `y=60` going +x returns `MISSED`.
- **Invalid surface type.**
  - `t.trace_surface(ray, "middle")` returns `MISSED` with nothing changed.
- **Unknown interaction.**
  - `t.trace_surface(ray, "front", "absorb")` returns `MISSED`.
  - However, origin is now `(0,0,0)` and OPL has increased by 10. This is a deliberate observable quirk.
- **Reflect.**
  - On-axis front hit with `"reflect"` returns `REFLECTED`.
  - Direction should become about `(-1,0,0)`; this relies on `Ray3D.reflect` being standard mirror reflection.
- **Zero-length hit.**
  - A ray whose origin is already at the vertex, with a non-empty path: the path length does not grow and OPL increases by 0.
  - The ray is still refracted.
- **x_offset.**
  - `LensRayTracer3D(lens, x_offset=100)` puts the front hit at `(100,0,0)`.
- **Flat surface.**
  - The surface is a plane through the vertex.
  - A ray parallel to it, e.g. direction `(0,1,0)` (|denom|<EPSILON), returns `MISSED`.
  - A plane hit behind the ray (`t < -EPSILON`) returns `MISSED`.
- **Parabolic surface with effective R that is ~0 or > 1e10.**
  - Treated as a plane.
- **Fresnel lens (facets non-empty).**
  - Only facet hits count. If no facet is hit, the result is `MISSED`.
  - Facet construction is in `openlens.geometry` (not shown), so avoid exact values.

## 7. Rules from comments
- "A zero-length hit (t ~ 0) is legitimate and must still refract" (cemented doublets), but the point must not be recorded twice in `path`.
- A no-intersection result must be `MISSED` with the state unchanged. The ray must never be faked into a transmitted ray.
- OPL is accumulated per segment using the current medium's `n`.
- Normal convention: the front normal points to −axis and the back normal to +axis, whatever the sign of the curvature.

## 8. Testing hints
- Imports:
  - `from openlens.tracer_3d import LensRayTracer3D`
  - `from openlens.ray import Ray3D, RefractionResult`
  - `from openlens.vector3 import vec3`
- Compare floats with `pytest.approx(abs=1e-6)`. Compare results with `is` / `==` on the enum.
- Record `len(ray.path)` before the call and compare deltas. The initial contents of `Ray3D.path` are defined in `openlens.ray` and are probably `[origin]`.
- Pass `refractive_index=` explicitly. Lens defaults (type, material index) come from code not shown here.
- Safest assertions:
  - return values;
  - `ray.origin` coordinates;
  - the OPL delta on the front surface (n=1);
  - that the state is unchanged on `MISSED`.
- Assertions that depend on the `openlens.ray` implementation, so keep them loose:
  - `ray.n` after refraction;
  - exact refracted directions on oblique rays.
- Do not test backward-travelling rays. The result depends on `rank_cap_roots` (not shown).

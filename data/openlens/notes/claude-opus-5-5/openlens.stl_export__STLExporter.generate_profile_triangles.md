# Notes: `STLExporter.generate_profile_triangles(self, profile, num_segments=100)`

## 1. Purpose
- Revolves a meridional rim profile, a list of `(radius, z)` points ordered front rim to back rim, around the Z axis.
- Returns the triangle mesh as a list.
- `export_lens_stl` and `export_lens_to_stl(lens=...)` use it to build the lens edge from `LensGeometry.edge_profile_points`.

## 2. Inputs
- `profile`: a sequence of 2-item `(r, z)` pairs.
  - It must support `len()` and slicing (`profile[1:]`). Lists and tuples work; a generator raises `TypeError`.
  - Each item is unpacked as `r, z`, so lists like `[5, 0]` also work.
  - No validation of values. Negative `r` is used as-is.
- `num_segments`: int, default `100`. It is the number of angular points around the full revolution.
  - It is not validated beyond the `< 3` check.
  - A float such as `4.0` fails in `range()` with `TypeError`.
- `self` is not used. Any `STLExporter()` instance works.

## 3. Raises
- The function raises no exceptions of its own.
- Incidental errors:
  - `TypeError` for a non-sliceable or unsized `profile`, or a non-int `num_segments` (when it is ≥ 3).
  - `ValueError` from tuple unpacking if a point does not have exactly 2 items.

## 4. Returns and guarantees
- Returns a new `list` of triangles. Each triangle is a tuple of 3 points, each point is a tuple `(x, y, z)` of floats.
- Precomputed angles: `angles = [2*pi*i/num_segments for i in range(num_segments)]`.
  - `angles[0] == 0.0` exactly.
  - The wrap `angle2 = angles[(index+1) % n]` makes the last segment end at exactly `0.0`. This closes the seam: the last segment's `p2` is identical to the first segment's `p1` ring point.
- Loop order: the outer loop runs over consecutive profile pairs `(r1, z1), (r2, z2)`; the inner loop runs over the angle index `0..n-1`.
- Each angle index appends 2 triangles, in this order:
  - `p1 = (r1*cos a1, r1*sin a1, z1)`
  - `p2 = (r1*cos a2, r1*sin a2, z1)`
  - `p3 = (r2*cos a2, r2*sin a2, z2)`
  - `p4 = (r2*cos a1, r2*sin a1, z2)`
  - The triangles are `(p1, p2, p3)` then `(p1, p3, p4)`.
- Length of the result is `2 * num_segments * (number of non-degenerate consecutive pairs)`.
- z values are copied verbatim; there is no offset.

## 5. Side effects
- None. It does not touch `self.triangles` and does not mutate `profile`. No file I/O.

## 6. Branches and edge cases
- `len(profile) < 2` returns `[]`. Examples: `[]` and `[(5, 0)]`.
- `num_segments < 3` returns `[]`. Examples: `2`, `0`, `-1`.
  - `num_segments == 3` works and gives 6 triangles per pair.
- Degenerate pair: skipped only if `abs(r1-r2) <= EPSILON` **and** `abs(z1-z2) <= EPSILON`.
  - `EPSILON` comes from `openlens.constants` (not shown), so its exact value is unknown. Use clearly equal or clearly different values, e.g. differences of 0 or ≥ 0.1.
  - `[(5, 0), (5, 0), (5, 3)]` gives `2*n` triangles; the first pair is skipped.
  - `[(5, 0), (5, 0)]` gives `[]`.
- A pair with the same `r` but different `z` is a cylinder band and is not skipped.
- A pair with the same `z` but different `r` is a flat annulus and is not skipped.
- `r1 == r2 == 0` with different `z` is not skipped. It emits zero-area triangles, all with x = y = 0.
- Default `num_segments=100` with a 2-point profile gives 200 triangles.

## 7. Rules from docstrings and comments
- The docstring says: "The straight two-point profile is the old cylinder."
  - For example, `[(D/2, z_front), (D/2, z_back)]` reproduces a cylindrical barrel.
- Through the callers, the winding gives radially outward normals when `z` increases along the profile at constant `r`.
- Example: `[(5, 0), (5, 3)]` with `n=4`. The first triangle's `calculate_normal` is ≈ `(0.7071, 0.7071, 0.0)`, which is outward at the 45° mid-segment.
  - In general, with z increasing, `normal · (cos(mid), sin(mid), 0) > 0` for the midpoint angle `mid`.

## 8. Testing hints
- Construct the object with `from openlens.stl_export import STLExporter; e = STLExporter()`.
- Exact checks for `tris = e.generate_profile_triangles([(5.0, 0.0), (5.0, 3.0)], num_segments=4)`:
  - `len(tris) == 8`.
  - `tris[0][0] == (5.0, 0.0, 0.0)`, exact, because angle 0.
  - `tris[0][1]` is ≈ `(0, 5, 0)`; use `pytest.approx(abs=1e-12)`, since `cos(pi/2)` is `6e-17`, not 0.
  - `tris[1] == (tris[0][0], tris[0][2], p4)` with `p4 == (5.0, 0.0, 3.0)`, exact.
  - All z values are in `{0.0, 3.0}`.
  - All points satisfy `hypot(x, y) ≈ 5`.
- Seam closure, exact equality: the last segment's `p2` equals `tris[0][0]`.
  - `tris[-2][1] == tris[0][0]` holds exactly, because `cos(0)=1.0` and `sin(0)=0.0` are exact.
- Multi-pair ordering:
  - Profile: `[(5, 0), (5, 3), (7, 3)]` with `n=4` gives 16 triangles.
  - The first 8 have z only in `{0, 3}` and r = 5.
  - The last 8 have z = 3 and radii in `{5, 7}`.
  - `tris[8][0] == (5.0, 0.0, 3.0)`.
- Triangle count formula to parametrize: `2 * n * pairs`.
  - Example: a 4-point non-degenerate profile with `n=6` gives 36.
- Use `e.calculate_normal(*tri)` to check outward orientation.
  - Avoid degenerate (zero-area) triangles there; `Vector3.normalize` behaviour on a zero vector lives in another module and is not shown.
- Pitfalls:
  - Compare non-zero-angle coordinates with `approx`, not `==`.
  - Do not rely on the EPSILON boundary.
  - The result contains tuples, not lists.
  - The return value does not alias `profile`.

# Notes: `STLExporter.generate_edge_triangles(diameter, z_front_arr, z_back_arr, num_segments=60)`

## 1. Purpose
- Builds the triangles of the cylindrical side wall (barrel) of a lens. The wall runs between a front rim ring and a back rim ring, both at radius `diameter/2`.
- Returns a plain list of triangles. It does **not** write a file and does **not** touch `self.triangles`.

## 2. Inputs
- `diameter`: number. `radius = diameter / 2.0`. There is no validation: 0 or negative values are accepted and used as-is.
- `z_front_arr` / `z_back_arr`: the Z of the front and back edge rings. Each can be:
  - **a scalar**: the same z is used for every vertex on that ring. The value is put into the tuples unchanged, so an `int` stays an `int`.
  - **a `list`**: its length must be `num_segments` or `num_segments + 1`.
  - The check is `isinstance(values, list)`. A tuple or numpy array is therefore **not** treated as per-segment. It goes down the scalar path and the whole object becomes the z coordinate. No error is raised.
- `num_segments`: int, default 60. `angle_step = 2*pi/num_segments`.

## 3. Raises
- **List length check.** It runs first, front list before back list.
  - Exception: `ValueError(f"{name} must have {num_segments} or {num_segments + 1} values, got {len(values)}")`, where `name` is the literal `"z_front_arr"` or `"z_back_arr"`.
  - Example: `gen(10, [0,0], 5, num_segments=4)` raises `"z_front_arr must have 4 or 5 values, got 2"`.
  - Example: `gen(10, 0, [1,2,3], num_segments=4)` raises `"z_back_arr must have 4 or 5 values, got 3"`.
  - If both lists are bad, the front one is reported.
- **Zero segments.** `num_segments=0` gives `ZeroDivisionError` from `2*math.pi/0`. This happens after the list check, so `[]` passes that check when n = 0.
- **Non-int segment count.** A float such as `num_segments=4.0` gives a `TypeError` from `range`.
- **Negative segment count.** A negative `num_segments` raises nothing and returns `[]`, because the range is empty.

## 4. Returns and guarantees
- The result is a new `list` of exactly `2 * num_segments` triangles. Each triangle is a 3-tuple of 3-tuples `(x, y, z)`.
- Loop order is `seg = 0..n-1`. For each segment:
  - `angle1 = seg*step`.
  - `angle2 = ((seg+1) % n) * step`. For the last segment this is exactly `0.0`, so it uses `cos = 1.0` and `sin = 0.0` exactly, which is a watertight seam.
- The four corners of each segment are:
  - `p1 = (r*cos1, r*sin1, zf1)`
  - `p2 = (r*cos2, r*sin2, zf2)`
  - `p3 = (r*cos2, r*sin2, zb2)`
  - `p4 = (r*cos1, r*sin1, zb1)`
- They are appended in this order: `(p1, p2, p4)`, then `(p2, p3, p4)`.
- The z values come from:
  - `zf1 = z_front[seg]`
  - `zf2 = z_front[(seg+1) % len(z_front)]`
  - The same pattern applies to the back ring.
  - With a scalar, all four z values equal that scalar.
- The list length changes which value closes the ring:
  - **Length n (closed):** the last segment's `zf2` wraps to index 0.
  - **Length n+1 (open):** the last segment's `zf2` is index `n`, the last element, even though its x/y equal those of segment 0's start.
- **Normals.** When back z > front z, the normals point radially outward, at the segment's mid-angle. This is checkable with `exporter.calculate_normal(*tri)`.
  - Example: `n=4`, `d=10`, front 0, back 5. Triangle 0 has normal ≈ `(0.7071, 0.7071, 0)`. Triangle 1 has the same normal.
  - If front z > back z, the normals point inward. The function does not check this.

## 5. Side effects
- None. Inputs are not mutated, `self.triangles` is unchanged, and no I/O happens.

## 6. Branches and edge cases
- Scalars `gen(10, 0.0, 5.0, num_segments=4)` give 8 triangles:
  - Triangle 0 = `((5.0,0.0,0.0), (5*cos(pi/2), 5.0, 0.0), (5.0,0.0,5.0))`. Here `5*cos(pi/2)` ≈ 3.06e-16, which is not 0, so use `approx`.
  - Triangle 7 = `(p2, p3, p4)` of seg 3, with `p2 = p3[:2]+... = (5.0, 0.0, 0.0)` / `p3 = (5.0, 0.0, 5.0)` exactly.
- Closed list `gen(10, [0,1,2,3], 5, 4)`:
  - Seg 3 has `p1.z == 3` and `p2.z == 0` (wraps).
- Open list `gen(10, [0,1,2,3,9], 5, 4)`:
  - Seg 3 has `p2.z == 9`. Index 4 is used, not index 0.
- `num_segments=1`: `angle1 = angle2 = 0`, so the result is 2 degenerate triangles, all at x = r and y = 0.
- `diameter=0`: every x and y is 0. The result is still `2n` triangles.

## 7. Rules from comments and docstring
- The last segment must wrap to an exact 0 angle. `sin(2*pi)` is about 6.4e-16, and using it would leave the barrel open.
- The edge normals must point radially outward. This is why the winding is `(p1,p2,p4)` and `(p2,p3,p4)`.
- Allowed list lengths: `n` is a closed profile and `n+1` is an open profile.

## 8. Testing hints
- Setup: `e = STLExporter(); tris = e.generate_edge_triangles(10.0, 0.0, 5.0, num_segments=4)`.
- **Count:** `len(tris) == 2*n`. Also check that `e.triangles == []` afterwards.
- **Radius:** every vertex has `hypot(x, y) ≈ diameter/2`.
- **Exact z:** every vertex z is in `{zf, zb}`.
- **Order:** check `tris[0][0] == tris[1][2]` (p4 is shared at the end of both). Also check `tris[0][1] == tris[1][0]` (p2 is shared).
- **Seam:** `tris[-1][0][:2] == (10/2, 0.0)` holds exactly, because the last segment's p2 x/y is `(r*1.0, r*0.0)` = `(5.0, 0.0)`. The first triangle's p1 is the same exact value.
- **Normal direction:** for each triangle, the dot product of `calculate_normal` with the radial vector at the triangle's centroid should be > 0.
- **Error messages:** match the full text with `pytest.raises(ValueError, match=r"^z_front_arr must have 4 or 5 values, got 2$")`.
- **Pitfalls:**
  - Do not expect exact zeros at angles π/2 or π. Use `pytest.approx(abs=1e-12)` for any x/y other than those at angle 0.
  - A tuple passed for z does not raise.
  - Integer z inputs stay `int`. `==` with floats still works.

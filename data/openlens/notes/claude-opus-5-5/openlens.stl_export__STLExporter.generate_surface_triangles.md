# Notes: `STLExporter.generate_surface_triangles(profile_points, z_offset, num_segments=60, is_front=True)`

## 1. Purpose
- Revolves a half-profile of `(radius_from_axis, z)` points around the Z axis into a triangle list (a surface cap).
- Each triangle is a 3-tuple of `(x, y, z)` float tuples.
- The winding order depends on `is_front`, so the normals point out of the glass:
  - front cap: normals along −Z;
  - back cap: normals along +Z.

## 2. Inputs
- **`profile_points`**: a sequence supporting `len()` and indexing, such as a list or tuple of `(r, z)` pairs, ordered from the axis to the rim.
  - The emptiness check is `if not profile_points:`, so `[]` and `None` both return `[]`.
  - A numpy array with more than one element makes `not arr` raise `ValueError` (ambiguous truth value). Use lists.
- **`z_offset`**: a float added to every profile z. Vertex z = `z_i + z_offset`.
- **`num_segments`**: an int, default 60. It sets the number of angular segments, with `angle_step = 2*pi/num_segments`.
- **`is_front`**: a bool, default `True`. It only changes the vertex order inside each triangle.
- **No validation is done.** It is not checked that `r` is non-negative or that the points are sorted.

## 3. Raises
- `num_segments=0` with a non-empty profile raises `ZeroDivisionError` (`2*math.pi/0`).
  - With an empty profile, the early `return []` happens first, so no error is raised.
- A float `num_segments` (e.g. `4.0`) with a non-empty profile raises `TypeError` from `range()`.
- A negative `num_segments` gives an empty `range`, so it returns `[]` and does not raise.

## 4. Returns and guarantees
- The result is a new list. `self.triangles` is **not** touched.

**Loop order:**
- The outer loop is `seg` in `0..num_segments-1`.
- The inner loop is the radial index `i` in `0..len(profile)-2`, over the pair `(profile[i], profile[i+1])`.

**Angles:**
- `angle1 = seg*step`.
- `angle2 = ((seg+1) % num_segments)*step`.
- For the last segment `angle2 == 0` exactly, so `cos2 = 1.0` and `sin2 = 0.0`. Its p3/p4 equal segment 0's p2/p1 exactly; the seam is watertight.

**Points per quad**, using `(r1,z1)=profile[i]` and `(r2,z2)=profile[i+1]`:

```
p1=(r1*cos1, r1*sin1, z1+off)
p2=(r2*cos1, r2*sin1, z2+off)
p3=(r2*cos2, r2*sin2, z2+off)
p4=(r1*cos2, r1*sin2, z1+off)
```

**Triangles per pair:**

| Case | `is_front=True` | `is_front=False` |
|---|---|---|
| `r1 < 1e-9` (centre fan, 1 triangle) | `(p1,p3,p2)` | `(p1,p2,p3)` |
| otherwise (quad, 2 triangles, in this order) | `(p1,p3,p2)`, `(p1,p4,p3)` | `(p1,p2,p3)`, `(p1,p3,p4)` |

- Only `r1` is tested. A negative `r1` also takes the fan branch. A pair with `r2 ≈ 0` is still a quad.

**Triangle count:**
- In general: `num_segments * Σ_i (1 if r_i < 1e-9 else 2)`, summed over `i = 0..len-2`.
- For a profile starting at r=0 with N points: `num_segments*(2N-3)`.
  - Example: 51 points and 60 segments give 5940 triangles.
- For a profile not starting at 0: `num_segments*2*(N-1)`.

## 5. Side effects
- None. The input is not mutated, no file is written, and `self.triangles` is unchanged.

## 6. Branches and edge cases
- `[]` returns `[]`.
- `[(0,0)]` (one point) returns `[]`, because the inner loop is empty.
- `[(0,0),(1,0)]` with `num_segments=4` returns 4 triangles (fans only).
- `[(1,0),(2,0)]` with `num_segments=4` returns 8 triangles (quads only).
- A worked example: `[(0,0),(1,0.1),(2,0.4)]`, `z_offset=5`, `num_segments=4`, `is_front=True` gives 12 triangles.
  - `tri[0] = ((0,0,5), (≈0,1,5.1), (1,0,5.1))`. The middle point is p3 at 90°; its x is `6.1e-17`, so use approx.
  - `tri[1] = ((1,0,5.1), (≈0,2,5.4), (2,0,5.4))`.
  - `tri[2] = ((1,0,5.1), (≈0,1,5.1), (≈0,2,5.4))`.
  - With `is_front=False`: `tri[0] = (p1,p2,p3) = ((0,0,5),(1,0,5.1),(≈0,1,5.1))`.
- Segment-0 values are exact. With `cos(0)=1.0` and `sin(0)=0.0`, p1 = `(r1, 0.0, z1+off)` and p2 = `(r2, 0.0, z2+off)` exactly.
- The last segment's `angle2` is exactly 0, so its p3 is `(r2, 0.0, z2+off)` exactly.

## 7. Rules from comments and docstring
- Normals must point out of the glass: −Z for the front, +Z for the back.
  - Winding both caps the same way was a past bug.
- The last segment wraps to angle 0 exactly, rather than using `n*step = 2π`, to avoid a non-watertight seam.
- At the centre (r1 ≈ 0), p1 and p4 coincide, so a single fan triangle is used instead of a degenerate quad.

## 8. Testing hints
- Construct the exporter with `from openlens.stl_export import STLExporter; e = STLExporter()`. Profiles can be literal lists. Real ones come from `e.calculate_surface_profile(50.0, 25.0, resolution=N)`, which returns N+1 points starting at `(0.0, 0.0)`.

**Checking normals:**
- Use `e.calculate_normal(*tri)`.
- For a flat profile, e.g. `[(0,0),(1,0),(2,0)]` with `num_segments >= 3`, every triangle's normal z should be ≈ −1 when `is_front=True` and ≈ +1 when `False`.
- Checking the sign of the z component (`< 0` / `> 0`) is robust.
- Avoid `num_segments` of 1 or 2: the triangles are degenerate or near-degenerate.

**Other checks:**
- Every vertex z equals `profile z + z_offset`.
- Every vertex radius `hypot(x,y)` matches one of the profile radii (use approx).
- Front and back outputs have the same vertex sets; each triangle is reordered with the second and third vertices swapped. For example, `front[k] == (b[0], b[2], b[1])` where `b = back[k]`.
- Check that `e.triangles == []` after the call.

**Pitfalls:**
- Compare intermediate-angle coordinates with `pytest.approx`, because `cos(pi/2)` is not exactly 0.
- A negative `r` can produce `-0.0`, which still compares equal to `0.0`.
- Don't assume a fixed count without accounting for the fan rule. Only the first pair is usually a fan.
- Don't expect validation errors for unsorted or negative radii.

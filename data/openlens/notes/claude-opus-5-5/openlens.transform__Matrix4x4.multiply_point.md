# Notes: `openlens.transform.Matrix4x4.multiply_point(self, point)`

## 1. Purpose
- Applies the affine 4x4 matrix to a 3D point, treating the point as homogeneous with w=1.
- Translation is included. Row 3 of the matrix is ignored, so there is no perspective divide.
- Returns a new `Vector3`.

## 2. Inputs
- `self.m`: a list of 4 lists of 4 numbers, row-major (`m[row][col]`).
  - `Matrix4x4()` gives the identity matrix.
  - `Matrix4x4(data)` stores `data` **by reference**; it is not copied.
  - A falsy `data` such as `None` or `[]` falls back to the identity matrix.
- `point`: any object with numeric `.x`, `.y`, `.z` attributes, normally `openlens.vector3.Vector3(x, y, z)`.
  - There is no validation or type check (duck typing).
- The `Vector3` implementation is in another module and is not shown. Do not rely on `Vector3.__eq__` or `__repr__`. Compare `.x`, `.y`, `.z` individually, or use `.to_tuple()`.

## 3. Raises
There are no explicit raises. Only incidental errors occur:
- `AttributeError` if `point` has no `.x`, `.y` or `.z`.
  - Example: `Matrix4x4().multiply_point((1, 2, 3))`.
- `IndexError` if `m` has fewer than 3 rows or a row has fewer than 4 entries.
  - Example: `Matrix4x4([[1, 0, 0]]).multiply_point(Vector3(1, 2, 3))`.
- `TypeError` for non-numeric entries or coordinates.
  - Example: a point with `x="a"` and a non-integer matrix entry.

## 4. Returns and guarantees
The result is `Vector3(x', y', z')`, computed exactly as:
```
x' = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3]
y' = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3]
z' = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3]
```
- The sum is evaluated left to right in this order. With integer-valued floats the result is exact.
- `m[3][*]` is never read. A matrix with row 3 = `[5, 6, 7, 8]` gives the same result as one with `[0, 0, 0, 1]`.
- The returned object is always new, never `point` itself. This holds even for the identity matrix.

## 5. Side effects
- None. Neither `point` nor `self.m` is mutated.

## 6. Branches and edge cases
There is a single code path. Useful concrete cases:
- **Identity:** `Matrix4x4().multiply_point(Vector3(1, 2, 3))` gives `(1, 2, 3)`.
- **Translation:** `Matrix4x4.from_translation(10, -5, 2.5).multiply_point(Vector3(1, 2, 3))` gives `(11, -3, 5.5)`.
  - Compare with `multiply_vector`, which ignores translation and gives `(1, 2, 3)`.
- **Origin:** a translation applied to `Vector3(0, 0, 0)` gives `(tx, ty, tz)`.
- **Scale:** `from_scale(2, 3, 4)` applied to `(1, 1, 1)` gives `(2, 3, 4)`.
- **Non-symmetric matrix (catches transposition):**
  - Matrix: `m = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]`.
  - Point: `(1, 1, 1)` gives `(10, 26, 42)`.
  - Point: `(1, 0, 0)` gives `(1+4, 5+8, 9+12) = (5, 13, 21)`.
  - Transposed use (`m[j][i]`) would give different numbers.
- **Row 3 ignored:**
  - Matrix: `[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,2]]`.
  - Point `(2, 4, 6)` gives `(2, 4, 6)`, not divided by 2.
- **Rotations** (use `pytest.approx`, since cos 90° ≈ 6e-17):
  - `from_euler(0, 0, 90)` on `(1, 0, 0)` gives about `(0, 1, 0)`.
  - `from_euler(90, 0, 0)` on `(0, 1, 0)` gives about `(0, 0, 1)`.
  - `from_euler(0, 90, 0)` on `(1, 0, 0)` gives about `(0, 0, -1)`.

## 7. Rules from comments/docstrings
- Docstring: "Transforms a point (w=1)."
- Comment: w is assumed to be 1 for affine transforms, and no perspective divide is done.

## 8. Testing hints
- Imports: `from openlens.transform import Matrix4x4` and `from openlens.vector3 import Vector3`.
- Build arbitrary matrices with `Matrix4x4([[...], ...])` using 4 rows of 4.
- **Composition:** `(A * B).multiply_point(p)` ≈ `A.multiply_point(B.multiply_point(p))`. `__mul__` is the standard row-by-column product.
  - Example: `T = from_translation(1, 0, 0)` and `S = from_scale(2, 2, 2)`, with `p = (1, 1, 1)`.
  - `(T*S)` gives `(3, 2, 2)`.
  - `(S*T)` gives `(4, 2, 2)`.
- **Inverse round trip:** `M.inverse().multiply_point(M.multiply_point(p))` ≈ `p` for an invertible `M`.
- **Non-mutation:** after the call, check that `point`'s coordinates and `M.m` are unchanged.
- **Comparisons:**
  - Use exact `==` only for integer-valued data.
  - Use `pytest.approx` for rotations and non-integer floats, for example `0.1 + 0.2`-style sums.
- **Likely regressions to pin down:**
  - Translation column dropped.
  - Matrix transposed.
  - w-divide or row 3 used.
  - Axes swapped.
  - Use non-symmetric matrices, nonzero translation, and a non-trivial row 3 to catch these.

# Notes: `openlens.transform.Matrix4x4.multiply_vector(self, vector)`

## 1. Purpose
- Transforms a direction vector by the matrix, treating it as homogeneous `w=0`.
- Only the upper-left 3×3 block is used. Translation and the bottom row are ignored. There is no perspective divide and no normalisation.

## 2. Inputs
- `self.m`: the matrix, a 4×4 list of lists, indexed as `m[row][col]`.
  - `Matrix4x4(data)` stores `data` **by reference** when `data` is truthy.
  - `Matrix4x4()`, `Matrix4x4(None)` and `Matrix4x4([])` all give the identity, because an empty list is falsy.
  - Entries may be ints or floats. `from_euler` builds its sub-matrices with int `0`/`1`.
- `vector`: a `Vector3`, or any object with numeric attributes `.x`, `.y`, `.z` (duck-typed).
  - There is no validation and no conversion.
  - `Vector3` is defined in `openlens.vector3`, whose source is not shown. Do not rely on `Vector3.__eq__` or `repr`.

## 3. Raises
There are no explicit raises.
- `AttributeError` if `vector` lacks `.x`, `.y` or `.z`. Example: `Matrix4x4().multiply_vector((1, 2, 3))`.
- `TypeError` if the components are non-numeric. Example: `Matrix4x4().multiply_vector(Vector3("a", 0, 0))` may raise.
  - It only fails if `Vector3` keeps the string as given; that behaviour is in the other module and is not shown, so do not test it.
- `IndexError` if `m` is smaller than 3×3. Example: `Matrix4x4([[1]]).multiply_vector(Vector3(1, 1, 1))`.

## 4. Returns and guarantees
- Returns a **new** `Vector3(x, y, z)` where:
  ```
  x = m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z
  y = m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z
  z = m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z
  ```
- Rows multiply the vector; the matrix is **not** transposed.
- Never used: `m[0][3]`, `m[1][3]`, `m[2][3]` (translation) and the whole of row 3 (`m[3][*]`).
- The result is never normalised; its length scales with the matrix.

## 5. Side effects
- None. Neither `self.m` nor `vector` is mutated.
- The result is a different object from `vector`, even for the identity matrix.

## 6. Branches and edge cases
There is a single code path.

- **Identity:** `Matrix4x4().multiply_vector(Vector3(1, 2, 3))` gives `(1, 2, 3)`.
- **Translation ignored:** `Matrix4x4.from_translation(10, 20, 30).multiply_vector(Vector3(1, 2, 3))` gives `(1, 2, 3)`.
  - Contrast: `multiply_point` on the same inputs gives `(11, 22, 33)`.
- **Non-symmetric matrix** (detects a transposition regression):
  ```python
  M = Matrix4x4([[1,2,3,10],[4,5,6,20],[7,8,9,30],[0,0,0,1]])
  M.multiply_vector(Vector3(1,2,3))  # -> (14, 32, 50)
  ```
  - A transposed implementation would give `(30, 36, 42)`.
  - Including translation would give `(24, 52, 80)`, which is what `multiply_point` returns.
- **Bottom row irrelevant:** setting `m[3] = [5, 5, 5, 5]` in the matrix above still gives `(14, 32, 50)`.
- **Scale, no normalisation:** `Matrix4x4.from_scale(2, 3, 4).multiply_vector(Vector3(1, 1, 1))` gives `(2, 3, 4)`.
  - `from_scale(2, 2, 2)` applied to `(1, 0, 0)` gives `(2, 0, 0)`, with magnitude 2.
- **Zero vector:** any matrix gives `(0, 0, 0)`.
- **Rotations** (compare with `pytest.approx`, since cos 90° ≈ 6.1e-17):
  - `from_euler(0, 0, 90)` applied to `(1, 0, 0)` gives ≈ `(0, 1, 0)`.
  - `from_euler(90, 0, 0)` applied to `(0, 1, 0)` gives ≈ `(0, 0, 1)`.
  - `from_euler(0, 90, 0)` applied to `(1, 0, 0)` gives ≈ `(0, 0, -1)`. Note the minus sign: `Ry` row 2 is `[-sy, 0, cy]`.
  - Composition is `R = Rz*Ry*Rx`, so X is applied first. `from_euler(90, 0, 90)` applied to `(0, 1, 0)`:
    - Rx maps it to `(0, 0, 1)`.
    - Rz leaves that unchanged, giving `(0, 0, 1)`.
- **Rotation preserves length:** `from_euler(30, 45, 60)` applied to `(1, 2, 3)` gives a magnitude of `sqrt(14)`, within approx.
- **Linearity:** `M·(a+b) == M·a + M·b`, componentwise, using the integer matrix above.

## 7. Rules from comments and docstrings
- Docstring: "Transforms a direction vector (w=0). Translation is ignored."
- `from_euler` docstring: order is `Z * Y * X`.

## 8. Testing hints
- Construct inputs with `from openlens.transform import Matrix4x4` and `from openlens.vector3 import Vector3`.
- Compare `r.x`, `r.y` and `r.z` individually, not whole `Vector3` objects (their equality is unknown).
  - With integer-valued matrices and vectors the results are exact; use `==` on components.
  - For rotations, use `pytest.approx(..., abs=1e-12)`.
- Always include a test with a **non-symmetric** matrix and a non-zero translation column. Identity and diagonal matrices cannot catch index swaps or translation leaking in.
- Check that the result is a new object: `r is not v`. Also check that `v.x`, `v.y`, `v.z` and `M.m` are unchanged afterwards (deep-copy `M.m` first and compare).
- Pitfall: `Matrix4x4(data)` aliases `data`. Mutating `data` after construction changes the matrix.
- Pitfall: do not assume the result is unit length.

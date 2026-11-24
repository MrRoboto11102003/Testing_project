# Notes: `openlens.transform.Matrix4x4.from_translation(x, y, z)`

## 1. Purpose
- A `@staticmethod` that builds a 4x4 affine translation matrix.
- The result is the identity matrix with `x`, `y`, `z` placed in the last column (rows 0–2, column 3).

## 2. Inputs
- `x`, `y`, `z` are required and positional or keyword. There are no defaults.
- There is **no validation or conversion**. Values are stored exactly as passed.
  - An `int` stays an `int`: `from_translation(1, 2, 3).m[0][3] == 1` and `type(...) is int`.
  - NaN, inf, negatives and even non-numbers are stored without error.
- Implementation:
  ```python
  mat = Matrix4x4()      # data=None -> identity() -> fresh nested lists of floats
  mat.m[0][3] = x; mat.m[1][3] = y; mat.m[2][3] = z
  return mat
  ```

## 3. Raises
- The function itself never raises. It does not use its arguments in arithmetic.
- Errors only appear later, when the matrix is used.
  - Example: `from_translation("a", 0, 0).multiply_point(Vector3(1, 2, 3))` raises `TypeError`, from `float + str` inside `multiply_point`.

## 4. Returns and guarantees
- Returns a **new** `Matrix4x4` on every call. Its `.m` is a list of 4 row lists.
- The exact content of `from_translation(5, -2, 3.5).m` is:
  ```
  [[1.0, 0.0, 0.0, 5],
   [0.0, 1.0, 0.0, -2],
   [0.0, 0.0, 1.0, 3.5],
   [0.0, 0.0, 0.0, 1.0]]
  ```
- All entries other than the three translation cells are float `1.0` or `0.0`. The bottom row is `[0.0, 0.0, 0.0, 1.0]`.
- Two calls never share row lists. Mutating one result does not affect another, or a later plain `Matrix4x4()`.
- `Matrix4x4` uses `__slots__ = ("m",)`, so it has no other attributes.
- `Matrix4x4` does not define `__eq__`. Compare `.m` lists, not the matrix objects.

## 5. Side effects
- None. There is no global state and no caching.

## 6. Behaviour when combined with other methods (all in this module)
- **`multiply_point(Vector3(px, py, pz))`** adds the translation.
  - It computes `1.0*px + 0.0*py + 0.0*pz + x`, and likewise for y and z.
  - So `from_translation(10, 20, 30).multiply_point(Vector3(1, 2, 3))` gives components `(11, 22, 33)`, as floats.
- **`multiply_vector(...)`** ignores the translation.
  - `from_translation(10, 20, 30).multiply_vector(Vector3(1, 2, 3))` gives `(1, 2, 3)`.
- **`from_translation(0, 0, 0)`** has `.m` equal to the identity, i.e. `Matrix4x4().m`.
- **Composition:** `(T(a) * T(b)).m[i][3]` equals `a_i + b_i`. The rotation block stays the identity, and the result is commutative.
- **`inverse()`** of `T(x, y, z)` gives last column `[-x, -y, -z, 1.0]` and the identity elsewhere.
  - With simple values this is exact, e.g. `T(2, -3, 4).inverse().m[0][3] == -2.0`.
  - Some entries may come out as `-0.0` where the result is zero. Use `==` (where `-0.0 == 0.0`) or `pytest.approx`.
- **Combining with `from_euler`:**
  - `from_translation(...) * from_euler(...)` keeps the translation column unchanged.
  - The reverse order, `from_euler(...) * from_translation(...)`, rotates the translation.
  - Example: `(from_euler(0, 0, 90) * from_translation(1, 0, 0)).m` has its last column ≈ `(0, 1, 0)` (use approx).

## 7. Rules from docstrings and comments
- `multiply_point` assumes w = 1 and does no perspective divide. The bottom row is never read.
- `multiply_vector`'s docstring says translation is ignored for directions (w = 0).

## 8. Testing hints
- Assert the full `.m` list exactly. Integer 1 equals 1.0 in Python, so list equality works with mixed types.
- Good regression checks:
  - Each of x, y, z lands in **its own row** (0, 1, 2) of **column 3**. Use distinct values such as `(1, 2, 3)` to catch swapped rows or columns.
  - Nothing is written to row 3 or to the diagonal. The diagonal stays `1.0`.
  - It is a fresh object: `from_translation(1, 2, 3) is not from_translation(1, 2, 3)`, and mutating `a.m[0][3]` does not change `b`.
  - Negative and float values are stored unchanged.
- Use `pytest.approx` for anything involving `from_euler` or non-trivial products.
- I could not see the `Vector3` source. Read the `.x`, `.y`, `.z` attributes (used by this module) or call `.to_tuple()`. Do not rely on `Vector3.__eq__`.
- Do not assert any exception from `from_translation` itself.

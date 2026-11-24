# Notes: `openlens.vector3.Vector3.normalize`

## 1. Purpose
- Returns a **new** unit-length `Vector3` pointing the same way as `self`.
- If the magnitude is below `1e-15`, it returns the zero vector instead of dividing.

## 2. Inputs
- `self` only. Components `x`, `y`, `z` are stored exactly as passed to `Vector3(x, y, z)`. The constructor does no validation or conversion, so ints, floats and numpy scalars are all kept unchanged.
- How the magnitude is computed:
  - `magnitude()` is `math.sqrt(self.x**2 + self.y**2 + self.z**2)`.
  - That is, `math.sqrt(magnitude_sq())`.

## 3. Raises
None of these are raised explicitly; they come from the arithmetic.
- **`TypeError`** for non-numeric components, raised by `**`.
  - Example: `Vector3("a", 0, 0).normalize()`.
- **`OverflowError`** (`(34, 'Numerical result out of range')`) for huge Python float components, raised by float `**`.
  - Example: `Vector3(1e200, 0, 0).normalize()`.
  - It does not return inf or 0. This is CPython float-pow behaviour.

## 4. Returns and guarantees
The code is exactly:
```python
mag = self.magnitude()
if mag < 1e-15:
    return Vector3(0, 0, 0)      # int zeros!
return self / mag                # Vector3(x/mag, y/mag, z/mag)
```
- The result is always a new `Vector3` object, never `self`. This holds even for the zero case and for vectors that are already unit length.
- **Normal path:** each component is `component / mag`, using Python true division, so the results are floats.
  - The direction and the signs of the components are preserved.
  - `magnitude()` of the result is ≈ 1.0. Use `pytest.approx` for this check.
- **Degenerate path:** returns `Vector3(0, 0, 0)` with **int** zeros.
  - `to_tuple()` gives `(0, 0, 0)`.
  - `type(result.x) is int`.

## 5. Side effects
- `self` is not mutated. Its `x`, `y`, `z` are unchanged after the call.
- There is no global state.

## 6. Branches and edge cases
| Input | Result `to_tuple()` |
|---|---|
| `Vector3(3, 4, 0)` | `(0.6, 0.8, 0.0)`. Exact, because `sqrt(25) == 5.0` |
| `Vector3(0, -2, 0)` | `(0.0, -1.0, 0.0)` |
| `Vector3(0, 0, 5.0)` | `(0.0, 0.0, 1.0)` |
| `Vector3(1, 1, 1)` | each component ≈ `1/sqrt(3)` ≈ 0.5773502691896258 |
| `Vector3(1.0, 0, 0)` (already unit) | `(1.0, 0.0, 0.0)`, as a new object |
| `Vector3()` / `Vector3(0, 0, 0)` | `(0, 0, 0)`, ints |
| `Vector3(1e-16, 0, 0)` | mag 1e-16 < 1e-15, so `(0, 0, 0)` |
| `Vector3(1e-16, 1e-16, 1e-16)` | mag ≈ 1.73e-16, so `(0, 0, 0)` |
| `Vector3(1e-200, 0, 0)` | the square underflows to 0.0, so `(0, 0, 0)` |
| `Vector3(1e-14, 0, 0)` | mag 1e-14 ≥ threshold, so x ≈ 1.0 (approx). Small but nonzero vectors are normalized |
| `Vector3(1e-10, 2e-10, 0)` | ≈ `(1/√5, 2/√5, 0.0)` |
| `Vector3(float('nan'), 0, 0)` | `nan < 1e-15` is False, so the vector is divided and all components become nan |
| `Vector3(1e200, 0, 0)` | `OverflowError` |

- The threshold uses strict `<`. Do not test at exactly `1e-15`, because the rounding of `sqrt` makes that boundary unreliable. Use `1e-16` for the zero branch and `1e-14` for the normalize branch.

## 7. Rules from comments/docstrings
- There are none beyond the class docstring "3D Vector class with basic operations".

## 8. Testing hints
- **No `__eq__` exists.** `v.normalize() == Vector3(1, 0, 0)` is False because it compares identity.
  - Compare with `v.to_tuple()` or with `.x`, `.y`, `.z`.
  - Use `pytest.approx` for non-exact values, e.g. `assert r.to_tuple() == pytest.approx((0.6, 0.8, 0.0))`.
- Exact equality is safe for the Pythagorean-style cases above (3-4-5, single-axis).
- **Check the "new object" guarantee:** `r is not v`. Also confirm `v.to_tuple()` is unchanged after the call, e.g. it stays `(3, 4, 0)`.
- **The zero branch returns ints.** `r.to_tuple() == (0, 0, 0)` passes, and `0 == 0.0` is also True. To pin the exact branch, assert `type(r.x) is int`.
  - The normal path gives floats even for int input. `Vector3(0, 0, 5).normalize().x` is `0.0` and its type is float.
- **Pin the threshold** with a pair of tests on either side, so a regression in it (e.g. to `1e-12` or `<=`) is caught:
  - `Vector3(1e-14, 0, 0).normalize().x == pytest.approx(1.0)`.
  - `Vector3(1e-16, 0, 0).normalize().to_tuple() == (0, 0, 0)`.
  - A threshold change to `1e-12` would make the `1e-14` case return `(0, 0, 0)`.
- **Scale invariance:** `Vector3(6, 8, 0)` and `Vector3(0.3, 0.4, 0)` both give ≈ `(0.6, 0.8, 0)`.
- **Sign preservation:** `Vector3(-3, 0, -4)` gives `(-0.6, 0.0, -0.8)`. The y component is `0/5.0 == 0.0`.
- **Idempotence:** `v.normalize().normalize()` gives approximately the same components.
- `Vector3` uses `__slots__`, so you cannot set extra attributes on instances.
- `vec3(x, y, z)` is an equivalent constructor.

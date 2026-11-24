# Notes: `openlens.polarization.PolarizationState.elliptical`

## 1. Purpose
- Classmethod. It builds a normalised (unit-intensity) Jones vector for an elliptical polarization state.
- The ellipse has semi-axis `a` along direction `angle_deg` and semi-axis `b` perpendicular to it, with a +90° phase on the `b` component.

## 2. Inputs
- `a: float`: semi-axis amplitude along the rotated x' axis. Any real number, including 0 and negative values.
- `b: float`: semi-axis amplitude along the rotated y' axis. Any real number, including 0 and negative values.
- `angle_deg: float = 0`: ellipse orientation in **degrees**. It is converted with `np.radians`.
- There is no validation, clamping or type coercion. Ints work.

Exact code:
```python
c, s = cos(rad), sin(rad)
jones = np.array([a*c - 1j*b*s, a*s + 1j*b*c])
norm  = np.sqrt(|jones[0]|**2 + |jones[1]|**2)
return cls(jones / norm)
```
- For real `a` and `b`, `norm == sqrt(a**2 + b**2)`. The norm is independent of the angle.

## 3. Raises
- No explicit raises.
- `a = b = 0` does **not** raise. It gives a 0/0 division, so `jones_vector` is `[nan+nanj, nan+nanj]`. numpy emits a `RuntimeWarning` ("invalid value encountered in divide").
- Non-numeric `a` or `b` (e.g. `"x"`) fails inside numpy or Python arithmetic, typically with `TypeError`. The message is not pinned down.

## 4. Returns and guarantees
- Returns `cls(...)`, which is a `PolarizationState` (or the subclass it is called on).
- `.jones_vector` is a new `np.ndarray` with shape `(2,)` and dtype `complex128`. It is complex even when `b == 0`.
- For nonzero `(a, b)`, `intensity()` is ≈ 1.0.
- Closed-form Stokes parameters, with N = a² + b²:
  - S0 = 1
  - S1 = (a² − b²)·cos2θ / N
  - S2 = (a² − b²)·sin2θ / N
  - S3 = −2ab / N
- The result is invariant to a common positive scale: `elliptical(2,2)` equals `elliptical(1,1)`.

## 5. Side effects
- None. There is no global state and no mutation.

## 6. Branches / concrete values
There is a single code path. Concrete values:
- `elliptical(1, 0)` → `[1+0j, 0+0j]`. This is the same as `linear_horizontal()`.
- `elliptical(1, 1)` → `[1, 1j]/√2`. This equals `circular_right().jones_vector`, and S3 = −1.
- `elliptical(1, -1)` → `[1, -1j]/√2`. This equals `circular_left()`, and S3 = +1.
- `elliptical(0, 1)` → `[0, 1j]`. Note this is **not** `[0, 1]`: the Ey component is purely imaginary.
- `elliptical(3, 4)` → `[0.6+0j, 0.8j]`. Stokes ≈ `[1, -0.28, 0, -0.96]`.
- `elliptical(1, 0, 90)` → ≈ `[0, 1]`. The cos(90°) term is about 6e-17, so compare with `np.allclose`.
- `elliptical(2, 1, 45)` → ≈ `[0.632456-0.316228j, 0.632456+0.316228j]`. Stokes ≈ `[1, 0, 0.6, -0.8]`.
- `elliptical(-1, 0)` → `[-1, 0]`. The sign is preserved; there is no abs().
- `elliptical(1, 0, 30)` → ≈ `linear_angle(30)` vector `[0.866025, 0.5]`.
- `elliptical(0, 0)` → all-NaN vector, as described in section 3.

## 7. Rules from docstring/module comments
- Fixed conventions: `circular_right = [1,+i]/√2` with S3 = −1, and `circular_left = [1,−i]/√2` with S3 = +1.
- Therefore a positive `a*b` gives S3 < 0 ("right" in this module's labelling).
- Module docstring says: do not "correct" signs in isolation. The `-1j*b*s` and `+1j*b*c` signs are deliberate.

## 8. Testing hints
- Compare vectors with `np.allclose(state.jones_vector, expected)`. Do **not** use `==` on two `PolarizationState` objects: the dataclass `__eq__` compares ndarrays and raises "truth value of an array is ambiguous".
- Good regression pins:
  - The exact vector for `(3,4,0)`.
  - The `(2,1,45)` vector.
  - The `elliptical(0,1)[1] == 1j` imaginary component.
  - The S3 sign: `elliptical(1,1).stokes_parameters()[3] ≈ -1`.
  - Equality with `circular_right` and `circular_left`.
  - `intensity() ≈ 1` for arbitrary `(a, b, angle)`.
  - Angle in degrees, not radians: `elliptical(1,0,90)` gives ≈ `[0,1]`.
  - Normalisation independent of the angle.
  - `dtype == np.complex128`.
- For the zero case, wrap in `np.errstate(invalid="ignore")` (or `pytest.warns(RuntimeWarning)`) and assert `np.all(np.isnan(v))`.
- Use `abs=1e-12` tolerances. sin and cos of 90°/180° are not exactly 0.

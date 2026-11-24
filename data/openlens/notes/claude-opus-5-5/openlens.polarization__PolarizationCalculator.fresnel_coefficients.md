# Notes: `PolarizationCalculator.fresnel_coefficients(n1, n2, angle_deg)`

## 1. Purpose
- Returns the Fresnel amplitude coefficients (r_s, r_p, t_s, t_p) for light going from index `n1` into index `n2` at incidence angle `angle_deg`.
- When total internal reflection (TIR) occurs, it returns complex reflection coefficients with magnitude 1 and zero transmission.

## 2. Inputs
- `n1`, `n2`: scalar indices. Nothing is validated, clamped or converted.
- `angle_deg`: scalar incidence angle in degrees. It is converted with `np.radians`. There is no range check: negative angles and angles above 90 are accepted.
- `self` state is not used (`self.matrices` is ignored). Build the object with `PolarizationCalculator()`, which takes no arguments.
- Only pass scalars. A numpy array input fails at `if sin_theta2 > 1.0` with numpy's "truth value of an array is ambiguous" `ValueError`.

## 3. Raises
- There are no explicit raises.
- `n2 == 0` with plain Python numbers raises `ZeroDivisionError` from `n1 / n2`. Example: `fresnel_coefficients(1.0, 0.0, 10.0)`. Don't assert the exact message, because it differs between Python versions.

## 4. Returns and guarantees
Each call builds a new `dict`. The steps are `sin_theta2 = (n1/n2)*np.sin(theta1)`, then the strict check `sin_theta2 > 1.0`.

**Normal branch** (`sin_theta2 <= 1.0`). The dict has exactly 6 keys:
`{"r_s", "r_p", "t_s", "t_p", "theta_transmitted_deg", "total_internal_reflection"}`
- The formulas, with c1 = cos θ1 and c2 = cos(arcsin(sin_theta2)):
  - `r_s = (n1 c1 - n2 c2)/(n1 c1 + n2 c2)`
  - `t_s = 2 n1 c1/(n1 c1 + n2 c2)`
  - `r_p = (n2 c1 - n1 c2)/(n2 c1 + n1 c2)`
  - `t_p = 2 n1 c1/(n2 c1 + n1 c2)`
- The values are **real `np.float64`**, not complex (the annotation says complex). They are still instances of `float`.
- `theta_transmitted_deg = np.degrees(theta2)`.
- `total_internal_reflection` is the Python bool `False`.
- Identities that hold exactly, up to float rounding:
  - `t_s == 1 + r_s`
  - `t_p == (n1/n2)*(1 + r_p)`
- Sign convention: at normal incidence `r_p == -r_s`. With n1 < n2, r_s is negative and r_p is positive.

**TIR branch** (`sin_theta2 > 1.0`). The dict has exactly 5 keys, and **no** `theta_transmitted_deg`:
- `cos_theta2 = cmath.sqrt(1 - sin_theta2**2)`. This is purely imaginary with a **positive** imaginary part.
- `r_s` and `r_p` use the same formulas as above, so they are complex (`complex` or `np.complex128`; both pass `isinstance(x, complex)`).
- `abs(r_s) == abs(r_p) == 1` within about 1e-12, and both have imaginary part < 0.
- `t_s` and `t_p` are `0j`. `total_internal_reflection` is `True`.

## 5. Side effects
None. Inputs are not mutated and no global state changes.

## 6. Branches and edge cases (concrete values)
- **Normal incidence**, `(1.0, 1.5, 0.0)`:
  - r_s = -0.2, r_p = 0.2, t_s = t_p = 0.8
  - theta_transmitted_deg = 0.0, total_internal_reflection is False
- **45°**, `(1.0, 1.5, 45.0)`:
  - r_s ≈ -0.303337, r_p ≈ 0.092013
  - t_s ≈ 0.696663, t_p ≈ 0.728009
  - theta_transmitted_deg ≈ 28.1255
- **Brewster angle**, `(1.0, 1.5, np.degrees(np.arctan(1.5)))` (about 56.3099°):
  - |r_p| < 1e-12
  - r_s = (1 - 2.25)/3.25 ≈ -0.384615
  - t_s ≈ 0.615385, t_p ≈ 2/3
  - theta_transmitted_deg ≈ 33.6901, so θ1 + θ2 = 90
- **Matched indices**, `(1.5, 1.5, 30.0)`: r_s = r_p = 0, t_s = t_p = 1, theta_transmitted_deg ≈ 30.0.
- **TIR**, `(1.5, 1.0, 60.0)` (the critical angle is about 41.81°):
  - total_internal_reflection is True; t_s == t_p == 0j
  - the dict has no `theta_transmitted_deg` key (`"theta_transmitted_deg" not in d`)
  - |r_s| ≈ |r_p| ≈ 1; the phase of r_s ≈ -1.671 rad, the phase of r_p ≈ -2.377 rad
- **Dense to rare medium below the critical angle**, `(1.5, 1.0, 30.0)`: normal branch, sin θ2 = 0.75, theta_transmitted_deg ≈ 48.5904.
- **Grazing incidence**, `(1.5, 1.0, 90.0)`: TIR. r_s ≈ -1 (magnitude 1).
- **Grazing incidence**, `(1.0, 1.5, 90.0)`: normal branch. c1 ≈ 6e-17, so r_s ≈ -1, r_p ≈ +1 and t ≈ 0.
- **Negative angle**, `(1.0, 1.5, -30.0)`: the coefficients equal those at +30, and theta_transmitted_deg is negative (≈ -19.4712).
- **Quirk with a negative angle in the TIR regime**, `(1.5, 1.0, -60.0)`: sin_theta2 ≈ -1.299 is not > 1, so the code takes the normal branch. `np.arcsin` returns `nan` with a RuntimeWarning, every coefficient is `nan`, and total_internal_reflection is `False`. Test it with `np.isnan` and `pytest.warns(RuntimeWarning)`, or don't test it at all.
- **n1 = 0**, `(0.0, 1.5, 30.0)`: r_s = -1.0, r_p = 1.0, t_s = t_p = 0.0.

## 7. Rules stated in comments
- Comment on the TIR branch: the same Fresnel formulas give complex coefficients with unit magnitude. Their relative phase is the s/p retardance (the basis of the Fresnel rhomb), and the power reflectance stays 1.
- `reflectance_transmittance` depends on this function. When `total_internal_reflection` is True it returns `R_s = R_p = 1.0` and `T_s = T_p = 0.0`. Otherwise it reads `coeffs["theta_transmitted_deg"]`, so that key must be present in the normal branch.

## 8. Testing hints
- Compare floats with `pytest.approx(..., abs=1e-9)` or `np.isclose`. Never use `==` on computed values, except the exact 0j values in the TIR branch and the bool flag.
- Check the key sets exactly (6 keys in the normal branch, 5 in TIR). This catches regressions that add or drop keys.
- Pin the signs as well as the magnitudes:
  - r_s < 0 and r_p > 0 at normal incidence when n1 < n2
  - r_p changes sign across the Brewster angle: at 40°, r_p > 0; at 70°, r_p < 0 (for n1=1, n2=1.5)
- Use the identities `t_s == 1 + r_s` and `t_p == (n1/n2)*(1+r_p)` at several angles. They catch swapped n1/n2 or c1/c2 terms.
- Energy check in the normal branch: `|r_s|² + (n2 c2)/(n1 c1)·t_s² == 1`, and the same holds for p.
- Test the boundary with angles clearly on either side of the critical angle (for example 41° and 43° with 1.5 → 1.0). At exactly the critical angle, floating-point rounding decides the branch.
- Don't assert `isinstance(r_s, complex)` in the normal branch. Those values are real.

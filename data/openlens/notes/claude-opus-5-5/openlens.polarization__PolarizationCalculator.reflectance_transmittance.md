# Notes: `PolarizationCalculator.reflectance_transmittance(n1, n2, angle_deg)`

## 1. Purpose
- Returns the power reflectance and transmittance for s- and p-polarisation at a planar interface between medium `n1` (incident side) and medium `n2`.
- It uses the Fresnel amplitude coefficients from `self.fresnel_coefficients`.

## 2. Inputs
- `n1`, `n2`: refractive indices, plain floats. **No validation** is done.
- `angle_deg`: angle of incidence in degrees. It is converted with `np.radians`. No validation or clamping.
- Construct the object with `PolarizationCalculator()`. It takes no arguments and the constructor has no side effects that matter here.

## 3. Raises
- There are no explicit `raise` statements.
- `n2 == 0` with Python floats raises `ZeroDivisionError` from `n1 / n2` inside `fresnel_coefficients`.
  - Example: `calc.reflectance_transmittance(1.0, 0.0, 10.0)`.
  - Avoid this case in tests.

## 4. Returns and guarantees
The result is always a new `dict` with exactly the keys `"R_s", "R_p", "T_s", "T_p"`, in that order. No extra keys appear: `theta_transmitted_deg` and `total_internal_reflection` are not included.

**Total internal reflection (TIR)** happens when `(n1/n2)*sin(theta1) > 1.0`. The check is strict `>`.
- The function returns the literal `{"R_s": 1.0, "R_p": 1.0, "T_s": 0.0, "T_p": 0.0}`.
- The values are exact Python floats, so `==` comparisons are safe.

**Normal (non-TIR) path:**
- `R_s = |r_s|²` and `R_p = |r_p|²`. These are `np.float64`.
- When `|cos(theta1)| >= 1e-10`:
  - `T_s = (n2*cos θ2)/(n1*cos θ1) * |t_s|²`
  - `T_p = (n2*cos θ2)/(n1*cos θ1) * |t_p|²`
- θ2 is recovered as `np.radians(np.degrees(arcsin(sin θ2)))`. The round trip only introduces float noise.
- The underlying coefficients are:
  - `r_s = (n1c1 - n2c2)/(n1c1 + n2c2)`
  - `t_s = 2n1c1/(n1c1 + n2c2)`
  - `r_p = (n2c1 - n1c2)/(n2c1 + n1c2)`
  - `t_p = 2n1c1/(n2c1 + n1c2)`
- Energy is conserved: `R_s + T_s == 1` and `R_p + T_p == 1`, within about 1e-12. This holds for any non-TIR, non-grazing input.

## 5. Side effects
None. Nothing is mutated, nothing is logged, and no files are written.

## 6. Branches and edge cases, with concrete values
**Normal incidence, `(1.0, 1.5, 0.0)`:**
- `R_s = 0.04` and `R_p = 0.04` (r_s = -0.2, r_p = +0.2).
- `T_s = 0.96` and `T_p = 0.96`.
- Use `pytest.approx`.

**Reverse direction, `(1.5, 1.0, 0.0)`:**
- Gives the same values: R = 0.04 and T = 0.96 for both polarisations.

**Brewster angle:**
- Inputs: `angle = np.degrees(np.arctan(1.5))`, about 56.3099°, with `(1.0, 1.5)`.
- `R_p ≈ 0`; check with `abs < 1e-12`.
- `R_s = ((1-n²)/(1+n²))² = (1.25/3.25)² ≈ 0.147929`.
- `T_p ≈ 1.0`.
- `T_s ≈ 0.852071`.

**Equal indices, `(1.5, 1.5, 30.0)`:**
- R_s and R_p are about 0.
- T_s and T_p are about 1.

**TIR, `(1.5, 1.0, 60.0)`:**
- `sin θ2 = 1.299`, which is above 1.
- Returns the exact dict `{"R_s":1.0,"R_p":1.0,"T_s":0.0,"T_p":0.0}`.
- The critical angle for 1.5→1.0 is `arcsin(1/1.5)` ≈ 41.81°.
  - At 41.0° there is no TIR (normal path).
  - At 42.0° the result is TIR.
  - Do not test exactly at the critical angle; floating-point rounding decides the branch there.

**Grazing incidence, `(1.0, 1.5, 90.0)`:**
- `cos(radians(90)) ≈ 6.1e-17`, which is below `1e-10`.
- `T_s == 0.0` and `T_p == 0.0`, as exact Python floats.
- `R_s ≈ 1` and `R_p ≈ 1`; check with `approx(1, abs=1e-9)`.
- At 89.0° the normal formula is still used, so T is greater than 0.

**Sign of the angle:**
- A negative angle gives the same result as the positive one.
- Example: `(1.0, 1.5, -30.0)` equals `(1.0, 1.5, 30.0)`, approximately.

**Monotonicity for 1.0→1.5:**
- R_s increases monotonically with angle from 0° to 89°.
- R_p falls to about 0 at Brewster's angle and then rises again.

## 7. Rules from comments and docstrings
- The grazing guard has the comment "Avoid division by zero at grazing incidence (theta1 = 90 deg)". The threshold is `1e-10` on `|cos θ1|`.
- The comments on the TIR branch say "Power reflectance stays 1".

## 8. Testing hints
**Value checks:**
- Use `pytest.approx` for every non-TIR value.
- Exact `==` is only safe for:
  - all four TIR values;
  - the grazing `T_s` and `T_p` (0.0).

**Strong regression checks:**
- `set(result) == {"R_s","R_p","T_s","T_p"}`, plus `list(result) == ["R_s","R_p","T_s","T_p"]`.
- The 0.04/0.96 values at normal incidence. These catch a dropped `n2/n1` factor or squaring: without the factor, T would be 0.64.
- Energy conservation at several angles, for example 0, 20, 45 and 70°, with `(1.0, 1.5)`.
- `R_p ≈ 0` at Brewster's angle (catches a swapped s/p formula), together with `R_s ≈ 0.1479`.
- The TIR exact dict at 60° and a non-TIR case at 41°.
- The grazing 90° case: T is exactly 0.

**Pitfalls:**
- Non-TIR values are `np.float64`. Comparing them with `approx` is fine, but `type(x) is float` fails.
- Never call with `n2 = 0`.
- Do not assert the `theta_transmitted_deg` key in the result; it is absent.
- `fresnel_coefficients` is a method on the same object, so it can be monkeypatched. That is unnecessary, because the real function works as described.

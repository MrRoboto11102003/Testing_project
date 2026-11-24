# Notes: `Lens.calculate_front_focal_length()`

## 1. Purpose
- Returns the signed Cartesian x-coordinate of the front focal point, measured from the front vertex. +x is the light direction.
- For a converging lens the front focus is on the object side, so the result is **negative**.
- Formula: `FFL = -f * (1 - t*P2/n)` with `P2 = -(n-1)/R2`.
- Equivalent form: `FFL = -f * (1 + t*(n-1)/(n*R2))`.

## 2. Inputs (no arguments, only lens state)
- `f = self.calculate_focal_length()` uses the lensmaker equation:
  - `power = (n-1)*(1/R1 - 1/R2 + (n-1)*d/(n*R1*R2))`.
  - It returns `None` if `abs(power) < EPSILON` or if a `ZeroDivisionError` occurs.
  - Otherwise it returns `1/power`.
- `n = self.refractive_index`. This is the stored value; there is no database lookup and no wavelength dependence.
- `r2 = self.get_effective_radius_2()`:
  - If `is_parabolic_2` is set: returns `(diameter/2)**2 / (2*parabolic_sag_2)`. If `|sag| < EPSILON` or `|diameter/2| < EPSILON`, it returns `inf`.
  - Otherwise: returns `radius_of_curvature_2`.
- R1 is handled the same way via `get_effective_radius_1()`.
- The radius setters turn `0` (and `-0.0`) into `float('inf')`, which means a flat surface. A flat surface gives `1/R = 0` and `P2 = -0.0`.
- `t = self.thickness`. This is the centre thickness and is not validated (negative or zero is accepted).
- Ignored by this method:
  - `conic_constant_*`
  - `diameter` (except for parabolic surfaces)
  - `wavelength`, `temperature`, `pressure_atm`
  - clear apertures
  - `lens_type`

## 3. Raises
- In practice, nothing.
- A `ZeroDivisionError` in the FFL step is caught and gives `None`.
- An `n == 0` case already fails inside `calculate_focal_length`, which returns `None`.

## 4. Returns and guarantees
- Returns `None` when the focal length is `None` (afocal or undefined). It never returns `inf` for this case.
- Otherwise returns a plain Python `float`.
- Pure: computed fresh on each call, no caching.

## 5. Side effects
- None. No attribute changes; `modified_at` is not updated.

## 6. Branches and worked values
Always pass an explicit `refractive_index`. The default index comes from `constants.DEFAULT_MATERIAL_INDICES`, which is not shown here.

| Lens(...) kwargs | f | FFL |
|---|---|---|
| R1=100, R2=-100, t=5, n=1.5 | 12000/119 ≈ 100.840336 | -11800/119 ≈ **-99.159664** |
| R1=50, R2=0 (flat back), t=5, n=1.5 | 100.0 | **-100.0** |
| R1=0 (flat front), R2=-50, t=5, n=1.5 | 100.0 | -100*(1-1/30) ≈ **-96.666667** |
| R1=-100, R2=100, t=5, n=1.5 (diverging) | -12000/121 ≈ -99.173554 | 12200/121 ≈ **+100.826446** |
| R1=100, R2=-100, t=0, n=1.5 (thin) | 100.0 | **-100.0** (= -f) |
| R1=0, R2=0 (both flat), any t, n | None | **None** |
| R1=100, R2=-100, n=1.0 | None (power 0) | **None** |

Parabolic back surface:
- Build with `R1=0, radius_of_curvature_2=1000` (this radius is ignored), `is_parabolic_2=True, parabolic_sag_2=-1.0, diameter=20, t=5, n=1.5`.
- The effective R2 is `100/(2*-1) = -50`.
- The result is the same as the flat-front row: ≈ -96.666667.

Relationship checks:
- For a symmetric biconvex lens (R1 = -R2), `FFL == -BFL`. Compare against `calculate_back_focal_length()` using approx.
- Plano-convex orientation matters:
  - Curved front, flat back (R1=50, R2=flat): FFL = -100 and BFL = 100*(1 - 5*0.5/(1.5*50)) ≈ 96.6667.
  - Flipped (flat front, R2=-50): FFL ≈ -96.6667 and BFL = 100.0.
  - So `FFL != -BFL` here. This catches a swapped R1/R2 or a wrong sign.

## 7. Rules from comments/docstring
- Uses a signed Cartesian convention:
  - A converging lens gives a negative FFL.
  - A diverging lens gives a positive FFL.
- Returns `None` (not `inf`) when the lens is afocal, matching the system-level convention.
- It is the mirror of `BFL = +f*(1 - d*P1/n)`, where `P1 = (n-1)/R1`.
- A flat R2 gives `P2 = 0`, so `FFL = -f`.

## 8. Testing hints
- Construct the lens like this: `Lens(radius_of_curvature_1=..., radius_of_curvature_2=..., thickness=..., refractive_index=1.5)`.
  - Leave `use_type_defaults` at its default `False`; setting it to `True` would overwrite the radii.
- Compare results with `pytest.approx(..., rel=1e-9)`, not `==`.
- A flat surface can be given as `0` or `float('inf')`; both behave the same.
- Assert the sign explicitly. Negating or dropping the leading `-` flips every converging case.
- Assert `None` with `is None` for the both-flat lens and the n=1 lens.
- Use a non-zero thickness and an asymmetric lens (the flat-front R2=-50 case). This catches:
  - using R1 instead of R2,
  - a `+` vs `-` sign error inside the parentheses,
  - dropping the `/n`.
- Check `-96.666667`; this exact value is lost if `/n` is omitted.
- Changing `material`, `wavelength_nm`, or `conic_constant_2` must not change the result when `refractive_index` is explicit. This can be asserted as a non-regression check.

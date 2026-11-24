# Notes: `Lens.calculate_back_focal_length()`

## 1. Purpose
- Returns the thick-lens back focal length in mm: the distance from the back vertex to the rear focal point.
- Formula: `BFL = f * (1 - t * P1 / n)`, where `P1 = (n - 1) / R1_eff`.
- Returns `None` when the lens is afocal or undefined.

## 2. Inputs
- The method takes no arguments. It reads only these attributes:
  - `self.refractive_index` (plain attribute, read as-is; no database or wavelength lookup)
  - `self.thickness`
  - `get_effective_radius_1()`
  - `calculate_focal_length()`, which also uses `get_effective_radius_2()`
- Effective radius rules (shown for surface 1; surface 2 is the same with `_2`):
  - If `is_parabolic_1` is False: returns `radius_of_curvature_1`. The setter turns `0` (and `-0.0`) into `float('inf')`.
  - If `is_parabolic_1` is True: returns `(D/2)**2 / (2*parabolic_sag_1)`.
    - Returns `inf` if `|sag| < EPSILON` or `|D/2| < EPSILON`.
    - The stored `radius_of_curvature_1` is ignored in this case.
- `conic_constant_*` is **not used** at all, so it does not change the BFL.
- `calculate_focal_length()`:
  - `power = (n-1)*(1/R1 - 1/R2 + (n-1)*d/(n*R1*R2))`
  - Returns `None` if `abs(power) < EPSILON` or on `ZeroDivisionError`; otherwise returns `1/power`.
  - `1/inf == 0.0`, so a flat surface contributes zero power.
  - `EPSILON` lives in `openlens.constants` (value not shown; it is a tiny tolerance).

## 3. Raises
- Nothing in normal use. `ZeroDivisionError` is caught in both this method and the focal-length method, and turned into `None`.
- Example: `n=0` with finite radii makes `n*R1*R2 == 0.0`, so the focal length is `None` and BFL is `None`.

## 4. Returns and guarantees
- Returns a `float` (sign kept, so it can be negative) or `None`.
- Afocal or undefined cases return `None`, never `inf`.
- If `R1_eff` is infinite, `P1 = 0`, so `BFL == f` exactly.
- If `thickness == 0`, `BFL == f`.

Worked values (always pass `refractive_index` explicitly; `t` is thickness):

| R1 | R2 | t | n | f | BFL |
|---|---|---|---|---|---|
| 100 | -100 | 5 | 1.5 | 12000/119 ≈ 100.840336 | 11800/119 ≈ **99.159664** |
| 100 | 0 (→inf) | 5 | 1.5 | 200.0 | 200*59/60 ≈ **196.666667** |
| 0 (→inf) | -100 | 5 | 1.5 | 200.0 | **200.0** (== f) |
| -100 | 100 | 5 | 1.5 | -12000/121 ≈ -99.173554 | -12200/121 ≈ **-100.826446** |
| 10 | -10 | 40 | 1.5 | 30.0 | **-10.0** (negative BFL) |
| 0 | 0 | 5 | 1.5 | None | **None** |
| 100 | -100 | 5 | 1.0 | None (power 0) | **None** |

Parabolic example:
- Setup: `diameter=20, is_parabolic_1=True, parabolic_sag_1=1.0` gives `R1_eff = 100/2 = 50`.
- Any `radius_of_curvature_1` is ignored.
- With `R2=-100, t=5, n=1.5`:
  - power = `0.5*(0.02 + 0.01 + 0.5*5/(1.5*50*-100))` = `0.5*(0.03 - 1/3000)` = `0.5*(89/3000)`
  - f = `6000/89` ≈ 67.41573
  - P1 = 0.01
  - BFL = `f*(1 - 0.05/1.5)` = `f*(29/30)` = `5800/89` ≈ **65.168539**
- Parabolic with `parabolic_sag_1=0.0` gives `R1_eff = inf`, so `BFL == f`.

## 5. Side effects
- None. The lens is not mutated, `modified_at` is not touched, and there is no database access or I/O.

## 6. Branches and edge cases
- Focal length is `None` (both surfaces flat, or `n == 1`, or `abs(power) < EPSILON`): returns `None` early.
- Finite `R1`: the full formula is applied.
- Infinite `R1`: returns `f`.
- `ZeroDivisionError` in the BFL step can only come from `n == 0`, but that case already returns `None` in the focal-length step.
- A NaN radius propagates and returns NaN. Avoid testing this.

## 7. Rules from docstring/comments
- "None (not inf) matches OpticalSystem.calculate_back_focal_length."
- A flat surface (`r1 = inf`) gives `P1 = 0`.
- The formula uses the effective (vertex) radius, so parabolic surfaces are handled.

## 8. Testing hints
- Build lenses directly, for example: `Lens(radius_of_curvature_1=100, radius_of_curvature_2=-100, thickness=5, diameter=25, refractive_index=1.5)`.
  - Always pass `refractive_index`. The default comes from `DEFAULT_MATERIAL_INDICES`, whose values are not shown.
  - Always pass the radii explicitly. The default radii are unknown constants.
- Compare floats with `pytest.approx(..., rel=1e-9)`. Use the exact fractions above where possible.
- Good regression checks:
  - **Sign**: a negative lens gives a negative BFL; the `R1=10, t=40` case gives `-10.0`.
  - **Asymmetry**:
    - Plano-convex (curved side first) gives 196.67.
    - Convex-plano (flat side first, `R1=inf`) gives 200.0.
    - This shows R1 is used, not R2.
  - **Thickness dependence**: `t=0` gives `BFL == f`.
  - **Parabolic**: BFL uses the effective radius, so the stored `radius_of_curvature_1` has no effect.
  - **Afocal**: check `is None`, not `inf` or 0.
- You may mutate `lens.refractive_index` or `lens.thickness` after construction; the method reads the current values.
- Relation to check: `bfl == lens.calculate_focal_length() * (1 - t*(n-1)/(R1*n))`.

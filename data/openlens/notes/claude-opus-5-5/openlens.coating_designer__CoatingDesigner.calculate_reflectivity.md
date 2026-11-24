# Notes: `CoatingDesigner.calculate_reflectivity(self, layers, wavelength_nm, angle_deg=0)`

## 1. Purpose
- Computes the reflectance R of a thin-film stack on a substrate with the transfer-matrix method.
- Light is unpolarized: R is the mean of the s and p reflectances.
- The incident medium is air (`self.air_index`, which is 1.0).
- With no layers it returns the bare Fresnel reflectance.

## 2. Inputs
- `self.substrate_index`: set by `CoatingDesigner(substrate_index=1.5168)`. Default 1.5168 (BK7). It is not validated.
- `self.air_index`: always 1.0 after `__init__`. It is a plain attribute that tests could reassign.
- `layers`: a list of `CoatingLayer(material, refractive_index, thickness_nm)`.
  - Order is **substrate to air**. Index 0 touches the substrate; the last item faces the air.
  - Any falsy value (`[]`, `None`) takes the bare-substrate path.
  - Only `refractive_index` and `thickness_nm` are used. `material` is ignored.
- `wavelength_nm`: a number that must be > 0.
- `angle_deg`: angle of incidence in air, in degrees. It must satisfy `0 <= angle_deg < 90`. Default `0` (an int).
- Layer indices and thicknesses are not validated. A thickness of 0 is allowed.

## 3. Raises (checks run in this order; the wavelength check comes first)
- `ValueError(f"Wavelength must be positive, got {wavelength_nm}")` when `wavelength_nm <= 0`.
  - `calculate_reflectivity([], 0)` gives the message `"Wavelength must be positive, got 0"`.
  - `-5.0` gives `"... got -5.0"`.
- `ValueError(f"Angle of incidence must be in [0, 90), got {angle_deg}")` when the angle is outside the range.
  - `calculate_reflectivity([], 550, 90)` gives `"... got 90"`.
  - `-1` gives `"... got -1"`.
- If both values are bad, the wavelength message wins: `calculate_reflectivity([], -1, 95)` reports the wavelength.
- Both checks run even when `layers` is empty.
- NaN cases:
  - A NaN wavelength passes the wavelength check, because `nan <= 0` is False.
  - A NaN angle raises the angle error.
- Use `pytest.raises(ValueError, match=...)`. Escape the regex characters `[`, `(` and `)` in the angle message, for example with `re.escape`.

## 4. Returns and guarantees
- Returns a Python `float`, clamped to `[0.0, 1.0]`.
- The result never depends on `material`.
- On the bare path (no layers) the result does not depend on `wavelength_nm`.
- Reference values at normal incidence:
  - Bare, `substrate_index=1.5`: `0.04` (that is (0.5/2.5)²). Compare with `pytest.approx`.
  - Bare, default 1.5168: `((0.5168/2.5168))**2 ≈ 0.0421678`. The docstrings call this "4.216%".
  - Bare, `substrate_index=1.0`: `0.0`.
- Quarter-wave single layer (thickness `λ/(4 n_c)`), normal incidence:
  - Formula: `R = ((n_s - n_c**2)/(n_s + n_c**2))**2`.
  - MgF2 (n=1.38, thickness 550/(4·1.38)) on 1.5168 at λ=550: R ≈ 0.012836.
  - Substrate `1.38**2 = 1.9044` with the MgF2 quarter-wave layer: R ≈ 0. Use `abs=1e-12`.
- Layers that act as if absent (the result equals the bare value at any wavelength or angle):
  - A half-wave layer, thickness `λ/(2n)`, at normal incidence.
  - A zero-thickness layer (phase δ=0 gives the identity matrix), at any angle.
  - These match because the matrix path and the bare Fresnel path agree.
- Scale invariance: on the layered path the phase uses `thickness/λ`. Multiplying all thicknesses and the wavelength by the same k gives the same R.

## 5. Side effects
- None. `layers` is not mutated; the reversal uses `list(reversed(layers))`.
- No state is stored on `self`.

## 6. Branches and edge cases
- **Bare path** (`not layers`):
  - `cos_s = cmath.sqrt(1 - (sin0/n_s)**2)`.
  - `rs = (cos0 - n_s cos_s)/(cos0 + n_s cos_s)` and `rp = (n_s cos0 - cos_s)/(n_s cos0 + cos_s)`.
  - Returns the mean of |rs|² and |rp|².
- **Brewster check**: `substrate_index=1.5`, `angle_deg=math.degrees(math.atan(1.5))` (≈56.31°).
  - Here rp=0, so R = rs²/2.
  - The value is less than both the normal-incidence R and the R at 80°.
- **Grazing**: the bare R at 89.9° is close to 1 but ≤ 1.0. R increases with angle beyond Brewster.
- **Layered path**:
  - The stack is traversed in reverse (air side first).
  - For each layer: `cos_t = sqrt(1-(sin0/n)^2)` and `δ = 2π n t cos_t / λ`.
  - Admittance is η = n·cos (s polarization) or n/cos (p polarization).
  - Then `B,C = M·[1, η_sub]`, `r = (η_inc B − C)/(η_inc B + C)`.
- **Order matters**: swapping two different layers in a non-trivial stack generally changes R.
  - Example: TiO2 then MgF2 quarter-waves at λ=550 gives a different R from the reversed order.
  - Only assert `!=` (or compute the expected value independently); do not hard-code values you have not derived.
- **Integer inputs**: `angle_deg=0` (int) is fine. `wavelength_nm=550` (int) is fine.

## 7. Rules from docstrings and comments
- Layers are stored substrate to air; light enters from air.
- Films are lossless (real indices), so callers report T = 1 − R.
- The angle must be in [0, 90).

## 8. Testing hints
- Import it as `from openlens.coating_designer import CoatingDesigner, CoatingLayer`.
- Always use `pytest.approx(rel=1e-9)` or an `abs` tolerance; never compare floats exactly.
- Good regression anchors, each derivable by hand:
  - Exact bare values: 1.5 → 0.04; 1.5168 → 0.042168.
  - The quarter-wave formula value.
  - Zero-thickness and half-wave layers equal the bare value.
  - Oblique incidence: compute the bare Fresnel value independently with the formulas in §6.
  - Error messages and the order of the checks.
  - Boundaries: angle 0 is accepted; 90 is rejected; 89.999 is accepted.
  - Wavelength: `1e-9` is accepted; `0` is rejected.
- Cross-check: `coated_reflectance(layers, n_s, wl, angle)` returns exactly the same value (it wraps this method).
- Do not assume any particular `design_dual_layer_ar` or `design_v_coating` output values. Only `R(design) <= R([])` is guaranteed.

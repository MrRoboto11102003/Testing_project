# Notes: `openlens.polarization.PolarizationState.intensity`

## 1. Purpose
- Returns the total intensity of a Jones vector: `|Ex|**2 + |Ey|**2`.
- The value is **not** normalised. It is the raw squared magnitude.

## 2. Inputs
- `self` is a `PolarizationState`, which is a plain `@dataclass` with one field, `jones_vector`.
- The dataclass does no validation, conversion or normalisation of `jones_vector`.
- Only `self.jones_vector[0]` and `self.jones_vector[1]` are read.
  - Any further elements are ignored. For example, `[1, 0, 5]` gives `1.0`.
- `jones_vector` should be a 1-D numpy array, real or complex.
  - A plain list such as `[3, 4]` also works, because `np.abs` accepts Python scalars.
- `np.abs` is applied to each component before squaring. So for complex entries the result uses the modulus, not the real part.

## 3. Raises
- The method has no explicit raises.
- `IndexError` from indexing if the vector has fewer than 2 elements:
  - `PolarizationState(np.array([1.0])).intensity()` raises `IndexError`.
- `TypeError` if `jones_vector` is `None`, because it is not subscriptable:
  - `PolarizationState(None).intensity()`.

## 4. Returns and guarantees
- For numpy input the result is a numpy scalar, `np.float64`. `isinstance(result, float)` is True.
- The result is always real and `>= 0`. Its value is `abs(Ex)**2 + abs(Ey)**2`.
- Exact-value examples (integers or exactly representable values):
  - `[3, 4]` gives `25.0`.
  - `[3, 4j]` gives `25.0`.
  - `[0, 2j]` gives `4.0`.
  - `[-3, 0]` gives `9.0`.
  - `[0, 0]` gives `0.0`.
  - `[2, 0]` gives `4.0`, not `2.0`. The value is squared, not a norm.
- Factory states:
  - `linear_horizontal()` and `linear_vertical()` give exactly `1.0`.
  - `linear_45()`, `circular_right()`, `circular_left()`, `linear_angle(θ)` and `elliptical(a, b, θ)` give approximately `1.0`. Floating-point rounding is possible, e.g. `0.9999999999999998`.
- The result scales quadratically with the vector: `PolarizationState(k * v).intensity() == k**2 * PolarizationState(v).intensity()`. Compare with approx.
- The result is unchanged by a global phase: multiplying the vector by `np.exp(1j*φ)` gives the same intensity (approx).
- It equals `stokes_parameters()[0]`, which is S0 and uses the identical formula.

## 5. Side effects
- None. `jones_vector` is not mutated.

## 6. Branches and edge cases
- There is a single code path with no branches.
- Zero vector `[0, 0]` returns `0.0`. Related: `degree_of_polarization()` returns `0.0` in this case and `1.0` otherwise.
- Purely imaginary component `[0, 1j]` returns `1.0`.
  - This distinguishes the modulus from the real part, which would give `0`.
- Complex `[1+1j, 0]` returns `≈ 2.0`; use approx.
- Non-unit vectors, e.g. `[3, 4]` giving `25.0`, distinguish the squared sum from `sqrt` or a norm (which would give `5`).
- Combining with `apply_matrix`:
  - `linear_horizontal().apply_matrix(JonesMatrices.linear_polarizer(45)).intensity()` is approximately `0.5`.
  - With polarizer `90` the result is approximately `0` (about `1e-33`), so use `abs=1e-12`.
  - With polarizer `0` the result is exactly `1.0`.
  - This matches `PolarizationCalculator().malus_law(1.0, θ)` (approx).
- Wave plates and rotators preserve intensity:
  - `quarter_wave_plate(45)`, `half_wave_plate(22.5)`, `rotator(30)` and `retarder(60, 10)` applied to `linear_45()` give approximately `1.0`.

## 7. Rules stated in comments/docstrings
- The docstring says: "Calculate intensity (magnitude squared)". The value is squared, with no sqrt and no normalisation.
- The module docstring says Jones vectors are pure states, so intensity is the only amplitude measure.

## 8. Testing hints
- Build states directly, e.g. `PolarizationState(np.array([3.0, 4.0j]))`, or with the classmethod factories.
- Use `pytest.approx` for any value involving `sqrt(2)`, trig functions or complex products.
- Exact `==` is safe only for small integer components.
- Strong regression-catching cases:
  - `[3, 4]` gives `25.0`. This catches a missing square or a sqrt.
  - `[3, 0]` gives `9.0` and `[0, 4]` gives `16.0`. These catch use of only one component.
  - `[0, 2j]` gives `4.0`. This catches use of the real part instead of the modulus.
  - `[-3, 0]` gives `9.0`. This catches sign handling.
  - Scaling: `v` versus `2*v` should give a ratio of 4.
  - `intensity() == approx(stokes_parameters()[0])`.
- Do not assert `type(result) is float`; it is `np.float64`. Use `isinstance(result, float)` or compare values.
- Do not rely on extra elements beyond index 1 contributing; they are ignored.

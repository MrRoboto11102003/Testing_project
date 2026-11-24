# Notes: `openlens.polarization.PolarizationState.stokes_parameters`

## 1. Purpose
- Converts the state's Jones vector `(Ex, Ey)` into the Stokes vector `[S0, S1, S2, S3]`.
- Returns it as a `np.ndarray` of shape `(4,)`.

## 2. Inputs
- The method takes no arguments. It reads only `self.jones_vector`.
- `jones_vector` is unpacked as `Ex, Ey = self.jones_vector`, so it must have exactly 2 elements. A list or a numpy array both work.
- Nothing is normalised. Raw magnitudes are used, so the result scales with intensity.
- The dataclass does not validate input. `PolarizationState(np.array([2, 0]))` is accepted.

## 3. Raises
- The method raises nothing explicitly.
- A `jones_vector` that does not have exactly 2 elements raises `ValueError` from tuple unpacking:
  - `PolarizationState(np.array([1, 0, 0])).stokes_parameters()` gives `ValueError: too many values to unpack (expected 2)`.
  - `PolarizationState(np.array([1])).stokes_parameters()` gives `ValueError: not enough values to unpack (expected 2, got 1)`.
  - Use `pytest.raises(ValueError)` only. Do not match the message.

## 4. Returns and guarantees
The exact formulas are:
```
S0 = |Ex|^2 + |Ey|^2
S1 = |Ex|^2 - |Ey|^2
S2 = 2 * Re(Ex * conj(Ey))
S3 = 2 * Im(Ex * conj(Ey))     # note: Ex * conj(Ey), NOT conj(Ex) * Ey
```
- A new `np.ndarray` is built on every call, in the order `[S0, S1, S2, S3]`.
- For float or complex input the dtype is float64 (real values).
- For integer input the dtype may be integer. Do not assert dtype in that case.

Reference values (compare with `np.allclose` or `pytest.approx`):

| State | Expected `[S0, S1, S2, S3]` |
|---|---|
| `linear_horizontal()` | `[1, 1, 0, 0]` |
| `linear_vertical()` | `[1, -1, 0, 0]` |
| `linear_45()` | `[1, 0, 1, 0]` |
| `linear_angle(θ)` | `[1, cos2θ, sin2θ, 0]`, e.g. 30° gives `[1, 0.5, 0.8660254, 0]`; −45° gives `[1, 0, -1, 0]` |
| `circular_right()` = `[1, +i]/√2` | `[1, 0, 0, -1]` (S3 = **−1**, as the module docstring states) |
| `circular_left()` = `[1, −i]/√2` | `[1, 0, 0, +1]` |
| `elliptical(2, 1, 0)` = `[2, i]/√5` | `[1, 0.6, 0, -0.8]` |
| `elliptical(1, 1, 0)` | same as `circular_right`: `[1, 0, 0, -1]` |
| `PolarizationState(np.array([2+0j, 0j]))` | `[4, 4, 0, 0]` (not normalised) |
| `PolarizationState(np.array([1+0j, 1j]))` | `[2, 0, 0, -2]` |
| `PolarizationState(np.array([0j, 0j]))` | `[0, 0, 0, 0]` |
| `PolarizationState(np.array([1+1j, 0j]))` | `[2, 2, 0, 0]` |

## 5. Side effects
- None. `self.jones_vector` is not mutated.
- Calling the method twice gives equal arrays, and they are distinct objects.

## 6. Branches and edge cases
- There is a single code path with no branches.
- For real-valued input (`linear_45`, `linear_angle`), `np.imag` gives 0, so S3 is exactly 0.
- For any Jones vector the result is fully polarized: `S0**2 ≈ S1**2 + S2**2 + S3**2`.
- A global phase does not change the result. Multiplying the vector by `np.exp(1j*0.7)` gives the same Stokes vector, within allclose.
- Each component scales by `|k|^2` when the vector is multiplied by `k`. For example, `k = 3` multiplies every S by 9.

## 7. Rules from comments and docstrings
- The module docstring fixes the conventions:
  - S3 = `2*Im(Ex*conj(Ey))`.
  - `circular_right` has S3 = −1 and `circular_left` has S3 = +1.
  - These signs are deliberate, and the docstring says not to "correct" them.
- Consistency anchor from the docstring: horizontal light through a QWP at 45° gives `circular_left` up to a global phase.
  - Concretely, `PolarizationState.linear_horizontal().apply_matrix(JonesMatrices.quarter_wave_plate(45))` gives the vector `[(1+i)/2, (1−i)/2]`.
  - Its Stokes vector is `[1, 0, 0, +1]`.

## 8. Testing hints
- Always compare floats with `np.allclose(result, expected, atol=1e-12)`. For example, `linear_45` gives S0 ≈ 1.0000000000000002.
- Check `result.shape == (4,)` and `isinstance(result, np.ndarray)`.
- The highest-value tests are:
  - **The S3 sign for circular_right and circular_left.** A mutation to `conj(Ex)*Ey` flips it.
  - **The S1 sign for vertical light**, which must be −1.
  - **The S2 sign for linear at −45°**, which must be −1.
  - **The factor 2 in S2 and S3**, e.g. `linear_45` must have S2 = 1, not 0.5.
  - **The unnormalised scaling**, e.g. `[2, 0]` must give S0 = 4.
  - **The order of components.** The `elliptical(2, 1, 0)` vector `[1, 0.6, 0, -0.8]` distinguishes every position.
- More checks:
  - Assert the identity `S0 == intensity()` (from the same class) for arbitrary vectors.
  - Rotator: `linear_horizontal().apply_matrix(JonesMatrices.rotator(30))` gives `[cos30, -sin30]`, so the Stokes vector is `[1, 0.5, -0.8660254, 0]`.
- Do not assume a normalised output, and do not assert an integer dtype.

# Notes: `openlens.polarization.PolarizationState.apply_matrix`

## 1. Purpose
- Applies a Jones matrix to this state. It returns a **new** `PolarizationState` whose vector is `matrix @ self.jones_vector`.
- Full body:
  ```python
  new_vector = matrix @ self.jones_vector
  return PolarizationState(new_vector)
  ```

## 2. Inputs
- `self.jones_vector`: whatever the dataclass holds. The dataclass does no validation and no normalisation. The factory methods produce shape-(2,) arrays.
  - `linear_horizontal()` and `linear_vertical()` give complex vectors.
  - `linear_45()` and `linear_angle()` give **float** vectors.
  - `circular_*` and `elliptical` give complex vectors.
- `matrix`: anything that supports `@` with an ndarray.
  - A 2x2 `np.ndarray` works, real or complex.
  - A nested list also works, e.g. `[[1,0],[0,0]]`, through `ndarray.__rmatmul__`.
- The function does no shape check, no unitarity check and no normalisation.

## 3. Raises
apply_matrix has no explicit raises. Only numpy errors from `@` can occur. Their exact wording depends on the numpy version, so assert only the type:
- `ValueError` when the core dimensions mismatch, e.g. `PolarizationState.linear_horizontal().apply_matrix(np.eye(3))`.
- `ValueError` for a scalar matrix, e.g. `.apply_matrix(np.float64(2.0))`. Note that matmul rejects 0-d operands.
- Use `pytest.raises(ValueError)` only. Do not use `match=`.

## 4. Returns and guarantees
- The result is an instance of exactly `PolarizationState`. It is hard-coded, not `type(self)`.
- The result is a new object: `out is not s` and `out.jones_vector is not s.jones_vector`, even with `np.eye(2)`.
- The output is **not normalised**, so intensity can drop.
- dtype follows numpy promotion:
  - A real matrix times the float `linear_45` vector gives a float result.
  - Anything involving complex gives a complex result.
- Shape follows matmul. For example, a (1,2) matrix gives a shape-(1,) vector with no error.
- Chaining `s.apply_matrix(A).apply_matrix(B)` equals `(B @ A) @ v`, so order matters.

## 5. Side effects
- None. Neither `self.jones_vector` nor `matrix` is modified.
- Test this by keeping a `.copy()` of the vector and comparing it after the call.

## 6. Concrete expected values
Use `np.testing.assert_allclose(..., atol=1e-12)` for all of these.

| Input state | Matrix (`JonesMatrices`) | Result `jones_vector` |
|---|---|---|
| `linear_vertical()` | `linear_polarizer(0)` = `[[1,0],[0,0]]` | `[0, 0]` → `intensity()==0`, `degree_of_polarization()==0.0` |
| `linear_45()` | `linear_polarizer(0)` | `[0.70710678, 0]`, `intensity()≈0.5` (Malus) |
| `linear_horizontal()` | `linear_polarizer(90)` | ≈`[0, 0]` (tiny float residue ~6e-17; use atol) |
| `linear_horizontal()` | `half_wave_plate(22.5)` | `[0.70710678, 0.70710678]` (= linear_45) |
| `linear_horizontal()` | `half_wave_plate(0)` = diag(1,-1) | `[1, 0]` |
| `linear_vertical()` | `half_wave_plate(0)` | `[0, -1]` (sign matters) |
| `linear_horizontal()` | `quarter_wave_plate(45)` | `[0.5+0.5j, 0.5-0.5j]` |
| `linear_horizontal()` | `rotator(30)` = `[[c,s],[-s,c]]` | `[0.8660254, -0.5]` (note the **minus**) |
| `linear_horizontal()` | `rotator(90)` | ≈`[0, -1]` |
| `circular_right()` | `np.eye(2)` | equal values to the input, but a new object |
| `PolarizationState(np.array([1+0j, 2+0j]))` | `[[0,1],[1,0]]` (list) | `[2, 1]` |

Further checks on the QWP(45) case, i.e. `linear_horizontal().apply_matrix(quarter_wave_plate(45))`:
- `stokes_parameters()` is `[1, 0, 0, 1]`, so S3 = +1, matching `circular_left`.
- The result equals `circular_left().jones_vector * (1+1j)/np.sqrt(2)`. This is the module docstring's "consistency anchor".
- `intensity()` is 1.0 and `degree_of_polarization()` is 1.0.

Other edge cases:
- An all-zero matrix gives a zero vector, `intensity()==0`, and `degree_of_polarization()` returns exactly `0.0`.
- A scaling matrix `2*np.eye(2)` applied to horizontal gives `[2, 0]` and `intensity()==4`. This proves there is no normalisation.

## 7. Rules from comments and docstrings
- Docstring: "Apply Jones matrix to transform polarization". It is a pure left-multiplication.
- Module conventions:
  - `circular_right` is `[1,+i]/√2`.
  - `rotator` is `[[c,s],[-s,c]]`, clockwise-positive.
  - Horizontal light through QWP@45 gives `circular_left` up to a global phase.
- Do not "correct" these signs in expected values.

## 8. Testing hints
- Always compare arrays with `np.testing.assert_allclose`, not `==`. Trig residues such as `cos(90°)≈6e-17` exist, so pass `atol`.
- Compare complex results either against complex expected arrays or by checking real and imaginary parts separately.
- Global phase:
  - Compare exactly against the computed vector, e.g. `[0.5+0.5j, 0.5-0.5j]`.
  - Or compare phase-invariant quantities: the Stokes parameters, `intensity()`, and `abs(np.vdot(a, b))`.
- Test non-commutativity. Apply `linear_polarizer(0)` then `linear_polarizer(90)` to `linear_45()` and you get ≈0. That is different from applying a single polarizer.
- Test `type(out) is PolarizationState`, `out is not s`, and input immutability.
- Do not assert numpy error message text. Do not assert dtype for mixed real/complex cases beyond `np.iscomplexobj` in cases where it is obviously complex.

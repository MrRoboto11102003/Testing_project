# Notes: `openlens.polarization.JonesMatrices.linear_polarizer`

## 1. Purpose
- Returns the 2×2 Jones matrix of an ideal linear polarizer whose transmission axis is at `angle_deg` from the x (horizontal) axis.

## 2. Inputs
- `angle_deg: float`, a required positional argument in **degrees**.
- There is no validation, clamping or type check. The only processing is `theta = np.radians(angle_deg)`.
- It is a `@staticmethod`. All of these work:
  - `JonesMatrices.linear_polarizer(30)`
  - `JonesMatrices().linear_polarizer(30)`
  - `PolarizationCalculator().matrices.linear_polarizer(30)`
- `int` inputs are fine. `numpy` scalars are fine.
- With `c = cos(theta)` and `s = sin(theta)`, the body is:
  ```python
  np.array([[c**2, c*s], [c*s, s**2]])
  ```

## 3. Raises
- Nothing for numeric scalars. This includes negative values, values above 360, `nan` and `inf`. `nan` and `inf` give an all-`nan` matrix.
- Non-numeric input such as `"45"` makes `np.radians` raise a numpy `TypeError` (`ufunc 'radians' not supported...`). Do not assert on the message text.

## 4. Returns and guarantees
- A new `np.ndarray` on every call, with `shape == (2, 2)` and `dtype == float64`. It is **real, not complex**, so `np.isrealobj(M)` is True.
- Element layout:
  - `M[0,0] = cos²θ`
  - `M[0,1] = M[1,0] = cosθ·sinθ`. These two are bit-identical, so `M[0,1] == M[1,0]` holds exactly.
  - `M[1,1] = sin²θ`
- Key values (use `np.allclose` with `atol=1e-12`):

  | angle_deg | expected matrix | notes |
  |---|---|---|
  | `0` | `[[1,0],[0,0]]` | **exact**, since cos(0)=1 and sin(0)=0 |
  | `90` | ≈ `[[0,0],[0,1]]` | `M[0,0]` ≈ 3.7e-33 and `M[0,1]` ≈ 6.1e-17, so it is not exactly 0 |
  | `45` | ≈ `[[0.5,0.5],[0.5,0.5]]` | |
  | `-45` | ≈ `[[0.5,-0.5],[-0.5,0.5]]` | |
  | `30` | `[[0.75, √3/4≈0.4330127], [0.4330127, 0.25]]` | |
  | `60` | `[[0.25, 0.4330127], [0.4330127, 0.75]]` | |
  | `135` | ≈ `[[0.5,-0.5],[-0.5,0.5]]` | |

- Algebraic invariants hold for any θ, within about 1e-12:
  - Symmetric: `M == M.T`.
  - Idempotent: `M @ M ≈ M`.
  - `trace == 1`.
  - `det ≈ 0`.
  - Eigenvalues are ≈ {0, 1}.
- Periodicity:
  - Period 180°: `P(θ+180) ≈ P(θ)`, and also `P(θ+360)`.
  - `P(-θ)` has the same diagonal as `P(θ)` with the off-diagonal sign flipped.
  - `P(θ) + P(θ+90) ≈ I` (identity).
  - `P(θ) @ P(θ+90) ≈ 0` (crossed polarizers).
- Consistency with the module's rotation convention: with `R = [[c, s], [-s, c]]`, `P(θ) = R.T @ diag(1,0) @ R`. This is the same construction used by `quarter_wave_plate` and `half_wave_plate`.

## 5. Side effects
- None. There is no state and no mutation, and the matrix is freshly built on each call.

## 6. Branches and edge cases
- There is a single code path with no branches.
- An array input such as `np.array([0, 90])` gives an object of shape `(2, 2, N)`, not `(2, 2)`. Avoid this in tests unless it is asserted explicitly.
- Large angles reduce naturally: `P(405) ≈ P(45)`.

## 7. Rules from docstrings and comments
- The docstring says only "Linear polarizer at angle".
- The module docstring fixes the conventions:
  - Jones vectors are `(Ex, Ey)`.
  - `rotator()` is `[[c,s],[-s,c]]`.
  - The polarizer angle is measured from the x-axis toward +y. For example, `P(θ)` applied to `linear_angle(θ)` returns `linear_angle(θ)` unchanged.

## 8. Testing hints
Observable behaviour through other public API:
```python
from openlens.polarization import JonesMatrices as J, PolarizationState as S, PolarizationCalculator
h = S.linear_horizontal()
out = h.apply_matrix(J.linear_polarizer(60))
out.intensity()          # ≈ cos²60 = 0.25   (Malus)
PolarizationCalculator().malus_law(1.0, 60)  # ≈ 0.25, same value
S.linear_angle(30).apply_matrix(J.linear_polarizer(30)).jones_vector  # ≈ [cos30, sin30]
S.linear_vertical().apply_matrix(J.linear_polarizer(0)).intensity()   # exactly 0.0
h.apply_matrix(J.linear_polarizer(0)).jones_vector  # exactly [1+0j, 0+0j]
```
- General rule (Malus): for `linear_angle(φ)` passed through `P(θ)`, `intensity ≈ cos²(θ-φ)`. Parametrize this over several (θ, φ) pairs.
- Assertions that catch typical regressions:
  - The exact value at 0°.
  - The off-diagonal sign at -45° and 135°.
  - The sin² vs cos² placement at 30° (0.75 vs 0.25) and 60°.
  - Degrees vs radians, using the 30° values.
  - Idempotence.
  - Trace = 1.
  - Real dtype.
- Pitfalls that would make a test fail on the current code:
  - Do not use `==` for anything except 0° and the `M[0,1]==M[1,0]` symmetry.
  - Do not expect exact zeros at 90°.
  - Do not expect a complex dtype.
  - Do not expect the input to be interpreted as radians.

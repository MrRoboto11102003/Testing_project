# Notes: `GaussianBeam.w_y` (property)

## 1. Purpose
- Returns the beam radius (1/e² intensity) in the Y (sagittal) plane.
- It is computed from the complex beam parameter `q_y` and the local wavelength each time it is read.
- Source:
  ```python
  if self.q_y.imag == 0:
      return 0.0
  return math.sqrt(-self.wavelength / (math.pi * (1.0 / self.q_y).imag))
  ```

## 2. Inputs (dataclass fields read)
- `GaussianBeam` is a `@dataclass` with these fields, in order:
  - `wavelength: float` (the LOCAL wavelength, in mm)
  - `q_x: complex`
  - `q_y: complex`
  - `ray: Ray3D`
  - `n: float = 1.0`
- `w_y` reads only `self.q_y` and `self.wavelength`.
  - It does **not** read `q_x`, `n` or `ray`.
  - `ray` is not type-checked, so `None` works for `w_y` tests.
- Construct a beam like this:
  `GaussianBeam(wavelength, q_x, q_y, None)`
  or `GaussianBeam(wavelength=..., q_x=..., q_y=..., ray=None, n=1.0)`.
- No validation is done. A plain int or float `q_y` is accepted, because its `.imag` is 0, so the result is `0.0`.

## 3. Raises
- **`ValueError: math domain error`** (from `math.sqrt`) when the argument is negative. Example:
  - `GaussianBeam(1.0, 1j, -1j, None).w_y`
  - Here 1/(-1j) = 1j, the imaginary part is +1, so the sqrt argument is -1/π.
- **`ValueError`** also when `wavelength < 0` and `q_y` has positive imaginary part. Example: `GaussianBeam(-1.0, 1j, 1j, None).w_y`.
- **No `ZeroDivisionError` for `q_y = 0j`**: the `imag == 0` check runs before `1/q_y`, so the result is `0.0`.

## 4. Returns and guarantees
- Returns a Python `float`. Nothing is cached.
- When `q_y.imag != 0`:
  - Formula: `w_y = sqrt(-λ / (π · Im(1/q_y)))`.
  - For `q_y = z + i·zR` this equals `sqrt(λ·(z² + zR²) / (π·zR))`, which is `w0·sqrt(1 + (z/zR)²)` with `w0 = sqrt(λ·zR/π)`.
- Exact checkable values:

| wavelength | q_y | w_y |
|---|---|---|
| `math.pi` | `1j` | `1.0` (1/q = -1j) |
| `math.pi` | `1+1j` | `sqrt(2)` ≈ 1.41421356 (Im(1/q) = -0.5) |
| `math.pi` | `4j` | `2.0` |
| `0.001` | `1j*math.pi` | `sqrt(0.001)` ≈ 0.0316227766 |
| `0.0` | `1j` | `0.0` (sqrt(-0.0) is 0.0) |
| any | `5+0j`, `0j`, `complex(float('inf'), 0)`, `3.0` | `0.0` |

- Waist round-trip: with `zR = π·w0²/λ` and `q_y = 1j*zR`, `w_y ≈ w0`.
  - Example: λ = 0.00055, w0 = 0.05.

## 5. Side effects
- None. This is a pure read and does not modify the beam.

## 6. Branches and edge cases
- `q_y.imag == 0` gives `0.0`. This is an early return and includes the plane-wave value `complex(inf, 0)` that `refract` can set.
- Otherwise the result comes from the sqrt formula, which raises if its argument is negative.
- Independence from `q_x`: for `GaussianBeam(math.pi, 0j, 1j, None)`, `w_y == 1.0` and `w_x == 0.0`.
  - Use different `q_x` and `q_y` values so that an x/y mix-up would be caught.
- Independence from `n`: `GaussianBeam(math.pi, 1j, 1j, None, n=1.5).w_y == 1.0`.

## 7. Rules from comments and docstrings
- `wavelength` is the LOCAL wavelength (vacuum / n). `q` is defined against the local medium.
- `refract` rescales `wavelength` by `n1/n2` so that `w` stays correct inside glass.
- `advance_q(d)` adds `d / n` to both `q_x` and `q_y`. It falls back to adding `d` when `n` is 0.
  - It does not move the ray, so `ray=None` is fine.
- The docstring describes `1/q = 1/R - i·λ/(π·w²)`.

## 8. Testing hints
- Import with `from openlens.analysis.beam_synthesis import GaussianBeam`. Compare with `pytest.approx`.
- Propagation through `advance_q` (safe with `ray=None`):
  - `b = GaussianBeam(math.pi, 1j, 1j, None)`, then `b.advance_q(1.0)`, gives `q_y == 1+1j` and `b.w_y ≈ sqrt(2)`.
  - With `n=2.0`, `advance_q(2.0)` gives the same `sqrt(2)`, because the step is d/n = 1.
- Flat-surface refraction preserves `w_y`:
  - Steps: `b = GaussianBeam(math.pi, 1j, 1j, None)`, then `b.refract(1.0, 1.5, 0.0)`.
  - Expected state: `b.q_y ≈ 1.5j`, `b.wavelength ≈ math.pi/1.5`, `b.n == 1.5`.
  - Result: `b.w_y ≈ 1.0` (unchanged).
  - This catches regressions where `refract` forgets to rescale the wavelength. In that case `w_y` would become `sqrt(1.5)` ≈ 1.2247.
- Do not call `propagate()` with `ray=None`, because it calls `self.ray.propagate`. Use `advance_q` instead.
- Pitfalls:
  - The sign convention is that positive Im(q) gives a real result; negative Im(q) raises.
  - Zero imaginary part returns exactly `0.0`; it does not raise.

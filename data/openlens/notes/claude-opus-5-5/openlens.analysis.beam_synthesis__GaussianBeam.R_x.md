# Notes: `GaussianBeam.R_x` (property, `openlens.analysis.beam_synthesis`)

## 1. Purpose
- Read-only property. It returns the wavefront radius of curvature in the X (meridional) plane, computed from the complex beam parameter `q_x`.
- Formula: `R = 1 / Re(1/q_x)`. It returns `float("inf")` when `Re(1/q_x) == 0` (a flat wavefront).

## 2. Inputs
- No arguments. It reads only `self.q_x`.
- It does **not** use `q_y`, `wavelength`, `n` or `ray`.
- `GaussianBeam` is a plain `@dataclass(wavelength, q_x, q_y, ray, n=1.0)` and validates nothing.
  - `q_x` can be a `complex`, a `float` or an `int`. Python's `1.0 / x` and `.real` work for all three.
- Code:
  ```python
  inv_q = 1.0 / self.q_x
  if inv_q.real == 0: return float("inf")
  return 1.0 / inv_q.real
  ```

## 3. Raises
- `ZeroDivisionError` when `q_x == 0`. The message depends on the type and Python version, so match only the type:
  - `GaussianBeam(5e-4, 0j, 1j, None).R_x` raises on complex division by zero.
  - `q_x=0.0` (float) also raises `ZeroDivisionError`.
- `AttributeError` when assigning: `beam.R_x = 1.0`. The property has no setter.

## 4. Returns and guarantees
- The result is a Python `float`, computed fresh on every access. Nothing is cached.
- For `q = z + i*zR`: `R = (z² + zR²)/z = z + zR²/z`. The sign follows `z`:
  - `z > 0` gives a positive R.
  - `z < 0` gives a negative R.
- Concrete values (use `pytest.approx`):
  - `q_x = 1+1j`: `1/q = 0.5-0.5j`, so `R_x == 2.0`.
  - `q_x = 2+1j`: `1/q = 0.4-0.2j`, so `R_x == 2.5`.
  - `q_x = -3+4j`: `1/q = -0.12-0.16j`, so `R_x == -25/3 ≈ -8.3333`.
  - `q_x = 3+4j`: `R_x == 25/3`.
  - `q_x = 4+0j` (real q): `R_x == 4.0`.
  - `q_x = 5.0` (plain float): `R_x == 5.0`.
  - `q_x = 2j`, or any purely imaginary value (beam at its waist): `1/q = -0.5j`, real part `0.0`, so `R_x == float("inf")`.
  - `q_x = -7j`: also `inf`. It is never `-inf`, because the special case always returns `+inf`.

## 5. Side effects
- None. The beam's state is not mutated.

## 6. Branches and edge cases
- **`inv_q.real == 0` branch**: returns `+inf`.
  - Reached by pure-imaginary `q_x`, for example the initial pupil beamlet `q0 = complex(0, zR)`.
  - Test it with `math.isinf(r) and r > 0`.
- **Normal branch**: returns `1.0/inv_q.real`, which is finite and keeps the sign.
- **`q_x == 0`**: raises `ZeroDivisionError` (see section 3).
- `q_x = complex(inf, 0)`: `refract` sets this for a "plane wave". `1/q` evaluates to roughly `0j`, which gives `inf`. This depends on the details of CPython's complex division, so do **not** test it.
- NaN `q_x` returns NaN, because `nan == 0` is False. Avoid testing this.

## 7. Rules from comments/docstrings
- The class docstring says `1/q = 1/R - i*lambda/(pi*w²)`, so `R` is the reciprocal of the real part of `1/q`.
- The pupil beamlets start with a planar wavefront, `q = i*zR`, which means R is infinite.
- `advance_q(d)` adds `d/n` to both `q_x` and `q_y`. If `n` is falsy, it adds `d` instead.

## 8. Testing hints
- Construction: the `ray` argument is not touched by `R_x`, so `GaussianBeam(5e-4, q_x, q_y, None)` works.
  - If you want a real ray instead:
    ```python
    from openlens.ray import Ray3D
    from openlens.vector3 import vec3
    ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0))
    ```
- Pin **independence from `q_y`**:
  - `GaussianBeam(5e-4, 1+1j, 2j, None)` gives `R_x == 2.0` and `R_y == inf`.
  - Swapping the q values swaps the results.
- Pin that `wavelength` and `n` do not matter: `R_x` for `q_x=1+1j` is `2.0` with `wavelength=1.0` and with `n=1.5`.
- Via `advance_q`:
  - `b = GaussianBeam(5e-4, 1j, 1j, None)` starts with `R_x == inf`.
  - `b.advance_q(1.0)` makes `q_x = 1+1j`, so `R_x == 2.0`.
  - With `n=1.5`: `advance_q(3.0)` adds a step of 2, making `q_x = 2+1j`, so `R_x == 2.5`. A regression that ignores `n` would give `q = 3+1j`, so `R = 10/3`.
- Via `refract` on a flat surface (radius 0 means `Phi = 0`):
  - `b = GaussianBeam(5e-4, 1+1j, 1+1j, None)`, then `b.refract(1.0, 2.0, 0.0)`.
  - Now `1/q = 0.25-0.25j`, so `R_x == 4.0`.
- Via `propagate(d)`: it calls `ray.propagate`, so you need a real `Ray3D` (not `None`).
- Pitfalls:
  - Use `pytest.approx` for every finite value.
  - Compare infinity with `== float("inf")` or `math.isinf`; do not use `approx`.
  - Do not assert on `ZeroDivisionError` message text.
  - The property never returns `0.0` and never returns `-inf`.

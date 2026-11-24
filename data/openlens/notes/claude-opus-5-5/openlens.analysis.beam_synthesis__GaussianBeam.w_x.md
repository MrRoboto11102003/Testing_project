# Notes: `openlens.analysis.beam_synthesis.GaussianBeam.w_x` (property)

## 1. Purpose
- Returns the 1/e² beam radius in the X (meridional) plane, computed from the complex beam parameter `q_x` and the beam's local `wavelength`.

## 2. Inputs
- This is a read-only `@property` with no arguments. It reads only `self.q_x` and `self.wavelength`.
- It ignores `q_y`, `ray` and `n`. Changing them does not change `w_x`.
- `GaussianBeam` is a plain `@dataclass(wavelength, q_x, q_y, ray, n=1.0)` and validates nothing. The constructor accepts `ray=None`, and `w_x` never touches the ray.
- Exact formula from the source:
  ```python
  if self.q_x.imag == 0: return 0.0
  return math.sqrt(-self.wavelength / (math.pi * (1.0 / self.q_x).imag))
  ```
- `q_x` can be any object with `.imag` that supports `1.0 / q`. That includes `complex`, `float` and `int`; for a float or int, `.imag == 0`, so the result is `0.0`.

## 3. Raises
- **`ValueError: math domain error`** (from `math.sqrt`) when the argument under the root is negative:
  - `q_x` has a negative imaginary part, e.g. `GaussianBeam(1.0, -1j*math.pi, 1j, None).w_x`. Here 1/q = +i/π, so the root argument is −1.
  - `wavelength` is negative while `q_x` is normal, e.g. `GaussianBeam(-1.0, 1j*math.pi, 1j, None).w_x`.
- When `q_x.imag == 0`, there is no division and nothing is raised. This covers `q_x = 0j`, `5+0j` and `complex(inf, 0)`.

## 4. Returns and guarantees
- Returns a Python `float`. It is recomputed on every access (no caching).
- Math: with `q = z + i·zR`, the result is w = sqrt(λ·(z² + zR²) / (π·zR)).
- At a waist (q = i·zR), the result is sqrt(λ·zR/π).
- Concrete values (λ = `wavelength`):
  - `wavelength=1.0, q_x=1j*math.pi` → `1.0`
  - `wavelength=1.0, q_x=4j*math.pi` → `2.0`
  - `wavelength=1.0, q_x=math.pi + 1j*math.pi` → `sqrt(2)` ≈ 1.41421356
  - `wavelength=4.0, q_x=1j*math.pi` → `2.0` (scales as sqrt(λ))
  - `wavelength=0.0, q_x=1j` → `0.0`
- Round trip with the module's own convention: with `zR = math.pi*w0**2/wavelength` and `q_x = 1j*zR`, `w_x == w0` (within float tolerance). For example, `wavelength=0.00055, w0=0.1` gives `w_x ≈ 0.1`.

## 5. Side effects
- None. It does not mutate the beam and has no global state.

## 6. Branches and edge cases
- **Zero imaginary part:** `q_x.imag == 0` → returns exactly `0.0`.
  - Covers `GaussianBeam(1.0, 0j, 1j, None)`, `(1.0, 3+0j, …)`, `(1.0, complex(float('inf'),0), …)`, and `(1.0, 2.0, …)` with a float `q_x`.
- **Normal path:** positive imaginary part and positive wavelength → positive float, as in the table above.
- **Larger |z| gives a wider beam:**
  - For fixed zR = π and λ = 1:
    - z = 0 → 1.0
    - z = π → √2
    - z = −π → √2 (symmetric in the sign of z)

## 7. Rules stated in comments/docstrings
- The docstring says "Beam radius (1/e^2 intensity) in X plane".
- The class docstring gives the convention 1/q = 1/R − i·λ/(π·w²).
- `wavelength` is the **local** wavelength (vacuum/n).
- `refract()` rescales `wavelength` by n1/n2 together with q, so the beam radius is preserved across a flat interface.
- `advance_q` steps q by `distance / n` (or by `distance` if `n` is 0).

## 8. Testing hints
- Import with `from openlens.analysis.beam_synthesis import GaussianBeam`. Pass `ray=None` for pure `w_x` tests.
- Compare floats with `pytest.approx` / `math.isclose`. Use `== 0.0` only for the zero-imaginary branch.
- Useful invariants that tie `w_x` to the other methods (none of them need a ray):
  - **Flat refraction keeps the radius:** `b = GaussianBeam(1.0, 1j*math.pi, 1j*math.pi, None)`; `b.refract(1.0, 1.5, 0.0)`.
    - Afterwards `b.w_x ≈ 1.0`, `b.wavelength ≈ 1/1.5` and `b.n == 1.5`.
    - The radius 0.0 counts as flat (|R| < 1e-9).
  - **Advancing q divides by n:** `b = GaussianBeam(1.0, 1j*math.pi, 1j*math.pi, None, n=2.0)`; `b.advance_q(2*math.pi)`.
    - Then `q_x == math.pi + 1j*math.pi` and `w_x ≈ sqrt(2)`.
    - With `n=1.0`, `advance_q(math.pi)` gives the same result.
  - Do **not** call `propagate()` with `ray=None`, because it calls `self.ray.propagate`. To use `propagate`, build a real ray: `GaussianBeam.__module__`'s `Ray3D(vec3(0,0,0), vec3(1,0,0))`, with `vec3` from `openlens.vector3`.
- **Pitfalls to avoid:**
  - Do not expect `w_x` to depend on `q_y` or `n`.
  - Do not expect an exception for `q_x = 0j`; it returns `0.0`.
  - A negative-imaginary `q_x` raises `ValueError`; it does not return NaN.

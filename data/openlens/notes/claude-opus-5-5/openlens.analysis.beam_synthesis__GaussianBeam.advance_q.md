# Notes: `GaussianBeam.advance_q(self, distance: float) -> None`

## 1. Purpose
- Advances the complex beam parameters `q_x` and `q_y` by a physical distance inside the beam's local medium.
- Does **not** move the central ray. The BSP tracer has already moved the ray, so it must not be moved twice.

## 2. Inputs
- `self` is a `GaussianBeam` dataclass with fields `wavelength`, `q_x`, `q_y`, `ray`, and `n=1.0`.
  - The dataclass does not validate types.
- `distance`: a number (int or float). It is not validated.
  - Negative and zero values are accepted.
- The step is computed from `self.n` exactly as written:
  ```python
  step = distance / self.n if self.n else distance
  self.q_x += step
  self.q_y += step
  ```
  - If `self.n` is truthy (e.g. `1.0`, `1.5`, `-2`), then `step = distance / self.n`.
  - If `self.n` is falsy (`0`, `0.0`, `None`), then `step = distance` and no division happens.

## 3. Raises
- The function raises nothing itself.
- The only realistic error is a `TypeError` from the arithmetic.
  - Example: `beam.advance_q("a")` with `n=1.0` gives `TypeError: unsupported operand type(s) for /: 'str' and 'float'`.
  - With `n=0`, `"a"` is not divided; instead `q_x += "a"` raises a `TypeError` (complex + str).
- No `ZeroDivisionError` is possible from `n == 0`, because that case is guarded.

## 4. Returns and guarantees
- Returns `None`.
- `q_x` and `q_y` each get the **same real** increment, `step`.
  - The imaginary parts are unchanged, so the Rayleigh range (zR) is preserved.
- `q_x` and `q_y` are updated independently, so different starting values stay different by the same offset.
- These are unchanged: `wavelength`, `n`, and `ray` (its origin, direction and path).

## 5. Side effects
- Mutates `self.q_x` and `self.q_y` in place (they are rebound to new complex values).
- No global state, logging, or I/O.

## 6. Branches and edge cases
- `n=1.0`, `q=10j`, `advance_q(5.0)` → `q_x == q_y == (5+10j)`.
- `n=1.5`, `q=10j`, `advance_q(3.0)` → `q == (2+10j)`.
  - `3/1.5 == 2.0` exactly, so `==` is safe here.
- `n=2.0`, `advance_q(-4.0)` → real part decreases by `2.0`.
- `n=0.0`, `q=10j`, `advance_q(3.0)` → `q == (3+10j)` (falsy-index fallback, no division).
- `advance_q(0)` → `q` unchanged.
- Calls accumulate: `n=1.5`, `advance_q(1.5)` twice → real part `+2.0`.
- `n` changes after `refract(n1, n2, R)`, which sets `self.n = n2`. A later `advance_q` divides by the new `n`.

## 7. Rules from docstrings and comments
- "q is defined against the *local* medium … so the propagation step inside glass is d/n, not d."
  - Advancing by the full `d` was the old bug (w_x 0.1323 → 0.1621 mm for R=50).
- Kept separate from `propagate`:
  - `propagate(d)` calls `advance_q(d)` and then `self.ray.propagate(d)`.
  - So `propagate` moves the ray by the **physical** `d`, but advances `q` by `d/n`.

## 8. Testing hints
- Build a beam:
  ```python
  from openlens.analysis.beam_synthesis import GaussianBeam
  from openlens.ray import Ray3D
  from openlens.vector3 import vec3
  ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0))
  b = GaussianBeam(wavelength=5.5e-4, q_x=complex(0, 10), q_y=complex(0, 20), ray=ray, n=1.5)
  ```
  - `ray` is never touched by `advance_q`, but use a real `Ray3D` anyway.
- Key regression checks:
  - Real part increases by `distance / n`, not `distance` (use `n != 1` to tell them apart).
  - Both `q_x` and `q_y` change.
  - Imaginary parts are unchanged.
  - `n == 0` does not raise and uses the raw distance.
  - The ray origin is unchanged: compare `ray.origin.x/y/z` before and after.
  - `b.n` and `b.wavelength` are unchanged.
- Contrast with `propagate`:
  - Setup: `n=1.5` and `b.propagate(3.0)`.
  - Expected: `q_x.real` increases by 2.0.
  - The ray origin x moves by 3.0. This assumes `Ray3D.propagate` moves `origin` along `direction`, which is defined in another module; I have not verified it here.
- Derived property check: `w_x` depends only on `Im(1/q)`. After advancing, it is `sqrt(-λ/(π·Im(1/q)))` with the new q. Compute the expected value with the same formula and compare using `pytest.approx`.
- Use `pytest.approx` (or `abs(a-b) < 1e-12`) for non-exact quotients like `1/1.5`.
- Pitfall: do not assume `R_x` stays infinite. Once the real part is nonzero, `R_x` becomes finite.

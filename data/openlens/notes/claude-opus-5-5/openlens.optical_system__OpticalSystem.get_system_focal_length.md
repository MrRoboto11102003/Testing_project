# Notes: `OpticalSystem.get_system_focal_length()`

## 1. Purpose
- Returns the system effective focal length (EFL) in mm.
- An empty system gives `None`.
- A single element gives that lens's own thick-lens focal length.
- Two or more elements give the exact paraxial EFL `-1/C` from a surface-by-surface ABCD matrix.

## 2. Inputs
- `self` only; there are no parameters.
- It reads:
  - `self.elements` (list of `LensElement`, in order).
  - `self.air_gaps[i].thickness`.
  - For each lens: `lens.refractive_index`, `lens.thickness`, `lens.get_effective_radius_1()` and `lens.get_effective_radius_2()`.
- It ignores:
  - Decenter and tilt.
  - Element `position` values (only gap thicknesses matter).
  - Wavelength. It uses the stored `refractive_index`, not `refractive_index_at`.
  - Aperture stop, casing and `image_plane_x_mm`.

## 3. Raises
- Nothing is raised by this method itself.
- Exceptions could only come from the Lens methods it calls (lens module, not shown).

## 4. Returns and guarantees
- The return type is `Optional[float]`.
- `len(elements) == 0`: returns `None`.
- `len(elements) == 1`: returns `self.elements[0].lens.calculate_focal_length()` unchanged.
  - This may itself be `None`; its behaviour is defined in `openlens.lens`, which is not shown.
  - The matrix is **not** used here, and air gaps are irrelevant.
- `len(elements) >= 2`:
  - `_calculate_system_matrix()` is called with no overrides, giving `(A, B, C, D)`.
  - If the matrix is falsy (only possible with no elements), it returns `None`.
  - If `abs(C) < 1e-10`, it returns `None` (afocal).
  - Otherwise it returns `-1.0 / C`.
- Nothing is cached; each call recomputes.

### Matrix algorithm (private `_calculate_system_matrix`)
- The ray vector is (y, n·θ). The matrix starts at `A, B, C, D = 1, 0, 0, 1`, with `n_current = 1.0`.
- For each element `i`, with `n = lens.refractive_index`:
  1. Read `R1 = get_effective_radius_1()`. If `R1 != 0` and it is finite:
     - `P1 = (n - n_current) / R1`
     - `C -= P1*A`, `D -= P1*B`
  2. Set `d = thickness / n` (or `thickness` if `n` is falsy). Then `A += d*C`, `B += d*D`.
  3. Read `R2 = get_effective_radius_2()`. If `R2 != 0` and it is finite:
     - `P2 = (1 - n) / R2`
     - `C -= P2*A`, `D -= P2*B`
  4. Set `n_current = 1.0`.
  5. If `i < len(elements)-1` and `i < len(air_gaps)`, apply the gap: `A += g*C`, `B += g*D`.
- Infinite (flat) radii contribute no power.
- Non-negative focusing (converging) systems give a negative `C`, so the EFL is positive.

## 5. Side effects
- None. Nothing is mutated, including `modified_at`.

## 6. Branches and edge cases

**Empty system**
- `OpticalSystem().get_system_focal_length()` returns `None`.

**Single lens**
- `s.add_lens(L)` gives `s.get_system_focal_length() == L.calculate_focal_length()`.
- `air_gap_before` on the first lens creates no gap and has no effect.

**Two lenses, worked example (hand-computed from the code)**
```python
L1 = Lens(name="a", radius_of_curvature_1=100.0, radius_of_curvature_2=-100.0,
          thickness=5.0, diameter=25.0, refractive_index=1.5)
L2 = Lens(name="b", radius_of_curvature_1=100.0, radius_of_curvature_2=-100.0,
          thickness=5.0, diameter=25.0, refractive_index=1.5)
s = OpticalSystem(); s.add_lens(L1); s.add_lens(L2, air_gap_before=10.0)
# C = -0.018519375  ->  EFL = 1/0.018519375 ≈ 53.99750
```
- Use `pytest.approx(53.9975, rel=1e-5)`.
- Intermediate checks for L1 alone: `C = -0.0099166667`, which matches thick-lens 1/f with f ≈ 100.840.

**Gap sensitivity**
- Setting `s.air_gaps[0].thickness = 0.0` and recalling changes the result. The gap value is read live; no rebuild is needed.
- The cemented case (gap 0) should be recomputed by hand rather than reused from the 10 mm example.

**Afocal**
- If `|C| < 1e-10`, the result is `None`.
- Easiest way to reach it: monkeypatch the private matrix on the instance:
```python
s._calculate_system_matrix = lambda n_overrides=None: (1.0, 0.0, 0.0, 1.0)  # -> None
s._calculate_system_matrix = lambda n_overrides=None: (1.0, 0.0, 5e-11, 1.0) # -> None
s._calculate_system_matrix = lambda n_overrides=None: (1.0, 0.0, -0.01, 1.0) # -> 100.0
s._calculate_system_matrix = lambda n_overrides=None: (1.0, 0.0, 0.01, 1.0)  # -> -100.0
s._calculate_system_matrix = lambda n_overrides=None: None                   # -> None
```
- Monkeypatching only matters when there are 2 or more elements. With a single lens the matrix is never called, so a patched matrix has no effect.
- A real afocal case: two plano-plano lenses (both radii `float('inf')`) give C exactly 0, so `None`. Whether `Lens` accepts infinite radii is decided in `openlens.lens` (not shown); prefer the monkeypatch.

**Sign**
- A negative-power pair (e.g. R1=-100, R2=100 on both lenses) gives a negative EFL.
- The result is not passed through `abs()`.

**Ordering and tilt**
- Tilt and decenter (`add_lens(..., tilt_x=5)`) do not change the result.

## 7. Rules stated in comments and docstrings
- The old two-lens thin-lens combination formula (`1/f1 + 1/f2 - d/(f1 f2)`) was deliberately removed. Every multi-element system, including two lenses, must use the matrix.
- For thick or cemented pairs, the old formula disagrees by about 10%.
- Glass propagation uses the reduced thickness `d/n`; this is correct, not a bug.
- Zero-gap (cemented) pairs are exact.
- Parabolic surfaces use the vertex radius via `get_effective_radius_*` (see the lens module).

## 8. Testing hints
- Always pass `refractive_index=` explicitly. Otherwise the index comes from material defaults in the lens module.
- For two or more elements, compute the expected value with your own copy of the ABCD loop above, and compare with `pytest.approx(rel=1e-9)`.
- Two good regression checks:
  - A thick two-lens system must **not** equal the thin-combination value `f1*f2/(f1+f2-d)`, using `f_i = L_i.calculate_focal_length()` and `d` = gap. Verify by your own computation that they differ by more than 1e-6 relative before asserting this.
  - The result must depend on lens thickness through `d/n`. For example, changing `L2.thickness` changes the EFL.
- Related public methods give consistency checks:
  - `calculate_optical_power() == 1000/EFL`.
  - `get_f_number() == abs(EFL)/elements[0].lens.diameter`.
- Pitfalls:
  - Do not assume the single-lens result equals a matrix computation; compare it to `lens.calculate_focal_length()`.
  - Use `approx`, not `==`, for floats in the multi-element case.

# Notes: `OpticalSystem.calculate_chromatic_aberration(self) -> Dict[str, Any]`

## 1. Purpose
- Computes longitudinal (axial) chromatic aberration of the whole system.
- It rebuilds the paraxial ABCD matrix at the F, d and C Fraunhofer lines, then reports back focal lengths (BFL), effective focal lengths (EFL), their difference, and a "corrected" flag.

## 2. Inputs
- No parameters. It reads the following:
  - `self.elements`, in order.
  - Each `el.lens.refractive_index_at(wl)`. This is defined in `openlens.lens`, which is not shown here; it probably uses the material DB or model glass.
  - `lens.get_effective_radius_1()` / `get_effective_radius_2()`.
  - `lens.thickness`.
  - `self.air_gaps[i].thickness`.
- Wavelengths come from `openlens.constants`: `WAVELENGTH_F_LINE`, `WAVELENGTH_D_LINE`, `WAVELENGTH_C_LINE`. The exact values are not shown, so import them instead of hard-coding them.
- Loop order is F, then d, then C. For each line, `refractive_index_at` is called once per element, in element order.
- `lens.refractive_index` is **ignored** whenever the system has at least one element. In that case the override map is non-empty, and the override index is always used.

## 3. Raises
- The method itself raises nothing explicitly.
- Exceptions from `refractive_index_at` or the radius getters would propagate. That code lives in `lens.py` and is not visible here.

## 4. Returns and guarantees
For each line, the private helper `_calculate_system_matrix(n_overrides)` builds the matrix:
- It starts from the identity matrix.
- Per element, in order:
  - **Surface 1:** if R1 is finite and non-zero, `P1 = (n - 1)/R1`, then `C -= P1*A` and `D -= P1*B`.
  - **Glass:** propagate the reduced thickness `d = t/n` (use plain `t` if n is 0): `A += d*C`, `B += d*D`.
  - **Surface 2:** if R2 is finite and non-zero, `P2 = (1 - n)/R2`, applied the same way.
  - **Air gap:** for every element except the last, propagate `air_gaps[i].thickness`.
- If `abs(C) > 1e-10`: `bfl = -A/C` and `efl = -1/C`. Otherwise both are `None`.

**Full result** (when `bfl_F` and `bfl_C` are both not None) is a new dict with exactly these 8 keys:
```
{"longitudinal": bfl_C - bfl_F, "bfl_F", "bfl_d", "bfl_C",
 "f_F", "f_d", "f_C", "corrected": abs(longitudinal) < 0.1}
```
- `corrected` uses a strict `<`, and its value is a Python `bool`.
- `bfl_d` and `f_d` can be `None` independently, but only if the d-line matrix alone is afocal.

**Fallback result** (otherwise): exactly `{"longitudinal": 0.0, "corrected": False}`, with no other keys.

Closed form for a single lens with spherical surfaces (n is the index at that line):
- `1/f = (n-1)(1/R1 - 1/R2) + (n-1)**2 * t / (n*R1*R2)`
- `bfl = f * (1 - (t/n)*(n-1)/R1)`

## 5. Side effects
- None.
- No mutation of lenses or elements, `modified_at` is not touched, and nothing is cached.
- A fresh dict is returned on every call.

## 6. Branches and edge cases
- **Empty system:** `OpticalSystem().calculate_chromatic_aberration() == {"longitudinal": 0.0, "corrected": False}`.
- **Afocal at F and C** (e.g. all surfaces flat): the fallback dict is returned.
- **Same index at all three lines** (index patched to a constant): F and C are computed identically, so `longitudinal == 0.0` exactly and `corrected is True`.
- **Positive BK7-like singlet with normal dispersion** (n_F > n_d > n_C):
  - `f_F < f_d < f_C`, and the same order holds for the BFLs.
  - So `longitudinal > 0`.
  - For f ≈ 100 mm BK7, it is about 1.5 mm, so `corrected is False`.
- **Negative lens:** the sign of `longitudinal` flips (it becomes negative).
- **Gaps:** the last element's following gap is never applied. Only `air_gaps[0 .. len-2]` are used.

## 7. Rules from docstrings/comments
- The `f_*` keys are true EFLs (`-1/C`). They are not copies of the BFL.
- BFL is `-A/C`, the observable focus position.
- The reduced thickness `d/n` is intentional (ray vector `(y, n·θ)`). Cemented zero-gap pairs are treated exactly.
- `is_achromatic()` returns `bool(result.get("corrected", False))`.

## 8. Testing hints
Pin indices exactly by overriding the bound method on the instance. This assumes `Lens` has no `__slots__`, which is likely; it is a normal class.

```python
from openlens.constants import WAVELENGTH_F_LINE as F, WAVELENGTH_D_LINE as D, WAVELENGTH_C_LINE as C
lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0, thickness=5.0, diameter=25.0)
idx = {F: 1.522, D: 1.5168, C: 1.514}
lens.refractive_index_at = lambda wl: idx[wl]
s = OpticalSystem(); s.add_lens(lens)
r = s.calculate_chromatic_aberration()
```

What to check:
- Compare `r["f_F"]`, `r["bfl_C"]` and so on with the closed form above, using `pytest.approx(rel=1e-9)`.
- Check `r["longitudinal"] == approx(r["bfl_C"] - r["bfl_F"])`.
- Check that `set(r)` equals the 8 keys.
- Verify `lens.refractive_index` is ignored: set it to something absurd (e.g. 3.0) and confirm the result does not change.
- Two-element test: add a second patched lens with `air_gap_before=10.0`. Compute the expected value by multiplying 2x2 matrices by hand. Changing the gap should change the BFL.
- Threshold test: choose indices so that |longitudinal| is about 0.05 (expect `corrected` True) versus about 0.2 (expect False). Avoid values near exactly 0.1.
- Verify ordering: record the wavelengths passed to `refractive_index_at`. For one element it should be `[F, D, C]`.

Pitfalls:
- Do not hard-code wavelength numbers or catalog indices. Without patching, `refractive_index_at` depends on code you cannot see, so only assert qualitative relations (signs and ordering).
- Use `approx` for floats. The one exception is the constant-index case, where exact `0.0` is safe.
- Avoid parabolic or conic lenses; their effective radius comes from `lens.py`, which is not shown.
- Flat surfaces (`float('inf')`) are skipped as having no power. Whether `Lens` accepts them is decided in `lens.py`.

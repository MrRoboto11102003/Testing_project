# Notes: `openlens.aberrations.analyze_lens_quality(lens, field_angle=5.0, **kwargs)`

## 1. Purpose
- Runs `AberrationsCalculator(lens).calculate_all_aberrations(...)`.
- Turns the result into a 0–100-style quality score, a rating string and a list of issue strings.

## 2. Inputs
- `lens`: a `Lens`.
  - `AberrationsCalculator.__init__` reads `refractive_index`, `radius_of_curvature_1/2`, `thickness` and `diameter`.
  - The object must NOT have an `elements` attribute, or it is treated as a system.
  - No `entrance_pupil_mm` is passed, so the pupil is `lens.diameter`.
- `field_angle` (float, default `5.0`, degrees).
  - It is overridden by `kwargs["field_angle_deg"]` if present: `field_angle = kwargs.get("field_angle_deg", field_angle)`.
- `kwargs["wavelength_nm"]`: defaults to `constants.WAVELENGTH_GREEN`. That value is not visible here; import it rather than hardcoding.
- Other kwargs are ignored. There is no validation of any input.
- The exact call made is: `calc.calculate_all_aberrations(field_angle=<angle>, wavelength_nm=<wl>)`.
  - Both are keyword arguments; `object_distance_mm` keeps its default `None`.

## 3. Raises
- The function itself has no explicit `raise`.
- `ValueError` cannot come from the object-distance check, because the distance is always `None`.
- `TypeError` is possible if a traced metric comes back `None` (trace failed), because of `abs(None)` or `None > 1.0`:
  - `results["distortion"]` is `None` → `abs(dist)` fails.
  - `results["astigmatism"]` is `None` → `ast > 1.0` fails.
- Don't rely on this in tests; it only happens for geometries whose trace fails.
- Spherical and chromatic are never `None` for a singlet:
  - Spherical falls back to Seidel.
  - Chromatic is `|f|/abbe`.

## 4. Returns
**Error branch:** taken if `results.get("error")` is truthy (focal length is `None`). The result is exactly:
```python
{"quality_score": 0, "rating": "Error",
 "issues": ["Cannot calculate focal length (zero optical power)"]}
```
- There is no `"aberrations"` key in this branch.
- When `Lens.calculate_focal_length()` returns `None` is defined in `openlens.lens`, which is not visible here (presumably zero optical power).

**Normal branch:** `{"quality_score": int, "rating": str, "issues": list[str], "aberrations": results}`.
- `"aberrations"` is the very dict returned by `calculate_all_aberrations` (same object, unchanged).
- `"issues"` is `["No significant aberrations detected"]` when no penalty applies.

**Scoring:** start at 100 and apply the checks in this order (all comparisons strict `>`). Issues are appended in the same order.

| Check | Penalty and issue text |
|---|---|
| `sa = abs(results["spherical"])`; `sa > SPHERICAL_ABERRATION_EXCELLENT` | −40, `f"High spherical aberration ({sa:.4f} mm)"` |
| elif `sa > SPHERICAL_ABERRATION_EXCELLENT/10` | −20, `f"Moderate spherical aberration ({sa:.4f} mm)"` |
| `ca = results["chromatic"]` (no abs); `ca > 0.5` | −40, `f"High chromatic aberration ({ca:.4f} mm)"` |
| elif `ca > 0.1` | −20, `"Moderate chromatic aberration (...)"` |
| `d = abs(results["distortion"])`; `d > 5` | −15, `f"High distortion ({d:.2f}%)"` |
| elif `d > 1` | −5, `f"Moderate distortion ({d:.2f}%)"` |
| `ast = results["astigmatism"]` (no abs); `ast > 1.0` | −15, `f"High astigmatism ({ast:.4f} mm)"` |
| elif `ast > 0.1` | −5, `f"Moderate astigmatism ({ast:.4f} mm)"` |

**Rating:** checked top-down with `>=`. The constants are in `openlens.constants` (not shown); the numbers in the comments imply EXCELLENT=85, GOOD=70, FAIR=50.

| Condition | Comment value | Rating |
|---|---|---|
| `score >= QUALITY_EXCELLENT_THRESHOLD+5` | 90 | `"Excellent"` |
| `score >= QUALITY_GOOD_THRESHOLD+5` | 75 | `"Good"` |
| `score >= QUALITY_FAIR_THRESHOLD+10` | 60 | `"Fair"` |
| `score >= QUALITY_FAIR_THRESHOLD-10` | 40 | `"Poor"` |
| otherwise | — | `"Very Poor"` |

- The minimum possible score is 100−40−40−15−15 = −10. The score is not clamped.

## 5. Side effects
- `calculate_all_aberrations` temporarily calls `lens.update_refractive_index(wavelength_nm=...)`.
- In `finally` it restores `lens.wavelength` and `lens.refractive_index`, so the lens is unchanged afterwards.
- A fresh calculator is used, so no cache persists between calls.

## 6. Branches / real-lens facts
**Chromatic for a singlet** is `abs(f)/V`.
- `V` is looked up by `lens.material`: BK7 64.17, "Fused Silica" 67.8, "Crown Glass" 60.0, "Flint Glass" 36.0, SF11 25.76, Sapphire 72.0.
- If the material is not in that table, `V` is estimated from the index: n<1.5→65, n<1.6→55, n<1.7→40, else 30.
- That index is `lens.refractive_index` read at calculator construction, i.e. before the wavelength update.
- `f` is computed at the requested wavelength.
- Consequence for BK7: |f| > ~32.1 mm gives "High chromatic"; 6.4 < |f| < 32.1 gives "Moderate".

**Field angle:**
- With `field_angle=0`, coma and astigmatism are exactly `0.0`, so there is no astigmatism issue.
- Distortion and field curvature still come from a ray trace.
- The exact traced values are not predictable from this module.

## 7. Comment-stated intent
- The comments give the rating cut-offs 90/75/60/40.
- Penalties are "major" (−40) and "minor" (−20) for spherical aberration, and the same pattern is reused for chromatic aberration.

## 8. Testing hints
**Best approach: monkeypatch the aberration call** for deterministic scoring.
```python
import openlens.aberrations as ab
from openlens.lens import Lens
fake = {"spherical": 0.0, "chromatic": 0.0, "distortion": 0.0, "astigmatism": 0.0}
monkeypatch.setattr(ab.AberrationsCalculator, "calculate_all_aberrations",
                    lambda self, **kw: (seen.update(kw), fake)[1])
r = ab.analyze_lens_quality(Lens())
```
- With all four metrics at 0 the result is: score 100, `"Excellent"`, `["No significant aberrations detected"]`, and `r["aberrations"] is fake`.
- The fake must accept keyword args; capture `kw` to assert `field_angle` and `wavelength_nm`.
  - Default call: `field_angle=5.0`.
  - `field_angle_deg=7` overrides `field_angle=3`.
- Import `SPHERICAL_ABERRATION_EXCELLENT` and the `QUALITY_*` constants from `openlens.constants`; don't hardcode them. Use spherical values like `1.5*E` and `0.5*E`.

**Boundaries:**
- Values exactly at a threshold do not penalize (strict `>`):
  - chromatic 0.5 → only −20 (because 0.5 > 0.1);
  - chromatic 0.1 → no penalty;
  - distortion 5 → −5; distortion 1 → none.
- Negative values:
  - Distortion −6 → "High distortion (6.00%)", because abs is applied.
  - Chromatic −1 → no penalty (no abs).
  - Spherical −x → penalized via abs, and the message shows the positive value.

**Error path:** return `{"error": "boom"}` from the fake. Expect `{"quality_score": 0, "rating": "Error", "issues": ["boom"]}` and no `"aberrations"` key.
- An empty-string `error` is falsy, so the normal path runs (and needs the four metric keys present).

**Exact strings to check:**
- Issue order: spherical, chromatic, distortion, astigmatism.
- Formatting: `.4f mm` for spherical/chromatic/astigmatism, `.2f%` for distortion (no space before `%`).
- Example: score 100−40−20−5−5 = 30 → `"Very Poor"`; score 60 → `"Fair"`; score 75 → `"Good"`.

**Real (un-mocked) runs:**
- Only assert structure, the restored lens index, and chromatic = `abs(r["aberrations"]["focal_length"])/64.17` for BK7.
- Don't assert traced values or the exact score.

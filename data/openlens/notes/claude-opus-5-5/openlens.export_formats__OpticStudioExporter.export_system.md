# Notes: `openlens.export_formats.OpticStudioExporter.export_system(system, filename)`

## 1. Purpose
- Writes a human-readable "OpticStudio-style" sequential surface table for a multi-element `OpticalSystem` to a text file.
- Returns `None`.

## 2. Inputs
- `system`: an `OpticalSystem`. The function reads these members:
  - `.name`, `.elements` (each has `.lens`), `.air_gaps` (each has `.thickness`)
  - `get_total_length()`, `get_system_focal_length()`, `calculate_back_focal_length()`
  - `getattr(system, "image_plane_x_mm", None)`
- Per lens it reads `radius_of_curvature_1/2` (nominal radii, not the effective ones), `thickness` and `material`.
- `filename`: a path (str). It is written through `atomic_write_text(filename)`, which defaults to utf-8 with `newline=None`. The atomic module is not shown; its docs say the target is replaced only on success.
- Nothing is validated in this function.

## 3. Raises
- The function raises nothing explicitly. Errors come from:
  - the system/lens methods, or
  - the file write (for example, a missing directory; the behaviour of `atomic_write_text` is not visible).
- If an exception happens inside the `with` block, the target file is not replaced.

## 4. Returns / exact file content
The function returns `None`. Lines are written in this exact order (`\n` endings):
```
# OpenLens Export - OpticStudio Format
# System: {system.name}
#============================================================   ("#" + 60 "=")
<blank>
SYSTEM PRESCRIPTION DATA
----------------------------------------                        (40 "-")
Title: {system.name}
Units: Millimeters
Type: Sequential
Number of Elements: {len(system.elements)}
Total Length: {total_length:.4f} mm        <- only if get_total_length() is truthy (nonzero)
System Focal Length: {fl:.4f} mm           <- only if get_system_focal_length() is truthy (not None/0)
<blank>
SURFACE DATA DETAIL
----------------------------------------                        (40 "-")
f"{'Surface':<10} {'Type':<12} {'Radius':<14} {'Thickness':<12} {'Glass':<10}"
------------------------------------------------------------    (60 "-")
f"{'OBJ':<10} {'STANDARD':<12} {'Infinity':<14} {'Infinity':<12} {'':<10}"
... element rows ...
f"{'IMA':<10} {'STANDARD':<12} {'Infinity':<14} {'':<12} {'':<10}"
<blank>
# End of OpenLens Export
```
- Every table row ends with padding spaces before `\n`, because the last column is `<10`. For example, the header ends in `"Glass     "`.
- The name is **not** XML-escaped; `<&` is written verbatim.

**Element rows.** `surf_num` starts at 1 and increments after every row. For each element `i`:
1. **Gap row**, written only if `gap_before > 0`:
   - `gap_before` is `0.0` for `i == 0`, or when `i-1 >= len(air_gaps)`. Otherwise it is `system.air_gaps[i-1].thickness`.
   - Row: `f"{surf_num:<10} {'STANDARD':<12} {'Infinity':<14} {gap:<12.4f} {'':<10}"`
2. **Front row**: `f"{surf_num:<10} {'STANDARD':<12} {r1_str:<14} {lens.thickness:<12.4f} {lens.material:<10}"`
3. **Back row**: `f"{surf_num:<10} {'STANDARD':<12} {r2_str:<14} {back:<12.4f} {'':<10}"`
   - `back` is `0.0` for every element except the last one, which gets `image_distance`.
- `r_str` is `f"{R:.4f}"` if `abs(R) < 1e10`, else `"Infinity"`. So `inf` and `1e10` both give `"Infinity"`, and `-50` gives `"-50.0000"`.
- The material is never truncated. Names longer than 10 characters widen the row.

**Example front row** (surf 1, R1=50, t=5, BK7):
`"1          STANDARD     50.0000        5.0000       BK7       "`

**`image_distance`** (from `_image_distance`, computed once before the loop, even with 0 elements):
- If `system.image_plane_x_mm` exists, converts with `float()`, and is finite: `max(0.0, plane - get_total_length())`.
- Otherwise, `bfl = system.calculate_back_focal_length()`. If `bfl` is `None` or not finite, the result is `0.0`; else `max(0.0, bfl)`.
- A virtual image (negative distance) is therefore clamped to `0.0000`.

## 5. Side effects
- Creates or overwrites `filename`.
- Does not mutate `system` or the lenses.

## 6. Branches / edge cases
- **First element:** never gets a gap row, whatever value `air_gap_before` had in `add_lens`.
- **Zero-thickness gap:** produces no row, so `surf_num` does not advance.
- **Single element:** its back row carries `image_distance`. For multiple elements, all back rows except the last show `0.0000`.
- **Zero elements:** `Number of Elements: 0`, then OBJ is followed directly by IMA. The Total Length line is skipped if the total is 0/falsy, and the System Focal Length line is skipped if the focal length is None. Whether the empty system's methods behave this way is in `optical_system` (not shown), so check them rather than assume.
- **Detector plane:**
  - `image_plane_x_mm = total + 20` gives a last back thickness of `20.0000`.
  - A plane below `total` gives `0.0000`.
  - A non-numeric value (if the attribute accepts it) falls back to the BFL path.
- **Surface count:** rows between OBJ and IMA = `2*N + (number of gaps i>=1 with thickness > 0)`. Numbering is consecutive from 1.

## 7. Rules from comments/docstrings
- "An intermediate element ends at the next gap row, so its own thickness is zero; only the last surface carries the distance to the image."
- "`air_gaps[i]` is the gap *after* element i", so the gap before element i is `air_gaps[i-1]`. Before a fix, the code used `elem.air_gap_before`, which raised `AttributeError`.
- "A configured detector plane wins over the paraxial fallback". The result is clamped at zero ("a virtual image cannot be represented as a negative thickness").

## 8. Testing hints
- **Build:**
  ```python
  s = OpticalSystem(name="Sys")
  s.add_lens(Lens(name="A", radius_of_curvature_1=50, radius_of_curvature_2=-50,
                  thickness=5, diameter=25, material="BK7"))
  s.add_lens(Lens(...), air_gap_before=10)
  ```
  Pass explicit lens values, because the defaults are not visible.
- **Write and read:** use `tmp_path`; read with `open(p, encoding="utf-8").read().splitlines()`. Text mode normalises Windows `\r\n`. Don't compare raw bytes.
- **Use the exporter's own formulas for expected values.** Don't hard-code optics:
  - `f"{s.get_total_length():.4f}"`, `s.get_system_focal_length()`
  - `s.air_gaps[0].thickness`, `s.calculate_back_focal_length()`
  - How `add_lens` fills `air_gaps` is not shown. Assert gap rows against `s.air_gaps[i-1].thickness`, not the literal 10.
- **Pin `image_distance` deterministically:**
  ```python
  s.image_plane_x_mm = s.get_total_length() + 20.0
  ```
  The last back row's thickness column is then `"20.0000"`. This could fail if `OpticalSystem` defines a validating property; that class is not shown.
- **Rebuild expected rows** with the identical f-strings from section 4 and compare whole lines. If you `.strip()`, the trailing padding is lost; that is fine, but then compare the stripped versions on both sides.
- **High-value regression checks:**
  - an intermediate back row shows `0.0000`
  - the gap row exists for `i >= 1` and is absent for `i == 0` and for zero gaps
  - surface numbering is consecutive
  - the `"Infinity"` threshold is `abs(R) < 1e10`
  - nominal R1 is used, not the effective radius
  - the `Number of Elements` count is correct
  - the header and footer text is exact
  - the `"#"+"="*60` and `"-"*40` / `"-"*60` rule lines are exact
- Don't assert on the Total Length or Focal Length lines without first checking whether those values are truthy.

# Notes: `openlens.export_formats.OpticStudioExporter.export_lens(lens, filename)`

## 1. Purpose
- Static method. Writes a human-readable "OpticStudio"-style text prescription for a single `Lens` to `filename`.
- Returns `None`.
- The file is written through `openlens.atomic.atomic_write_text(filename)`, so the target is only replaced if writing succeeds.

## 2. Inputs
- `lens`: a `Lens`. There is no validation. The function reads these attributes and methods:
  - `name`, `material`, `refractive_index`, `diameter`, `bevel_1`, `bevel_2`
  - `radius_of_curvature_1`, `radius_of_curvature_2`, `thickness`
  - `get_clear_aperture_1()`, `get_clear_aperture_2()`, `coating_label(1)`, `coating_label(2)`, `calculate_focal_length()`
- `filename`: a path, either `str` or `tmp_path / "x.txt"`.
  - `atomic_write_text` uses its defaults: `encoding='utf-8'`, `newline=None`.
  - Its internals are in `openlens.atomic`, which was not shown.
- Lens defaults (`DEFAULT_RADIUS_1` etc.) live in `openlens.lens`, which was not shown. Pass every value explicitly, for example `Lens(name="T", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0, thickness=5.0, diameter=25.0, material="BK7")`.
- Build expected numbers from the lens's own attributes, not from your inputs. `refractive_index`, the clear apertures and the focal length are computed inside `Lens`.

## 3. Raises
- The function contains no explicit `raise`.
- Errors can only come from attribute access, from `Lens` methods, or from the file writer. For example, writing to a non-existent directory raises an `OSError` from `atomic_write_text`; the exact behaviour is not shown, so don't test it.

## 4. Returns and guarantees
- Returns `None`. The file content is exactly the following, with each item on its own line ending in `"\n"`:

```
# OpenLens Export - OpticStudio Format
# Lens: {lens.name}
#============================================================      <- "#" + 60 "="
                                                                   <- blank line
SYSTEM PRESCRIPTION DATA
----------------------------------------                           <- 40 "-"
Title: {lens.name}
Units: Millimeters
Type: Sequential

LENS DATA SUMMARY
----------------------------------------
Material: {lens.material}
Refractive Index (d-line): {lens.refractive_index:.6f}
Outer Diameter: {lens.diameter:.4f} mm
Clear Aperture 1/2: {ca1:.4f} / {ca2:.4f} mm
Bevel 1/2 (45 deg): {lens.bevel_1:.4f} / {lens.bevel_2:.4f} mm
Coating 1/2: {lens.coating_label(1)} / {lens.coating_label(2)}
Effective Focal Length: {fl:.4f} mm      <- ONLY if fl is truthy
                                         <- blank
SURFACE DATA DETAIL
----------------------------------------
<header row>
----------------------------------------------------------------------   <- 70 "-"
<OBJ row>
<STO row>
<row 1>
<row 2>
<IMA row>
                                         <- blank
# End of OpenLens Export
```

### Table rows
Each row is an exact f-string. Columns are separated by single spaces, and trailing padding is kept.

- Header: `f"{'Surface':<10} {'Type':<12} {'Radius':<14} {'Thickness':<12} {'Glass':<10} {'Diameter':<10}\n"`
- OBJ: `f"{'OBJ':<10} {'STANDARD':<12} {'Infinity':<14} {'Infinity':<12} {'':<10} {'':<10}\n"`
- STO: `f"{'STO':<10} {'STANDARD':<12} {'Infinity':<14} {'0.0000':<12} {'':<10} {lens.diameter:<10.4f}\n"`
  - This uses the mechanical **diameter**, not the clear aperture.
- Row 1: `f"{'1':<10} {'STANDARD':<12} {r1_str:<14} {lens.thickness:<12.4f} {lens.material:<10} {lens.get_clear_aperture_1():<10.4f}\n"`
- Row 2: `f"{'2':<10} {'STANDARD':<12} {r2_str:<14} {'100.0000':<12} {'':<10} {lens.get_clear_aperture_2():<10.4f}\n"`
  - The thickness column is always the literal `100.0000`.
  - The Glass column is blank.
- IMA: `f"{'IMA':<10} {'STANDARD':<12} {'Infinity':<14} {'':<12} {'':<10} {'':<10}\n"`

### Radius strings
- `r1_str = f"{lens.radius_of_curvature_1:.4f}"` if `abs(r1) < 1e10`, otherwise `"Infinity"`. `r2_str` follows the same rule.
- These are the **nominal** `radius_of_curvature_*` values, not `get_effective_radius_*`. Parabolic and conic settings do not change the table.

### Other content rules
- `lens.name` and `lens.material` are written verbatim: no escaping, no truncation.
- A material longer than 10 characters simply widens its column.

## 5. Side effects
- Creates or replaces `filename`.
- The lens is not mutated by this code. Whether `Lens` methods cache anything is not visible.

## 6. Branches and edge cases
- **Focal-length line.** It is written only if `lens.calculate_focal_length()` is truthy.
  - `None` or `0.0` omits the line, and the blank line follows `Coating 1/2:` directly.
  - A negative focal length (diverging lens) **is** written, e.g. `-xx.xxxx`.
- **Radius threshold.** `abs(r) < 1e10` prints the number; `abs(r) >= 1e10` prints `Infinity`.
  - Example: `radius_of_curvature_1=1e10` gives `Infinity`, while `9e9` gives `9000000000.0000`.
  - This only applies if the `Lens` setter stores the value unchanged; check the stored value via `lens.radius_of_curvature_1`.
  - Plano surfaces given as `float('inf')` print `Infinity`. The importer builds Lenses with `inf` radii, so `inf` is accepted.
- **Negative radius** prints with a sign: `-50.0000`.
- **Coatings.** For an uncoated lens, use `lens.coating_label(1)` for the expected value. The API doc says it returns e.g. `"Uncoated"`.

## 7. Rules from comments
- The STO row's comment says: "entrance pupil at front surface". That is why the stop sits at thickness `0.0000` with the full diameter.

## 8. Testing hints
- **How to read the file.** Read with `open(path, encoding="utf-8").read().splitlines()`.
  - Text mode with universal newlines, plus `splitlines`, avoids Windows `\r\n` issues.
  - Do **not** `.strip()` table rows if you compare exact padding. Instead, rebuild the expected row with the identical f-string using the lens's own values.
- **Exact lines worth asserting:**
  - `lines[0] == "# OpenLens Export - OpticStudio Format"`
  - `lines[1] == f"# Lens: {lens.name}"`
  - `lines[2] == "#" + "="*60`
  - `lines[3] == ""`
  - `"Units: Millimeters"` is present.
  - The last line is `"# End of OpenLens Export"`.
- **Content checks:**
  - `f"Refractive Index (d-line): {lens.refractive_index:.6f}"` (6 decimals; the others use 4).
  - `f"Clear Aperture 1/2: {lens.get_clear_aperture_1():.4f} / {lens.get_clear_aperture_2():.4f} mm"`.
  - Row 2 contains `"100.0000"`.
  - The STO row ends with `f"{lens.diameter:<10.4f}"`.
- **Distinguish clear aperture from diameter.** Use `clear_aperture_1=20.0`, `clear_aperture_2=18.0` with `diameter=25.0`. Assert that row 1 ends with `f"{20.0:<10.4f}"`, row 2 with `f"{18.0:<10.4f}"`, and STO with `25.0000`.
  - This assumes `get_clear_aperture_*` returns the set value; assert via the getters.
- **Bevel check.** Set `bevel_1=0.5`, `bevel_2=0.25`, then expect `"Bevel 1/2 (45 deg): 0.5000 / 0.2500 mm"`.
- **Line counts.** With the focal-length line present the file has 33 lines; without it, 32. Count it yourself before asserting, or assert presence or absence of the `Effective Focal Length:` prefix instead.
- **Focal-length value.** Compute the expected value as `lens.calculate_focal_length()`; never hardcode lensmaker results.
- **Pitfalls:**
  - `refractive_index` depends on the lens wavelength and material handling, which is code you cannot see. Read it back from the lens.
  - The surface table never uses the effective radius, so a parabolic lens still shows its nominal radius.

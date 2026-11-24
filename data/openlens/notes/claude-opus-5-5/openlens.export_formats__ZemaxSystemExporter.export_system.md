# Notes: `ZemaxSystemExporter.export_system(system, filename) -> None`

## 1. Purpose
- Writes a multi-element `OpticalSystem` to a text file in a simplified Zemax `.zmx` dialect.
- Each element produces a front and a back `SURF` block.
- An optional STOP surface is added inside the stop gap, and the file ends with an image surface.
- Returns `None`.

## 2. Inputs
- **`system`**: an `OpticalSystem`. The function reads these attributes:
  - `system.name`
  - `system.elements`, a list of `LensElement` with `.lens`, `.position`, `.decenter_y/z` and `.tilt_x/y/z`
  - `system.air_gaps`, a list of `AirGap` with `.thickness`
  - `system.get_aperture_stop()`, which returns `None` or a dict with the keys `"gap_index"`, `"diameter"` (read with `.get`) and `"position"`
- **`filename`**: a path. It is written through `atomic_write_text(filename)` (default `encoding="utf-8"`, `newline=None`).
- There is no validation of either argument.

## 3. Raises
- The function raises nothing itself.
- Errors from `atomic_write_text` can propagate, for example a missing parent directory. That code is in `openlens.atomic` and was not shown.
- Per the `atomic_write_text` docs, the target file is replaced only on success.

## 4. Output text (exact, `\n`-joined; `{}` = value)
```
! OpenLens System Export
! {system.name}          <- raw, not escaped
VERS 140000
MODE SEQ
<blank>
SURF 0 / "  TYPE STANDARD" / "  CURV 0.0" / "  DISZ INFINITY" / <blank>
```
Surfaces are numbered sequentially from `surf = 1`.

**Front surface of each element `i`** (lines in this order):
- `SURF {n}`
- `  TYPE STANDARD`
- `  CURV {c1}`
- optional `  CONI {k:g}`
- `  DISZ {lens.thickness}`
- `  DIAM {lens.get_clear_aperture_1()/2.0}`
- `  GLAS {mat}`
- `  WAVE {lens.wavelength:g}`
- optional `  DECE {dy:g} {dz:g}`
- optional `  TILT {tx:g} {ty:g} {tz:g}`
- optional `  COAT {label}`
- blank line

**Back surface** (note that DISZ comes LAST and there is no GLAS/WAVE/DECE/TILT):
- `SURF {n+1}`
- `  TYPE STANDARD`
- `  CURV {c2}`
- optional `  CONI`
- `  DIAM {ca2/2.0}`
- optional `  COAT {label2}`
- `  DISZ {back_disz}`
- blank line

**Stop surface** (only for the stop gap's element):
- `SURF {m}`
- `  TYPE STANDARD`
- `  STOP`
- `  CURV 0.0`
- optional `  DIAM {stop_dia/2.0}`
- `  DISZ {stop_remainder}`
- blank line

**End of file**: `SURF {last}\n  TYPE STANDARD\n\n`

### Value formatting
- **Numbers written with plain `{x}`** use Python's float `str`:
  - DISZ thickness: `5.0` → `5.0`
  - DIAM: `25.0` → `12.5`
- **CURV `c`** comes from `_zemax_curv(lens.get_effective_radius_N())`:
  - `"0.0"` if the radius is `== 0`, `abs > 1e10`, or non-finite (inf/nan).
  - Otherwise `f"{1.0/radius}"`. Examples: R=50 → `0.02`, R=-50 → `-0.02`, R=3 → `0.3333333333333333`.
  - For non-parabolic surfaces the effective radius is presumably the nominal radius (`Lens` code was not shown).
- **CONI** comes from `_zemax_conic(is_parabolic, conic_constant)`:
  - Parabolic → `-1.0`, printed as `CONI -1`.
  - Otherwise, if `abs(conic) > EPSILON` it prints the conic, e.g. `-0.5` → `CONI -0.5`.
  - Otherwise the line is omitted.
  - The value of `EPSILON` is in `constants` and not shown; it is tiny.
- **GLAS `mat`** comes from `_zemax_material`:
  - `(material or "BK7").strip() or "BK7"`.
  - If the name contains any whitespace it is wrapped in double quotes: `Fused Silica` → `"Fused Silica"`, `N-BK7` → `N-BK7`.
- **DECE** is written iff `abs(dy) > EPSILON or abs(dz) > EPSILON`. Example: dy=1.5, dz=0 → `  DECE 1.5 0`.
- **TILT** is written iff any tilt `> EPSILON`. Example: `  TILT 0 2 0`.
- **COAT** is written iff `lens.coating_label(s) != "Uncoated"`.

### `back_disz` for element `i` (`gaps = list(system.air_gaps)`)
1. If a stop exists and `stop["gap_index"] == i`:
   - `back_disz = stop["position"] - (element.position + lens.thickness)`
   - `stop_remainder = gaps[i].thickness - back_disz`
   - The stop surface is emitted after the back surface.
2. Else if `i < len(gaps)`: `back_disz = gaps[i].thickness`.
3. Else: `back_disz = 100.0`, written as `DISZ 100.0`.

### Stop DIAM
- Written only if `stop_dia` is truthy, so `None` or `0` means no DIAM line.

## 5. Side effects
- Creates or replaces `filename`.
- Does not mutate `system` or its lenses.

## 6. Branches and edge cases
- **Empty system** (`OpticalSystem(name="E")`): the whole file is exactly
  `"! OpenLens System Export\n! E\nVERS 140000\nMODE SEQ\n\nSURF 0\n  TYPE STANDARD\n  CURV 0.0\n  DISZ INFINITY\n\nSURF 1\n  TYPE STANDARD\n\n"`
- **One element, no stop**: surfaces are SURF 1 (front), SURF 2 (back), SURF 3 (image).
  - Whether `air_gaps` has an entry after the last element is defined in `optical_system` (not shown).
  - The docstring of `_gap_before` says `air_gaps[i]` is the gap after element i. That suggests N elements → N-1 gaps, so the last back DISZ is `100.0`.
  - Assert against `system.air_gaps` rather than assuming.
- **Two elements with gap g**: SURF 2 has `DISZ {air_gaps[0].thickness}`, then SURF 3/4 for element 2, then image SURF 5.
- **Stop set** with `system.set_aperture_stop(0, diameter=10.0)`:
  - An extra block containing `  STOP` and `  DIAM 5.0` appears.
  - Every later surface number shifts by +1.
  - The back DISZ plus the stop DISZ equals `gaps[0].thickness`, up to float rounding.
  - The stop's `position` rule lives in `OpticalSystem` (not shown). Read it from `system.get_aperture_stop()["position"]`.
- **Flat surface** (radius `float('inf')`, if `Lens` accepts it): `CURV 0.0`.
- **Parabolic** (`is_parabolic_1=True`): `CONI -1`.
  - CURV uses `get_effective_radius_1()`, which may differ from the nominal radius (not shown). Compute the expected value from that method.

## 7. Rules from comments and docstrings
- The front surface carries the center thickness.
- The back surface carries the distance to the next surface: the stop plane, the next front vertex, or the image.
- DIAM is a semi-diameter, i.e. full aperture / 2.
- The mechanical diameter is not exported; the importer rebuilds it as the max of the two apertures.

## 8. Testing hints
Build inputs with the public API:
```python
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem

s = OpticalSystem(name="T")
s.add_lens(Lens(name="A", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                thickness=5.0, diameter=25.0, material="BK7"))
```

How to check the output:
- Read the file with `Path(p).read_text(encoding="utf-8").splitlines()`.
  - `newline=None` text mode may write `\r\n` on Windows, so do not compare raw bytes.
- Build expected strings from the objects instead of hardcoding them:
  - `f"  WAVE {lens.wavelength:g}"`
  - `f"  DIAM {lens.get_clear_aperture_1()/2.0}"`
  - `f"  CURV {1.0/lens.get_effective_radius_1()}"`
  - The default clear aperture probably equals the diameter, but that is not verified here.
- For computed DISZ values (stop case), parse the float and compare with `pytest.approx`.
- Strong checks worth writing:
  - exact header lines
  - exact SURF numbering sequence
  - back block has DISZ as its last line
  - front block has no DISZ after DIAM, and `GLAS` appears only on front surfaces
  - `CONI` absent for spherical surfaces
  - quoting of `"Fused Silica"`
  - no `DECE`/`TILT` when alignment is zero
  - `DECE 1.5 0` after `s.set_element_alignment(0, decenter_y=1.5)`
  - image block is the final `SURF`
  - count of `SURF ` lines = 2 + 2·N (+1 with a stop)
- Round trip: `ZemaxSystemImporter.import_system(p)` should give back the radii, thicknesses and gaps.
  - The refractive index comes from the catalog, and the name becomes `"Element k"`.
  - Lens diameter equals the max clear aperture.
- Pitfalls:
  - `system.name` is written unescaped.
  - Leading two spaces on every keyword line.
  - `CONI -1`, not `-1.0`.
  - `CURV` uses full float repr, not `:g`.

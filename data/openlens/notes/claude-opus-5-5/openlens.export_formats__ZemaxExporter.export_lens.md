# Notes: `openlens.export_formats.ZemaxExporter.export_lens(lens, filename)`

## 1. Purpose
- Static method that writes one `Lens` as a simplified sequential Zemax `.zmx` text file.
- The file has four surfaces: OBJ (0), front (1), back (2) and image (3).
- It returns `None`. The only output is the file.

## 2. Inputs
- `lens`: a `Lens`. The method reads these members:
  - `name`, `thickness`, `material`, `wavelength`
  - `is_parabolic_1/2`, `conic_constant_1/2`
  - `get_effective_radius_1/2()`, `get_clear_aperture_1/2()`, `coating_label(1/2)`
- `filename`: a path (str; `atomic_write_text` takes `Any`, so a `pathlib.Path` probably works).
  - The file is written through `atomic_write_text(filename)` with `encoding='utf-8'` and `newline=None`.
- No validation is done on any input.

## 3. Raises
- The method raises nothing explicitly. Errors from attribute access or formatting propagate:
  - a lens without the attributes above raises `AttributeError`;
  - `lens.wavelength = None` raises `TypeError`, because `f"{None:g}"` is invalid.
- A missing target directory makes the error come from `atomic_write_text` (module not shown). That module's docstring says the target is replaced only on success.

## 4. Returns / exact file content
Returns `None`. Every line ends in `\n`. A blank line follows each block, including the last one. Template:
```
! OpenLens Export
! {lens.name}
VERS 140000
MODE SEQ

SURF 0
  TYPE STANDARD
  CURV 0.0
  DISZ INFINITY

SURF 1
  TYPE STANDARD
  CURV {_zemax_curv(lens.get_effective_radius_1())}
  CONI {k:g}                 <- only if conic present (see below)
  DISZ {lens.thickness}      <- plain str(), e.g. "5.0"
  DIAM {lens.get_clear_aperture_1() / 2.0}   <- semi-diameter, plain str(), e.g. "12.5"
  GLAS {material}
  WAVE {lens.wavelength:g}
  COAT {lens.coating_label(1)}   <- only if label != "Uncoated"

SURF 2
  TYPE STANDARD
  CURV {_zemax_curv(lens.get_effective_radius_2())}
  CONI {k:g}                 <- optional
  DIAM {lens.get_clear_aperture_2() / 2.0}
  COAT {lens.coating_label(2)}   <- optional
  DISZ 100.0

SURF 3
  TYPE STANDARD

```
- The file therefore ends with `"SURF 3\n  TYPE STANDARD\n\n"`.
- Line order differs between the two surfaces:
  - SURF 1 is `CURV, [CONI], DISZ, DIAM, GLAS, WAVE, [COAT]`.
  - SURF 2 is `CURV, [CONI], DIAM, [COAT], DISZ 100.0`. SURF 2 has no GLAS and no WAVE line.
- Surface indentation is exactly two spaces.

Private helpers:
- **`_zemax_curv(r)`**
  - Returns `"0.0"` if `r == 0`, `abs(r) > 1e10`, or `r` is not finite.
  - Otherwise returns `f"{1.0/r}"` (repr of the float). Examples: `100` → `"0.01"`, `-50` → `"-0.02"`, `-100` → `"-0.01"`.
- **`_zemax_conic(is_parabolic, k)`**
  - Parabolic surface → `-1.0`, written as `CONI -1`.
  - Otherwise, if `abs(k) > EPSILON` → `k`. Example: `-0.5` gives `CONI -0.5`.
  - Otherwise → `None`, and no CONI line is written.
  - The `EPSILON` value is in `constants`, which is not shown. Use clearly nonzero values such as `-0.5`.
- **`_zemax_material(m)`**
  - `name = (m or "BK7").strip() or "BK7"`.
  - If the name contains any whitespace it is wrapped in double quotes. Examples:
    - `"BK7"` → `GLAS BK7`
    - `"  N-BK7 "` → `GLAS N-BK7`
    - `"Fused Silica"` → `GLAS "Fused Silica"`
    - `""` or `None` → `GLAS BK7`
- **`lens.name`** is written raw. There is no escaping or quoting.

## 5. Side effects
- The method creates or replaces `filename`.
- It does not mutate the lens.
- It changes no global state.

## 6. Branches / edge cases
| Case | How to reach it | Observable result |
|---|---|---|
| Spherical surfaces | Default lens with conic 0 | No `CONI` line anywhere |
| Front conic | `lens.conic_constant_1 = -0.5` | `  CONI -0.5` right after SURF 1's CURV |
| Parabolic front | `lens.is_parabolic_1 = True` | `  CONI -1`. CURV uses `get_effective_radius_1()`, whose parabolic behaviour lives in `lens.py` (not shown) |
| Flat surface | Radius `float('inf')` or `>1e10` | `  CURV 0.0` |
| Coated surface | `lens.set_coating(1, layers)` with a label other than `"Uncoated"` | `  COAT <label>` as the last SURF 1 line; on SURF 2 it comes before `DISZ 100.0` |
| Uncoated | Default lens | No COAT lines |
| Material with a space | `lens.material = "Fused Silica"` | `  GLAS "Fused Silica"` |

## 7. Rules from comments and docstrings
- A parabolic surface is written as CONI -1 at its vertex radius, not as a sphere at the nominal radius.
- Zemax DIAM is a semi-diameter, so the full aperture is halved. A 25 mm lens is written as `DIAM 12.5`.
- Glass names that contain whitespace are quoted, and the importer strips the quotes again.

## 8. Testing hints
- Use `tmp_path`: `p = tmp_path / "l.zmx"`, then `ZemaxExporter.export_lens(lens, str(p))`.
- Read the file back with `p.read_text(encoding="utf-8")` and `.splitlines()`.
  - Do **not** compare raw bytes. On Windows, text mode turns `\n` into `\r\n`.
- Compute expected values from the lens instead of hard-coding Lens-internal defaults:
  - `f"  CURV {1.0/lens.get_effective_radius_1()}"`
  - `f"  DIAM {lens.get_clear_aperture_1()/2.0}"`
  - `f"  WAVE {lens.wavelength:g}"`
  - `f"  DISZ {lens.thickness}"`
  - The sign convention of R2 and the default for `clear_aperture=None` (likely the full diameter) live in `lens.py`, which is not shown.
- Pin these exact values:
  - header lines `! OpenLens Export`, `! <name>`, `VERS 140000`, `MODE SEQ`;
  - SURF 0's `CURV 0.0` / `DISZ INFINITY`;
  - `  DISZ 100.0` as the last SURF 2 line;
  - SURF ids 0..3 and the exact line order inside each block;
  - the total number of lines.
- Explicit-aperture example: `Lens(diameter=25.0, clear_aperture_1=20.0)` should give `  DIAM 10.0` on SURF 1. The `Lens` constructor validates this; its exact behaviour is not shown.
- A good regression check is that DIAM equals half the clear aperture and not the full aperture.
- Assert that SURF 2 has no GLAS/WAVE lines and that only one `DISZ` appears in each of SURF 1 and SURF 2.
- Optional round trip: `ZemaxSystemImporter.import_system(str(p))` should give one element named `"Element 1"`.
  - Its thickness and radii should match the original, within approx for radii.
  - Its diameter is the max of the two clear apertures.
- Pitfalls:
  - `DISZ`/`DIAM` use `str(float)`, so they print `5.0`, not `5`. A test that writes `"DISZ 5"` is wrong.
  - Only `WAVE` and `CONI` use `:g`. `CURV` uses the full repr, so `1/3` would print `0.3333333333333333`.
  - Set `material`, conic and parabolic flags on the instance after construction if the constructor might reject them. For parabolic lenses, also set a valid `parabolic_sag_*`, or check the result with `in` rather than an exact CURV value.

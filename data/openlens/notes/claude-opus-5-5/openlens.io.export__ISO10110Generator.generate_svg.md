# Notes: `ISO10110Generator.generate_svg(filename, width=800, height=600)`

## 1. Purpose
- Renders an SVG "ISO 10110 drawing" of `self.system` (an `OpticalSystem`) and writes it to `filename` using `openlens.atomic.atomic_write_text`.
- Returns `None`.

## 2. Inputs
- `self.system`: the object given to `ISO10110Generator(system)`. It is stored as-is and never copied or validated. The function reads:
  - `.name`, `.id`, `.elements`
  - `.get_aperture_stop()`, `.get_system_focal_length()`, `.calculate_back_focal_length()`
- `filename`: a path passed straight to `atomic_write_text(filename)`. Its behaviour is defined in `openlens.atomic`, which is not shown. Its docstring says the file is replaced only on success.
- `width`, `height`: not validated. They are put into f-strings exactly as given.
  - Pass ints. `width=800.0` would print `"800.0"`.
  - `center_y = height/2` is always a float, e.g. `300.0`.

## 3. Raises
- The SVG text is fully built before the file is opened. If rendering fails, the file is not created or changed.
- If `get_aperture_stop()` returns a dict without `'gap_index'`: `KeyError`.
- If the stop dict lacks `'position'` and the system has elements: `KeyError`.
- If the stop `diameter` is truthy but not a number: `ValueError` or `TypeError` from the `:.2f` format.
- Errors when writing the file (for example, the directory does not exist) come from `atomic_write_text`, which is not shown. Do not assert a specific exception type.

## 4. Returns / file content
- Returns `None`.
- The file content is lines joined with `"\n"`. The last line is `</svg>`, with no trailing newline added here.
- Read the file back in text mode (`Path.read_text(encoding="utf-8")`). Newline translation on Windows is not visible from this module.

Line order:
1. `<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">`, then `<defs>`, `<style>`.
2. Five style lines. They are plain strings, not f-strings, so the doubled braces appear literally:
   - `  .lens {{ fill: #E0E0E0; stroke: black; stroke-width: 2; opacity: 0.8; }}`
   - `.axis`, `.text`, `.title`, `.table` follow in the same form.
   - Then `</style>`, `</defs>`, `<rect width="100%" height="100%" fill="white"/>`.
3. Title line: `<text x="20" y="30" class="title">ISO 10110 Drawing: {escape(name)}</text>`.
   - `escape` only converts `&`, `<` and `>`. `"A&B"` becomes `A&amp;B`.
4. `<text x="20" y="50" class="text">Date: YYYY-MM-DD</text>`, using `datetime.now()`.
5. Only if a stop exists: `<text x="20" y="68" class="text">Aperture stop: gap {gap_index}[, Ø{dia:.2f}mm]</text>`.
   - The `, Ø…mm` part is omitted when the diameter is `None` or `0`.
6. Axis line: `<line x1="20" y1="300.0" x2="780" y2="300.0" class="axis"/>` (defaults shown).
7. Stop marker, only if a stop exists and there are elements:
   - A `<line …stroke-width="1.5"/>`.
   - A `…text-anchor="middle">STOP</text>`.
8. For each element `i`:
   - `<path d="M x y L x y … Z" class="lens"/>`. Coordinates are `:.2f` and come from `LensGeometry.get_lens_polyline`.
   - A label `<text x="{pos_x}" y="…" class="text" text-anchor="middle">L{i+1}</text>`. `pos_x` is not formatted.
9. ISO table at `x=width-490`, `y=height-150`, width 470. For the defaults:
   - `<rect x="310" y="450" width="470" height="{35+40*n}" fill="white" stroke="black" stroke-width="1"/>`, where `n` is the number of elements.
   - Header line, then headers `Surf Radius Thick Mat Diam CA Coat Bev`. Header x positions are `x+10, +50, +110, +160, +210, +260, +310, +375`, all at `y+18`.
   - Two rows per lens.
     - Front row values: `surf_idx`, `R1:.2f`, `thickness:.2f`, escaped material, `diameter:.2f`, `get_clear_aperture_1():.2f`, escaped `coating_label(1)`, `bevel_1:.2f`.
     - Back row values: `R2:.2f`, `-` for thickness, an empty material cell (`<text x="470" y="…" class="text"></text>` with defaults), then diameter, CA2, coat2, `bevel_2`.
   - Surface numbers run 1, 2, 3, … across lenses.
10. Title block at `(20, height-170)`, width 300.
    - Height is `24+8*18+6 = 174`: `<rect x="20" y="430" width="300" height="174" fill="white" stroke="black" stroke-width="1.5"/>`.
    - Then `TITLE BLOCK`.
    - Then rows, each written as `<text x="28" …>Key:</text>` followed by `<text x="130" …>value</text>` (value escaped). The rows are:
      - Title
      - Dwg No (`system.id[:8].upper()+"-A"`)
      - Material (`sorted(set(materials))` joined `", "`, or `—` if there are none)
      - `Units: mm`
      - Scale (`{scale:.2f} px/mm`)
      - EFL / BFL (`{v:.2f} mm`, or `—` when the value is `None`)
      - Surfaces (`2*len(elements)`)
      - Date / Rev (`YYYY-MM-DD / A`)
11. Notes block:
    - `<text x="20" y="100" class="text" font-weight="bold">ISO 10110 NOTES</text>`.
    - Eight notes at `y = 118, 133, …, 223`, each passed through `escape`.
    - Note 0: `0/ General: all dimensions in mm; surfaces numbered front to back.`
    - Note 4 when there is no tilt: `4/ Centering: TBD (assembly nominally centered).`
    - Note 4 with tilt: `4/ Centering: ≤ {max_tilt*60:.1f}' element tilt (max over assembly).`
      - `max_tilt` is the largest `abs(tilt_x)` or `abs(tilt_y)` over all elements. `tilt_z` is ignored.
      - Example: `tilt_x=0.5` gives `≤ 30.0'`.
    - Note 6: `6/ Coating: L1: {c1}/{c2}; L2: …` (entries joined `"; "`), or `6/ Coating: —.` when empty.
      - The labels come from `Lens.coating_label`. Per the API doc an uncoated surface is probably `"Uncoated"`, but that code is not shown here.
12. Dimensions, only if there are elements:
    - The total track line.
    - `{total_length:.2f} mm`, where `total_length = last.position + last.thickness`.
    - The OD line.
    - `<text x="…" y="300.00" class="text" text-anchor="end">Ø{max_diam:.2f}</text>`.
13. `</svg>`.

How the scale is computed:
- Empty system: `10.0`.
- Otherwise: `min((width-100)/max_x, (height-100)/max_diam) * 0.8`, where `max_x` is last position plus thickness. A value of 0 or less falls back to 10.
- Example: one lens with diameter 25, thickness 5, position ≈0 gives `min(140, 20)*0.8 = 16.0`, shown as `Scale: 16.00 px/mm`.

## 5. Side effects
- Creates or overwrites `filename` (atomically, per the `atomic` module).
- Does not mutate the system or its lenses (only getters are called).

## 6. Branches
- **Empty `OpticalSystem("Empty")`:**
  - Scale is `10.00`.
  - No `<path`, no `STOP`, no dimension lines, no `mm</text>` track text.
  - Table rect `height="35"`; `Surfaces: 0`; `Material: —`; `6/ Coating: —.`.
- **No stop set:** no `Aperture stop` text and no `STOP`.
- **After `set_aperture_stop(...)`:** both appear.
  - The stop dict is built in `optical_system`, which is not shown. Check for the substring `Aperture stop: gap ` rather than exact numbers, unless you also read `system.get_aperture_stop()` and format it yourself.
- **Several lenses with the same material:** the material appears once. `BK7` + `F2` gives `BK7, F2`.

## 7. Comment-stated rules
- Notes are XML-escaped because coating layer names are free-form text. The exported SVG must stay well-formed XML.

## 8. Testing hints
- Use pytest `tmp_path`. Build the system with:
  ```python
  s = OpticalSystem("T")
  s.add_lens(Lens(radius_of_curvature_1=50, radius_of_curvature_2=-50, thickness=5, diameter=25, material="BK7"))
  ```
- Good checks:
  - Parse the file with `xml.etree.ElementTree.fromstring(text)`; it must be well-formed.
  - Count `<path ` (one per lens) and the `L1`, `L2` labels.
  - Check table cells `>50.00<`, `>-50.00<`, `>5.00<`, `>25.00<`, `>BK7<`.
  - Check `>Surfaces:<` paired with `>2<`, `Scale: 16.00 px/mm` → really `>16.00 px/mm<`, `>Ø25.00<`, and the literal `{{`.
- A name with `&` or `<` must show up escaped (`&amp;`, `&lt;`) in both the title and the title block.
- Pitfalls:
  - Do not hardcode the date. Compute `datetime.now().strftime("%Y-%m-%d")`, or only check the `Date: ` prefix.
  - EFL/BFL values and polyline coordinates depend on other modules. Compare with `f"{s.get_system_focal_length():.2f} mm"` instead of literal numbers.
  - The element position after `add_lens` is set by `optical_system` (not shown). Derive expected totals from `s.elements[-1].position + s.elements[-1].thickness`.
  - The radius is printed with `:.2f`; a flat (infinite) radius prints as `inf`.
  - Calling it twice overwrites the file; the second content replaces the first.

# Notes: `SVGExporter.export_lens(lens, filename, width=400, height=300, show_rays=False, show_labels=True)`

## 1. Purpose
- Writes a 2D cross-section drawing of one `Lens` as an SVG text file.
- The drawing contains a title, a dashed optical axis, a filled and stroked outline path, optional rays and optional labels.
- Returns `None`.

## 2. Inputs
- `lens`: a `Lens`. The function reads these attributes:
  - `thickness`, `diameter`, `name`, `material`
  - `radius_of_curvature_1`, `radius_of_curvature_2` (the raw property values, not the "effective" radii)
  - `refractive_index`
  - `calculate_focal_length()`
- How `Lens` stores, validates or sign-converts radii and the index is in `openlens/lens.py`, which is not shown. Build expected values from the lens's own properties, not from constructor arguments.
- `filename`: path. It is written through `atomic_write_text(filename)` with the defaults `encoding='utf-8'` and `newline=None`.
- `width`, `height`: numbers, not validated. They are interpolated with plain `{}` formatting, so ints print as `400` and `300`.
- `show_rays`, `show_labels`: truthiness only.

## 3. Raises
- No explicit `raise` statements.
- `ZeroDivisionError` if `max(thickness*2, diameter) == 0`. This happens before the file is opened. `Lens` probably rejects such values first; that is unverified.
- Any I/O error comes from `atomic_write_text` (separate module, not shown). The docstring says the target is replaced only on success.
- Negative or zero available space (e.g. `width=80`) gives `scale <= 0`. No error is raised.

## 4. Returns and guarantees
Geometry, with `margin=40`:
- `scale = min(width-80, height-80) / max(lens.thickness*2, lens.diameter) * 0.8`
- `cx = width/2`, `cy = height/2`
- `t = thickness*scale`, `h = (diameter/2)*scale`
- `front_x = cx - t/2`, `back_x = cx + t/2`
- `top_y = cy - h`, `bot_y = cy + h`

Sag, per surface:
- `sag = abs(r) - sqrt(r*r - (D/2)**2)`, then multiplied by `scale`.
- Sag is 0 if `abs(r) > 1e9`.
- Sag is also 0 when `r*r < (D/2)**2`; the `ValueError` is caught. In that case the surface is still treated as curved.

Edge x positions:
- Front: flat (`abs(r1) > 1e9`) gives `front_x`. If `r1 > 0`, it is `front_x + sag1`. If `r1 <= 0`, it is `front_x - sag1`.
- Back: flat gives `back_x`. If `r2 < 0`, it is `back_x - sag2`. Otherwise it is `back_x + sag2`.

Path `d`: the parts are joined with single spaces and every number is formatted `:.2f`. Parts in order:
1. Front flat: `M fx top_y`, then `L fx bot_y`.
   Front curved: `M ftop top_y`, then `Q ctrl cy fbot bot_y`. Here `ctrl = front_x` if `r1 > 0`, else `front_top_x + 2*sag1`.
2. Bottom edge: `L back_bot bot_y`.
3. Back flat: `L back_top top_y`.
   Back curved: `Q ctrl cy back_top top_y`. Here `ctrl = back_x` if `r2 < 0`, else `back_bot_x - 2*sag2`.
4. `Z`.

Worked example: `width=400, height=300`, thickness 5, diameter 50, `r1=100`, `r2=-100`.
- `scale` = 3.52, `cx` = 200, `cy` = 150, `front_x` = 191.2, `back_x` = 208.8, `h` = 88, `sag` ≈ 11.1775.
- Resulting path: `M 202.38 62.00 Q 191.20 150.00 202.38 238.00 L 197.62 238.00 Q 208.80 150.00 197.62 62.00 Z`

File content, in order (each line ends `\n`):
1. `<?xml version="1.0" encoding="UTF-8"?>`
2. `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 400 300">`
3. A `<defs><style>` block with classes `.lens-fill`, `.lens-stroke`, `.axis`, `.ray`, `.label`, `.title`. These are always present.
4. Background: `  <rect width="400" height="300" fill="white"/>`
5. Title: `  <text x="200.0" y="25" class="title" text-anchor="middle">{escape(str(lens.name))}</text>`
   - `x` is `width/2`, which is always a float.
   - `xml.sax.saxutils.escape` converts only `&`, `<` and `>`.
6. Axis: `  <line x1="40" y1="150.0" x2="360" y2="150.0" class="axis"/>`
7. Outline, two lines with the same `d`:
   - `  <path d="{d}" class="lens-fill"/>`
   - `  <path d="{d}" class="lens-stroke"/>`
8. Optional rays and labels (see section 6).
9. A blank line, then `</svg>` and a final newline.

## 5. Side effects
- Creates or replaces `filename` atomically.
- Does not mutate `lens`. It only calls getters and `calculate_focal_length()`.

## 6. Branches and edge cases
- **`show_rays=True`:** adds exactly 3 lines of the form `<line x1="40" y1="{cy+ray_y}" x2="{front_top_x}" y2="{cy+ray_y}" class="ray"/>`.
  - `ray_y` takes the values `-h*0.7`, `0`, `h*0.7`.
  - These numbers use raw float repr, not `.2f`. For example `y1` may be `88.40000000000001`, so do not assert it exactly.
  - Default (`show_rays=False`): the substring `class="ray"` is absent.
- **`show_labels=True` (default):** adds these lines:
  - `<text x="50" y="{cy-h-5}" class="label">∅{diameter:.1f}mm</text>`, e.g. `∅50.0mm`.
  - `t={thickness:.1f}mm`, e.g. `t=5.0mm`, at `x="{cx}"`.
  - `{escape(material)} (n={refractive_index:.4f})`, e.g. `BK7 (n=1.5168)` if the index is 1.5168.
  - Only if `calculate_focal_length()` is truthy: `<text x="350" y="140.0" class="label" text-anchor="end">f={fl:.1f}mm</text>`. Nothing is written when it is `None` or `0`.
- **`show_labels=False`:** no `class="label"` text and no `∅`. The `.label {` CSS rule stays in the file.
- **Flat surfaces:** use `L` segments instead of `Q`. Whether `Lens` accepts `inf` or 1e12 radii is unknown (lens module not shown).
- **`r` smaller than half the diameter:** a `Q` segment is still drawn, with the control point and endpoints all at `front_x`. `Lens` may reject such a lens at construction.

## 7. Rules from comments
- Front: `r1 > 0` is convex and `r1 < 0` is concave.
- Back: `r2 < 0` is convex and `r2 >= 0` is concave.
- Curved surfaces are drawn as quadratic Béziers, not arcs. The `_generate_arc_path` and `_get_surface_x` helpers are not used here.

## 8. Testing hints
- **Writing and reading:** use `tmp_path / "l.svg"`. Read it back with `open(p, encoding="utf-8").read()` in text mode, because `newline=None` writes `\r\n` on Windows and `∅` is non-ASCII.
- **Building the lens:** pass explicit `thickness`/`diameter`, since the default constants are unseen. Then check `lens.thickness`, `lens.diameter` and the radii properties first, and compute expected strings from those values with the formulas above.
- **Robust assertions:**
  - `text.startswith('<?xml')` and `text.rstrip().endswith('</svg>')`.
  - `'width="400" height="300" viewBox="0 0 400 300"'` is present.
  - `text.count('class="lens-fill"/>') == 1` and the same for `class="lens-stroke"/>`.
  - Both `<path>` lines carry an identical `d`.
  - `text.count('class="ray"') == 3` with rays on, `0` with rays off.
  - `>A &amp; B<` appears for `name="A & B"`.
  - `f"n={lens.refractive_index:.4f}"` appears.
  - `x1="40"` and `x2="360"` appear on the axis line.
- **Custom size:** `width=600, height=400` gives title `x="300.0"`, axis `x2="560"` and `y1="200.0"`, with `scale = 320/maxdim*0.8`.
- **Focal-length label:** compare it against `lens.calculate_focal_length()` computed in the test. Do not hardcode the value.
- **Pitfalls:**
  - `cy` prints as `150.0`, not `150`.
  - `.2f` is used only inside the path `d`; everywhere else floats are printed raw.

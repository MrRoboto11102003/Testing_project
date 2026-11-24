# Notes: `openlens.export_formats.SVGExporter.export_system`

## 1. Purpose
- Writes a simplified SVG diagram of an `OpticalSystem` to `filename`.
- Each lens is drawn as two plain rectangles (fill and stroke), placed left to right with the air gaps between them.
- Returns `None`.

## 2. Inputs
- `system: OpticalSystem`. The function reads:
  - `system.name`
  - `system.elements`, specifically each `elem.lens.thickness` and `elem.lens.diameter`
  - `system.air_gaps`, through `_gap_before`
  - `system.get_total_length()`
- `filename: str`: the path written through `atomic_write_text(filename)` (utf-8, from `openlens.atomic`).
- `width: int = 600`, `height: int = 300`. Neither is validated. Both are interpolated with plain f-strings, so `600` prints as `600` and `600.0` prints as `600.0`.
- `margin = 40` is a hard-coded local.
- Element `position`, decenter and tilt are ignored. Placement uses only the gaps and the thicknesses.
- `_gap_before(system, i)`:
  - returns `0.0` when `i <= 0` or when `i-1 >= len(system.air_gaps)`;
  - otherwise returns `system.air_gaps[i-1].thickness`.
  - `air_gaps[k]` is the gap *after* element k.

## 3. Raises
There are no explicit raises. Possible failures:
- `ZeroDivisionError` if every element has `diameter == 0`, which makes `max_diameter == 0`. Lens validation probably prevents building such a lens, so this is unlikely to be reachable.
- I/O errors from `atomic_write_text` for an unwritable path. The exact type is defined in another module; it is probably `OSError`.
- Any exception raised inside the `with` block (for example from `get_total_length`) means the target file is **not** replaced. This follows `atomic_write_text`'s documented contract: it replaces the file only on success.

## 4. Returns and output text
The function returns `None`. The file content is exactly the following lines, each ending in `\n`, with no blank lines anywhere:
```
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <defs>
    <style>
      .lens-fill { fill: #b3d9ff; fill-opacity: 0.5; }
      .lens-stroke { stroke: #0066cc; stroke-width: 2; fill: none; }
      .axis { stroke: #999999; stroke-width: 1; stroke-dasharray: 5,5; }
      .title { font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; }
    </style>
  </defs>
  <rect width="{width}" height="{height}" fill="white"/>
  <text x="{width/2}" y="25" class="title" text-anchor="middle">{escape(str(system.name))}</text>
  <line x1="40" y1="{cy}" x2="{width-40}" y2="{cy}" class="axis"/>
  ...two rect lines per element...
</svg>
```

Header details:
- There is **no `viewBox`** on `<svg>`, unlike `export_lens`.
- The `.title` rule has no `fill:` and there is no `.ray` or `.label` class.
- With the defaults:
  - the title has `x="300.0"` (true division);
  - `cy = height/2`, so `y1="150.0"`;
  - `x2="560"` (an int).

Per element `i`, the function writes two lines, fill first and then stroke:
```
  <rect x="{x_pos}" y="{cy - h}" width="{t}" height="{h * 2}" class="lens-fill"/>
  <rect x="{x_pos}" y="{cy - h}" width="{t}" height="{h * 2}" class="lens-stroke"/>
```
The values come from this arithmetic, in this exact operation order:
- `total_length = system.get_total_length() or 100`, so `0`/`None` becomes `100`.
- `max_diameter = max(elem.lens.diameter ...)`, defaulting to `50` when there are no elements.
- `scale = min((width-80) / total_length * 0.8, (height-80) / max_diameter * 0.8)`
- `x_pos` starts at `60`. For each element, `x_pos += _gap_before(system, i) * scale` is applied first, and then `x_pos += t` after the element is drawn.
- `t = lens.thickness * scale`
- `h = lens.diameter / 2 * scale`
- The first element's `x` is `60 + 0.0*scale`, which is the float `60.0` and prints as `"60.0"`.

Number formatting:
- All numbers use the default `str(float)` repr. Nothing is rounded.

## 5. Side effects
- The function creates or overwrites `filename`.
- The system and its lenses are not mutated.

## 6. Branches and edge cases
- **No elements** (`OpticalSystem()`):
  - no element `<rect>` lines are written; only the background rect remains;
  - `max_diameter` defaults to 50;
  - `</svg>` immediately follows the axis line.
- **`get_total_length()` returns 0 or None:** the value 100 is used instead. What an empty or single system returns is defined in `optical_system` and is not visible here.
- **Gaps:** element 0 never gets a gap. Element i>0 is shifted by `air_gaps[i-1].thickness * scale`. A gap of 0 shifts nothing, so the next rect's `x` equals the previous `x + t`.
- **Scale choice:** the vertical limit wins for tall, thin systems and the horizontal limit wins for long systems.
- **Name escaping:** the name passes through `xml.sax.saxutils.escape`, which escapes only `&`, `<`, `>`.
  - Example: `"A&B<C>"` becomes `A&amp;B&lt;C&gt;`.
  - Quotes are **not** escaped.

## 7. Rules from comments and docstrings
- The `_gap_before` docstring says gaps live in `system.air_gaps`, not on the element. The gap before element i is `air_gaps[i-1]`, and the first element has gap 0.
- The comment "Simple rectangle for each lens (simplified)" states the intent: element curvature is not drawn.

## 8. Testing hints
- Build the input like this:
  ```python
  s = OpticalSystem(name="Sys")
  s.add_lens(Lens(thickness=5, diameter=25))
  s.add_lens(Lens(thickness=4, diameter=20), air_gap_before=10)
  ```
  - Pass `thickness` and `diameter` explicitly, because the default constants are not shown here.
  - Write to `tmp_path / "s.svg"` and read the file back with `encoding="utf-8"`.
- **Do not hard-code floats.** Recompute the expected values in the test with the identical expressions and order shown in section 4. For example, `(300-80)/25*0.8` may not be exactly `7.04`.
  - Take `total_length` from `s.get_total_length()`.
  - Take gap values from `s.air_gaps[k].thickness`. How `add_lens` fills `air_gaps` is not visible here.
  - Then build the strings with the same f-string patterns, e.g. `f'x="{x}"'`.
- Robust assertions:
  - `text.count("<rect") == 1 + 2*len(s.elements)`;
  - `class="lens-fill"` comes before `class="lens-stroke"` for each element;
  - `"viewBox" not in text`;
  - `text.endswith("</svg>\n")`;
  - `text.startswith('<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="600" height="300">\n')`;
  - `'<line x1="40" y1="150.0" x2="560" y2="150.0" class="axis"/>'` is present;
  - the title `x="300.0"` is present.
- Custom sizes: `width=800, height=400` gives `x="400.0"`, `x2="760"` and `cy="200.0"`.
- Positional checks:
  - the first element's rect starts with `x="60.0"`;
  - the second element's `x` is `60.0 + t0 + gap0*scale`, computed in that order.
- Use a name with `&` and `<` to check escaping, and a name with `"` to check that it passes through literally.
- Pitfalls to avoid:
  - Assuming a `viewBox`.
  - Assuming element `position` is used.
  - Rounding to 2 decimals.
  - Expecting blank lines; this function, unlike `export_lens`, writes none.

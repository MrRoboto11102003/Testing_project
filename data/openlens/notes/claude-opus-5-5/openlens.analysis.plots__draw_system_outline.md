# Notes: `openlens.analysis.plots.draw_system_outline(ax, system)`

## 1. Purpose
- Draws every lens element of an `OpticalSystem` onto a matplotlib Axes, in a Z (optical axis) vs Y side view.
- For each lens it draws one translucent filled polygon plus four outline lines (front surface, back surface, upper rim, lower rim).
- Lenses are placed one after another along Z.

## 2. Inputs
- `ax`: any object with `.plot(...)` and `.add_patch(...)`.
  - A real matplotlib Axes or a `unittest.mock.MagicMock()` both work.
  - The function never validates `ax`.
- `system`: an `OpticalSystem`. The function only reads:
  - `system.elements`: a list of `LensElement`. For each it uses `element.lens` and `element.lens.thickness`.
  - `system.air_gaps`: a list of `AirGap`. For each it uses `.thickness`.
  - There is no type check, so any duck-typed object with these attributes works.
- Geometry for each lens comes from `LensGeometry.lens_outline(lens, num_points=50, max_points=2000)`. That function lives in `openlens.geometry` and its code is not shown here. It returns a dict with these keys:
  - `"front"`, `"back"`, `"edge"`: lists of `(x, y)` pairs. For `"edge"` the second value is treated as a radius.
  - `"feasible"`: a bool.

## 3. Raises
- The function raises nothing itself.
- Exceptions of type `ImportError`, `TypeError` or `ValueError` raised while building or adding the fill polygon are swallowed:
  - This covers the `from matplotlib.patches import Polygon` import, the `Polygon(...)` constructor and `ax.add_patch`.
  - The error is logged with `logger.debug("Skipping lens fill: %s", e)` on logger `openlens.analysis.plots`.
  - The four outline lines are still drawn afterwards.
- Any other exception from the fill step propagates. Example: `ax.add_patch` raising `RuntimeError`, or an `ax` object with no `add_patch` attribute (gives `AttributeError`).
- Exceptions from `ax.plot` or from `lens_outline` propagate.

## 4. Returns and guarantees
- Returns `None`.
- `current_z` starts at `0.0`. After element `i` is drawn:
  - if `i < len(system.air_gaps)`: `current_z += lens.thickness + system.air_gaps[i].thickness`
  - otherwise: `current_z += lens.thickness`
- How `add_lens(air_gap_before=...)` fills `air_gaps` is defined in `optical_system.py`, which is not shown. In tests, read `system.air_gaps` directly to compute the expected offsets.
- For each element, with outline `o` and offset `cz`, the values are:
  - `z1 = [cz + x for x, _ in o["front"]]` and `y_front = [y for _, y in o["front"]]`
  - `z2` and `y_back` are built the same way from `o["back"]`.
  - `edge_z = [cz + x ...]` and `edge_y = [r ...]` come from `o["edge"]`.
- For each element, the calls are made in this exact order:
  1. `ax.add_patch(Polygon(verts, closed=True, facecolor=F, edgecolor="none", alpha=0.3))`
     - `verts = zip(z1 + edge_z[1:] + z2 + reversed(edge_z[1:-1]), y_front + edge_y[1:] + y_back + [-r for r in reversed(edge_y[1:-1])])`
  2. `ax.plot(z1, y_front, color=FC, alpha=0.9, linewidth=2)`
  3. `ax.plot(z2, y_back, color=BC, alpha=0.9, linewidth=2)`
  4. `ax.plot(edge_z, edge_y, color=COLOR_LENS_RIM, alpha=0.7, linewidth=1)`
  5. `ax.plot(edge_z, [-r for r in edge_y], color=COLOR_LENS_RIM, alpha=0.7, linewidth=1)`
- Colours depend on `bad = not o["feasible"]`:
  - `F = COLOR_LENS_BAD if bad else COLOR_LENS_FILL`
  - `FC = COLOR_LENS_BAD if bad else COLOR_LENS_R1`
  - `BC = COLOR_LENS_BAD if bad else COLOR_LENS_R2`
  - Rim lines are always `COLOR_LENS_RIM`.
- All colour constants come from `openlens.constants`; their values are not shown. Import them in the test rather than hard-coding strings.
- The function sets no title, labels or grid.

## 5. Side effects
- Draws on `ax` only.
- Does not mutate `system` or the lenses, assuming `lens_outline` is pure (its code is not visible).
- May emit one debug log record per lens whose fill fails.

## 6. Branches and edge cases
- **Empty system** (`OpticalSystem()` with no lenses): no `plot` or `add_patch` calls at all; returns `None`.
- **One lens**: exactly 1 `add_patch` call and 4 `plot` calls. Every z value is offset by `0.0`, so `z1` equals the `x` values of `lens_outline(lens, 50, 2000)["front"]`.
- **Two or more lenses**: each element adds 1 patch and 4 lines. Element `k`'s z values are shifted by the sum of the earlier thicknesses plus the matching `air_gaps[i].thickness`.
- **Fill fails with `ValueError`** (for example a MagicMock `ax` with `ax.add_patch.side_effect = ValueError("x")`): no exception escapes, and the 4 `plot` calls per lens still happen.
- **Infeasible outline** (`feasible` is False): front, back and fill all use `COLOR_LENS_BAD`; the rims stay `COLOR_LENS_RIM`.
  - To force this path without relying on geometry, monkeypatch `LensGeometry.lens_outline` (as imported in `openlens.analysis.plots`) to return a small fake dict.

## 7. Rules from docstring/comments
- `lens.thickness` is the centre (vertex-to-vertex) thickness.
- Rim (edge) thickness is `thickness - sag1 + sag2`; this is computed inside `LensGeometry`, not here.
- A degenerate profile must not stop the outlines being drawn. That is why the fill errors are caught.

## 8. Testing hints
- Use a real Axes on a non-GUI backend:
  ```python
  import matplotlib; matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots()
  ```
  Then check `len(ax.lines) == 4*n` and `len(ax.patches) == n`.
- Read line data with `ax.lines[0].get_xdata()` / `get_ydata()` and compare with `pytest.approx`.
- For exact call and kwarg checks, use a `MagicMock` ax:
  - `ax.plot.call_args_list[k].kwargs` should equal `{"color": ..., "alpha": 0.9, "linewidth": 2}`.
  - `ax.plot.call_count == 4*n`.
- Build systems with the public API:
  ```python
  s = OpticalSystem()
  s.add_lens(Lens())
  s.add_lens(Lens(), air_gap_before=5.0)
  ```
  Compute expected offsets from `s.elements[i].lens.thickness` and `s.air_gaps[i].thickness`, not from assumed values.
- Get the expected geometry by calling `LensGeometry.lens_outline(lens, num_points=50, max_points=2000)` yourself; it uses the same arguments.
- The lower rim's y values are exactly the negated upper rim's y values, and both rims share the same x data. This holds for every lens.
- Pitfalls:
  - Don't hard-code colour strings or outline point counts.
  - Don't expect a return value.
  - Don't expect titles or labels.
  - `plt` figures accumulate across tests, so call `plt.close(fig)`.

# Notes: `openlens.diffraction.DiffractionCalculator.plot_psf_2d`

## 1. Purpose
- Draws a two-panel matplotlib figure: a 2D Airy PSF image on the left and its horizontal centre cross-section on the right.
- It then saves the figure to a file or shows it, and closes it. It always returns `None`.

## 2. Inputs
- `self.wavelength_m`
  - Set by `DiffractionCalculator(wavelength_m=550e-9)` or by the `wavelength=` kwarg.
  - `self.wavelength` is an alias for it.
- `focal_length` (mm), `aperture_diameter` (mm)
  - Plain numbers. No validation is done.
  - They are put into the suptitle with `str()` formatting: `100` gives `"100"`, `100.0` gives `"100.0"`.
- `save_path: Optional[str] = None`
  - Truthy: `plt.savefig(save_path, dpi=300, bbox_inches="tight")`.
  - Falsy (`None` or `""`): `plt.show()`.

## 3. Raises
- `ZeroDivisionError` ("float division by zero") when `aperture_diameter=0`:
  - `airy_r` is computed first as `1.22*λ*(f/1000)/(D/1000)*1e6`.
  - This happens before the matplotlib check, so it raises even when matplotlib is unavailable.
  - Example: `DiffractionCalculator().plot_psf_2d(100, 0)`.
- There are no explicit `raise` statements. Any other errors come from numpy or matplotlib.

## 4. Returns and guarantees
- Always returns `None`.
- Values for the default λ = 550e-9, f = 100, D = 10:
  - `airy_r = 1.22*λ*f/D*1e6` = **6.71 µm**. Use `pytest.approx`.
  - `size = 10*airy_r` = **67.1 µm**.
- It calls `psf = self.point_spread_function_2d(size, focal_length, aperture_diameter)`:
  - Arguments are positional, with default `pixels=256`.
  - The result is a 256×256 array normalised so its max is 1.
- Figure layout: `plt.subplots(1, 2, figsize=(14, 6))`. After the colorbar is added, `fig.axes == [ax1, ax2, colorbar_ax]`.
- `ax1` (PSF image):
  - `imshow(psf, extent=[-size/2, size/2, -size/2, size/2], cmap="hot", origin="lower")`.
  - xlabel `"X (μm)"`, ylabel `"Y (μm)"`, title `"2D Point Spread Function"`.
  - The colorbar label `"Normalized Intensity"` is the colorbar axes' ylabel.
- `ax2` (cross-section):
  - `ax2.lines[0]` is the profile `psf[psf.shape[0]//2, :]`, i.e. row 128. Its x-data is `np.linspace(-size/2, size/2, 256)`. Style is `"b-"`, linewidth 2.
  - `ax2.lines[1]` is an axvline at `+airy_r`, red, dashed, label `f"Airy disk radius = {airy_r:.2f} μm"`. For the default case this is `"Airy disk radius = 6.71 μm"`.
  - `ax2.lines[2]` is an axvline at `-airy_r`, red, dashed, with no label.
  - xlabel `"Position (μm)"`, ylabel `"Normalized Intensity"`, title `"PSF Cross-Section"`.
  - Grid is on (alpha 0.3). The legend has exactly 1 text, the Airy label.
- Suptitle, exact template: `f"Point Spread Function Analysis\n(f={focal_length}mm, D={aperture_diameter}mm, λ={self.wavelength*1e9:.0f}nm)"`.
  - For (100, 10) this gives `"Point Spread Function Analysis\n(f=100mm, D=10mm, λ=550nm)"`.
  - For λ = 632.8e-9 the end is `"λ=633nm"`.
- `plt.tight_layout()` runs before save or show. `plt.close()` closes the current figure at the end.

## 5. Side effects
- Writes a PNG/other file at `save_path`. The format comes from the file extension; dpi is 300.
- Otherwise it calls `plt.show()`, which can block on an interactive backend.
- It does not mutate `self`. No figure is left open afterwards.

## 6. Branches and edge cases
- **matplotlib missing** (`openlens.diffraction.MATPLOTLIB_AVAILABLE` is False):
  - `airy_r` is still computed.
  - Logs WARNING `"plot_psf_2d unavailable: matplotlib is not installed. Skipping."` on logger `"openlens.diffraction"`.
  - Returns `None` without calling `point_spread_function_2d` or any `plt` function.
- **`save_path` truthy**: `savefig` is called and `show` is not.
- **`save_path` is `None` or `""`**: `show` is called and `savefig` is not.
- `focal_length=0`: `size` becomes 0, giving a degenerate extent. Avoid this case; matplotlib behaviour is uncertain.

## 7. Rules from comments/docstring
- Both inputs are in mm. The plot extent is 10× the Airy radius, i.e. ±5 Airy radii.

## 8. Testing hints
- Setup:
  - Use `pytest.importorskip("matplotlib")`.
  - Force a non-GUI backend before importing openlens: `import matplotlib; matplotlib.use("Agg")`. Alternatively set `MPLBACKEND=Agg`.
- Never let the real `plt.show()` run. Either pass `save_path=str(tmp_path/"psf.png")` or monkeypatch `openlens.diffraction.plt.show`.
- To inspect the figure before `plt.close()`, patch `savefig` (or `show`) to grab `plt.gcf()`:
  ```python
  import openlens.diffraction as d
  cap = {}
  monkeypatch.setattr(d.plt, "savefig", lambda *a, **k: cap.update(fig=d.plt.gcf(), args=a, kw=k))
  d.DiffractionCalculator().plot_psf_2d(100, 10, save_path="x.png")
  fig = cap["fig"]
  assert cap["args"] == ("x.png",)
  assert cap["kw"] == {"dpi": 300, "bbox_inches": "tight"}
  ```
- What to check on the captured figure:
  - `fig._suptitle.get_text()`.
  - `fig.axes[0].images[0].get_extent()` ≈ `[-33.55, 33.55, -33.55, 33.55]`.
  - `.get_cmap().name == "hot"`.
  - `fig.axes[1].lines[1].get_xdata()[0]` ≈ 6.71.
  - `fig.axes[1].get_legend().get_texts()[0].get_text()`.
- Real save: pass `tmp_path/"psf.png"` and assert the file exists with size > 0.
- Show branch: patch `show` and `savefig` with counters. With `save_path=None` or `""`, expect `show` once and `savefig` 0 times.
- Spy on PSF arguments:
  - `monkeypatch.setattr(calc, "point_spread_function_2d", fake)`.
  - `fake` must return a 2D array such as `np.ones((256, 256))`, because `.shape[0]` and indexing are used.
  - Assert it received `(approx 67.1, 100, 10)`.
- Missing-matplotlib branch:
  - `monkeypatch.setattr(d, "MATPLOTLIB_AVAILABLE", False)`.
  - Use `caplog.at_level("WARNING", logger="openlens.diffraction")` and check the exact message.
  - Make `point_spread_function_2d` raise to prove it is not called.
- Pitfalls:
  - Compare floats with `approx`.
  - The labels contain the Unicode `μ` (U+03BC) and `λ`; copy them exactly.
  - `plt.close()` runs after save or show, so do not inspect `plt.gcf()` after the call returns.

# Notes: `DiffractionCalculator.plot_airy_disk(focal_length, aperture_diameter, save_path=None)`

## 1. Purpose
- Draws the normalized 1-D Airy intensity profile with matplotlib.
- Marks the Airy-disk radius and a 50 % line on the plot.
- Saves the figure to `save_path`, or calls `plt.show()` when no path is given.
- Always closes the figure and returns `None`.

## 2. Inputs
- `focal_length` (mm) and `aperture_diameter` (mm): plain numbers.
  - No validation or conversion is done.
  - They are inserted into the title exactly as given: `100` prints as `100`, `100.0` prints as `100.0`.
- `save_path`: optional `str` or path-like.
  - The check is truthiness, so `None` and `""` both go to `plt.show()`.
  - Any truthy value goes to `plt.savefig(save_path, dpi=300, bbox_inches="tight")`.
- Instance state used:
  - `self.wavelength_m` drives the numbers. The default is `550e-9`. `DiffractionCalculator(wavelength=633e-9)` also sets it through the kwargs alias.
  - `self.wavelength` (an alias of `wavelength_m`) is used for the title.

## 3. Raises
Order of operations: compute `airy_r`, then check `MATPLOTLIB_AVAILABLE`, then plot. So division errors happen even when matplotlib is missing.
- **`aperture_diameter=0`** raises `ZeroDivisionError("float division by zero")`.
  - It comes from `airy_disk_radius_um`, which computes `1.22*λ*f/D` with Python floats.
  - Example: `DiffractionCalculator().plot_airy_disk(100, 0)`.
- **`focal_length=0`** does not raise.
  - numpy produces NaN with a RuntimeWarning, so the plot is meaningless.
  - Avoid it in tests.
- Errors from `savefig` (for example a bad directory) propagate from matplotlib.

## 4. Returns and guarantees
- Always returns `None`.
- Computed values:
  - `airy_r = 1.22 * wavelength_m * focal_length / aperture_diameter * 1e6`, in µm. Example: λ = 550 nm, f = 100, D = 10 gives `6.71`.
  - `max_radius = 5 * airy_r`.
  - The data comes from `self.airy_pattern(max_radius, focal_length, aperture_diameter)` with the default 1000 points.
- Figure content (one Axes, `figsize=(10, 6)`):
  - `lines[0]`: the curve `plot(r, np.real(intensity), "b-", linewidth=2)`.
    - 1000 points; x runs from 0 to `5*airy_r`.
    - y is normalized, so its max is `1.0` and `y[0]` ≈ 1.0. It has no label.
  - `lines[1]`: `axvline(x=airy_r, color="r", linestyle="--", label=f"Airy disk radius = {airy_r:.2f} μm")`. Its xdata is `[airy_r, airy_r]`.
  - `lines[2]`: `axhline(y=0.5, color="g", linestyle=":", alpha=0.5)`. Its ydata is `[0.5, 0.5]`.
  - xlabel is `"Radius (μm)"` and ylabel is `"Normalized Intensity"` (µ here is U+03BC Greek mu, not the micro sign).
  - Title template: `f"Airy Diffraction Pattern\n(f={focal_length}mm, D={aperture_diameter}mm, λ={self.wavelength*1e9:.0f}nm)"`. Example: `"Airy Diffraction Pattern\n(f=100mm, D=10mm, λ=550nm)"`.
  - Grid is on with `alpha=0.3`.
  - The legend has exactly one entry: `"Airy disk radius = 6.71 μm"` for the example above.

## 5. Side effects
- Creates a new pyplot figure.
- Then calls `savefig` (writes a file at dpi 300) or `show`.
- Finally calls `plt.close()`, which closes the current (new) figure. The count from `plt.get_fignums()` is unchanged afterwards.
- No instance attributes are changed.

## 6. Branches
- **Matplotlib unavailable**:
  - Triggered with `monkeypatch.setattr(openlens.diffraction, "MATPLOTLIB_AVAILABLE", False)`.
  - Logs a WARNING on logger `"openlens.diffraction"` with the exact text: `"plot_airy_disk unavailable: matplotlib is not installed. Skipping."`
  - Returns `None` without calling `airy_pattern` or any `plt` function.
  - `airy_r` is still computed, so `aperture_diameter=0` still raises here.
- **`save_path` truthy**: `plt.savefig(save_path, dpi=300, bbox_inches="tight")`. `plt.show` is not called.
- **`save_path` is `None` or `""`**: `plt.show()`. `savefig` is not called.
- `plt.close()` runs on both plotting paths.

## 8. Testing hints
- Use the non-GUI backend: `import matplotlib; matplotlib.use("Agg")` before importing pyplot.
- File test:
  - Call `calc.plot_airy_disk(100, 10, save_path=str(tmp_path/"a.png"))`.
  - Assert the file exists and has size > 0, and the return value is `None`.
- Inspecting the figure: it is closed before return, so capture it inside a patched `savefig`.
  ```python
  import openlens.diffraction as d
  cap = {}
  def fake_savefig(path, **kw):
      fig = d.plt.gcf(); ax = fig.axes[0]
      cap.update(kw=kw, path=path, title=ax.get_title(), xl=ax.get_xlabel(),
                 lines=[(l.get_xdata(), l.get_ydata()) for l in ax.get_lines()],
                 legend=[t.get_text() for t in ax.get_legend().get_texts()])
  monkeypatch.setattr(d.plt, "savefig", fake_savefig)
  d.DiffractionCalculator().plot_airy_disk(100, 10, save_path="x.png")
  ```
  Then assert:
  - `kw == {"dpi": 300, "bbox_inches": "tight"}`.
  - The title matches exactly.
  - `len(lines[0][0]) == 1000`.
  - `lines[0][0][-1]` ≈ `5*6.71` (use `pytest.approx`).
  - `max(lines[0][1])` ≈ 1.0.
  - `lines[1][0][0]` ≈ 6.71 and `lines[2][1][0] == 0.5`.
- To confirm the show branch, patch `d.plt.show` with a counter, call with `save_path=None` (and again with `""`), and assert it was called once per call and `savefig` was not.
- Use `pytest.approx` for numbers and exact strings for formatted labels.
  - The `:.2f` and `:.0f` formatting removes float noise: `550e-9*1e9` becomes `550`.
  - Other wavelengths: `633e-9` gives title `λ=633nm` and label `"Airy disk radius = 7.72 μm"` (f = 100, D = 10).
- Pitfalls:
  - Do not try to read the figure after the call returns; it is already closed.
  - The curve line has no label, so do not expect two legend entries.
  - `scipy.special.j1` or the numpy fallback may be used. The values are equal to roughly 1e-6, so use tolerances.

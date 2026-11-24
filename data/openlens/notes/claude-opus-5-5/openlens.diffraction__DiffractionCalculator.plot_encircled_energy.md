# Notes: `DiffractionCalculator.plot_encircled_energy(self, focal_length, aperture_diameter, save_path=None)`

## 1. Purpose
- Draws the analytic Airy encircled-energy curve (in %) with matplotlib.
- Marks the Airy radius and the 50% / 80% points.
- Then either saves the figure to `save_path` or calls `plt.show()`, and always closes it.
- Returns `None`.

## 2. Inputs
- `focal_length` (mm) and `aperture_diameter` (mm): plain numbers, not validated.
- They are passed unchanged to `self.encircled_energy(...)` and `self.airy_disk_radius(...)`.
- They appear verbatim in the title via `{focal_length}`. So `100` renders as `f=100mm` and `100.0` renders as `f=100.0mm`.
- `save_path`: optional. Its truthiness decides the branch:
  - Truthy: `plt.savefig(save_path, dpi=300, bbox_inches="tight")`.
  - `None` or `""`: `plt.show()`.
- Wavelength comes from `self.wavelength_m` (default `550e-9`). The `wavelength` property is an alias, and the title uses `self.wavelength*1e9`.

## 3. Raises
- **Check order:**
  1. `encircled_energy` is computed first, before the matplotlib check.
  2. Then the `MATPLOTLIB_AVAILABLE` check.
  3. Then `airy_disk_radius`.
- `aperture_diameter=0` (int or float) raises `ZeroDivisionError("float division by zero")`.
  - Source: `airy_disk_radius_um`, line `1.22*λ*f/D` with `D = 0/1000.0` (a Python float).
  - It is raised from inside `encircled_energy`, because `max_radius_um` defaults to 5× the Airy radius.
  - It is raised **even when matplotlib is unavailable**.
  - Example: `DiffractionCalculator().plot_encircled_energy(100, 0)`.
- `focal_length=0` does not raise in this module. Numpy array division gives NaN with a RuntimeWarning. Do not test this case.
- An unwritable or nonexistent directory in `save_path` makes matplotlib's `savefig` raise (behaviour lives in matplotlib, not here).

## 4. Returns and guarantees
- Always returns `None`, on both the plotting and the skip path.
- **Figure contents** (Axes = `plt.gca()` of the one figure, `figsize=(10, 6)`). `ax.get_lines()` holds 6 Line2D objects, in this order:
  1. Curve: x = `r` (μm, 100 points from `encircled_energy` default `num_points=100`), y = `encircled*100`, colour `"b"`, linewidth 2.
  2. `axvline(x=airy_r)`: colour `"r"`, linestyle `"--"`, label `f"Airy disk radius = {airy_r:.2f} μm"`.
  3. `axhline(y=50)`: colour `"g"`, linestyle `":"`, alpha 0.5.
  4. `axhline(y=80)`: colour `"orange"`, linestyle `":"`, alpha 0.5.
  5. Marker `"go"` at `(r[idx_50], 50)`, markersize 8, label `f"50% @ {r[idx_50]:.2f} μm"`.
  6. Orange `"o"` marker at `(r[idx_80], 80)`, label `f"80% @ {r[idx_80]:.2f} μm"`.
- `idx_50 = np.argmin(np.abs(encircled - 0.5))`, and `idx_80` likewise with 0.8. These use the fractions, not the percentages.
- Legend texts, in order: `["Airy disk radius = … μm", "50% @ … μm", "80% @ … μm"]`.
- Labels and title:
  - xlabel `"Radius (μm)"`.
  - ylabel `"Encircled Energy (%)"`.
  - Title `f"Encircled Energy Curve\n(f={focal_length}mm, D={aperture_diameter}mm, λ={self.wavelength*1e9:.0f}nm)"`.
- Grid is on (alpha 0.3) and `ylim == (0, 105)`.
- **Worked example:** default calculator, `f=100`, `D=10`.
  - `airy_r = 6.71` μm, so the label is `"Airy disk radius = 6.71 μm"`.
  - Title is `"Encircled Energy Curve\n(f=100mm, D=10mm, λ=550nm)"`.
  - The curve's x data runs from 0 to about 33.55 (= 5·6.71); y[0] = 0.0.

## 5. Side effects
- Creates one pyplot figure and closes it with `plt.close()` (the current figure).
- With a truthy `save_path`, writes that file.
- `self` is not mutated and there is no global state.
- On the skip path it logs a WARNING on logger `"openlens.diffraction"`, with the exact text `"plot_encircled_energy unavailable: matplotlib is not installed. Skipping."`.

## 6. Branches
- **`MATPLOTLIB_AVAILABLE` is False:** logs the warning and returns `None`. No figure is created and `airy_disk_radius` is not called, but `encircled_energy` already ran. Test with `monkeypatch.setattr(openlens.diffraction, "MATPLOTLIB_AVAILABLE", False)`.
- **`save_path` truthy:** `savefig(save_path, dpi=300, bbox_inches="tight")` is called exactly once and `show` is not called. `close` is called once.
- **`save_path` is `None` or `""`:** `show()` is called with no args, `savefig` is not called, then `close()`.

## 7. Rules from comments / docstrings
- The encircled-energy values are the closed form `EE = 1 - J0(v)^2 - J1(v)^2`, clipped to [0, 1].
- The plotting extent (5× Airy radius) must not change the values.
- EE at the first Airy zero ≈ 0.838, so the curve is not forced to 100% at the last sample. The last y is about 96–97%, not 100.

## 8. Testing hints
- Use a non-interactive backend: `import matplotlib; matplotlib.use("Agg")` before importing pyplot. Or monkeypatch `openlens.diffraction.plt.show` to a recorder; Agg's `show` may emit a UserWarning.
- **To inspect the figure:** monkeypatch `openlens.diffraction.plt.savefig` (or `show`) with a fake that grabs `plt.gcf()` and reads it before `close` runs:
  ```python
  import openlens.diffraction as d, matplotlib.pyplot as plt
  seen = {}
  def fake_save(path, **kw):
      seen["path"], seen["kw"], seen["ax"] = path, kw, plt.gcf().axes[0]
  monkeypatch.setattr(d.plt, "savefig", fake_save)
  d.DiffractionCalculator().plot_encircled_energy(100, 10, save_path="x.png")
  assert seen["kw"] == {"dpi": 300, "bbox_inches": "tight"}
  ax = seen["ax"]
  assert ax.get_title().startswith("Encircled Energy Curve\n(f=100mm, D=10mm, λ=550nm)")
  ```
  - `ax.get_title()` returns the title string exactly.
  - Compare `ax.get_ylim()` to `(0.0, 105.0)`.
- **Recompute expected values** with `r, ee = calc.encircled_energy(100, 10)`:
  - Curve x data equals `r`; y data equals `ee*100` (use `np.allclose`).
  - Marker x data equals `[r[np.argmin(abs(ee-0.5))]]`.
  - Don't hard-code those radii.
- A real save into `tmp_path / "ee.png"` works; assert the file exists and is non-empty.
- Mock `plt.close` to count calls; expect exactly 1 on both branches.
- **Pitfalls:**
  - The `μ` and `λ` characters are Unicode. Copy them exactly as `"μm"` and `"λ="`.
  - Float formatting in the title depends on the input type (`100` vs `100.0`).
  - Don't assert that the curve's last value is 100.
  - Wavelength formatting is `:.0f`, so `632.8e-9` renders as `633nm`.
  - Whether scipy is installed doesn't matter for these values (j0/j1 fallbacks agree to ~1e-10); use tolerances.

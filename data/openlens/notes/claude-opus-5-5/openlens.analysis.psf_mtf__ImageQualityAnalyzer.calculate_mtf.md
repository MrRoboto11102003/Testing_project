# Notes: `ImageQualityAnalyzer.calculate_mtf` (openlens/analysis/psf_mtf.py)

## 1. Purpose
- Returns the tangential and sagittal MTF as 1-D arrays against spatial frequency (lp/mm).
- The default is geometric: a ray-binned PSF, then the line spread function (LSF), then an FFT.
- With `use_diffraction=True` it takes the FFT of the pupil-function PSF instead.

## 2. Inputs
- `field_angle_deg` (float, 0.0): field angle in Y. It is passed to `SpotDiagram.trace_spot(field_angle_y_deg=...)`, or to the wavefront sensor.
- `wavelength_nm` (float, `WAVELENGTH_GREEN`).
- `focus_shift_mm` (float, 0.0): defocus.
- `max_freq` (float, 100.0): upper frequency cut, inclusive (`<=`).
- `use_diffraction` (bool, False).
- `**kwargs`: legacy aliases only.
  - `field_angle=` is popped and **overrides** `field_angle_deg`.
  - The `"max_freq"` alias branch is dead code: `max_freq=` always binds to the named parameter.
  - Any other key raises `TypeError`.
- No type or range validation is done on any argument.

## 3. Raises
- `TypeError(f"Unexpected kwargs: {list(kwargs.keys())}")`.
  - Example: `an.calculate_mtf(foo=1)` raises `TypeError: Unexpected kwargs: ['foo']`.
  - Order of checks: aliases are popped first, then this check runs, and only then the empty-system check.
  - So the error is raised even on an empty `OpticalSystem()`.
  - `calculate_mtf(field_angle=1, foo=2)` gives the message `Unexpected kwargs: ['foo']`.
- Anything else would come from other modules: the tracer, `WavefrontSensor`, `DiffractionPSFCalculator`.

## 4. Returns
- Always a new `dict` with exactly the keys `{"freq", "mtf_tan", "mtf_sag"}`.
- All three values are numpy arrays of the same length.
- **Empty system** (`not system.elements`): three empty float64 arrays, `shape == (0,)`.
  - Logs a warning: `"MTF: system has no elements; returning an empty MTF"`.
  - This holds for both values of `use_diffraction`.
- **Geometric branch** (default):
  1. It traces the spot with `num_rings=25` and reads `rms_radius`.
  2. Window size:
     - `window = max(0.256, 2*2.2*rms)` = `max(0.256, 4.4*rms)` when `rms` is finite and > 0.
     - Otherwise `window = 0.256`.
  3. Pixel count: `num_pixels = min(max(128, ceil(window/0.002)), 1024)`.
  4. It calls `self.calculate_psf(..., sensor_size_mm=window, pixels=num_pixels)`.
     - That function caps its own window at 8.0 mm.
     - So `step_size = min(window, 8.0)/num_pixels`.
  5. LSFs: `lsf_tan = psf.sum(axis=1)` (profile along Y) and `lsf_sag = psf.sum(axis=0)` (profile along Z).
  6. `mtf = |fft(lsf)|`, divided by `mtf[0]` only if `mtf[0] > 0`. An all-zero PSF (no rays) gives an all-zero MTF, not NaN.
  7. Frequencies: `freqs = fftfreq(num_pixels, d=step_size)`. The first `num_pixels//2` entries are kept, then masked to `0 <= f <= max_freq`.
- **Geometric guarantees**:
  - `freq[0] == 0.0`.
  - `mtf_tan[0] == mtf_sag[0] == 1.0` when any rays land.
  - Frequencies are strictly increasing with uniform step `df = 1/(num_pixels*step_size)`. For a small spot (rms ≤ 0.256/4.4 ≈ 0.0582 mm) that is ≈ 1/0.256 = 3.90625 lp/mm.
  - With the default `max_freq=100` and a small spot, the length is 26 (k = 0..25, last value ≈ 97.656).
  - All MTF values lie in [0, 1] (up to float tolerance), because the LSF is non-negative.
  - Frequency is capped at index `num_pixels//2 - 1`. For 128 px that is 63 × 3.90625 ≈ 246 lp/mm, so `max_freq=1000` still returns only ~64 entries.
- **Diffraction branch**:
  - Gets the wavefront with `grid_size=64`. If it is degenerate (`W` is None, not 2-D, or a side < 2), it returns three empty arrays and logs a warning.
  - Otherwise it adds defocus to the wavefront in place. This happens only if `focus_shift_mm` is truthy, with `W += dz*NA²*p²/(2*λ_mm)` and `NA = D1/(2|efl|)`.
  - It then builds the PSF with `pad_factor=4`, then the 2-D MTF (both from another module, not visible).
  - It slices from the centre outwards: `mtf_tan = mtf2d[cy:, cx]` and `mtf_sag = mtf2d[cy, cx:]`.
  - Frequency step: `df = (D1/(N-1)) / (wavelength_nm*1e-6*efl)`.
    - `D1` is the first element's lens diameter; `N = W.shape[0]`.
    - `efl = system.get_system_focal_length() or 100.0`.
  - `freq = arange(len)*df`, masked by `freq <= max_freq` only. There is no `>= 0` check here, so a negative EFL gives negative frequencies, all of which are kept.
  - `freq[0] == 0`.

## 5. Side effects
- The system and its lenses are not mutated.
- Only internal temporaries (the wavefront object) are modified.
- Logging only; no files are written.

## 6. Branches and edge cases
- `OpticalSystem()` with no lenses returns empty arrays (`len == 0`).
- `max_freq=-1` (geometric, real lens) returns three empty arrays, since 0 > -1.
- `max_freq=0` returns `freq == [0.]` and `mtf_tan == mtf_sag == [1.]`.
- Large `focus_shift_mm` gives a larger rms. The window becomes > 0.256, so `df` gets smaller and there are more entries below `max_freq`. MTF at non-zero frequencies drops.
- `field_angle=x` (alias) gives the same result as `field_angle_deg=x`.

## 7. Rules from comments
- The geometric window must scale with the spot (2.2 × RMS radius each side) so that light is not clipped. The floor is 0.256 mm.
- Sampling is ~2 µm per pixel, 128 to 1024 pixels.
- The diffraction branch must honour `focus_shift_mm`. A bug was fixed here: the MTF used to be identical at every defocus.
- Pupil spacing uses `N-1` intervals, not `N`.

## 8. Testing hints
- Setup:
  ```python
  from openlens.lens import Lens
  from openlens.optical_system import OpticalSystem
  from openlens.analysis.psf_mtf import ImageQualityAnalyzer
  s = OpticalSystem(); s.add_lens(Lens())   # default biconvex BK7
  an = ImageQualityAnalyzer(s)
  r = an.calculate_mtf()
  ```
- Safe geometric assertions:
  - Keys equal the set above.
  - Equal lengths.
  - `r["freq"][0] == 0`, and the array is non-decreasing.
  - `max(freq) <= max_freq`.
  - `np.isclose(r["mtf_tan"][0], 1.0)`.
  - `0 <= mtf <= 1 + 1e-9`.
  - Uniform spacing via `np.allclose(np.diff(freq), freq[1])`.
- To check `df` exactly, recompute it rather than hard-coding a value:
  - `p = an.calculate_psf(sensor_size_mm=W, pixels=N)` with the same `W` and `N` formula.
  - Then `np.allclose(r["freq"], np.fft.fftfreq(N, p["step_size"])[:N//2][mask])`.
  - `ceil(0.256/0.002)` may be 128 or 129 depending on float rounding; do not assume which.
- Pick `max_freq` small (e.g. 10 or 20) so the result does not depend on the cap.
- Compare runs with `np.allclose`; ray tracing is deterministic for the same inputs.
- Defocus test (geometric): `an.calculate_mtf(focus_shift_mm=2.0)["mtf_tan"][1:]` is generally lower than in focus. Prefer comparing a mean over the first few frequencies, and do not assert exact values.
- Diffraction branch: assert only structure — keys, equal lengths, `freq[0] == 0`, `freq <= max_freq`. Its values depend on unseen modules.
- Empty system: assert `r["freq"].size == 0` for both `use_diffraction` values.
- Warnings can be checked with `caplog`; the logger name is `openlens.analysis.psf_mtf`.

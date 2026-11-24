# Notes: `WavefrontSensor.get_pupil_wavefront(field_angle_deg=0.0, wavelength_nm=WAVELENGTH_GREEN, grid_size=64)`

## 1. Purpose
- Builds the exit-pupil wavefront-error map, in waves, for an `OpticalSystem` at a given wavelength and field angle.
- Before tracing, it temporarily sets every lens to `wavelength_nm`. It restores the lenses afterwards, even when an exception is raised.

## 2. Inputs
- `self`: build it as `WavefrontSensor(system)`.
  - The constructor stores `self.system` and `self.tracer = SystemRayTracer3D(system)`.
- `field_angle_deg` (float, default `0.0`):
  - The ray direction is `(cos θ, sin θ, 0)`, a tilt in the X–Y plane.
  - The value is not validated.
- `wavelength_nm` (float, default `WAVELENGTH_GREEN` from `openlens.constants`):
  - Its value is not shown here; import it rather than hard-coding it.
  - The trace converts it to mm as `wavelength_nm * 1e-6`.
  - The value is not validated.
- `grid_size` (int, default `64`):
  - Number of pupil samples per axis.
  - The value is not validated.
- No argument is normalised. All three are passed by keyword to the private `_compute_pupil_wavefront(field_angle_deg=..., wavelength_nm=..., grid_size=...)`.

## 3. Raises
- The function raises nothing itself. Exceptions from tracing, `refractive_index_at`, or `find_paraxial_focus` (other modules) propagate.
- In every case, the `finally` block first restores each lens's `refractive_index` and `wavelength`.

## 4. Returns and guarantees
The function returns a `WavefrontError` with attributes `.Y`, `.Z` and `.W`. The cases below are checked in this order.

1. **System has no elements** (`system.elements` is empty):
   - `Y`, `Z` and `W` are all the same object, `np.array([])`, so `Y is Z is W` and `W.size == 0`.
   - Paraxial focus is not computed and nothing is traced.
2. **Otherwise, the pupil grid is built:**
   - `r = system.elements[0].lens.diameter / 2`.
   - `y = z = np.linspace(-r, r, grid_size)` and `Y, Z = np.meshgrid(y, z)`. Both have shape `(grid_size, grid_size)`.
   - `Y[i, j] == y[j]`, so `Y[0, 0] == -r` and `Y[0, -1] == r`.
   - `Z[i, j] == z[i]`, so `Z[-1, 0] == r`.
3. **`find_paraxial_focus(system, wavelength_nm, tracer=self.tracer)` returns `None`:**
   - `W` is entirely NaN, with the same shape as `Y`.
   - A warning is logged: "Paraxial focus unavailable ...".
4. **Chief ray fails:**
   - The failure condition is: path has fewer than 2 points, or the ray is `terminated`, or `|direction.x| < 1e-9`.
   - `W` is entirely NaN and a warning is logged.
   - The same happens for a degenerate reference sphere, where `R_ref <= 1e-9`.
5. **Normal case:**
   - Samples outside the circular aperture (`py² + pz² > r²`, strictly greater) are NaN. All four corners are therefore always NaN.
   - Rays that are terminated (vignetted or TIR), or that miss the reference sphere, are also NaN.
   - A valid sample is `W = (OPL_ray + n·t − OPL_ref) / λ_mm`.
   - Over the valid samples, a least-squares plane `c0 + c1·(Y/r) + c2·(Z/r)` is subtracted. If `lstsq` throws, only the mean is subtracted instead.
   - **Consequences on the valid (non-NaN) samples:**
     - `nanmean(W) ≈ 0`.
     - `nansum(W * Y) ≈ 0` and `nansum(W * Z) ≈ 0`, up to about 1e-9 relative.
     - The residual is orthogonal to the constant column and the two tilt columns.
- The result is always a new `WavefrontError`. The sensor keeps no state, and `self.tracer` is not replaced.

## 5. Side effects
- For each `element` in `system.elements`, the lens is changed during the call:
  - `lens.wavelength = wavelength_nm`
  - `lens.refractive_index = lens.refractive_index_at(wavelength_nm)`
- After the call, both attributes are restored to the exact original values via `try/finally`. No net mutation of the lenses is observable.
- If `system.elements` is `None` or missing, it is treated as `[]` (`getattr(..., None) or []`).
- Element positions are not modified, and no files are written.

## 6. Branches and edge cases
- `OpticalSystem()` with no lenses returns an empty result, as in case 1 above.
- `grid_size=1`: `linspace` gives `[-r]`, so the only sample is the corner `(-r, -r)`.
  - That sample is outside the aperture, so `W.shape == (1, 1)` and it is NaN.
  - This assumes the focus and chief-ray steps succeed.
- `grid_size=2`: all 4 samples are corners, so `W` is all NaN.
- Odd `grid_size`: the centre value may be about 1e-16 rather than exactly 0. Do not compare it with `==`.
- If no valid samples exist, the plane fit is skipped.

## 7. Rules stated in docstrings and comments
- Asking for a wavefront must never change the system it was asked about.
- The wavefront must actually depend on the wavelength. Previously, the R, G and B wavefronts came out identical; this is the bug the method fixes.
- The tracer is intentionally not rebuilt. A tracer the caller installed deliberately must be kept (`sensor.tracer` stays the same object).
- An untraced chief ray must invalidate the whole map, producing an all-NaN `W`.

## 8. Testing hints
- **Building a system:**
  ```python
  from openlens.lens import Lens
  from openlens.optical_system import OpticalSystem
  from openlens.analysis.diffraction_psf import WavefrontSensor
  sys_ = OpticalSystem()
  sys_.add_lens(Lens(radius_of_curvature_1=50, radius_of_curvature_2=-50, thickness=5, diameter=25))
  wf = WavefrontSensor(sys_).get_pupil_wavefront(grid_size=9)
  ```
  - `wf.W.shape == (9, 9)`.
  - `wf.Y[0, 0] == -12.5` and `wf.Y[0, -1] == 12.5`.
  - `wf.Z[0, 0] == -12.5` and `wf.Z[-1, 0] == 12.5`.
  - `np.isnan(wf.W[0, 0])` is true (a corner).
- **Restoration:**
  - Record `lens.refractive_index` and `lens.wavelength` before the call.
  - Call with `wavelength_nm=486.1`.
  - Assert both values are equal afterwards.
- **Restoration on exception:**
  - Monkeypatch `sensor._compute_pupil_wavefront` with a function that raises `RuntimeError`.
  - Use `pytest.raises` and assert the lens values are restored.
- **Wavelength applied during the trace:**
  - Monkeypatch `_compute_pupil_wavefront` with a spy that records `lens.refractive_index`, `lens.wavelength` and its kwargs.
  - Expect `lens.wavelength == 486.1` and `n == lens.refractive_index_at(486.1)`.
  - Expect kwargs exactly `{'field_angle_deg': a, 'wavelength_nm': w, 'grid_size': g}`.
  - The spy should return a dummy object; the return value is passed through unchanged (`is` identity).
- **Plane-fit property** (normal case only, with at least 1 valid sample):
  - `abs(np.nanmean(W)) < 1e-6`.
  - `abs(np.nansum(W * Y)) < 1e-6 * np.nansum(np.abs(W * Y)) + 1e-9`.
- **Pitfalls:**
  - Do not assert exact numeric `W` values. They depend on the tracer and paraxial modules (not shown), and on `PARAXIAL_LAUNCH_OFFSET_MM`.
  - Do not assume any particular sample in the interior of the aperture is non-NaN unless the lens clearly passes those rays.
  - Use `np.isnan`, never `==`, for NaN checks.
  - Do not hard-code the value of `WAVELENGTH_GREEN`.

# Notes: `BeamSynthesisPropagator.propagate_to_image`

## 1. Purpose
- Splits the entrance pupil (the first element's full diameter) into a square grid of Gaussian beamlets.
- Traces each beamlet through every element and sums their complex fields on a square detector plane perpendicular to x.
- Returns the detector grids `(DY, DZ)` and the intensity `|E|²`.

## 2. Inputs
- `self`: built with `BeamSynthesisPropagator(system)`. The constructor stores `self.system` and creates `self.tracer = SystemRayTracer3D(system)`.
- `wavelength_mm` (default `WAVELENGTH_GREEN * NM_TO_MM`, from `openlens.constants`):
  - This is the vacuum wavelength in mm. It is not validated.
  - It is used for the beamlet Rayleigh range `zR = π·w0²/λ`, for the phase `k = 2π/λ`, and for the paraxial-focus search, which receives `wavelength_mm / NM_TO_MM` nm.
- `grid_size` (int, default 32):
  - Pupil samples per axis: `np.linspace(-r, r, grid_size)`, where `r = elements[0].lens.diameter/2`.
  - `delta = 2r/(grid_size-1)` when `grid_size > 1`, otherwise `2r`. The beamlet waist is `w0 = 1.5*delta`.
  - Only points with `py²+pz² <= r²` become beamlets; the boundary is included.
- `image_plane_x` (float or None, default None):
  - It is not validated.
  - If None, the plane comes from `_default_image_plane` (see §6).
- `detector_size` (float, default 0.1 mm) and `detector_pixels` (int, default 64):
  - Grid: `det = np.linspace(-size/2, size/2, pixels)` on both axes.
  - Neither value is validated.

## 3. Raises
- `ImportError("numpy is required for BSP calculation.")`: only when numpy is missing (`NUMPY_AVAILABLE` False). This is checked first and is not practically testable.
- A negative `grid_size` or `detector_pixels` raises numpy's `ValueError` from `linspace` ("Number of samples, -1, must be non-negative."). This is only reached when the system has elements.
- No other explicit raises. Tracing errors from `tracer_3d` or `paraxial` would propagate; that code is not visible here.

## 4. Returns and guarantees
- Empty system (`not system.elements`): returns the tuple `(np.array([]), np.array([]), np.array([]))`.
  - Each array has shape `(0,)`.
  - This path does no tracing and no image-plane lookup.
- Otherwise it returns the tuple `(DY, DZ, Intensity)`, built from `DY, DZ = np.meshgrid(det_y, det_z)`:
  - All three arrays have shape `(detector_pixels, detector_pixels)`.
  - `DY[i, j] == det_y[j]`: values vary along axis 1.
  - `DZ[i, j] == det_z[i]`: values vary along axis 0.
  - `DY[0,0] == -detector_size/2` and `DY[0,-1] == detector_size/2`.
  - `Intensity = np.abs(E_field)**2` is a real float64 array with values `>= 0`.
- With no surviving beamlets, `Intensity` is exactly all `0.0`.
- Per-beamlet details:
  - Each beamlet starts at `x = elements[0].position - 20`, at `(py, pz)`, travelling along `+x`.
  - It starts with `q0 = 1j*zR`.
  - Beamlets whose ray is terminated (front or back surface not `REFRACTED`) are dropped.
  - Beamlets with `|direction.x| < 1e-6` are skipped.
  - Each remaining beamlet is propagated by `t = (image_plane_x - origin.x)/direction.x`; `t` may be negative.
  - Its field is `(w0/wx)·exp(-ly²/wx² - lz²/wy²)·exp(-1j·k·(ly²/(2Rx)+lz²/(2Ry)+OPL))`.
- New arrays are created on every call, and the call is deterministic: the same inputs give an identical result.

## 5. Side effects
- New `LensRayTracer3D` objects are built per call, from the global node transforms when `system.root` has one flat element node per element, otherwise from `x_offset=element.position`.
- Beamlet rays are local objects, so the system is not mutated as far as this module shows.
- Logs at WARNING on logger `openlens.analysis.beam_synthesis`:
  - `"Beam synthesis: ignoring non-finite image plane %r"` when the configured plane is unusable.
  - `"Beam synthesis: no paraxial focus (afocal or untraceable system); placing the detector %.1f mm past the last surface at x=%.3f"` when the focus search fails.

## 6. Branches and edge cases
- `OpticalSystem()` with no lenses returns three empty arrays (§4).
- `grid_size=1`: `linspace` returns `[-r]`, and the point `(-r,-r)` lies outside the circle, so there are no beamlets and `Intensity` is all zeros. The image plane is still resolved.
- `grid_size=0` also gives all zeros.
- `detector_pixels=0` gives empty `(0,0)` arrays.
- `detector_pixels=1` gives `DY == [[-size/2]]`.
- Image-plane resolution when `image_plane_x is None`, in order:
  1. `getattr(system, "image_plane_x_mm", None)`. It is not visible here whether `OpticalSystem` defines this attribute. If the value is not None, it is converted with `float()`. If that succeeds and the result is finite, it is used.
  2. If the value is non-convertible (e.g. `"abc"`) or non-finite (`inf`, `nan`), the warning is logged and resolution falls through to step 3.
  3. `find_paraxial_focus(system, wavelength_mm/NM_TO_MM, tracer=self.tracer)`. A finite result is used.
  4. If the focus is None or non-finite, the plane is `system.get_total_length() + 50.0` (`DEFAULT_IMAGE_PLANE_PAST_LAST_SURFACE`), and the warning is logged.
- An explicit `image_plane_x` skips all of the above, including the warnings.

## 7. Rules from comments and docstrings
- Beamlet spacing is `diameter/(grid_size-1)`, deliberately not `/grid_size`.
- The default detector sits at the paraxial focus, the same plane `DiffractionPSF` uses. It is not "50 mm past the last vertex" unless the system is afocal or untraceable.
- `q` is advanced by `distance/n` inside glass.
- Refraction of a beamlet's `q`:
  - Front surface: from the beam's current `n` (air) into `lens.refractive_index`, using `R1`.
  - Back surface: from `lens.refractive_index` into 1.0, using `R2`.
  - After each refraction the beam's wavelength is rescaled by `n1/n2`.

## 8. Testing hints
Build a system:
```python
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.analysis.beam_synthesis import BeamSynthesisPropagator
import openlens.analysis.beam_synthesis as bs

s = OpticalSystem()
s.add_lens(Lens(diameter=25.0))
p = BeamSynthesisPropagator(s)
DY, DZ, I = p.propagate_to_image(grid_size=8, detector_pixels=16)
```
- Keep `grid_size` small (6–10) so tests run fast.
- Check `I.shape == (16,16)`, `np.isrealobj(I)`, `(I >= 0).all()`, and `np.isfinite(I).all()`.
- Check the grids with `np.allclose(DY[0], np.linspace(-0.05, 0.05, 16))` and `np.allclose(DZ[:,0], same)`.
- Check `grid_size=1` gives `I.sum() == 0` (exact), and that an empty system gives all `.size == 0`.
- Check the default plane equals the paraxial focus:
  - Compare `propagate_to_image(grid_size=6)` against an explicit `image_plane_x=find_paraxial_focus(s)` (from `openlens.analysis.paraxial`) using `np.allclose`.
  - Use `allclose`, not exact equality: `WAVELENGTH_GREEN*NM_TO_MM/NM_TO_MM` may differ in the last bit.
- Check the fallback plane with `monkeypatch.setattr(bs, "find_paraxial_focus", lambda *a, **k: None)`:
  - The default call should equal `image_plane_x=s.get_total_length()+50.0`, using `np.array_equal` or `allclose`.
  - `caplog` should contain `"no paraxial focus"`.
  - Repeat with the patch returning `float("nan")`.
- Check the configured plane by setting `s.image_plane_x_mm = <focus value>`:
  - The default call should equal the explicit call with the same x.
  - With `s.image_plane_x_mm = "abc"` or `float("inf")`, the call should fall back, and `caplog` should contain `"ignoring non-finite image plane"`.
  - Whether attribute assignment works on `OpticalSystem` is not visible here; prefer the monkeypatch tests.
- Determinism: two identical calls should give `np.array_equal` intensities.
- Changing `image_plane_x` far from focus (e.g. focus ± 20 mm) should change the intensity: assert `not np.allclose(...)`.
- Pitfalls:
  - Do not assert exact peak locations or values.
  - Do not assert exact y/z symmetry; use a loose tolerance if you check it.
  - Do not pass `wavelength_mm` in nm.

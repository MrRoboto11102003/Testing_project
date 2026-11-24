# Notes: `WavefrontSensor`: source not available, notes cannot be grounded

**`WavefrontSensor` is not defined in `openlens/analysis/beam_synthesis.py`.** The module I was given defines only `GaussianBeam` and `BeamSynthesisPropagator`. It never imports or mentions `WavefrontSensor`. The public API index lists it in **`openlens.analysis.diffraction_psf`**:

```python
# openlens.analysis.diffraction_psf
class WavefrontSensor:  # Reconstructs wavefront from ray trace data (OPL).
    def __init__(self, system: OpticalSystem)
    def get_pupil_wavefront(self, field_angle_deg: float=0.0,
                            wavelength_nm: float=WAVELENGTH_GREEN,
                            grid_size: int=64) -> WavefrontError
```

I don't have the source of `diffraction_psf.py`, so I cannot state any of the following:
- Behaviour, validation, exceptions or messages.
- Grid layout, units or the reference sphere.
- What happens for an empty system or an untraceable ray.

Any notes on those points would be guesses. Tests built on guesses would likely fail on the current code and be false alarms.

## What is safe to rely on (signature and index only)
- Import path: `from openlens.analysis.diffraction_psf import WavefrontSensor, WavefrontError`. It is **not** `openlens.analysis.beam_synthesis`.
- The constructor takes one `OpticalSystem`.
- `get_pupil_wavefront(field_angle_deg=0.0, wavelength_nm=WAVELENGTH_GREEN, grid_size=64)` is annotated to return a `WavefrontError`.
- `WavefrontError(Y, Z, W)` takes three `np.ndarray` arguments. The index does not show whether they are stored as attributes `.Y`, `.Z` and `.W`.
- `DiffractionPSFCalculator.calculate_psf(wavefront, intensity_map=None, pad_factor=4)` consumes a `WavefrontError`.

## Testing hints (if a test file must still be written now)
- Only assert things that follow from the signature:
  - `WavefrontSensor(system)` constructs without error for a system built with `OpticalSystem(); system.add_lens(Lens())`.
  - The result of `get_pupil_wavefront()` is an instance of `WavefrontError`.
- Do **not** assert any of these, because they are unverified:
  - Array shapes such as `(64, 64)`.
  - NaN masking outside the pupil.
  - Units (waves vs. mm).
  - Zero wavefront on axis.
  - Behaviour on an empty system.
  - Exceptions.
- Such shallow tests are unlikely to catch regressions. The real fix is below.

## Needed to write proper notes
Please provide the full source of `openlens/analysis/diffraction_psf.py`, plus any private helpers or constants it uses. Alternatively, if you meant a target inside `beam_synthesis.py`, I can write full notes for one of these:
- `GaussianBeam`, with its properties `w_x`, `w_y`, `R_x`, `R_y` and methods `advance_q`, `propagate`, `refract`.
- `BeamSynthesisPropagator.propagate_to_image`.

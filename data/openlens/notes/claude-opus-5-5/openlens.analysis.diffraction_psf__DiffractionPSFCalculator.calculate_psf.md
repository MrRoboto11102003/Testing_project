# Notes: `DiffractionPSFCalculator.calculate_psf(wavefront, intensity_map=None, pad_factor=4)`

## 1. Purpose
- Static method. Computes a diffraction PSF by taking the |FFT|² of the pupil function `A·exp(i·2π·W)`.
- `W` is the wavefront error in **waves**. The pupil is zero-padded and the result is normalised so that its peak equals 1.0.
- Call it as `DiffractionPSFCalculator.calculate_psf(wf)`. No instance is needed.

## 2. Inputs
- `wavefront`: any object with a `.W` attribute. Normally this is `WavefrontError(Y, Z, W)`.
  - Only `.W` is read. `Y` and `Z` are ignored, so `WavefrontError(None, None, W)` works.
  - `W` must be a numpy 2-D array. The code assumes it is square: `N = W.shape[0]`.
  - NaN entries mean "outside the aperture".
  - Use a float dtype for `W` if you need NaNs.
- `intensity_map`: pupil apodisation (intensity, not amplitude). The default `None` means uniform amplitude 1.
  - When given, `amplitude = np.sqrt(intensity_map)`. This creates a new array, so the caller's map is not modified.
  - Its shape must match `W`.
- `pad_factor`: an int, default 4. The padded size is `M = N * pad_factor`.
- No explicit validation is done anywhere.

## 3. Raises
There are no explicit raises. Errors only come from numpy or Python:
- `W` is a Python list → `AttributeError` (`'list' object has no attribute 'shape'`).
- `W` is non-square, e.g. shape (4, 6) → `ValueError` from numpy broadcasting.
  - The pupil is assigned into an (N, N) slice of the padded array.
- `pad_factor=0` with N=4 → `ValueError` "could not broadcast input array from shape (4,4) into shape (0,0)".
- Float `pad_factor` (e.g. 2.0) → `TypeError` from `np.zeros` (shape must be ints).
- An `intensity_map` shape that does not match `W` → `IndexError` or `ValueError` from boolean masking or broadcasting.

## 4. Returns and guarantees
- The result is a **new** `np.ndarray`.
  - dtype is float64.
  - shape is `(N*pad_factor, N*pad_factor)`. For example, a 4×4 `W` with the default pad gives (16, 16), and a 32×32 `W` with pad 2 gives (64, 64).
- Every value is ≥ 0 exactly. It is computed as `real(z*conj(z))`.
- If any pupil amplitude is non-zero, the result is normalised: `psf.max() == 1.0`.
- **Zero or constant-phase pupil** (W all 0, no NaN, `intensity_map` None):
  - The peak is at the centre `psf[M//2, M//2] == 1.0`, and that is the argmax.
  - The PSF is symmetric: `psf[M//2+k, M//2] ≈ psf[M//2-k, M//2]` for small k.
- **Parseval check** for a flat-phase pupil with K unmasked samples of amplitude 1:
  - `psf.sum() ≈ M²/K`.
  - Example: N=4, pad=4, no NaN → M=16, K=16, sum ≈ 16.0.
  - Example: N=4, pad=1 → M=4, K=16, sum ≈ 1.0. In this case the PSF is a single delta: `psf[2,2]=1` and everything else ≈ 0.
- **Uniform-pupil delta:** with `pad_factor=1` and a full uniform pupil, the result is a delta at `[N//2, N//2]`.

## 5. Side effects
- `wavefront.W` is not mutated. `np.nan_to_num` returns a copy, and the mask is applied to `amplitude`, not to `W`.
- `intensity_map` is not mutated, because `sqrt` creates a new array.
- There is no global state and no I/O.

## 6. Branches and edge cases
- **numpy missing** → returns `[]`. This cannot be tested in a normal environment.
- **`intensity_map is None`** → amplitude is `ones_like(W)`.
  - The `else` branch gives the same output for `intensity_map=np.ones((N,N))`.
- **NaNs in W:**
  - The amplitude at those samples becomes 0.
  - The phase there becomes 0.
  - The NaNs do not propagate: the result has no NaN.
- **All-NaN W** (or `intensity_map` all zeros):
  - The pupil is all zero and the max is 0, so normalisation is skipped.
  - Returns `np.zeros((M, M))` with no division error. `psf.max() == 0.0`.
- **Phase is in waves:**
  - Adding any constant (piston) to W gives the same PSF, e.g. W = 0.3 everywhere or W = 1.0 everywhere matches W = 0.
  - Adding an integer to individual samples also leaves the PSF unchanged.
- **Tilt:** `W[i,j] = j * s / M` (a linear ramp along axis 1) moves the peak by `s` pixels along axis 1. The row stays at `M//2`.
  - Assert `abs(col - M//2) == s` and `row == M//2`, so the test does not depend on the sign.
- **Intensity scaling:** multiplying `intensity_map` by a constant (e.g. 4×) leaves the normalised PSF unchanged (`allclose`).
- **Non-trivial phase** (e.g. a quadratic `W = 0.5*(y²+z²)`):
  - The max is still 1.0.
  - The unnormalised peak drops, which you can see through `psf.sum()`: it becomes larger than the flat case's M²/K.

## 7. Rules from comments and docstring
- NaN means outside the aperture, so its amplitude is 0.
- The zero padding exists to sample the PSF more finely. A higher `pad_factor` gives a smoother PSF.
- `fftshift` puts zero frequency at the centre.
- The output is described as "normalized intensity".

## 8. Testing hints
- Build inputs directly:
  ```python
  W = np.zeros((8, 8))
  wf = WavefrontError(None, None, W)
  ```
- To make a circular aperture:
  ```python
  y = np.linspace(-1, 1, N)
  Y, Z = np.meshgrid(y, y)
  W = np.where(Y**2 + Z**2 <= 1, 0.0, np.nan)
  ```
  Then `K = np.count_nonzero(~np.isnan(W))`, and `psf.sum() ≈ M²/K`.
- Use `np.isclose` or `allclose` with roughly `atol=1e-9` for FFT results.
  - The peak exactly equals 1.0 after division, but other values carry round-off error.
  - Use `np.unravel_index(psf.argmax(), psf.shape)` to find the peak position.
- Pin down the shape `(N*pad, N*pad)`. Use several pad factors (1, 2, 4).
- Check that `W` is unchanged after the call: `np.array_equal(W, W_copy, equal_nan=True)`.
- Do not use `WavefrontSensor` to build inputs. It needs a system trace and is slow and indirect.

# Notes: `ImageQualityAnalyzer.simulate_image` (openlens/analysis/psf_mtf.py)

## 1. Purpose
- Blurs an image by convolving each channel with the system's **diffraction** PSF at that channel's wavelength.
- Returns a float64 array with the same shape as the input, clipped to [0, 1].

## 2. Inputs
- `image_array`: must have a `.shape` attribute **before** conversion, because `np.zeros(image_array.shape, dtype=float64)` runs before `np.asarray`.
  - After `np.asarray`, it must be 2-D `(H, W)` or 3-D `(H, W, C)`.
  - Integer dtypes are divided by `np.iinfo(dtype).max`, so uint8 255 becomes 1.0.
  - Every other dtype (float, bool) is cast to float64 unchanged.
- `pixel_size_mm=0.005`: used to size the PSF kernel.
  - `psf_size_mm = max(0.2, pixel_size_mm*64)`
  - `psf_pixels = int(psf_size_mm / pixel_size_mm)`; if even, `+1`, so it is always odd.
- `wavelengths_nm=(WAVELENGTH_C_LINE, WAVELENGTH_D_LINE, WAVELENGTH_F_LINE)`: any sequence. Import the constants from `openlens.constants`; do not hard-code the values.
- `field_angle_deg=0.0` and `focus_shift_mm=0.0`: passed straight through to `calculate_psf`.
- `**kwargs`, legacy aliases:
  - `pixel_size` overrides `pixel_size_mm`.
  - `wavelengths` overrides `wavelengths_nm`.
  - Any other key raises `TypeError`.

## 3. Raises (in this order)
1. `TypeError(f"Unexpected kwargs: {list(kwargs.keys())}")`.
   - `sim(img, foo=1)` → `"Unexpected kwargs: ['foo']"`.
   - Aliases are popped first, so `sim(img, pixel_size=0.01, foo=1)` → `"Unexpected kwargs: ['foo']"`.
2. `AttributeError` for a plain list such as `sim([[0.1]])`, because the list has no `.shape`. This is not a ValueError.
3. `ValueError(f"image_array must be 2-D (H, W) or 3-D (H, W, C); got shape {image_array.shape}")`.
   - `np.zeros(5)` → `"... got shape (5,)"`.
   - `np.array(1.0)` → `"... got shape ()"`.
   - `np.zeros((2,2,2,2))` → `"... got shape (2, 2, 2, 2)"`.
4. `ValueError(f"image_array has {C} channels but {len(wavelengths_nm)} wavelengths were given")`.
   - `np.zeros((4,4,2))` with the default 3 wavelengths → `"image_array has 2 channels but 3 wavelengths were given"`.
   - `np.zeros((4,4,3))` with `wavelengths=(550.0,)` → `"image_array has 3 channels but 1 wavelengths were given"`.
- All four checks happen before any PSF is computed, so they fire even with an empty `OpticalSystem()`.
- If a mocked `calculate_psf` returns an **integer** kernel with a positive sum, `kernel /= kernel_sum` raises numpy's casting error. Mocks should return float arrays.

## 4. Returns and guarantees
- Returns `np.clip(output_image, 0, 1)`: a new float64 ndarray with `shape == input.shape`.
- For each wavelength `wl` at index `i`, it calls:
  ```
  self.calculate_psf(field_angle_deg=..., wavelength_nm=wl, focus_shift_mm=...,
                     sensor_size_mm=psf_pixels*pixel_size_mm, pixels=psf_pixels,
                     use_diffraction=True)
  ```
  - There is exactly one call per wavelength, in order, including for grayscale input.
- Kernel handling: `kernel = psf_data["image"]`. If `sum > 0` it is normalised in place to sum 1; if `sum == 0` it is left unchanged, so convolution gives zeros.
- Convolution is `scipy.signal.fftconvolve(channel, kernel, mode="same")`. If scipy is missing, a numpy FFT fallback with the same "same" cropping is used.
- Grayscale (2-D) output is the **mean** of the per-wavelength convolutions.
- 3-D output: channel `i` is the input channel `i` convolved with kernel `i`.
- Clipping: values above 1 become 1 and values below 0 become 0 (e.g. float input 2.0 → 1.0, -0.5 → 0.0).

## 5. Side effects
- The input array is not mutated (`astype` makes a copy).
- The PSF kernel object returned by `calculate_psf` **is** divided in place. A mock that returns a shared array will see it modified.
- Logging only; no files written.

## 6. Branches and edge cases
- **Empty system:** `ImageQualityAnalyzer(OpticalSystem())`. `calculate_psf` returns all-zero images (its empty-system branch runs before the diffraction branch). Every kernel is zero and not normalised, so the result is `np.zeros(shape)`.
  - Example: `sim(np.ones((8,8)))` → all 0.0, float64, shape (8,8).
  - The same holds for uint8 and for 3-D input.
- **Integer vs float input:** with an identity kernel, `np.array([[0,255],[51,102]], uint8)` → `[[0,1],[0.2,0.4]]`.
- **Grayscale with mixed kernels:** mock an identity kernel for one wavelength and zero kernels for the other two. The output is `input/3`.
- **3-D:** mock an identity kernel for wavelength 0 and zero kernels otherwise. Output channel 0 equals the input; channels 1 and 2 are zero.
- **Non-normalised kernel:** `[[0,0,0],[0,4.0,0],[0,0,0]]` is scaled to sum 1, so it acts as identity (output is not 4× the input).
- **Alias check:** `wavelengths=(500.0, 600.0)` with a 2-channel image is accepted. Check the call list `[500.0, 600.0]` via the mock.

## 7. Rules from comments and docstring
- Output is always float64 and normalised 0–1. The old `zeros_like` dtype bug truncated uint8 results.
- 2-D input must give 2-D output, with no invented channel axis.
- Integer input is scaled by its dtype max so the clip does not saturate the image.
- The diffraction PSF is used deliberately, because geometric PSFs are sparse.

## 8. Testing hints
- **Recommended approach:** build `iqa = ImageQualityAnalyzer(OpticalSystem())`, then monkeypatch the instance:
  ```python
  calls = []
  def fake(**kw): calls.append(kw); k = np.zeros((3,3)); k[1,1] = 1.0; return {"image": k}
  monkeypatch.setattr(iqa, "calculate_psf", fake)
  ```
  - Return a **fresh float array** on every call.
  - Assert `use_diffraction is True` in each recorded call.
  - Assert `kw["pixels"] % 2 == 1` and `kw["sensor_size_mm"] == kw["pixels"] * pixel_size`.
- Do **not** pin the exact `pixels` value. `0.32/0.005` may floor to 63 or 64, giving 63 or 65.
- Compare convolution results with `np.allclose(..., atol=1e-9)`, because FFT noise means outputs are not exact.
- Check `result.dtype == np.float64`, `result.shape == input.shape`, and that `result is not input`.
- **Real lens:** the full diffraction pipeline (WavefrontSensor, DiffractionPSFCalculator) depends on other modules not shown here. Only assert shape, dtype, `0 <= result <= 1`, and finiteness; keep images small, e.g. 16×16.
- **Exceptions:** use `pytest.raises(ValueError, match=re.escape(msg))` with the exact text above.

# Notes: `ImageSimulator.simulate_image(input_image, object_distance, image_distance=None, wavelength=WAVELENGTH_D_LINE, pixel_pitch_mm=0.01)`

## 1. Purpose
Runs an image through the simulated optics in this order:
1. image distance (lens equation)
2. aberration blur
3. diffraction blur
4. per-channel chromatic zoom (RGB only)
5. cos⁴ vignetting
6. quality metrics

It returns a dict.

## 2. Inputs
- `input_image`: numpy array, `(H, W)` or `(H, W, C)`. Use **float** arrays with values in [0, 1].
  - `gaussian_filter` keeps the input dtype, so integer input stays integer until the vignetting multiply.
- `object_distance`: float in mm. Not validated.
- `image_distance`: `None` means it is computed. Any value given is used as-is and returned unchanged (no `float()` conversion).
- `wavelength`: nm. Forwarded to `refractive_index_at`, `get_aberrations` and the Airy radius.
- `pixel_pitch_mm`: mm per pixel.
  - `<= 0` disables aberration blur.
  - It also gives diffraction sigma 0, which is floored to 0.1 px.
  - It also skips vignetting.
- `self.optical_system` (set in the constructor) is duck-typed. Attributes it uses:
  - `.elements[i].lens.refractive_index` (read and written)
  - `.lens.refractive_index_at(wl)`
  - `.elements[0].lens.diameter`
  - `get_system_focal_length()`
  - optionally `get_aberrations(wl)`

## 3. Raises
Checks run in this order:
1. **When `image_distance is None`:** `optical_system.elements` is accessed directly. A target without `.elements` raises `AttributeError`.
2. **Focal length.** `get_system_focal_length()` is callable and returns falsy (`None` or `0`):
   - `ValueError("OpticalSystem reports no focal length; cannot simulate an image through it.")`
   - Raised from the image-distance step, or later from the diffraction step if `image_distance` was given.
3. **No focal-length method.** None of `get_system_focal_length`, `calculate_focal_length` or `effective_focal_length` exists, or the last two return falsy:
   - `TypeError(f"{type(system).__name__} is not a supported optical target for ImageSimulator.")`
4. **No aperture** (scipy only, in diffraction). The pupil diameter check runs **before** the focal length is read there. It fails when there is no element with diameter > 0 and no positive `aperture_diameter` or `diameter` attribute:
   - `ValueError("Optical target has no usable aperture; cannot size the diffraction spot.")`
   - Trigger: `elements=[]` with `get_system_focal_length` returning 100 and `image_distance=50`.
5. **`object_distance == 0`:** `ZeroDivisionError`. It comes from `1.0/object_distance`, or from the magnification `-image_distance/object_distance` even when `image_distance` is given.
6. **No `get_aberrations` on the target:** `AberrationsCalculator` from `openlens.aberrations` is used. Its behaviour and errors are not visible here, so avoid it in tests.

## 4. Returns
A new dict with exactly 4 keys:

- `"output_image"`: new ndarray with the same shape as the input.
- `"magnification"`: `-image_distance / object_distance`.
- `"image_distance"`:
  - When computed: `abs(1/(1/f - 1/object_distance))`.
  - If `abs(object_distance - f) < 0.001`: `float("inf")`, so magnification is `-inf`.
- `"metrics"`: dict with keys `psnr`, `global_ssim`, `nyquist_spectral_ratio`, `sharpness`.
  - `psnr = 10*log10(1/mse)`, or `inf` when `mse == 0`.
  - `global_ssim`: global SSIM formula with `c1=1e-4`, `c2=9e-4`.
  - `nyquist_spectral_ratio`:
    - Mean FFT magnitude over the ring `|r - min(h,w)//4| <= 2`, divided by DC.
    - Returns 0.0 if DC <= 0.
    - On tiny images the ring includes DC, so use ≥ 16×16 images.
  - `sharpness`: mean Sobel gradient magnitude of the grayscale (channel-mean) image.

The focal length `f` is measured after every lens's `refractive_index` is set to `refractive_index_at(wavelength)`.

## 5. Side effects
- Each `lens.refractive_index` is temporarily overwritten and **restored in a `finally` block**. After the call the values equal their originals, even if the call raised.
- `input_image` is not mutated: aberrations operate on `.copy()`, and the filters return new arrays.

## 6. Branches and edge cases
**Image distance (f = 100):**

| `object_distance` | `image_distance` | `magnification` |
|---|---|---|
| 200 | 200.0 | -1.0 |
| 50 | 100.0 (abs of -100) | -2.0 |
| 100 | inf | -inf |

- **Given distance:** `image_distance=50`, `object_distance=100` gives `image_distance == 50` and magnification -0.5. `refractive_index_at` is not called for a grayscale image.

**Aberrations** (skipped entirely without scipy). Values are read from `get_aberrations(wl)`.
- Each sigma is `abs(value)/pitch` in pixels. None or non-numeric values are skipped.
- `spherical_aberration` or `spherical`: blurs both spatial axes.
- `coma`: blurs both spatial axes.
- `astigmatism` gives a half-width h:
  - axis 0 (rows) is blurred with `h*√2`;
  - axis 1 (columns) is blurred with `h/√2`.
- On RGB images the colour axis is never blurred (sigma `(s, s, 0)`).
- `{}` means no aberration blur.

**Diffraction:**
- `airy = 1.22*wavelength*1e-6*f/D` (mm).
- `sigma = max(0.1, airy*0.844/2.355/pitch)`.
- Example: f=100, D=25, λ=550, pitch=0.01 gives about 0.096 px, so sigma 0.1.
- That leaves an effectively identity kernel (kernel radius 0), but do not rely on exact equality for non-constant images.

**Chromatic step** (only when `ndim == 3` and `shape[2] >= 3`):
- `f_base = get_system_focal_length()` at the current index.
- For channels R, G, B at wavelengths 650, 550, 450: `scale = f_base / f_at(wl)`.
- If `|scale - 1| > 1e-4`, the channel is zoomed with order 1, then centre-cropped or edge-padded back to the original size.
- If the index does not vary with wavelength, nothing changes.
- `(H, W, 1)` images skip this step.

**Vignetting:**
- If `image_distance` is inf, the fallback is `abs(get_system_focal_length())`. `effective_focal_length(WAVELENGTH_GREEN)` is tried first if present.
- If L ≤ 0 or `pitch <= 0`, the image is unchanged.
- Otherwise the factor is `(L/sqrt(L² + r²))⁴`, where:
  - `r = hypot(x - w/2, y - h/2) * pitch`;
  - the centre is the **float** point `(w/2, h/2)`, so pixel `(h/2, w/2)` has factor 1 when h and w are even.
  - RGB images are broadcast over channels.

## 7. Rules from comments and docstrings
- Asking a question must never leave the system changed (index restore).
- The wavelength must be forwarded to the aberration lookup, not dropped.
- Blur sigma is in pixels: millimetres divided by `pixel_pitch_mm`.
- Astigmatism blurs both meridians by different amounts.
- RGB blur must not mix channels.
- Vignetting never forces the corners to zero.

## 8. Testing hints
Use a fake system for determinism, and do not define `effective_focal_length` on it:

```python
class FakeLens:
    def __init__(s, n=1.5, d=25.0):
        s.refractive_index = n; s.diameter = d; s.calls = []
    def refractive_index_at(s, wl):
        s.calls.append(wl); return s.refractive_index
class FakeSys:
    def __init__(s, f=100.0, ab=None):
        s.lens = FakeLens(); s.elements = [type("E", (), {"lens": s.lens})()]
        s.f = f; s.ab = ab or {}
    def get_system_focal_length(s): return s.f
    def get_aberrations(s, wl): return s.ab
```

**Exact checks:**
- **Identity case:** `np.ones((64, 64))`, `object_distance=200`, `pixel_pitch_mm=0`. Expected:
  - `output_image` equals the input;
  - `psnr == inf`, `global_ssim == 1.0`;
  - `nyquist_spectral_ratio == 0.0`, `sharpness == 0.0`;
  - `image_distance == 200.0` (use `pytest.approx` for any computed distance), `magnification == -1.0`.
- **Vignetting:** `np.ones((4, 4))`, `object_distance=200` (L=200), `pixel_pitch_mm=10`.
  - Every filter preserves a constant image, so the output equals the vignette map.
  - `out[2, 2] == 1.0`.
  - `out[0, 0] ≈ (40000/40800)**2 ≈ 0.961169`.
  - Values decrease toward the corners.
- **Wavelength forwarding:** `wavelength=486.1` with a grayscale image gives `lens.calls == [486.1]`. An RGB `(8, 8, 3)` image gives `[486.1, 650, 550, 450]`.

**Pitfalls:**
- Use `pytest.approx` or `np.allclose` for blurred images.
- Each requirement below needs a separately built fake: a target without `get_aberrations` pulls in the unseen `AberrationsCalculator`; `f=None` triggers the ValueError; `elements=[]` triggers the aperture error (pass `image_distance`).
- These results assume scipy is installed; without it the blurs are skipped.

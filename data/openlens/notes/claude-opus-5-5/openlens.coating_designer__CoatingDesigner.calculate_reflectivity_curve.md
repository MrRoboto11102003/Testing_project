# Notes: `CoatingDesigner.calculate_reflectivity_curve(layers, wavelength_range, num_points=100)`

## 1. Purpose
- Samples the reflectivity of a coating stack at evenly spaced wavelengths.
- Returns a list of `(wavelength_nm, R)` tuples.
- Each `R` is exactly `self.calculate_reflectivity(layers, wl)` at normal incidence (angle 0).

## 2. Inputs
- `self`: a `CoatingDesigner(substrate_index=1.5168)`. The default substrate is BK7. `self.air_index` is always `1.0`.
- `layers`: a list of `CoatingLayer(material, refractive_index, thickness_nm)` in substrate-to-air order.
  - `[]` is allowed and means an uncoated surface (bare Fresnel reflection).
  - It is only read, never validated.
- `wavelength_range`: unpacked as `min_wl, max_wl = wavelength_range`, so it must have exactly 2 items.
  - It is not validated or sorted.
  - `min > max` is allowed and gives descending wavelengths.
  - `min == max` is allowed and repeats the same wavelength.
- `num_points`: an int, default `100`.
  - It is checked for `< 1` first.
  - It is used in `range()`, so a non-int such as `2.5` gives a `TypeError` from `range`.
- There is no angle parameter. The angle is always the `calculate_reflectivity` default, `0`.

## 3. Raises (in this order)
1. `ValueError("num_points must be at least 1")` when `num_points < 1`.
   - Example: `d.calculate_reflectivity_curve([], (400, 700), 0)`; `-3` also triggers it.
   - This check runs before the range is unpacked, so `curve([], (1, 2, 3), 0)` still gives this message.
2. Tuple-unpacking `ValueError` when `wavelength_range` does not have 2 items.
   - Examples: `(400,)` and `(400, 500, 600)`. This is Python's own message ("not enough / too many values to unpack").
3. From `calculate_reflectivity`, for the first sample with `wl <= 0`: `ValueError(f"Wavelength must be positive, got {wavelength_nm}")`.
   - `d.calculate_reflectivity_curve([], (0, 700), 3)` gives `"Wavelength must be positive, got 0.0"`. The value is a float because it was computed.
   - `d.calculate_reflectivity_curve([], (0, 700), 1)` gives `"...got 0"`. The original int is passed through unchanged.
   - Use `pytest.raises(ValueError, match="Wavelength must be positive")`.
- The angle check in `calculate_reflectivity` can never fire, because the angle is 0.

## 4. Returns and guarantees
- A new `list` of length exactly `num_points`.
- Each element is a `tuple` `(wl, R)`, not a list.
- Wavelength formula:
  - `num_points == 1`: `wavelengths = [min_wl]`. This is the original object, so an int stays an int.
  - `num_points > 1`: `wl_i = min_wl + (max_wl - min_wl) * i / (num_points - 1)` for `i = 0..n-1`. The multiply happens before the divide, and every result is a float.
  - With integer inputs the values are exact. `(400, 700), 4` gives `[400.0, 500.0, 600.0, 700.0]`. `(400, 700), 2` gives `[400.0, 700.0]`.
  - The first point equals `min_wl`, and the last equals `min_wl + (max_wl - min_wl)`, which is exact for ints.
- `R` is a Python `float` clamped to `[0, 1]`. It equals `d.calculate_reflectivity(layers, wl)` bit-for-bit, so `==` is safe for that comparison.
- Order follows `i` ascending, i.e. from `min_wl` toward `max_wl`.

## 5. Side effects
- None. `layers` and the designer are not mutated, and nothing is cached.
- Calling the method twice gives equal lists.

## 6. Branches and edge cases
- **Single point:** `num_points == 1` gives `[(min_wl, R(min_wl))]`, and `max_wl` is ignored. `curve([], (500, 900), 1) == [(500, R)]`.
- **Uncoated (`layers == []`):** R does not depend on wavelength.
  - It equals `((n-1)/(n+1))**2`.
  - BK7 1.5168 gives ≈ `0.042164` (approx 4.216%). `n = 1.5` gives `0.04`; use approx.
  - So all R values in the curve are identical.
- **Quarter-wave MgF2 on BK7 at design λ = 550:**
  - The layer is `CoatingLayer("MgF2", 1.38, 550/(4*1.38) ≈ 99.6377)`; `design_single_layer_ar(550)` returns exactly this.
  - At wl = 550, R = `((1.5168-1.38**2)/(1.5168+1.38**2))**2 ≈ 0.012835`.
  - At 400 and 700, R is larger than at 550 but still below the bare value of 0.042164.
- **Reversed range:** `(700, 400), 4` gives wavelengths `[700.0, 600.0, 500.0, 400.0]`.
- **Equal bounds:** `(550, 550), 3` gives three entries at `550.0` with equal R.

## 7. Rules from comments and docstrings
- The single-sample case deliberately avoids dividing by `num_points - 1` and sits at the start of the range.
- `layers` are in substrate-to-air order. `calculate_reflectivity` reverses them internally because light enters from air.
- Reflectivity is the unpolarized average of s and p, with lossless films.

## 8. Testing hints
- `from openlens.coating_designer import CoatingDesigner, CoatingLayer`, then `d = CoatingDesigner()`.
- Useful oracles:
  - `len(curve) == num_points`.
  - `[w for w, _ in curve]` equals the expected list.
  - `all(isinstance(p, tuple) for p in curve)`.
  - `r == d.calculate_reflectivity(layers, w)` for every point.
- Check the default length: `len(d.calculate_reflectivity_curve([], (400, 700))) == 100`.
  - The second point is `400 + 300*1/99`; compare it with `pytest.approx`.
- Use `pytest.approx` for physics values such as 0.042164 and 0.012835. Use exact `==` only for integer-friendly wavelength grids or for comparisons against `calculate_reflectivity`.
- Pitfalls:
  - Do not expect the range to be sorted or validated.
  - Do not expect an error for `min > max`.
  - Do not pass an angle; there is no such argument.
  - The `num_points` check comes before any wavelength check.

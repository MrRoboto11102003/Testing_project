# Notes: `MaterialDatabase.get_transmission(material_name, wavelength_nm, thickness_mm=10.0)`

## 1. Purpose
- Returns the internal transmission (0–1) of a glass at one wavelength and thickness.
- It linearly interpolates the material's `transmission_data`, then scales by Beer–Lambert: `T = t_base ** (thickness_mm / mat.transmission_thickness)`.

## 2. Inputs
- `material_name: str` is resolved via `self.get_material(name)`:
  - First it tries the exact `name.upper()` key.
  - Then it tries the alphanumeric-only uppercase form, so `"bk7"`, `"Fused Silica"`, `"fused-silica"` and `"FUSEDSILICA"` all resolve.
  - Normalised alias: the first catalog key wins. `"nbk7"` resolves to `"N-BK7"`, not BK7.
- `wavelength_nm: float` is not validated.
- `thickness_mm: float = 10.0` is not validated. Zero and negative values are accepted.
- Results are not cached, unlike `get_refractive_index`.

## 3. Raises
There are no explicit raises. Incidental errors:
- `material_name` not a str: `AttributeError` from `.upper()`. Example: `db.get_transmission(None, 500)`.
- A material with data and `transmission_thickness == 0`: `ZeroDivisionError`. This needs a custom material.
- Non-numeric `wavelength_nm` with data present: `TypeError` from comparison or subtraction.

## 4. Returns
The result is a float. The code path, in order:
1. `mat` is None (unknown name), or `mat.transmission_data` is empty: return `0.99 ** (thickness_mm / 10.0)`. In this case `transmission_thickness` is ignored.
2. Compute `t_base = _interpolate_transmission(data, wl)`:
   - `data` is sorted with `sorted(data)`. This makes a new list; the stored list is not mutated.
   - `wl <= data[0][0]`: return the first T (clamped).
   - `wl >= data[-1][0]`: return the last T (clamped).
   - Otherwise, use the first segment `w1 <= wl <= w2`: `t1 + (wl-w1)/(w2-w1)*(t2-t1)`.
   - If no segment matches, return `0.95`. In practice only `wl = float('nan')` reaches this.
3. `t_base <= 0`: return `0.0`. This happens before exponentiation, so it holds even for thickness 0.
4. Otherwise return `t_base ** (thickness_mm / mat.transmission_thickness)`.

## 5. Side effects
None. No file I/O, no cache changes, and no mutation of the material.

## 6. Built-in data and concrete values
The default `transmission_thickness` is 10.0.

**Materials with data:** only `"BK7"` and `"FUSEDSILICA"`.
- BK7: (380,.95) (400,.97) (500,.995) (600,.998) (700,.998) (800,.998) (1000,.997) (1500,.99) (2000,.98) (2500,.5)
- FUSEDSILICA: (200,.9) (250,.95) (300,.98) (400,.995) (600,.998) (1000,.998) (2000,.98) (3000,.5)

**Materials with no data** (they use the fallback): `N-BK7`, `SF11`, `F2`, `S-LAH79`, `E-FD60`.

Expected values (use `pytest.approx`):

| Call | Result |
|---|---|
| `("BK7", 500)` | 0.995 |
| `("BK7", 500, 20)` | 0.995**2 = 0.990025 |
| `("BK7", 500, 5)` | 0.995**0.5 |
| `("BK7", 450)` | 0.97 + 0.5·0.025 = 0.9825 |
| `("BK7", 2250)` | 0.98 + 0.5·(0.5−0.98) = 0.74 |
| `("BK7", 300)` (below range) | 0.95 |
| `("BK7", 3000)` (above range) | 0.5 |
| `("BK7", 380)` / `("BK7", 2500)` | 0.95 / 0.5 |
| `("BK7", 500, 0)` | 1.0 |
| `("bk7", 500)` | 0.995 (same as BK7) |
| `("FUSEDSILICA", 2500)` | 0.98 + 0.5·(0.5−0.98) = 0.74 |
| `("Fused Silica", 250)` | 0.95 |
| `("FUSEDSILICA", 100)` | 0.9 |
| `("FUSEDSILICA", 350)` | 0.98 + 0.5·0.015 = 0.9875 |
| `("N-BK7", 500)` | 0.99 (fallback, not BK7 data) |
| `("Unobtainium", 500)` | 0.99 |
| `("Unobtainium", 500, 20)` | 0.9801 |
| `("Unobtainium", 500, 0)` | 1.0 |
| `("Unobtainium", 500, -10)` | 1/0.99 |
| `("BK7", float('nan'))` | 0.95 (falls through the loop; then 0.95**1.0) |

Fallback calls are wavelength-independent: `("SF11", 300)` and `("SF11", 900)` both return 0.99.

## 7. Rules from docstring and comments
- "Normalizes from the catalog reference thickness using Beer-Lambert law: T2 = T1^(thickness2 / thickness1)".
- "Default fallback: 99% per 10mm".

## 8. Testing hints
- **Always build the database with a nonexistent temp path:** `db = MaterialDatabase(db_path=str(tmp_path / "m.json"))`.
  - `MaterialDatabase()` loads the user's real catalog, which could override BK7.
  - Avoid `get_material_database()` for the same reason.
- **Custom materials:** add them with `db.add_material(MaterialProperties(...), save=False)` so nothing is written to disk. Required fields are `name`, `catalog`, `nd`, `vd`.
  - `transmission_data=[(600, 0.8), (400, 0.6)]`, i.e. unsorted. At wl 500, thickness 10, the result is 0.7. This proves the data is sorted.
  - `transmission_thickness=5.0` with data `[(500, 0.81)]` at thickness 10 gives 0.81**2 = 0.6561. This proves the divisor is the material's thickness, not 10.
  - Data `[(400, 0.0), (600, 0.0)]` gives exactly `0.0`, even with `thickness_mm=0`.
  - Data `[(400, -0.2), (600, -0.2)]` gives `0.0`.
  - `transmission_data=[]` gives the fallback, 0.99 at thickness 10.
  - Afterwards, `mat.transmission_data` still has its original (unsorted) order.
- **Pitfalls:**
  - Use `pytest.approx` for interpolated and exponentiated values.
  - Exact equality is safe for the clamped endpoints (0.95, 0.5) at thickness 10, because the exponent is exactly 1.0.
  - Do not assume N-BK7 shares BK7's transmission.

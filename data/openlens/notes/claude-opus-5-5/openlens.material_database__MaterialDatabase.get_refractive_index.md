# Notes: `MaterialDatabase.get_refractive_index(material_name, wavelength_nm, temperature_c=20.0)`

## 1. Purpose
- Returns the refractive index of a catalog glass at a wavelength (nm) and temperature (°C).
- Uses a 3-term Sellmeier equation plus an optional Schott TIE-19 thermal correction.
- Results are cached per instance.

## 2. Inputs
- `material_name: str`
  - Looked up with `get_material`. The exact `name.upper()` key is tried first.
  - If that misses, the alphanumeric-only uppercase form is tried. So `"bk7"`, `"Fused Silica"` and `"fused-silica"` all resolve.
  - Unknown names are not an error (see §6).
- `wavelength_nm: float`
  - Converted to µm: `wl_um = wavelength_nm/1000`, then `lambda_sq = wl_um**2`.
  - It is not validated.
- `temperature_c: float = 20.0`
  - The reference temperature is exactly `20.0`.
  - The thermal correction runs only if `temperature_c != 20.0`.
- Cache key: the raw tuple `(material_name, wavelength_nm, temperature_c)`.
  - The name is **not** normalised, so `"bk7"` and `"BK7"` are separate cache entries with equal values.

## 3. Raises
There are no explicit raises. The guards replace ZeroDivision and sqrt-domain errors. Incidental errors:
- `get_refractive_index(123, 500)` → `AttributeError` ('int' has no `.upper`).
- `get_refractive_index("BK7", [500])` → `TypeError` (unhashable list used as cache key). This fails before any computation.
- `get_refractive_index("BK7", "500")` → `TypeError` (str / float).

## 4. Returns and guarantees
- Returns a `float`. Repeated calls with the same key return the cached value, without recomputation.
- Sellmeier step:
  - Start with `n_sq = 1.0`.
  - For each `(B, C)` in `(B1,C1),(B2,C2),(B3,C3)`:
    - Skip the term if `B == 0.0` or `C <= 0.0`.
    - Skip the term (with a logged warning) if `abs(lambda_sq - C) < 1e-12`.
    - Otherwise add `B*lambda_sq/(lambda_sq - C)`. Negative terms are kept.
- If `n_sq <= 0.0`, return **exactly `1.0`**. No thermal correction is applied in that case.
- Otherwise `n = sqrt(n_sq)`.
- Thermal correction, applied only if `temperature_c != 20.0` and any of `D0, D1, D2, E0, E1` is nonzero:
  - `dT = T - 20`
  - `bracket = D0*dT + D1*dT**2 + D2*dT**3`
  - If `(E0 or E1)` and `lambda_tk_um > 0`, compute `den = lambda_sq - lambda_tk_um**2`.
    - If `abs(den) > 1e-12`, add `(E0*dT + E1*dT**2)/den` to `bracket`.
    - Otherwise skip the E-term and log a warning.
  - `n += (n**2 - 1)/(2*n) * bracket`. This uses the Sellmeier `n`.

## 5. Side effects
- Stores the value in `self._index_cache`.
- On every cache miss it sets `self._alias_keys = None`. This is harmless and not externally visible.
- If the cache already holds `>= 1024` entries (`INDEX_CACHE_MAX`), the cache is cleared **before** the new value is inserted. Its length then becomes 1.
- No file I/O and no mutation of `materials`.

## 6. Branches and edge cases (all reachable)
- **Unknown name:** `get_refractive_index("NOPE", 500)` returns exactly `1.5168`, which is the BK7 fallback. It is also cached.
- **Built-ins:** `BK7`, `N-BK7`, `SF11`, `F2`, `FUSEDSILICA`, `S-LAH79`, `E-FD60`.
  - At 587.6 nm, 20 °C they give approximately their `nd`: 1.5168, 1.5168, 1.78472, 1.62004, 1.45846, 2.00330, 1.52249.
  - Compare with `abs=1e-3`, not exactly.
- **BK7 vs N-BK7:** they have identical coefficients, so they give identical results at any wavelength and temperature.
- **Dispersion:** for BK7, `n(486.1) > n(587.6) > n(656.3)`.
- **Temperature, BK7:** `n(30 °C) > n(20 °C)`. The increase is about 1.4e-5, so assert the direction or use a tolerance rather than an exact value.
- **No thermal data:** `S-LAH79` and `E-FD60` have no D/E coefficients, so `n(…, 80.0) == n(…, 20.0)` exactly.
- **Custom glass** (see §8), with `B1=1.0, C1=0.01` and all other coefficients 0, at `wavelength_nm=1000.0`:
  - At 20 °C the result is `math.sqrt(1 + 1/0.99)`, about 1.41778.
  - `C1=1.0` hits the resonance at 1000 nm: the term is skipped and the result is `1.0`.
  - `C1=-1.0` or `B1=0.0`: the term is skipped and the result is `1.0`.
  - `B1=-2.0, C1=0.01`: `n_sq < 0`, so the result is `1.0`. This holds even with D0 set and T=50.
  - Adding `D0=1e-5` at `T=30.0`: `n = n0 + (n0**2-1)/(2*n0)*1e-4`.
  - Adding `E0=1e-6, lambda_tk_um=0.5` at `T=30.0`: the bracket gains `1e-5/0.75`.
  - Same E0 but `lambda_tk_um=0.0`: the E-term is ignored.
  - `lambda_tk_um=1.0` at 1000 nm: the E-term is skipped because `den == 0`.
- **Zero wavelength:** `wavelength_nm=0` gives `lambda_sq=0`, every term is 0, and the result is `1.0` with no error.

## 7. Rules from comments and docstrings
- A bad catalog entry must degrade gracefully, never raise.
- The negative third Sellmeier term must be kept. Dropping it would put BK7 at about 1.5223.
- Callers that need to detect an unknown name should use `has_material` first.
- After mutating `materials` directly, call `clear_cache()`. `add_material` clears the cache itself.

## 8. Testing hints
- **Always isolate the database:** `db = MaterialDatabase(db_path=str(tmp_path/"m.json"))`.
  - With the default path, the user's real catalog is loaded and could override the built-ins.
- **Custom glass:**
  - `MaterialProperties(name="TESTG", catalog="Custom", nd=1.4, vd=50, B1=1.0, C1=0.01)`
  - Register it with `db.add_material(mat, save=False)` to avoid writing a file.
  - Remember that default D/E are 0, so there is no thermal effect unless you set them.
- **Staleness:**
  - Call once, then set `db.materials["TESTG"].B1 = 2.0`. The same call returns the **old** value.
  - After `db.clear_cache()` it returns the new value.
  - `add_material` of a replacement also invalidates the cache.
- **Cache size:** fill the cache with 1024 distinct wavelengths, then make one more new call. `len(db._index_cache) == 1` afterwards.
- **Assertions:** use `pytest.approx` with `rel=1e-12` for the custom-glass formulas, and loose tolerances for the catalog nd comparisons.
- **Logging:** warnings go through the logger `openlens.material_database` (observable with `caplog`). Message text includes, for example, `"Sellmeier resonance for"` and `"using n = 1.0"`.

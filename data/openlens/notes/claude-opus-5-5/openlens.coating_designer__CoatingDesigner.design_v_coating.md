# Notes: `CoatingDesigner.design_v_coating(self, wavelength_nm)`

## 1. Purpose
- Builds a 3-layer broadband visible "V-coat" in substrate-to-air order: Al2O3 / SiO2 / MgF2.
- The inner layer is always quarter-wave.
- The middle and outer thicknesses come from a greedy, deterministic grid search. The search minimises the **sum** of reflectance at 7 fixed probe wavelengths.

## 2. Inputs
- `self.substrate_index`
  - Set in `__init__`, default `1.5168`.
  - Used only through `self.calculate_reflectivity`, so it changes only the searched middle/outer thicknesses.
  - It never changes the materials or the inner thickness.
- `self.air_index` is always `1.0`.
- `wavelength_nm` (number, no validation) is the design wavelength. It sets:
  - the inner thickness `wavelength_nm/(4*1.63)`;
  - the start point `(wavelength_nm/(4*1.46), wavelength_nm/(4*1.38))`;
  - the initial step `step = wavelength_nm/(4*1.38)/4`. This is always based on MgF2's index.
- `wavelength_nm` is **not** passed to `calculate_reflectivity`. Reflectance is only evaluated at the module constants:
  - `V_COATING_PROBE_WAVELENGTHS_NM = (400, 450, 500, 550, 600, 650, 700)`
  - `V_COATING_SEARCH_STEPS = 12` (maximum number of rounds)
  - `V_COATING_TOLERANCE = 1e-9`
- Indices come from `COATING_MATERIALS`: Al2O3 = 1.63, SiO2 = 1.46, MgF2 = 1.38.

## 3. Raises
- No explicit raises. Because the probes are fixed and positive, `calculate_reflectivity` never raises its "Wavelength must be positive" error here.
- `wavelength_nm=0` or negative values do **not** raise.
  - `0`: step is 0, every neighbour equals (0, 0) and is skipped, so the result is three layers with `thickness_nm == 0.0`.
- A non-numeric value (e.g. `"550"`) raises `TypeError` from arithmetic such as `"550"/(4*1.63)`.

## 4. Returns and guarantees
- Returns a new `list` of 3 new `CoatingLayer` objects, always in this order:
  - `[0]`: `CoatingLayer("Al2O3", 1.63, wavelength_nm/(4*1.63))`. For 550 nm the thickness is 84.355828…
  - `[1]`: `CoatingLayer("SiO2", 1.46, t_middle)`
  - `[2]`: `CoatingLayer("MgF2", 1.38, t_outer)`
- Both searched thicknesses satisfy `t_middle >= 0` and `t_outer >= 0` (each candidate is clamped with `max(0.0, …)`).
- The result is deterministic: repeated calls, or a new instance with the same substrate, give equal lists (dataclass `==`).

**Search algorithm (exact):**
- `best_cost` starts as the **uncoated** cost: `sum(calculate_reflectivity([], p) for p in probes)`.
- `best_t` starts at the quarter-wave pair. The start point itself is never scored.
- Each round loops over `d_middle in (-step, 0, step)` (outer loop) and `d_outer in (-step, 0, step)` (inner loop), skipping `(0, 0)`. That is 8 candidates.
- Each candidate is `(max(0, best_t[0]+dm), max(0, best_t[1]+do))`.
- It is accepted only if `cost < best_cost - 1e-9`. Acceptance is greedy: `best_t` updates mid-round, and later candidates offset from the new best.
- If a round has no acceptance, the search breaks. Otherwise `step /= 2`.

**Consequence:** if nothing ever beats the uncoated cost, the result is the unscored quarter-wave start. In that case the docstring's claim "beats the bare substrate" does not hold. If anything was accepted, `band_cost(result) < uncoated_cost - 1e-9`.

## 5. Side effects
- Does not change `self.substrate_index`, `self.air_index` or `COATING_MATERIALS`.
- No I/O.
- Calls `self.calculate_reflectivity(layers, probe)` with 2 positional arguments. An instance-level monkeypatch therefore takes effect.

## 6. Branches and edge cases (use monkeypatching for exact values)
Set `d = CoatingDesigner()`, `wl = 550`, `qm = 550/(4*1.46) = 94.178082…`, `qo = 550/(4*1.38) = 99.637681…`, `step = qo/4 = 24.909420…`.

- **No improvement ever** (`d.calculate_reflectivity = lambda L, w, a=0: 0.5`)
  - The candidate cost 3.5 is not less than 3.5 − 1e-9, so the search breaks after round 1.
  - Result thicknesses: `(qm, qo)`.
  - 63 calls in total (7 uncoated + 8×7).
- **Single improvement** (`lambda L, w, a=0: 0.0 if L else 1.0`)
  - The first candidate `(-step, -step)` is accepted. In round 2, `0 < 0 - 1e-9` is false, so the search breaks.
  - Result: `t_middle = qm - step ≈ 69.268662`, `t_outer = qo - step ≈ 74.728261`.
  - 119 calls in total.
- **Always improving** (`lambda L, w, a=0: L[1].thickness_nm + L[2].thickness_nm if L else 1e9`)
  - All 12 rounds run, and each round accepts `(-step, -step)` first.
  - Result: `t = start - step*(2 - 2**-11)`, i.e. `t_outer ≈ qo/2 + tiny`. Use `pytest.approx`.
  - 7 + 12×56 = 679 calls.
- **`wavelength_nm=0`**: all thicknesses are 0.0 and there is no exception.

## 7. Rules from the docstring and comments
- The inner layer is held quarter-wave.
- The two outer thicknesses are solved numerically, minimising mean reflection over the 400–700 nm band.
- The search "starts from" the uncoated cost, so an accepted stack always beats bare glass.
- The old ×1.2/×0.9 Ta2O5 design is gone. Ta2O5 is never used.

## 8. Testing hints
- Check `[l.material for l in res] == ["Al2O3", "SiO2", "MgF2"]` and the indices `[1.63, 1.46, 1.38]`.
- Check the inner thickness with `pytest.approx(wl/(4*1.63))`. Test several wavelengths and substrates; the inner thickness is independent of the substrate.
- `coating_label(res) == "3L AR (Al2O3+SiO2+MgF2)"`.
- `design_preset("V-coating", 1.5168, 550) == CoatingDesigner(1.5168).design_v_coating(550)`.
- For the real (unpatched) result, do not hard-code `t_middle` or `t_outer`; they are not given here. Instead assert either:
  - `cost(res) < cost([]) - 1e-9`, **or**
  - the thicknesses equal `(qm, qo)` exactly.

  Compute `cost` with `sum(d.calculate_reflectivity(L, p) for p in (400, 450, …, 700))`.
- Use the monkeypatch scenarios in section 6 to pin the following, with `pytest.approx(rel=1e-9)`:
  - the step size `qo/4`;
  - the tolerance (e.g. a patch whose improvement is only 1e-10 must leave the start unchanged);
  - the greedy order;
  - the 12-round cap;
  - the call counts.

  Patch on the instance, e.g. `d.calculate_reflectivity = fn`, with an `fn` that takes `(layers, wl, angle=0)`.
- Tolerance check in detail: with a patch returning `1.0` for `[]` and `1.0 - 1e-10/7` otherwise, the result is the start point.
- Do not assert that the thicknesses differ from quarter-wave, and do not compare against reflectance at `wavelength_nm` itself; it is not what the search optimises.

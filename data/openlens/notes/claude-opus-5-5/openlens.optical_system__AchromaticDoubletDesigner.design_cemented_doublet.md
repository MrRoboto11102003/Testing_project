# Notes: `AchromaticDoubletDesigner.design_cemented_doublet`

## 1. Purpose
- A `@staticmethod` that designs a two-element cemented achromat.
- It uses thin-lens achromat power splitting from catalog Abbe numbers.
- It returns a new `OpticalSystem` with a crown lens followed by a flint lens and no gap between them.

## 2. Inputs
- **`focal_length`** (float): target system focal length in mm. It is not validated.
- **`diameter`** (float): used for both lenses' `diameter`. Crown thickness is `diameter*0.15`; flint thickness is `diameter*0.08`. It is not validated here; `Lens.__init__` may validate it, but `openlens/lens.py` is not visible.
- **`crown_material`** (str, default `'BK7'`) and **`flint_material`** (str, default `'SF11'`):
  - Each is looked up with `get_material_database().get_material(name)`, which returns `MaterialProperties` or `None`.
  - Whether the lookup ignores case is not visible, so use exact catalog names.
- Values used from the materials: `v1 = crown.vd`, `v2 = flint.vd`, `n1 = crown.nd`, `n2 = flint.nd`.

## 3. Raises
- **`ValueError(f"Materials not found: {crown_material}, {flint_material}")`**
  - Raised if either lookup returns a falsy value, or if the material database failed to import.
  - The message always lists **both** names, even when only one is missing.
  - Example: `design_cemented_doublet(100, 50, crown_material="NOPE")` gives the message `"Materials not found: NOPE, SF11"`.
- **`ZeroDivisionError`** when `focal_length == 0`:
  - `f2 = 0`, so `term = 1.0/(f2*(n2-1))` fails before any `Lens` is built.
  - Same-material case with `focal_length=0` also gives `f2 = 0`, so the same error.
- Any exception raised by `Lens(...)` validation, for example a non-positive diameter or thickness, propagates unchanged. Those rules live in `lens.py`, which is not shown.

## 4. Returns and guarantees
The result is a **new** `OpticalSystem` on every call, with a fresh uuid `id`.

**Math, in code order:**
```
if v1 == v2:  f1 = f2 = focal_length*2
else:         f1 = focal_length*(v1-v2)/v1 ;  f2 = focal_length*(v2-v1)/v2
R1_crown = 2*f1*(n1-1);  R2_crown = -R1_crown;  R1_flint = R2_crown
term   = 1.0/(f2*(n2-1));  inv_R2 = 1.0/R1_flint - term
R2_flint = inf if abs(inv_R2) < 1e-10 else 1.0/inv_R2
```

**Lenses** (`system.elements[0].lens` and `system.elements[1].lens`):

| Field | Crown (element 0) | Flint (element 1) |
|---|---|---|
| `name` | `f"{crown_material} Element"`, e.g. `"BK7 Element"` | `f"{flint_material} Element"`, e.g. `"SF11 Element"` |
| `radius_of_curvature_1` | `R1_crown` | `R1_flint` (equals `-R1_crown`) |
| `radius_of_curvature_2` | `-R1_crown` | `R2_flint` |
| `thickness` | `diameter*0.15` | `diameter*0.08` |
| `diameter` | `diameter` | `diameter` |
| `material` | `crown_material` | `flint_material` |
| `refractive_index` | `n1` (catalog `nd`) | `n2` (catalog `nd`) |

- Both lenses are built with `wavelength=WAVELENGTH_D_LINE`.

**System fields:**
- `name == "Achromatic Doublet"`
- `len(elements) == 2` and `len(air_gaps) == 1`
- `air_gaps[0].thickness == 0.0` (exactly) and `air_gaps[0].position == diameter*0.15`
- `elements[0].position == 0.0` and `elements[1].position == diameter*0.15`
- `get_total_length() == diameter*0.23` (use approx)
- Decenter and tilt are all 0.
- `aperture_stop_gap is None`, `casing is None`, `image_plane_x_mm is None`, `metadata == {}`

## 5. Side effects
- No inputs are mutated.
- It reads, and may initialise, the material database singleton.
- No files are written by this function itself.

## 6. Branches and edge cases
- **Normal case:** BK7 (high vd) with SF11 (low vd) and `f > 0`.
  - `f1 > 0` and `f2 < 0`.
  - `R1_crown > 0`, `R2_crown < 0`, `R1_flint < 0`.
- **Equal Abbe numbers:** for example `crown_material="BK7", flint_material="BK7"`.
  - `f1 = f2 = 2f` and `n1 == n2 = n`.
  - `R1_crown = 4f(n-1)` and `R1_flint = -4f(n-1)`.
  - `inv_R2 = -3/(4f(n-1))`, so `R2_flint = -4f(n-1)/3`.
  - Both lens names are `"BK7 Element"`.
- **Flat flint back (`R2_flint = inf`):** only reached when `|1/R1_flint - term| < 1e-10`. This is hard to hit with real catalog values.
- **Negative `focal_length`:** all radii flip sign; no error is raised.

## 7. Rules from comments
- The lens index **must be the catalog `nd`** used to solve the radii:
  - For SF11 that is ≈ 1.78472.
  - It must not be the `DEFAULT_MATERIAL_INDICES` fallback (SF11 → 1.78).
  - It must not be the Sellmeier value from `Lens.for_material` (≈ 1.785107).
- Crown thickness is about 15% of the diameter; the flint is thinner at 8%.
- The second lens is cemented, so the air gap is zero.

## 8. Testing hints
- Do not hard-code catalog numbers. Compute expected values from the database:
  ```python
  from openlens.material_database import get_material_database
  db = get_material_database()
  c, fl = db.get_material("BK7"), db.get_material("SF11")
  ```
  Then recompute `f1`, `f2` and the radii with the formulas above and compare using `pytest.approx(rel=1e-9)`.
- Good regression checks:
  - `elements[0].lens.refractive_index == approx(c.nd)` and `elements[1].lens.refractive_index == approx(fl.nd)`.
  - `radius_of_curvature_2 == -radius_of_curvature_1` for the crown.
  - The flint `R1` equals the crown `R2`.
  - Thicknesses are `0.15*D` and `0.08*D`.
  - The gap is `0.0`.
  - Lens and system names.
  - The exact error message.
- `create_doublet(f, D)` is equivalent to `design_cemented_doublet(f, D)` with the defaults (BK7/SF11), so its radii must be the same.
- Pitfalls:
  - **Do NOT** assert `system.get_system_focal_length() == focal_length`. The design is thin-lens and the model is thick-lens, so the values differ. At most, assert that it is positive for `f > 0`.
  - **Do NOT** assert `is_achromatic()` is True.
  - The `Lens` radius setters may normalise values (for example 0 → inf, per a comment). Use realistic, non-zero values such as `focal_length=100, diameter=50`.

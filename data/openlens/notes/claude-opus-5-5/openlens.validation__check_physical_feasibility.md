# Notes: `openlens.validation.check_physical_feasibility`

## 1. Purpose
- Checks whether a lens spec is physically buildable.
- Returns `(is_feasible, message)`.
- It never raises a `ValidationError`. It probes the geometry through a temporary `Lens`, then adds soft heuristic warnings.

## 2. Inputs
- `radius1`, `radius2`: front and back radius in mm. Negative is concave. `float('inf')` means flat.
  - Not validated here.
  - `abs()` is called on both unconditionally, so `None` or a string raises `TypeError` (see Raises).
- `thickness`, `diameter`: centre thickness and diameter in mm. Used raw in the heuristics. `h = diameter / 2`.
- `is_parabolic_1/2`, `parabolic_sag_1/2`, `clear_aperture_1/2`, `conic_constant_1/2`:
  - These are only forwarded to the probe `Lens(...)`.
  - They affect the result only through `edge`.
- The probe also gets `Lens` defaults for everything else (material `'BK7'` and so on).
- Edge maths is in `openlens/lens.py` (not shown). `MIN_EDGE_THICKNESS` and `EPSILON` come from `openlens.constants` (values not shown).

## 3. Raises
- The probe is wrapped in `try/except Exception`. Any error from `Lens(...)` or `calculate_edge_thickness()` is swallowed and gives `edge = None`.
- `TypeError` from `min(abs(radius1), abs(radius2))` when a radius is not numeric:
  - Example: `check_physical_feasibility(None, 100, 5, 25)` raises `TypeError` (`abs(None)`).
  - If `edge` is None, a string radius may already raise `TypeError` inside `math.isfinite`.
- No `ValidationError` is ever raised.

## 4. Returns and guarantees
- Always returns a 2-tuple:
  - `(True, None)` only when there is no hard error AND no warnings.
  - Otherwise `(False, str)`. **Soft warnings alone also give `False`.**
- Message assembly:
  - hard error and warnings: `hard + " " + " ".join(warnings)`
  - hard error only: `hard`
  - warnings only: `" ".join(warnings)`
- Hard error, decided from `edge = probe.calculate_edge_thickness()`. The comment says edge = t − sag1 + sag2.
  - **`edge is None`**: loop over `(radius1, radius2)` in order. The first `r` that is not None, finite, has `|r| >= EPSILON` and `|h| > |r|` gives:
    `f"Aperture overhangs a surface (|R| = {abs(r):.1f}mm < D/2 = {h:.1f}mm) - no real surface exists at the rim. Reduce the diameter or flatten the radii."`
  - If no radius qualifies: `"Lens geometry is undefined for these parameters. Reduce the diameter or flatten the radii."`
  - **`edge <= 0`**: `f"Surfaces intersect within the clear aperture (edge thickness {edge:.2f}mm <= 0). Increase center thickness, reduce diameter, or flatten the radii."`
- Warnings, appended in this fixed order:
  1. Edge warning, only if there is no hard error and `edge < MIN_EDGE_THICKNESS`:
     `f"Edge thickness ({edge:.2f}mm) is below {MIN_EDGE_THICKNESS}mm. May be fragile."`
  2. If `thickness > 0.5*min_radius`:
     `f"Thickness ({thickness:.1f}mm) is very large compared to minimum radius ({min_radius:.1f}mm). Lens may be impractical."`
  3. If `diameter > 0.8*min_radius`:
     `f"Diameter ({diameter:.1f}mm) is large compared to minimum radius ({min_radius:.1f}mm). Edge effects may be significant."`
  4. If `thickness < 0.02*diameter`:
     `f"Lens is very thin ({thickness:.1f}mm) relative to diameter ({diameter:.1f}mm). May be fragile."`
- `min_radius = min(|R1|, |R2|)`. With both radii inf it is inf, so warnings 2 and 3 can never fire.
- All comparisons are strict (`>`, `<`). Exactly equal to a threshold gives no warning.

## 5. Side effects
- No inputs are mutated and no global state is changed by this module.
- A throw-away `Lens` is constructed. Whether `Lens` touches the material database is in `lens.py` (not visible).

## 6. Branches and edge cases
These examples are computed assuming the usual spherical sag (`R − sqrt(R² − h²)`) inside `Lens`. Values marked "≈" depend on lens.py.

- **Fine biconvex:** `(200, -200, 10, 25)` → `(True, None)`.
  - edge ≈ 9.2; min_radius 200, and no heuristic fires.
- **Flat/flat:** `(inf, inf, 5, 25)` → expected `(True, None)`, with edge ≈ 5.
- **Thickness warning only:** `(float('inf'), -20, 15, 10)`.
  - edge ≈ 14.4; 15 > 10; 10 > 16 is false.
  - Result: `(False, "Thickness (15.0mm) is very large compared to minimum radius (20.0mm). Lens may be impractical.")`
- **Overhang:** `(10, -10, 5, 30)`. h = 15 > 10, so edge is expected None. The diameter warning is also added (30 > 8). Result:
  `(False, "Aperture overhangs a surface (|R| = 10.0mm < D/2 = 15.0mm) - no real surface exists at the rim. Reduce the diameter or flatten the radii. Diameter (30.0mm) is large compared to minimum radius (10.0mm). Edge effects may be significant.")`
- **Overhang picks the first qualifying radius:** `(-8, 100, 5, 20)` → message starts `"Aperture overhangs a surface (|R| = 8.0mm < D/2 = 10.0mm)"`.
- **Intersecting surfaces:** `(50, -50, 1, 40)`, edge ≈ −7.35.
  - Message starts `"Surfaces intersect within the clear aperture (edge thickness "` and contains `"mm <= 0)"`.
  - No heuristics fire: 40 > 40 is false, and 1 < 0.8 is false.
- **Hard error suppresses the edge warning:** when edge <= 0, the "Edge thickness (…) is below" text is never present.
- **Thin-lens warning:** fires only when `thickness < 0.02*diameter`, e.g. t = 0.9, d = 50.
  - The edge-fragile warning may also precede it, depending on `MIN_EDGE_THICKNESS`. Use `in` checks here.

## 7. Rules from comments
- The centre thickness is vertex-to-vertex. The edge is `thickness - sag1 + sag2`; a plus-sign variant would wrongly reject concave lenses.
- `edge <= 0` means the surfaces intersect within the aperture.
- `edge None` means the aperture overhangs a surface (|R| < D/2).
- Clear apertures are judged at the polished radius (`Lens.get_manufacturing_radius`), not at D/2.
- The edge floor is a soft warning, not a hard error.

## 8. Testing hints
- Import with `from openlens.validation import check_physical_feasibility`. Pass positional floats.
- Use exact string equality only for heuristic-only or overhang messages whose numbers come from the inputs (`.1f` formatting, e.g. `30.0mm`).
- For edge-dependent text (`{edge:.2f}`), assert the prefix or substrings, not the full number.
- Pin the ordering and separators: one space between parts, hard error first, and warnings in order edge → thickness → diameter → thin.
- Pin strictness at boundaries:
  - `(inf, 100, 50, 10)` → thickness 50 > 50 is false. No thickness warning; expect `(True, None)` if the edge is OK.
  - `(inf, 100, 50.1, 10)` → the thickness warning appears.
- Pin that soft warnings make the first element `False`, not `True`.
- Pitfalls:
  - Do not assume `MIN_EDGE_THICKNESS` has any specific value. Keep edges large (> 5 mm) when expecting `(True, None)`.
  - Do not expect `ValidationError` for bad geometry; it returns a tuple.
  - `None` radii raise `TypeError`.

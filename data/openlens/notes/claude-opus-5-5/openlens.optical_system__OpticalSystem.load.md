# Notes: `OpticalSystem.load(filename)` (staticmethod, `openlens/optical_system.py`)

## 1. Purpose
- Reads an optical system from a JSON file and returns a new `OpticalSystem` built by `OpticalSystem.from_dict`.
- It **never raises** an `Exception` subclass. Any failure is logged and the function returns `None`.

## 2. Inputs
- `filename: str` is a path (a `pathlib.Path` from `tmp_path` also works with `open`).
- Processing steps, in order, all inside one `try`:
  1. `reject_oversized_file(filename)` checks a size cap. It lives in `openlens.validation`, which is not shown; its exact limit is unknown.
  2. `open(filename, "r")` then `json.load(f)`. `json.load` accepts the bare `NaN`, `Infinity` and `-Infinity` literals.
  3. `data = json_safe_radii(data)` comes from `openlens.lens`, not shown. Per the docstring, it converts infinite *radii* (the legacy flat-surface spelling) to `null`.
  4. `reject_non_finite_numbers(data, "optical system")` rejects any remaining NaN or Infinity anywhere. Its implementation is not shown.
  5. `return OpticalSystem.from_dict(data)`. No `lens_lookup` is passed, so every lens is deserialized fresh.

## 3. Raises
- **Nothing.** The code is `except Exception as e: logger.error("Error loading optical system: %s", e); return None`.
- The logger name is `"openlens.optical_system"` and the level is ERROR.
- Inputs that return `None`:
  - Missing file: `OpticalSystem.load(str(tmp_path/"nope.json"))` returns `None`.
  - Invalid JSON text `"not json"` returns `None`.
  - Top-level JSON list `[]` returns `None`, because `data.get` raises AttributeError.
  - Non-finite value outside a radius, e.g. `{"name":"x","aperture_stop_diameter": NaN}`, returns `None`.
  - Legacy flat element (no `node_tree`) whose dict lacks the `"lens"` key returns `None` (KeyError in `elem_data["lens"]`).
    - This also happens when `node_tree` is present, because the elements are parsed first.
  - An invalid `image_plane_x_mm` makes `validate_image_plane_x` (in `openlens.lens`, not shown) raise, and the result is `None`.

## 4. Returns and guarantees (what `from_dict` restores)
- Each call returns a **new** `OpticalSystem`.
- Values copied straight from the file, when present:
  - `id`
  - `created_at`
  - `modified_at`: it is not touched afterwards, since `add_lens` and the rebuild steps do not update it.
  - `name`: default `"Optical System"`.
  - `aperture_stop_gap` and `aperture_stop_diameter`: raw values, default `None`.
  - `thermal_state_temperature`: default `DEFAULT_TEMPERATURE`, a constant whose value is not shown.
- `image_plane_x_mm = validate_image_plane_x(data.get("image_plane_x_mm"))`. `None` stays `None`.
- `casing = Casing.from_dict(...)` only if the value is a dict, else `None`. It is **not** validated against the lenses.
- `metadata` is `data["metadata"]` with every key in this set removed:
  - `{"id","name","type","created_at","modified_at","aperture_stop_gap","aperture_stop_diameter","elements","air_gaps","metadata","node_tree","thermal_state_temperature","image_plane_x_mm","casing"}`
  - Example: saving with metadata `{"tol": 1, "name": "zzz"}` loads back as `{"tol": 1}`.
- Elements are first added via `add_lens(lens, air_gap_before=gaps[i-1]["thickness"])` with decenter and tilt values converted by `float(x or 0.0)`.
- If `node_tree` is present (always true for files written by `save`), the tree is rebuilt from it:
  - The element lenses then come from `node_tree[...]["element_model"]`.
  - Node positions and rotations come from the tree.
  - Modifiers are restored, and nesting is preserved.
- Without `node_tree` (legacy file):
  - Each saved `"position"` that is not `None` overrides the root child's x.
  - Then the flat lists are rebuilt.
  - With no positions at all, the layout runs from x=0 using the gaps.
- Air gaps are recomputed from positions: `next.x - (x + thickness)`, clamped to ≥0.

## 5. Side effects
- The function only reads the file. It does not modify or create files.
- On failure it logs one ERROR record.

## 6. Branches / edge cases
- `{}` in the file gives a system with name `"Optical System"`, `elements == []`, `air_gaps == []`, `metadata == {}` and `aperture_stop_gap is None`. A fresh `id` is generated.
- A file containing a bare `Infinity` radius (legacy plano) still loads, because `json_safe_radii` converts it before the finite check. This relies on unseen helpers, so assert only `is not None`.
- A legacy dict with elements at saved positions 15 and 30 (thickness 5) loads with element positions 15 and 30 and gap thickness 10 at gap position 20.
- An empty or `null` `node_tree` takes the legacy path.

## 7. Rules from docstrings/comments
- The file is size-capped, and NaN/Infinity anywhere is rejected because NaN would silently defeat sanity checks.
- Legacy `Infinity` radii are the one accepted exception.
- `save` writes `json_safe_radii(to_dict())` with `indent=2` and `allow_nan=False`, so `save` followed by `load` must round-trip.

## 8. Testing hints
- Round trip:
  ```python
  s = OpticalSystem(name="T", metadata={"tol": 1, "name": "zzz"})
  s.add_lens(Lens(name="A", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0, thickness=5.0, diameter=25.0))
  s.add_lens(Lens(name="B", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0, thickness=5.0, diameter=25.0), air_gap_before=10.0)
  s.set_aperture_stop(0, 8.0)
  p = tmp_path/"s.json"; assert s.save(str(p))
  r = OpticalSystem.load(str(p))
  ```
- Assertions that hold for this round trip:
  - `r is not s`.
  - `r.id == s.id`, `r.name == "T"`, and `r.created_at`/`r.modified_at` equal the originals.
  - `[e.position for e in r.elements] == approx([0.0, 15.0])`.
  - `r.air_gaps[0].thickness == approx(10.0)` and `r.air_gaps[0].position == approx(5.0)`.
  - `r.get_aperture_stop() == {"gap_index": 0, "position": approx(5.0), "diameter": 8.0}`.
  - `r.metadata == {"tol": 1}`.
  - `[e.lens.name for e in r.elements] == ["A", "B"]`.
  - `r.get_system_focal_length() == approx(s.get_system_focal_length())`.
- Also check that `r.image_plane_x_mm` round-trips (set it to `None` or e.g. `120.0` before saving).
- Legacy files: build them by hand with `json.dump` from `Lens(...).to_dict()` dicts and finite radii, leaving out `node_tree`.
- Failure cases:
  - Use `caplog` (level ERROR) and assert `"Error loading optical system"` appears in `caplog.text`.
  - Use `assert result is None`, **not** `pytest.raises`.
- Pitfalls:
  - Write NaN by hand as text (`'{"aperture_stop_diameter": NaN}'`); `json.dump` with defaults does emit it.
  - Do not assume `DEFAULT_TEMPERATURE`'s value; compare against `s.thermal_state_temperature`.
  - Use `pytest.approx` for positions.

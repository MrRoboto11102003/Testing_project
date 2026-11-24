# Notes: `OpticalSystem.from_dict(data, lens_lookup=None)` (classmethod)

## 1. Purpose
- Builds a new `OpticalSystem` from a dict such as the output of `to_dict()`.
- It has two layout paths. If `data["node_tree"]` is truthy, the tree is the authority. Otherwise it uses the flat `elements`/`air_gaps` lists, optionally pinned to the saved `position` values.

## 2. Inputs
- `data`: a dict. Every key is optional and read with `.get`, except `elem_data["lens"]` (see Raises).
  - `name`: default `"Optical System"`.
  - `metadata`: a dict or None. Keys in `_SYSTEM_SERIALIZED_KEYS` are dropped. These are `id, name, type, created_at, modified_at, aperture_stop_gap, aperture_stop_diameter, elements, air_gaps, metadata, node_tree, thermal_state_temperature, image_plane_x_mm, casing`. The copy is shallow.
  - `id`, `created_at`, `modified_at`: copied verbatim. If absent, a fresh uuid hex or the current ISO timestamp is used.
  - `aperture_stop_gap`, `aperture_stop_diameter`: copied verbatim, default None. They are **not validated**, so a bad index is accepted.
  - `thermal_state_temperature`: default `DEFAULT_TEMPERATURE` (a constant from `openlens.constants`; compare it to `OpticalSystem().thermal_state_temperature`).
  - `image_plane_x_mm`: passed through `openlens.lens.validate_image_plane_x`. That function is in another module, so its behaviour is not visible here; None is accepted.
  - `casing`: if it is a dict, `Casing.from_dict(...)`, else None. `validate_casing` is **not** called.
  - `elements`: a list of dicts with keys `lens`, `lens_id`, `position`, `decenter_y/z`, `tilt_x/y/z`.
  - `air_gaps`: a list of `{"thickness": ...}`.
  - `node_tree`: see Branches.
- `lens_lookup`: an optional dict mapping lens id to a `Lens`. A falsy value (None or `{}`) disables reuse. A matched lens is used **by identity**, with no copy.

## 3. Raises
Checks are not ordered up front; errors surface as the code reaches each value.
- `AttributeError`, if `data` is not a dict or `metadata` is truthy but not a dict.
  - Example: `from_dict({"metadata": [1]})`.
- `KeyError: 'lens'`, if an element dict has no `"lens"` and no lookup hit.
  - Example: `from_dict({"elements": [{}]})`.
  - This happens even when `node_tree` is present, because the elements list is processed first.
- `ValueError` or `TypeError` from `float()` on decenter or tilt values.
  - Example: `{"decenter_y": "abc"}`. None and 0 are fine (`or 0.0`).
- On the flat path, `float(saved position)` can raise the same errors.
- Errors from `Lens.from_dict`, `Casing.from_dict` and `validate_image_plane_x` live in other modules and are not visible here.

## 4. Returns and guarantees
- Always returns a new `OpticalSystem`. The `data` dict itself is not stored.
- The flat lists are rebuilt from the tree (`_rebuild_from_tree`):
  - `elements[i].position` is the node's global x.
  - Decenter and tilt come from the node's position y/z and rotation.
  - `lens_id` comes from `lens.id`, via `LensElement.__post_init__`, **not** from `elem_data["lens_id"]`.
- Air gaps follow the element positions:
  - `air_gaps[i].thickness = next.x - (x_i + lens_i.thickness)`.
  - `air_gaps[i].position = x_i + lens_i.thickness`.
  - An overlap below `-1e-12` gives thickness `0` (int); a value within `±1e-12` gives `0.0`.
- `modified_at` is not bumped; it equals `data["modified_at"]` when given.

## 5. Side effects
- `data` is not mutated.
- Lookup lenses are shared, so later edits on the system's lens show up in the lookup objects.

## 6. Branches and edge cases
- **Empty dict** `{}` gives:
  - name `"Optical System"`;
  - `elements == []` and `air_gaps == []`;
  - `metadata == {}`, `casing is None`, `aperture_stop_gap is None`, `image_plane_x_mm is None`;
  - `get_total_length() == 0.0`.
- **Flat path with no `position` keys.** Elements are stacked by gaps: `gap_before` for element i is `air_gaps[i-1]["thickness"]`, or 0.0 if missing or the list is too short.
  - Example: two lenses with thickness 5 and `air_gaps=[{"thickness": 10}]` give positions `[0.0, 15.0]`, gap thickness 10, gap position 5.
- **Flat path where some `position` is not None.**
  - Each root child takes `x = float(saved)` and the gaps are recomputed from those positions. Saved gap thicknesses are therefore effectively overridden.
  - Example: positions `[15, 30]` with thickness 5 give elements at 15 and 30, gap 10 at x=20.
  - A None entry keeps its stacked x. For example, `[None, 40]` gives x 0 and 40.
  - Overlapping saved positions give gap `0`.
- **`node_tree` truthy.** The flat elements are discarded and the system is rebuilt from the tree:
  - A node with `type == "OpticalElement"` becomes an element. Its lens is `Lens.from_dict(element_model)`, or `Lens()` if that key is missing or empty.
  - The lens is then swapped for `lens_lookup[model.id]` if present.
  - Any other or unknown `type` becomes an `OpticalAssembly` container.
  - `position` and `rotation` are 3-sequences, default `(0,0,0)`. The root's own position and rotation are restored too.
  - Modifiers are rebuilt via `modifier_from_dict`.
  - Nested global x depends on `OpticalNode.get_flat_list`, which is in another module.
- **`lens_lookup` hit on the flat path.** `system.elements[0].lens is lookup[id]`.

## 7. Rules from comments
- `to_dict` "elements" is a flattened list; `node_tree` preserves nesting and wins when present.
- Saved absolute positions must be restored. A first element at x=15 must not come back at 0.
- Metadata keeps only keys "that have no home on the model".

## 8. Testing hints
- Build inputs like this:

  ```python
  L = lambda n: Lens(name=n, radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0, thickness=5.0, diameter=25.0)
  s = OpticalSystem("S"); s.add_lens(L("a")); s.add_lens(L("b"), air_gap_before=10.0)
  d = s.to_dict(); r = OpticalSystem.from_dict(d)
  ```

- Round trip: check `r.id == s.id`, `r.name == "S"`, `[e.position for e in r.elements] == [0.0, 15.0]` and `r.air_gaps[0].thickness == pytest.approx(10.0)`.
- To test the flat path, delete the tree first: `d.pop("node_tree")`. Then set `d["elements"][0]["position"] = 15.0` and `d["elements"][1]["position"] = 30.0`, and assert positions `[15.0, 30.0]`.
- Metadata filtering: `{"metadata": {"name": "x", "tol": 1}}` gives `r.metadata == {"tol": 1}` and `r.name == "Optical System"`.
- A stale stop is accepted: `{"aperture_stop_gap": 5}` sets `r.aperture_stop_gap == 5`, while `r.get_aperture_stop() is None`.
- Lookup identity: `OpticalSystem.from_dict(d, lens_lookup={lens.id: lens}).elements[0].lens is lens`. This works on both paths, as long as the lens id round-trips through `Lens.to_dict`/`from_dict` (that is the lens module's behaviour, so verify `Lens.from_dict(lens.to_dict()).id == lens.id` if unsure).
- Pitfalls:
  - Use `pytest.approx` for derived gap thicknesses.
  - Don't assert `modified_at` changed.
  - Don't expect `ValueError` for a bad stop index or an interfering casing.
  - Don't rely on the default thickness or radius constants; pass explicit values.

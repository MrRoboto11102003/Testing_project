# Notes: `OpticalSystem.refresh_references(lens_lookup: dict) -> None`

## 1. Purpose
- Replaces each element's `Lens` with the shared instance from `lens_lookup`, matched on `element.lens_id`.
- Points the node tree at the same new lens objects.
- Re-lays out the axial positions using the new lens thicknesses.

## 2. Inputs
- `lens_lookup`: a mapping `lens_id -> Lens`. Not validated.
  - Per element, it calls `LensElement.refresh(lens_lookup)`: `if lens_lookup and self.lens_id in lens_lookup: self.lens = lens_lookup[self.lens_id]`.
  - A falsy lookup (`None`, `{}`) means no element is swapped. The tree sync and layout steps below still run.
  - Keys that match no element are ignored. Extra entries are harmless.
- `element.lens_id` is set in `LensElement.__post_init__` as `lens.id` when the lens has an `id` attribute. Otherwise it stays `None`.
  - `Lens.id` lives in `openlens/lens.py`, which is not shown here.
  - Safest key to use: `system.elements[i].lens_id`.

## 3. Raises
There is no explicit `raise`. Incidental errors:
- `refresh_references(5)` raises `TypeError` (`argument of type 'int' is not iterable`) from the `in` test, if the system has elements.
- A lookup value without `.thickness` (e.g. `{lid: object()}`) raises `AttributeError` in `_update_positions` (`element.thickness -> self.lens.thickness`).
- An empty system with any lookup does not raise.

## 4. Returns and guarantees
- Returns `None`.
- The `LensElement` objects in `system.elements` are the same objects as before. The list is not rebuilt. Only their `.lens` attribute is reassigned.
- `element.lens_id` is NOT updated. It keeps the old key even if the new lens has a different `.id`. `to_dict()["elements"][i]["lens_id"]` therefore stays the old id.
- Decenter and tilt fields on elements and nodes are unchanged.
- `air_gaps[i].thickness` is unchanged.

**Tree sync:**
- Element nodes are taken in flat order from `root.get_flat_list()` and zipped with `elements`.
- If `node.element_model is not element.lens`, then `node.element_model = element.lens`.
- After the call, `system.root.children[i].element_model is system.elements[i].lens` (flat system).

**Layout (`_update_positions`):**
- `current = first element node's global x`. This is the old anchor, not 0.
- For each element `i`:
  - `element.position = current`
  - the node's `position.x` is set to `current` (minus the parent's global x if nested); node `y`/`z` are kept
  - `current += element.lens.thickness`
  - `air_gaps[i].position = current`
  - `current += air_gaps[i].thickness`
- Effect: a thicker or thinner replacement lens shifts every later element. The gaps keep their size.

## 5. Side effects
- Mutates `elements[i].lens`, the tree nodes' `element_model` and `position.x`, `elements[i].position` and `air_gaps[i].position`.
- Does NOT change `modified_at`, `aperture_stop_*`, `casing`, `image_plane_x_mm` or `metadata`.
- Does not mutate `lens_lookup` or the lens objects themselves.

## 6. Branches and edge cases
**Swap with a thicker lens:**
- Setup: `a` has thickness 5, `b` has thickness 3, `s.add_lens(a); s.add_lens(b, air_gap_before=10)`.
  - Positions before: `[0, 15]`.
  - `air_gaps[0]` before: thickness 10, position 5.
- Call: `s.refresh_references({s.elements[0].lens_id: a2})` with `a2.thickness = 8`.
- Results:
  - `elements[0].lens is a2`
  - positions `[0.0, 18.0]`
  - `air_gaps[0].position == 8.0`, `.thickness == 10`
  - `get_total_length() == 21.0`

**Anchor is kept:**
- First lens added with `air_gap_before=3` has position 3.
- After any refresh it stays at 3, not 0.

**Empty, `None` or non-matching lookup:**
- Lenses are unchanged.
- Positions are recomputed but come out identical for a consistent system.

**Sync without a lookup:**
- `s.elements[0].lens = other; s.refresh_references({})` updates `root.children[0].element_model` to `other`.
- Positions are re-laid with `other.thickness`.

**Empty system:**
- `OpticalSystem().refresh_references({"x": Lens()})` is a no-op.
- `elements == []`, `air_gaps == []`.

**Partial lookup:**
- Only the matching element is swapped.
- The others keep their original lens object (check with `is`).

## 7. Rules from comments
- The tree is the authority for geometry.
- Refreshing only the flat list used to leave the tree on the old `Lens`. The 2D (flat) calculations and the 3D (tree) tracer then analysed different optics, so the tree must be updated too.
- Nodes are matched to elements by the identity of the wrapped lens model. Positions are only axial; decenter lives on the node.

## 8. Testing hints
- Build lenses explicitly: `Lens(name="A", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0, thickness=5.0, diameter=25.0)`. Do not rely on `DEFAULT_THICKNESS`, which is not shown here.
- Observe the swap:
  - `s.elements[i].lens is new`
  - `s.root.children[i].element_model is new`
  - `s.root.children[i].position.x == s.elements[i].position`
- Check that the following stay equal to their values before the call:
  - `elements` (identity, e.g. `id(s.elements[0])`)
  - `lens_id`
  - `air_gaps[i].thickness`
  - `modified_at`
  - decenter values (e.g. `add_lens(..., decenter_y=1.5)` keeps 1.5 on both element and node)
- A downstream effect you can assert: `get_system_focal_length()` uses the new lens. For a single element it equals `new.calculate_focal_length()`.
- Use `pytest.approx` for positions. Avoid asserting exact `Lens.id` formats, which are not visible here.

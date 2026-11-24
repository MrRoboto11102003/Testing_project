# Notes: `openlens.database.DatabaseManager.load_all`

## 1. Purpose
- Reads the SQLite database and returns plain dicts.
- It returns every **library** lens (`is_library` truthy) and every assembly.
- Each assembly comes back with its ordered elements and air gaps.
- It is read-only and returns a new list of new dicts on each call.

## 2. Inputs
- Takes no arguments; it uses `self.db_path`.
- Build the manager with `db = DatabaseManager(str(tmp_path / "t.db"))`.
- `resolve_db_path` and `ensure_parent_dir` live in `openlens.paths`, which is not shown here. Assume an explicit path is used as given.
- A fresh file is created at schema v7 by `__init__`.

## 3. Raises
- `load_all` raises nothing on purpose.
- Corrupt metadata blobs do **not** raise. They log a warning and produce `metadata == {}`.
- Only raw `sqlite3` errors can escape, for example if the file is not a database.
- `UnsupportedSchemaVersionError` comes from `__init__`, not from `load_all`.

## 4. Returns and guarantees
The result is a `list[dict]` built in this order:
1. All lens rows with truthy `is_library`, in `SELECT * FROM lenses` order. This is normally rowid order. `INSERT OR REPLACE` re-inserts a row, so a re-saved lens moves to the end.
2. All assemblies, in `SELECT * FROM assemblies` order.

Empty database → `[]`.

**Lens dict:**
- **Columns:** every column of the table, except that `radius1`/`radius2` are renamed to `radius_of_curvature_1`/`radius_of_curvature_2`. The keys `radius1` and `radius2` are absent.
  - Other columns: `id, name, thickness, material, refractive_index, diameter, created_at, modified_at, is_parabolic_1, parabolic_sag_1, is_parabolic_2, parabolic_sag_2, conic_constant_1, conic_constant_2, clear_aperture_1, clear_aperture_2, bevel_1, bevel_2, coating_1, coating_2, is_library`.
- **Metadata merge:** the blob is merged **flat** onto the row with `dict.update`, so blob keys override column values.
  - `lens["metadata"]` is a copy of the **whole** parsed blob. It is `{}` when the blob is NULL, empty, invalid JSON, or valid JSON that is not an object.
- **`row_type`:** `lens["row_type"] == "Lens"`, set **after** the merge, so a blob cannot spoof it.
  - A lens's `"type"` key, such as `"Biconvex"` from the blob, is left untouched.
- **Coatings:** if `coating_1`/`coating_2` is a `str`, it is decoded.
  - Valid non-empty JSON → the decoded list.
  - `"[]"`, `"null"`, `""`, or invalid JSON → `None`.
  - A non-str value, such as `None` or a list coming from the blob, is kept as is.
- **SQLite types:**
  - `is_parabolic_*` and `is_library` come back as `int` 0/1, not `bool`.
  - REAL columns come back as `float` (`thickness=5` → `5.0`).
  - Missing optional columns come back as `None`.

**Assembly dict:**
- **Columns:** `id, name, created_at, modified_at, aperture_stop_gap, aperture_stop_diameter`.
- **Metadata:** the blob is merged flat, and `metadata` is a copy of the blob dict.
- **`row_type`:** `"OpticalSystem"`, set after the merge.
- **`elements`:** a list ordered by `order_index` (the save order). Each item is `{"lens", "lens_id", "position", "decenter_y", "decenter_z", "tilt_x", "tilt_y", "tilt_z"}`.
- **`air_gaps`:** a list of `{"thickness", "position"}` ordered by `order_index`.

**Element `"lens"` value:**
- It is the **same dict object** as the lens lookup entry, because the JOIN guarantees the lens exists in the lookup.
- For a library lens, `elements[i]["lens"] is <that lens dict in results>`.
- For an assembly-only lens, it is a full lens dict, including `row_type: "Lens"` and `is_library: 0`, that does not appear at top level.
- The fallback branch that rebuilds the lens from `lens_*` columns is effectively unreachable.

## 5. Side effects
- None on the database. It opens one connection and closes it.
- The only side effect is a `logger.warning` when a metadata blob is bad.

## 6. Branches and edge cases
**Membership (which lenses appear at top level):**
- **Standalone save:** `save_lens(...)` stores `is_library=1`, so the lens appears in results.
- **Assembly member:** a lens first written through `save_assembly` gets `is_library=0`. It is absent from top-level results but present inside `elements[*]["lens"]`.
- **Lens saved first, then used in an assembly:** the existing `is_library=1` is kept. The lens appears once at top level and is also referenced, by identity, from the element.
- **Explicit flag:** a `lens_dict` with `"is_library": False` saves as 0 and is hidden.
- **Promotion:** `set_library_membership(["L2"])` promotes L2 to 1, so it then appears.
- **Blob override:** a blob key `is_library` would override the column before the inclusion check. This only happens with hand-written blobs; `save_lens` strips that key.

**Metadata examples:**
- Corrupt blob, e.g. via raw `sqlite3`: `UPDATE lenses SET metadata='{bad'`. Result: `metadata == {}`, the row is still returned, nothing raises.
- `metadata='[1,2]'` → `{}` as well.

**Flattening on save, then on load:**
```python
save_lens({..., "foo": "bar", "metadata": {"tol": 1}})
```
After `load_all()`:
- `lens["foo"] == "bar"`
- `lens["tol"] == 1`
- `lens["metadata"] == {"foo": "bar", "tol": 1}`
- The key `"metadata"` is never nested.

**Coatings:**
- Saving `coating_1=[]` or `None` → `None` after load.
- Saving `[{"material": "MgF2", ...}]` → an equal list after load.

**Empty assembly:** no elements or gaps → `elements == []` and `air_gaps == []`.

## 7. Rules from comments/docstrings
- Only "standalone" library lenses are returned at top level; assembly members are not.
- The row marker uses the key `row_type`, never `type`.
- The marker is set after the merge, so a blob cannot spoof it (#488).
- A bad blob must not empty the whole library.
- The element lens is taken from the lookup "to ensure identity".

## 8. Testing hints
**Minimum dict for `save_lens`:** the NOT NULL columns need `id, name, radius_of_curvature_1, radius_of_curvature_2, thickness, material`. Missing any of these gives an `IntegrityError`.
```python
L = {"id": "L1", "name": "A", "radius_of_curvature_1": 50.0,
     "radius_of_curvature_2": -50.0, "thickness": 5.0,
     "material": "BK7", "diameter": 25.0}
```

**Minimum assembly for `save_assembly`:**
```python
{"id": "A1", "name": "Sys",
 "elements": [{"lens": L, "position": 0.0}],
 "air_gaps": [{"thickness": 10.0, "position": 5.0}]}
```

**Pitfalls:**
- **Ordering:** find rows by `row_type` and `id` rather than relying on list order. The guaranteed part is that lenses come before assemblies.
- **Int flags:** compare `is_parabolic_1 == 1` or `== True`, never with `is True`.
- **Default values:** unspecified `decenter_*`/`tilt_*` load as `0.0`. Unspecified `aperture_stop_gap` loads as `None`.
- **Fresh objects:** each call builds new dicts, so mutating a result does not affect the next call.

**What to check:**
- `"radius1" not in lens`.
- `lens["row_type"] == "Lens"`, even if the saved dict had `"type": "Biconvex"`, in which case `lens["type"] == "Biconvex"`.
- Element identity with `is`.

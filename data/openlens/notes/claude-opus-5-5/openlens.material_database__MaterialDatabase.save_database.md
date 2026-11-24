# Notes: `MaterialDatabase.save_database(self) -> None`

## 1. Purpose
- Writes the whole in-memory catalog (`self.materials`) to the JSON file at `self.db_path`.
- The file is replaced atomically, and its old contents are discarded. The new contents are not merged with the old ones.

## 2. Inputs
- No parameters. It uses `self.db_path` and `self.materials`.
- `self.db_path` is set in `__init__` as `db_path or str(default_materials_path())`.
  - A falsy `db_path` (`None` or `""`) means the real per-user catalog file.
- `self.materials` is a `Dict[str, MaterialProperties]`.
  - A fresh instance holds 7 built-ins in this insertion order: `"BK7", "N-BK7", "SF11", "F2", "FUSEDSILICA", "S-LAH79", "E-FD60"`.
  - Entries from an existing file at `db_path` follow, with keys set to `name.upper()`.

## 3. Raises
- The method has no checks of its own.
- `ensure_parent_dir(self.db_path)` (in `openlens.paths`, not shown) creates the parent directory. Per the code comment, it reports an unwritable directory instead of raising.
- `atomic_write_json(path, payload, indent=2)` (in `openlens.atomic`, not shown) does the write. Any `OSError` or `TypeError` propagates. Because the write is atomic, the target is left untouched on failure. The exact messages cannot be seen here, so do not assert on them.
- Possible trigger: a non-JSON-serialisable field value, e.g. `db.materials["X"] = MaterialProperties("X","C",1.5,60, density={1})` (a set). This would likely raise `TypeError`; the exact behaviour depends on the unseen code.

## 4. Returns and guarantees
- Returns `None`.
- The payload is `{key: mat.to_dict() for key, mat in self.materials.items()}`.
  - Keys are the dict keys, not `mat.name`.
  - Values are `dataclasses.asdict(mat)`, which holds all 23 fields: `name, catalog, nd, vd, B1..B3, C1..C3, D0, D1, D2, E0, E1, lambda_tk_um, thermal_expansion, transmission_data, transmission_thickness, density, climate_resistance, acid_resistance, alkali_resistance`.
- The payload is serialised as JSON with `indent=2`. Whether keys are sorted, plus the exact whitespace, depends on the unseen `atomic_write_json`. Parse the file with `json.load`; do not compare raw text.
- `transmission_data` tuples become JSON arrays, so the parsed value is `[[380, 0.95], ...]` (lists, not tuples).
- Floats round-trip exactly. For example, FUSEDSILICA `C1 == 0.0684043**2`, and BK7 `B1 == 1.03961212`.

## 5. Side effects
- Creates the parent directory (via the helper) and creates or overwrites the file at `db_path`.
- Does not mutate `self.materials`.
- Does not clear `_index_cache`.
- `add_material(m)` with the default `save=True` calls this method. `import_*_catalog` calls it once when `count > 0`.

## 6. Branches / edge cases
- **Fresh database, nothing loaded:** the file contains exactly the 7 built-in keys.
- **After `db.add_material(MaterialProperties("myglass","Custom",1.6,40), save=False)`:** the file contains 8 keys, including `"MYGLASS"`. The value's `"name"` stays `"myglass"`.
- **After `del db.materials["SF11"]`:** saving leaves `"SF11"` out of the file. On reload, SF11 reappears because built-ins load before the file.
- **Modified built-in:** set `db.materials["BK7"].nd = 1.6`, then save, then reload with `MaterialDatabase(db_path=same)`. The reloaded `.materials["BK7"].nd` is `1.6`, because the file overrides the built-in.
- **Key differs from name:** after `db.materials["FOO"] = MaterialProperties(name="bar", ...)`, the file key is `"FOO"` and `["FOO"]["name"] == "bar"`.
- **Snapshot:** mutating a material after saving does not change the file until the next save.
- **Missing parent directory:** a nested path such as `tmp_path/"a"/"b"/"m.json"` should get its directories created, via the unseen helper.

## 7. Rules from comments
- The user data directory may not exist yet. `ensure_parent_dir` reports an unwritable directory instead of raising, matching the contract used for the database path.
- Imports batch their writes: one `save_database` per import rather than one per glass. A write failure from the import paths is meant to reach the caller (issue #474).

## 8. Testing hints
- **Always pass an explicit `db_path`**, e.g. `MaterialDatabase(db_path=str(tmp_path / "materials.json"))`. With `None` or `""`, the test would write the developer's real catalog.
- Read the result with `json.load(open(path, encoding="utf-8"))`. Assert on:
  - the set of keys;
  - values such as `data["BK7"]["nd"] == 1.5168`, `data["SF11"]["vd"] == 25.68`, `data["FUSEDSILICA"]["climate_resistance"] == 4`;
  - `data["BK7"]["transmission_data"][0] == [380, 0.95]`;
  - `data["N-BK7"]["transmission_data"] == []`.
- **Round trip:** reload from the same path and compare fields.
  - Reloaded `transmission_data` items are lists, so `[list(t) for t in orig]` equals the reloaded value, but the original tuples do not.
  - Compare individual fields, not whole dataclasses: `==` on a dataclass would fail on tuple vs list.
- `save_database()` returns `None`.
- The file exists after the call, and calling it twice overwrites the file cleanly. After the second save, the file reflects only the current `materials`.
- To compare key order, use `list(data.keys())` against `list(db.materials.keys())`. This is only safe if `atomic_write_json` does not sort keys, which cannot be seen. Prefer set comparison.
- Do not assert exact file text, indentation, or a trailing newline.

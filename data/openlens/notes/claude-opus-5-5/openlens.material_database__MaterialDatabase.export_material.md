# Notes: `MaterialDatabase.export_material(self, material_name: str, output_file: str)`

## 1. Purpose
- Writes one catalog material to `output_file` as a JSON object (`MaterialProperties.to_dict()`).
- If the name cannot be resolved, it silently does nothing.

## 2. Inputs
- `material_name: str` is resolved with `self.get_material(name)`:
  1. First it tries the exact key `name.upper()` in `self.materials`. For example, `"bk7"` resolves to `"BK7"` and `"n-bk7"` resolves to `"N-BK7"`.
  2. Otherwise it tries the alphanumeric-only uppercase form, `"".join(c for c in name.upper() if c.isalnum())`. That form is looked up in an alias map built from the catalog keys, and the first key wins.
  - Alias examples:
    - `"Fused Silica"`, `"fused-silica"` and `"fused_silica"` all resolve to `"FUSEDSILICA"`.
    - `"n bk7"` and `"NBK7"` resolve to `"N-BK7"`.
    - `"s lah79"` resolves to `"S-LAH79"`.
    - `"E FD60"` resolves to `"E-FD60"`.
- Built-in keys: `BK7, N-BK7, SF11, F2, FUSEDSILICA, S-LAH79, E-FD60`. Further keys come from the JSON file at `db_path` and from `add_material`.
- `output_file`: a path. It is passed straight to `openlens.atomic.atomic_write_json(output_file, payload)` with the default `indent=2`. There is no validation, and `ensure_parent_dir` is not called.

## 3. Raises
- The function raises nothing itself.
- A non-string name (e.g. `db.export_material(None, p)`) raises `AttributeError` from `None.upper()`.
- Write failures, such as a missing parent directory or an unwritable path, come from `atomic_write_json` in `openlens.atomic`. That code is not shown, so the exact exception is unknown. Do not test it.

## 4. Returns and guarantees
- Always returns `None`, whether found or not.
- When the material is found, the file holds the JSON of `dataclasses.asdict(mat)`.
  - Keys, in field order: `name, catalog, nd, vd, B1, B2, B3, C1, C2, C3, D0, D1, D2, E0, E1, lambda_tk_um, thermal_expansion, transmission_data, transmission_thickness, density, climate_resistance, acid_resistance, alkali_resistance`. That is 23 keys.
  - `name` is the stored material's own `name` (e.g. `"FUSEDSILICA"`), not the string the caller passed.
  - `transmission_data` tuples become JSON lists of lists. For BK7 the value is `[[380, 0.95], [400, 0.97], …, [2500, 0.5]]` (10 entries). For N-BK7 it is `[]`.
- BK7 values:
  - `nd=1.5168`, `vd=64.17`, `catalog="Schott"`
  - `B1=1.03961212`, `C3=103.560653`
  - `D0=1.86e-6`, `lambda_tk_um=0.17`
  - `thermal_expansion=7.1e-6`, `density=2.51`
  - `alkali_resistance=2`, `climate_resistance=1`
- N-BK7 values: `climate_resistance=2`, `acid_resistance=2`.
- FUSEDSILICA values: `nd=1.45846`, `vd=67.82`, `C1=0.0684043**2`, `thermal_expansion=0.55e-6`, `climate_resistance=4`, `alkali_resistance=1`.
- S-LAH79 has `catalog="Ohara"`. Its D0–E1 and `lambda_tk_um` are all `0.0`.
- E-FD60 has `catalog="Hoya"`.

## 5. Side effects
- Writes or overwrites only `output_file`.
- Does not write `db_path` (it does not call `save_database`).
- Does not mutate `self.materials` or the index cache. It may build the internal alias map, which is not observable.
- If the material is not found, no file is created and an existing `output_file` is left untouched.

## 6. Branches
- **Found (exact key)**: `db.export_material("BK7", p)` writes the file and `json.load` gives `["name"] == "BK7"`.
- **Found (alias)**: `db.export_material("Fused Silica", p)` writes the file with `["name"] == "FUSEDSILICA"`.
- **Not found**: `db.export_material("Unobtainium", p)` returns `None` and `p` does not exist afterwards.
  - `""` is also not found, because the empty key matches nothing.
  - A pre-existing file at `p` keeps its old content.
- **Custom material**: after `db.add_material(MaterialProperties(name="MyGlass", catalog="Custom", nd=1.6, vd=40.0), save=False)`, `export_material("myglass", p)` writes `name == "MyGlass"` with defaults:
  - `B1 = 0.0`
  - `thermal_expansion = 7.1e-6`
  - `transmission_thickness = 10.0`
  - `density = 2.51`
  - `transmission_data = []`
  - resistances `= 1`
- **Overwrite**: exporting `"BK7"` then `"SF11"` to the same path leaves SF11 data (`nd == 1.78472`).

## 7. Rules from docstrings
- The docstring says "Export material data to file".
- `get_material` treats spacing, punctuation and case variants as the same glass.

## 8. Testing hints
- Always build the database as `MaterialDatabase(db_path=str(tmp_path / "materials.json"))`, pointing at a non-existent file.
  - `MaterialDatabase()` with no argument reads the user's real catalog.
  - Use `add_material(..., save=False)` to avoid writing `db_path`.
- Read the output back with `json.load(open(p, encoding="utf-8"))` and compare values.
  - Do not assert the exact text, whitespace or indent, because `atomic_write_json` internals are not shown.
  - Compare floats with `==` against the literal values above (a JSON round-trip of a Python float is exact) or with `pytest.approx`.
- Round-trip checks:
  - For N-BK7 (empty transmission), `MaterialProperties.from_dict(json.load(...)) == db.get_material("N-BK7")` holds.
  - For BK7 it does NOT hold, because the loaded data has lists instead of tuples. Compare `[tuple(x) for x in d["transmission_data"]]` instead.
- Assert `export_material(...) is None` in both branches.
- Assert `not (tmp_path / "materials.json").exists()` after an export, to show the catalog was not saved.
- Use a write target in an existing directory (`tmp_path / "out.json"`).

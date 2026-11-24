# Notes: `openlens.material_database.MaterialProperties`

## 1. Purpose
- A plain `@dataclass` that holds one optical glass: its index and Abbe number, Sellmeier coefficients, TIE-19 thermal coefficients, transmission table, and physical and chemical data.
- It has two methods, `to_dict` and `from_dict`. Neither validates anything.

## 2. Inputs (constructor fields, in declaration order)
Positional construction follows this exact order:

| # | field | type | default |
|---|-------|------|---------|
| 1 | `name` | str | required |
| 2 | `catalog` | str | required |
| 3 | `nd` | float | required |
| 4 | `vd` | float | required |
| 5–10 | `B1, B2, B3, C1, C2, C3` | float | `0.0` each |
| 11–15 | `D0, D1, D2, E0, E1` | float | `0.0` each |
| 16 | `lambda_tk_um` | float | `0.0` |
| 17 | `thermal_expansion` | float | `7.1e-6` |
| 18 | `transmission_data` | List[Tuple[float, float]] | `field(default_factory=list)`, a new `[]` per instance |
| 19 | `transmission_thickness` | float | `10.0` |
| 20 | `density` | float | `2.51` |
| 21–23 | `climate_resistance, acid_resistance, alkali_resistance` | int | `1` each |

- That is 23 fields in total.
- No `__post_init__`, so there is no normalisation or validation:
  - types are not checked;
  - `name` is not uppercased;
  - negative or NaN values are accepted and stored as given.
- `from_dict(data)` is exactly `cls(**data)`.

## 3. Raises
- **Constructor, missing required field:** `TypeError`.
  - Trigger: `MaterialProperties("X", "Custom", 1.5)` (no `vd`).
  - Assert only the type; Python versions word the message differently.
- **`from_dict` with an unknown key:** `TypeError` (unexpected keyword argument).
  - Trigger: `MaterialProperties.from_dict({"name":"X","catalog":"C","nd":1.5,"vd":60,"foo":1})`.
- **`from_dict` with a missing required key:** `TypeError`.
  - Trigger: `from_dict({"name":"X"})`.
- **`from_dict` with a non-mapping:** `TypeError`.
  - Trigger: `from_dict([1, 2])` (`**` needs a mapping).
- **Hashing:** `hash(m)` raises `TypeError`. The dataclass is `eq=True` and not frozen, so `__hash__` is `None`.
  - `m` cannot be used in a set or as a dict key.

## 4. Returns and guarantees
**`to_dict()`**
- Returns `dataclasses.asdict(self)`: a new `dict` with all 23 keys, in field-declaration order.
- `list(d.keys())` equals the field order in the table above.
- `transmission_data` is copied recursively:
  - the dict's value is a new `list`, not the instance's own list (`d["transmission_data"] is not m.transmission_data`);
  - its elements stay `tuple`s, e.g. `[(380, 0.95)]` → `[(380, 0.95)]`;
  - mutating the returned dict or list does not affect the instance.
- Values are stored as given: no float conversion, so `nd=2` stays the int `2`.

**`from_dict(data)`**
- Returns a new `MaterialProperties`.
- It does not copy anything: `m.transmission_data is data["transmission_data"]` holds when that key is supplied.
- Extra keys are rejected (see Raises). Missing optional keys get the defaults.

**Round trip**
- `MaterialProperties.from_dict(m.to_dict()) == m`.
- Equality is the generated `__eq__`, comparing all fields in order.

**Other generated behaviour**
- `__repr__` starts with `MaterialProperties(name='...', catalog='...', nd=...`.
- Instances are mutable: assigning an attribute works.

## 5. Side effects
- None. No I/O, and no global state is touched.
- Instances do not share their default list: `MaterialProperties("A","C",1.5,60).transmission_data is not MaterialProperties("B","C",1.5,60).transmission_data`.

## 6. Branches and edge cases
- **Minimal construction:** `m = MaterialProperties("X", "Custom", 1.5, 60.0)` gives:
  - `m.B1 == 0.0`
  - `m.thermal_expansion == 7.1e-6`
  - `m.transmission_thickness == 10.0`
  - `m.density == 2.51`
  - `m.climate_resistance == m.acid_resistance == m.alkali_resistance == 1`
  - `m.transmission_data == []`
- **Positional Sellmeier values:** `MaterialProperties("X","C",1.5,60, 1.0,2.0,3.0, 0.1,0.2,0.3)` gives `B1=1.0, B2=2.0, B3=3.0, C1=0.1, C2=0.2, C3=0.3`. All B's come before all C's.
- **`from_dict({"name":"x","catalog":"c","nd":1.0,"vd":1.0})`:** works, and `name` stays `'x'` (lowercase).
- **Equality:** two instances that differ in any one field, e.g. `density`, compare unequal.

## 7. Rules from comments and docstrings
- Sellmeier model: n² = 1 + Σ Bᵢλ²/(λ² − Cᵢ), with λ in µm.
- TIE-19 temperature model for the absolute index:
  - reference temperature is T0 = 20 °C;
  - `lambda_tk_um` is the resonance wavelength in µm;
  - all-zero D/E coefficients mean "no thermal data": no correction is applied.
- Units:
  - `thermal_expansion` is in 1/K;
  - `transmission_data` pairs are (nm, fraction from 0 to 1) at `transmission_thickness` mm;
  - `density` is in g/cm³;
  - resistances are on a 0–4 scale.
- `from_dict` does no key filtering. `MaterialDatabase._load_database` relies on catching that `TypeError`, logging it, and skipping the bad entry.

## 8. Testing hints
- Import with `from openlens.material_database import MaterialProperties`.
- **Pin every default** (all 23 fields) and check `list(m.to_dict())` against the exact key order.
- **Pin `to_dict` values exactly**, e.g. `d["thermal_expansion"] == 7.1e-6` and `d["transmission_data"] == [(500, 0.99)]`.
  - Use tuples in the comparison: a list of lists does not equal a list of tuples.
- **Test defensive copying:** `d = m.to_dict(); d["transmission_data"].append((1, 1)); assert m.transmission_data == [...]` (unchanged).
- **Test the per-instance default list:** append to one instance's `transmission_data`; another default instance should still have `[]`.
- **Test the round trip and the `TypeError`s** listed in Raises.
- **Optional integration test with `MaterialDatabase`:**
  - Use `MaterialDatabase(db_path=str(tmp_path / "m.json"))`. With no path it reads and writes the real user catalog, so never construct it without a path.
  - `db.add_material(m, save=False)` stores the instance under `m.name.upper()`, and `db.get_material("x") is m`.
- **Pitfalls:**
  - Do not expect validation or coercion.
  - Do not call `hash()` and expect success.
  - Do not assert exact `TypeError` message text.
  - Use `pytest.approx` only for computed values. Stored values come back exactly as given.

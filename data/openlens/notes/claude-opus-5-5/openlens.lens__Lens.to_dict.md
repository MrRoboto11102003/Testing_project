# Notes: `openlens.lens.Lens.to_dict`

## 1. Purpose
- Serialises a `Lens` into a fresh plain `dict` of 37 keys.
- It only reads attributes. There is no validation, no database or I/O access, and no timestamp update.

## 2. Inputs
- `self` only. Every value is read from the current instance attributes at call time.
- Attribute changes made after construction show up in the output. For example, after `lens.name = "X"`, `d["name"] == "X"`.
- Normalisation happens at construction or in setters, not in `to_dict`:
  - Radius setter: assigning `0` stores `float("inf")`. Any other value is stored unchanged, so an `int` stays an `int`.
  - `refractive_index`:
    - If given explicitly, it is `float(refractive_index)`.
    - Otherwise it is `DEFAULT_MATERIAL_INDICES.get(material, REFRACTIVE_INDEX_BK7)`. These constants are in `openlens.constants`, which is not shown.
    - In model-glass mode it comes from the material database.
    - **Always pass `refractive_index=` explicitly in tests.**
  - `wavelength`: the `wavelength` argument if it is not `None`, else `wavelength_nm`.
  - `mount_type`: `str(mount_type or DEFAULT_MOUNT_TYPE)`. The constant's value is not shown.
  - `image_plane_x_mm`: `float(value)` or `None`.
  - Parabolic flags: `bool(...)`.
  - Parabolic sags, conic constants and clear apertures: converted with `float(...)`. A clear aperture of `None` stays `None`.
  - Bevels: `float(x or 0.0)`.
  - `num_grooves`:
    - If `is_fresnel=True` and `num_grooves` is `None`, it becomes `int((diameter/2)/groove_pitch)`.
    - If `groove_pitch <= 0`, it becomes `0`.
    - If `is_fresnel=False`, it stays as passed (default `None`).
  - Coatings: `None` or `[]` becomes `None`. A list of dicts or `CoatingLayer` objects becomes a list of new dicts.
  - `metadata`: `dict(metadata)` if truthy, else `{}`.

## 3. Raises
- Nothing under normal use.
- Only pathological attribute values could raise, e.g. an `edge_profile` whose items are not iterable (raises `TypeError` from `list(point)`).

## 4. Returns and guarantees
The result is a new `dict` on every call, with keys in exactly this insertion order:

```
id, name, radius_of_curvature_1, radius_of_curvature_2, thickness, diameter,
refractive_index, type, material, mount_type, wavelength, temperature,
pressure_atm, thermal_state_temperature, image_plane_x_mm, is_fresnel,
groove_pitch, num_grooves, model_glass_mode, model_nd, model_vd,
is_parabolic_1, parabolic_sag_1, is_parabolic_2, parabolic_sag_2,
conic_constant_1, conic_constant_2, clear_aperture_1, clear_aperture_2,
bevel_1, bevel_2, coating_1, coating_2, edge_profile, created_at,
modified_at, metadata
```

Key-set facts:
- `"type"` holds `self.lens_type`, exactly as passed. It is not `classify_lens_type()`.
- There is **no** `"wavelength_nm"` key and **no** `"use_type_defaults"` key.
- `set(d) == lens._LENS_SERIALIZED_KEYS - {"wavelength_nm"}`. `_LENS_SERIALIZED_KEYS` is a module-level frozenset of 38 names, so `len(d) == 37`.

How values are copied:
- `coating_1` / `coating_2` are `copy.deepcopy(self.coating_N)`: either `None` or a new list of new dicts. Mutating the output does not touch the lens.
- `edge_profile` is `[list(p) for p in self.edge_profile]` (a list of lists) if truthy, else `None`. An empty tuple becomes `None`.
- `metadata` is `dict(self.metadata)`, a **shallow** copy:
  - Adding or removing keys in the output does not affect the lens.
  - Nested mutable values are shared with the lens.
- Radii are output raw. A flat surface is `float("inf")` and is **not** converted to `None`; that is the job of `json_safe_radii`.
- `id` is a 32-character hex string (`uuid4().hex`). `created_at` and `modified_at` are ISO strings from `datetime.now().isoformat()`.

## 5. Side effects
- No mutation of the lens. `modified_at` is not changed, despite the comment.
- No global state is touched and no files are written.
- Two calls give equal but distinct dicts:
  - `d1 == d2` and `d1 is not d2`.
  - `d1["metadata"] is not lens.metadata`.

## 6. Branches and edge cases
- **Uncoated lens:** `Lens(refractive_index=1.5)` gives `d["coating_1"] is None` and `d["coating_2"] is None`.
- **Coated lens:** `coating_1=[{"material":"MgF2","refractive_index":1.38,"thickness_nm":99.6}]` gives an equal list.
  - `d["coating_1"][0]["thickness_nm"] = 0` leaves `lens.coating_1` unchanged.
  - `d["coating_1"] is not lens.coating_1`.
- **Edge profile set:**
  - Set it directly: `lens.edge_profile = ((0.0, 12.5), (5.0, 12.5))`. This bypasses geometry validation, which is fine for `to_dict`.
  - Result: `d["edge_profile"] == [[0.0, 12.5], [5.0, 12.5]]`, with inner items of type `list`.
- **No edge profile:** `d["edge_profile"] is None`.
- **Zero radius:** `Lens(radius_of_curvature_1=0, ...)` gives `d["radius_of_curvature_1"] == float("inf")` (check with `math.isinf`).
- **Fresnel:** `Lens(is_fresnel=True, diameter=50, groove_pitch=2.0, refractive_index=1.5)` gives `d["num_grooves"] == 12`, since `int(25/2.0)` is 12.
- **`wavelength` argument wins:** `Lens(wavelength=550.0, wavelength_nm=600.0, refractive_index=1.5)` gives `d["wavelength"] == 550.0`.
- **Image plane:** `image_plane_x_mm="42"` gives `d["image_plane_x_mm"] == 42.0` as a float.
- **Bevel `None`:** `bevel_1=None` gives `d["bevel_1"] == 0.0`.
- **Empty metadata:** `metadata=None` gives `d["metadata"] == {}`.

## 7. Rules from comments
- `to_dict` must not hand out live references to model state. The `metadata` copy and the coating deep copies exist for this reason.
- `_LENS_SERIALIZED_KEYS` is meant to equal what `to_dict` emits, plus `wavelength_nm`, which only `from_dict` reads. The comment says this is "asserted against it in tests".

## 8. Testing hints
Build test lenses like this:

```python
from openlens.lens import Lens, _LENS_SERIALIZED_KEYS
lens = Lens(name="T", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
            thickness=5.0, diameter=25.0, refractive_index=1.5,
            lens_type="Biconvex", material="BK7", temperature=20.0,
            model_nd=1.6, model_vd=40.0, metadata={"tol": [1, 2]})
d = lens.to_dict()
```

Exact checks that are safe:
- `d["name"] == "T"`, `d["refractive_index"] == 1.5`, `d["type"] == "Biconvex"`.
- `d["model_nd"] == 1.6`, `d["model_vd"] == 40.0`.
- `d["id"] == lens.id`, `d["created_at"] == lens.created_at`.
- `list(d)` equals the key order listed in section 4.

Round trip:
- `Lens.from_dict(d).to_dict()` keeps `id`, `created_at`, `modified_at`, the radii, `refractive_index` and `metadata` (user keys only).
- Avoid model-glass mode in round-trip tests.

Mutation-isolation tests:
- Mutating `d["metadata"]["new"] = 1` leaves `lens.metadata` without `"new"`.
- But `d["metadata"]["tol"] is lens.metadata["tol"]`, because the copy is shallow.

Pitfalls:
- Do not hard-code these defaults: `mount_type`, `pressure_atm`, `thermal_state_temperature`, the `lens_type` constant, and the default `refractive_index`. They come from modules that are not shown. Read them from the instance or the imported constants instead.
- Do not assert `"wavelength_nm" in d`; it is absent.
- Do not expect `modified_at` to change after `to_dict`.
- `inf` radii stay `inf`. `json.dumps(d)` would emit `Infinity`, so do not assert JSON validity.
- Avoid `model_glass_mode=True` without an explicit `refractive_index`, because the index then depends on the material database.

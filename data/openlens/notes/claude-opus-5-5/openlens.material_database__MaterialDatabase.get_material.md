# Notes: `MaterialDatabase.get_material(self, name: str) -> Optional[MaterialProperties]`

## 1. Purpose
- Looks up a material by name. It tries the exact uppercased key first. If that misses, it tries a punctuation-, space- and case-insensitive alias.
- It returns the stored `MaterialProperties` object, or `None` if no match is found.

## 2. Inputs
- `name`: must be a `str`. The function does no other validation.
- **Step 1 (exact):** `self.materials.get(name.upper())`.
- **Step 2 (alias):** normalise `name` with the private helper `_material_key(name)`:
  ```python
  "".join(c for c in name.upper() if c.isalnum())
  ```
  - It uppercases the name, then keeps only alphanumeric characters (Unicode `isalnum`).
  - So spaces, `-`, `_`, `.` and similar are dropped.
  - Examples: `"Fused Silica"` → `"FUSEDSILICA"`; `"n_bk7"` → `"NBK7"`; `"S-LAH79"` → `"SLAH79"`.
- **Alias table (`_normalised_keys()`):**
  - Built lazily into `self._alias_keys` the first time it is needed.
  - Maps `_material_key(catalog_key)` → `catalog_key` for every key in `self.materials`, in insertion order.
  - It uses `setdefault`, so the **first key wins** when two keys normalise to the same string.
  - The table is reset to `None` by `clear_cache()`, which `add_material` and `_load_database` call. `get_refractive_index` also resets it on every cache miss.

## 3. Raises
- A non-str `name` raises `AttributeError` from `.upper()`.
  - Example: `db.get_material(None)` → `AttributeError: 'NoneType' object has no attribute 'upper'`.
  - Example: `db.get_material(7)` → `AttributeError` (`'int' object has no attribute 'upper'`).
- Nothing else raises, and the function does not log.

## 4. Returns and guarantees
- On a hit it returns **the same object** stored in `self.materials`; it does not make a copy.
  - Example: `db.get_material("bk7") is db.materials["BK7"]` is `True`.
- On a miss it returns `None`. It never falls back to BK7; that fallback exists only in `get_refractive_index`.
- Built-in keys, loaded before any JSON file: `BK7`, `N-BK7`, `SF11`, `F2`, `FUSEDSILICA`, `S-LAH79`, `E-FD60`.
- Useful built-in field values:

  | Key | `nd` | `vd` | `catalog` |
  |---|---|---|---|
  | BK7 | 1.5168 | 64.17 | "Schott" |
  | N-BK7 | 1.5168 | 64.17 | "Schott" |
  | SF11 | 1.78472 | 25.68 | "Schott" |
  | F2 | 1.62004 | 36.37 | "Schott" |
  | FUSEDSILICA | 1.45846 | 67.82 | "Schott" |
  | S-LAH79 | 2.00330 | 28.27 | "Ohara" |
  | E-FD60 | 1.52249 | 59.48 | "Hoya" |

- The returned object's `.name` equals the key for all built-ins (e.g. `"FUSEDSILICA"`, `"N-BK7"`).

## 5. Side effects
- It may populate `self._alias_keys`; this happens only when the exact lookup misses.
- It does not change `materials` or `_index_cache`, does no file I/O, and does not log.

## 6. Branches and edge cases
- **Exact hit:**
  - `"BK7"`, `"bk7"` and `"n-bk7"` → the BK7 / N-BK7 objects.
  - When the exact lookup hits, the alias table is not built: `db._alias_keys` stays `None` on a fresh db.
- **Alias hit:**
  - `"Fused Silica"`, `"fused-silica"`, `"fused_silica"` → FUSEDSILICA.
  - `"NBK7"`, `"n bk7"` → N-BK7.
  - `"slah79"`, `"S LAH 79"` → S-LAH79.
  - `"e.fd.60"` → E-FD60.
- **Miss:** `"Unobtainium"`, `""` and `"   "` → `None`. All built-in keys normalise to non-empty strings.
- **First key wins:**
  ```python
  db.add_material(MaterialProperties(name="BK-7", catalog="Custom", nd=1.6, vd=50), save=False)
  ```
  - The new material is stored under key `"BK-7"`.
  - `get_material("bk-7")` is the new object (exact hit).
  - `get_material("bk 7")` and `get_material("BK_7")` return the **built-in BK7** (nd 1.5168), because the alias `"BK7"` → `"BK7"` was registered first.
- **New material reachable by alias after `add_material`** (the cache is cleared):
  - Add a material named `"My Glass"` (key `"MY GLASS"`).
  - Then `get_material("my-glass")` and `get_material("MYGLASS")` return it.
- **Stale alias table after direct dict mutation** (a quirk of current behaviour):
  - If the table was already built (e.g. after `get_material("fused silica")`), then `db.materials["MY GLASS"] = mat` without `clear_cache()` leaves the table stale.
  - `get_material("MYGLASS")` → `None`, while `get_material("my glass")` → `mat` (exact hit).
  - After `db.clear_cache()`, `get_material("MYGLASS")` → `mat`.
- **Alias pointing to a deleted key:**
  - Build the table, then `del db.materials["F2"]` without `clear_cache()`.
  - `get_material("F 2")` → `None`, because the final `self.materials.get(key)` misses.
- **Non-uppercase key inserted directly** (`db.materials["abc"] = m`, then `clear_cache()`):
  - `get_material("abc")` misses the exact step (it looks for `"ABC"`).
  - It then hits via the alias `"ABC"` → `"abc"`, and returns `m`.

## 7. Rules from docstrings/comments
- "The exact uppercased key wins, then any spacing/punctuation variant: "Fused Silica", "fused-silica" and "FUSEDSILICA" all name the same glass."
- Alias table: "The first catalog key wins, so the built-in glasses keep their names when an imported catalog repeats one under different punctuation."
- `has_material(name)` is exactly `get_material(name) is not None`.

## 8. Testing hints
- **Isolate from the user's real catalog.** Always use `db = MaterialDatabase(str(tmp_path / "materials.json"))`.
  - The file does not exist, so only built-ins load.
  - Do not pass `""` as the path: it falls back to the default user path.
- Use `add_material(..., save=False)` to avoid writing files.
  - `save=True` with a tmp path is also safe, and writes JSON there.
- **Catalogs loaded from JSON:** the keys are `name.upper()` of the JSON *dict key*, not the material's `name` field.
  - Write `{"my glass": {...full MaterialProperties dict...}}`; it then loads as key `"MY GLASS"`.
- Build `MaterialProperties` with the required args `name`, `catalog`, `nd`, `vd`; everything else has defaults.
- **Prefer checks that pin behaviour down:**
  - identity checks (`is db.materials[...]`);
  - the `.name` / `.nd` of the result;
  - `None` for misses;
  - the first-key-wins example above.
- **Do not assume:**
  - that the result is a copy (it is the stored object);
  - that unknown names give BK7 (they give `None`);
  - that `get_material(None)` returns `None` (it raises `AttributeError`).
- Avoid the singleton `get_material_database()`: it reads the user's real data path.

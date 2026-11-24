# Notes: `MaterialDatabase.clear_cache(self)`

## 1. Purpose
- Empties **this instance's** refractive-index cache (`self._index_cache.clear()`).
- Drops the name-alias map (`self._alias_keys = None`) so it is rebuilt on the next lookup miss.
- Call it after mutating `db.materials` directly.

## 2. Inputs
- No parameters besides `self`. No validation is done.

## 3. Raises
- Nothing. It is safe on an empty cache and safe to call repeatedly.

## 4. Returns and guarantees
- Returns `None`.
- Afterwards, `db._index_cache == {}` and `db._alias_keys is None`.
- `db.materials` is not touched. The same dict and the same `MaterialProperties` objects remain.
- The effect is per instance: other `MaterialDatabase` objects keep their caches. This is the point of the docstring; it used to be a class-wide `lru_cache.cache_clear()`.

## 5. Side effects
- Only the two private attributes above change.
- No file I/O and no global state.

Callers inside the module:
- `add_material` calls `clear_cache()` after storing the material, before the optional save.
- `_load_database` calls it at the end of a successful load.

## 6. Branches and edge cases
- There is a single code path with no branching.

How the cache behaves, which is what makes `clear_cache` observable:
- `get_refractive_index(name, wl, T)` uses the key `(name, wl, T)` with the **exact** name string, so `"BK7"` and `"bk7"` are separate entries.
- On a hit (stored value not `None`), it returns the stored value without recomputing, even if `materials` changed meanwhile.
- On a miss, it computes the value and stores it. It also sets `_alias_keys = None` on every miss.

How the alias map behaves:
- `get_material(name)` first tries `materials[name.upper()]`.
- If that misses, it uses `_normalised_keys()`. That map is built once, from the catalog keys at that moment, as an alphanumeric-uppercase key mapped to the real key. It is cached until reset.
- So a key added directly to `materials` after the map was built is invisible to the normalised lookup until `clear_cache()` or a `get_refractive_index` miss.

## 7. Rules from comments/docstrings
- "Call clear_cache() after mutating `materials` directly." (from `get_refractive_index`)
- "one instance adding a material wiped every other instance's cache" was the old bug. Clearing must not affect other instances.

## 8. Testing hints
Build an isolated database so the real user catalog is never read or written. A non-existent file means only the built-ins load (BK7, N-BK7, SF11, F2, FUSEDSILICA, S-LAH79, E-FD60):
```python
from openlens.material_database import MaterialDatabase, MaterialProperties
db = MaterialDatabase(db_path=str(tmp_path / "m.json"))
```

**Deterministic way to make stale vs fresh values visible:** replace BK7 with a material whose B coefficients are all zero. Then `n_sq` stays `1.0`, and with no temperature correction at T=20.0 the result is exactly `1.0`.
```python
before = db.get_refractive_index("BK7", 587.6)        # ≈1.5168, use abs tol 1e-3
db.materials["BK7"] = MaterialProperties(name="BK7", catalog="X", nd=1.6, vd=40.0)
assert db.get_refractive_index("BK7", 587.6) == before   # stale cache hit
db.clear_cache()
assert db.get_refractive_index("BK7", 587.6) == 1.0      # recomputed
```

**Per-instance isolation:**
- Create `db1` and `db2` with separate tmp paths and warm `"BK7"@587.6` on both.
- Replace `materials["BK7"]` on both with the zero-B material.
- Call `db1.clear_cache()` only.
- Expect `db1` to return `1.0` and `db2` to still return about `1.5168`.

**Alias reset.** Do not call `get_refractive_index` between these steps, because a miss there also resets aliases:
```python
assert db.get_material("foo bar") is None             # builds alias map
m = MaterialProperties(name="FOO_BAR", catalog="X", nd=1.5, vd=50.0)
db.materials["FOO_BAR"] = m
assert db.get_material("foo bar") is None             # stale alias map
db.clear_cache()
assert db.get_material("foo bar") is m
```

Other checks:
- **Private state:** `db._index_cache == {}` and `db._alias_keys is None` after the call. This is acceptable to assert, but behavioural tests like the ones above are stronger.
- **Materials untouched:** `list_materials()` and the identity of `db.materials` objects are unchanged after `clear_cache()`.
- `db.clear_cache()` returns `None`, and calling it twice in a row does not raise.

Pitfalls:
- Do not use `MaterialDatabase()` without `db_path`; it reads the real per-user file.
- Do not use `get_material_database()`, because it returns a shared singleton.
- Use `add_material(..., save=False)` if you add materials; otherwise it writes the JSON file. It already calls `clear_cache` itself, so it cannot demonstrate staleness.
- Compare real glass indices with a tolerance, not equality.

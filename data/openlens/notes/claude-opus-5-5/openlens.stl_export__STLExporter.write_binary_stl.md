# Notes: `openlens.stl_export.STLExporter.write_binary_stl(self, filename)`

## 1. Purpose
- Serialises `self.triangles` into a **binary STL** file at `filename`.
- Writes through `atomic_write_bytes` (from `openlens.atomic`), so the file is only replaced when the write succeeds.
- Returns `None`.

## 2. Inputs
- `filename`: a path, passed straight to `openlens.atomic.atomic_write_bytes`.
  - That function's signature takes `path: Any`, so `str` and `pathlib.Path` should both work. Its internals are not shown here.
  - There is no validation in this method.
- `self.triangles`: an iterable of triangles.
  - Each triangle must unpack into exactly 3 points: `p1, p2, p3 = triangle`.
  - Each point must support indexing `[0]`, `[1]`, `[2]` with numbers that `struct` can pack as float32.
  - Tuples, lists and numpy floats all work.
  - It starts as `[]` after `STLExporter()`. You can assign it directly.

## 3. Raises
No exception is raised explicitly. These come from Python or `struct`:
- **Triangle with the wrong number of points** → `ValueError` from unpacking.
  - `[((0,0,0),(1,0,0))]` → `"not enough values to unpack (expected 3, got 2)"`.
  - 4 points → `"too many values to unpack (expected 3)"`.
- **Point with fewer than 3 coordinates** → `IndexError`.
  - The first failure is inside `calculate_normal`, which runs before packing.
- **Non-numeric coordinate** (e.g. a string) → `TypeError` raised by the subtraction in `calculate_normal`.
- **Finite coordinate beyond the float32 range** (e.g. `1e39`) → `OverflowError` from `struct.pack('<fff', ...)`.
  - `float('inf')` packs without error.
- **Degenerate (zero-area) triangle**: behaviour depends on `Vector3.normalize()` in `openlens.vector3`, which is not visible here. Do not test it.
- **Nonexistent parent directory or other I/O errors**: behaviour depends on `atomic_write_bytes`. Do not assert it.
- **Failure part-way through**: the docs say the target is replaced "only on success". The exception propagates; whether an old file survives is decided by the atomic module.

## 4. Returns and guarantees
The method returns `None`. The file layout is exact, all little-endian:

| Section | Size | Content |
|---|---|---|
| Header | 80 bytes | `b"openlens STL export"` (19 bytes) followed by 61 ASCII spaces `b" "`. No NUL padding. |
| Count | 4 bytes | `struct.pack("<I", len(self.triangles))` |
| Each triangle | 50 bytes | see below |

Each triangle record, in order:
1. Normal: `"<fff"`.
2. `p1`: `"<fff"`.
3. `p2`: `"<fff"`.
4. `p3`: `"<fff"`.
5. Attribute: `"<H"` with value `0`.

Further guarantees:
- **Total file size** is `84 + 50*N`. An empty list gives an 84-byte file with count 0.
- **Normal** is `calculate_normal(p1,p2,p3)`, i.e. `normalize((p2-p1) × (p3-p1))`.
  - Example: `((0,0,0),(1,0,0),(0,1,0))` → normal `(0,0,1)`.
  - Reversing the vertex order → `(0,0,-1)`.
- **Normals are recomputed** from the vertices on every write. No stored normal is used.
- **Triangle order** in the file equals the order of `self.triangles`.
- **Vertex order** is kept exactly as given (p1, p2, p3).
- **Precision**: values are stored as float32, so they read back float32-rounded.

## 5. Side effects
- Creates or overwrites `filename`.
- Does not modify `self.triangles`.
- No other state is changed.

## 6. Branches and edge cases
- **Empty `triangles`**: file is the header plus `b"\x00\x00\x00\x00"`, 84 bytes total.
- **N triangles**: the loop writes N records; the count field equals N.
- **Non-unit or scaled triangle**: the normal is still unit length.
  - Example: `((0,0,0),(2,0,0),(0,3,0))` → `(0,0,1)`.
- **Triangle in the XZ plane**: `((0,0,0),(0,0,1),(1,0,0))` → v1=(0,0,1), v2=(1,0,0), cross=(0,1,0).

## 7. Rules from comments and docstrings
- The docstring reads "Write triangles to binary STL file".
- Atomic replacement is the documented intent of `atomic_write_bytes` ("replace path only on success").

## 8. Testing hints
Write a fixed triangle list to `tmp_path` and parse the bytes back:

```python
ex = STLExporter()
ex.triangles = [((0.0,0.0,0.0),(1.0,0.0,0.0),(0.0,1.0,0.0))]
p = tmp_path/"t.stl"; ex.write_binary_stl(str(p))
data = p.read_bytes()
assert data[:80] == b"openlens STL export" + b" "*61
assert struct.unpack("<I", data[80:84])[0] == 1
rec = struct.unpack("<12fH", data[84:134])   # normal, p1, p2, p3, attr
```

Things worth asserting:
- `len(data) == 84 + 50*N`.
- `attr == 0`.
- The normal equals `(0,0,1)`; for the reversed winding it is `(0,0,-1)`.
- Vertices come back in the same order.
- Multiple triangles appear in list order.

Pitfalls:
- **Float32 precision**: use exactly representable values (0.5, 1.0, -2.25), or compare with `pytest.approx(rel=1e-6)`. `0.1` reads back as `0.10000000149011612`.
- **Normals with irrational components** (e.g. 1/√2) also need `approx`.
- **Return value**: the method returns `None`. The triangle count is returned by `export_lens_to_stl` and `export_lens_stl`, not by this method.
- **Overwrite**: writing twice to the same path leaves only the second content, so the size reflects the second list.
- **Integration check**: `n = export_lens_stl(Lens(), str(p))` → the count field in the file equals `n`, and the size is `84 + 50*n`.
- **Header**: do not assume the header is NUL-padded or that it starts with `b"solid"`.

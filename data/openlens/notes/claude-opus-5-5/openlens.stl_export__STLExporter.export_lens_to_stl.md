# Notes: `STLExporter.export_lens_to_stl(r1, r2, thickness, diameter, filename, resolution=50, lens=None) -> int`

## 1. Purpose
- Builds a closed triangle mesh of one lens with Z as the optical axis: front cap at z=0, back cap at z=`thickness`, plus the rim.
- Writes the mesh as a binary STL file and returns the number of triangles.

## 2. Inputs
- `r1`, `r2`: signed radii in mm. They are used only when `lens is None`.
  - A radius counts as **flat** when `abs(R) < 1e-10` or `abs(R) > 1e10`. So `0`, `0.0` and `float('inf')` are flat.
  - Positive R gives sag `+s`. Negative R gives sag `-s`, where `s = y²/(|R| + sqrt(R²-y²))`.
- `thickness`: added as the z offset of the back cap. Not validated.
- `diameter`: rim radius is `h = diameter/2`. Used by the spherical profiles, the flat caps and the legacy edge.
- `filename`: written through `atomic_write_bytes` (in `openlens.atomic`, not shown). Per its docstring, the file is replaced only on success.
- `resolution`: number of radial steps.
  - A curved profile has `resolution+1` points, at `y = i/resolution*h`.
  - The angular segment count is `S = resolution*2`.
  - No validation is done. `resolution=0` raises `ZeroDivisionError`, for curved and for flat surfaces.
- `lens` (optional): a model `Lens`.
  - Surfaces come from `lens.get_effective_radius_1/2()` and `LensGeometry.surface_profile(lens, s, num_points=resolution)`. The `(z, r)` pairs are swapped to `(r, z)`.
  - The edge comes from `LensGeometry.edge_profile_points(lens)` and is revolved with `generate_profile_triangles`.
  - In this mode `r1`/`r2` are **ignored**. `thickness` and `diameter` are still used: `thickness` for the back offset, `diameter` for flat caps.
  - A non-finite or flat effective radius gives a flat cap.
  - `LensGeometry` lives in another module, so triangle counts in this mode are not derivable from this file.

## 3. Raises (legacy path, `lens=None`)
- **`ValueError`** when `R² - y² < 0` at some sample point, i.e. `|R| < diameter/2`.
  - It is raised at the first offending `i`.
  - Exact template: `f"Surface of radius {radius} mm does not extend to {y:.4f} mm: |R| must be at least half the diameter ({h:.4f} mm). Widen the radius or reduce the diameter."`
  - Example: `export_lens_to_stl(10, -50, 5, 25, p, resolution=50)` gives `"Surface of radius 10 mm does not extend to 10.2500 mm: |R| must be at least half the diameter (12.5000 mm). Widen the radius or reduce the diameter."`
  - The samples are `y = 0.25*i`, and `i=41` is the first with `y > 10`.
  - Passing `10.0` prints `10.0`; passing `-10` prints `-10`.
- **Check order**: the front (`r1`) is checked before the back (`r2`).
  - If only `r2` is bad, `self.triangles` already holds the front triangles when the error is raised.
  - In both cases `write_binary_stl` is never reached, so no file is written.
- `|R| == h` exactly is allowed, because the radicand at the rim is 0.
- **`ZeroDivisionError`** when `resolution=0`.

## 4. Returns and guarantees
- Returns an `int` equal to `len(self.triangles)`, which also equals the triangle count stored in the file.
- `self.triangles` is reset to `[]` at the start of every call, so calls do not accumulate. After the call it keeps the mesh.
- Triangle counts in the legacy path, with N = resolution:
  - Curved cap: `2N*(2N-1)`. The center ring gives 1 triangle per segment; the other rings give 2.
  - Flat cap: `2N`.
  - Edge: `4N`.
  - Both curved: `8N²`. N=50 gives **20000**; N=2 gives 32; N=1 gives 8.
  - Both flat: `8N`. N=50 gives **400**; N=4 gives 32.
  - One curved, one flat: `4N²+4N`. N=50 gives **10200**; N=2 gives 24.
- File layout:
  - 80-byte header: `b"openlens STL export" + b" "*61`.
  - `<I` triangle count.
  - Then each triangle takes 50 bytes: normal `<fff`, three vertices `<fff`, and `<H` 0.
  - Total file size is `84 + 50*n`.
- Geometry for `r1=50, r2=-50, t=5, D=25`:
  - Front vertex is at (0,0,0). Front rim is at z=1.58771. The sag is `156.25/(50+sqrt(2343.75))`.
  - Back vertex is at (0,0,5). Back rim is at z=`5-1.58771 = 3.41229`. The source comment says 3.412, not 6.588.
  - Overall z range is [0, 5]. The maximum |x| is 12.5, at angle 0.
  - The rim ring vertices all have `x²+y² = 12.5²`.
- Legacy edge band:
  - It runs from `z_front = profile1[-1][1]`, or 0 if the front is flat.
  - It ends at `z_back = profile2[-1][1] + thickness`, or `thickness` if the back is flat.
- Normals point outward:
  - The front cap faces -Z. Its center-fan normal z is negative.
  - The back cap faces +Z.
  - Edge normals point radially outward.
  - Signed volume `Σ p1·(p2×p3)/6` is positive and close to the solid volume.
  - Both-flat case: the volume is `π(D/2)²·t`, approximated by a 2N-gon.
- Every vertex of segment `S-1` that sits at angle `2π` uses angle exactly 0, so the mesh is watertight. Vertex `(h, 0.0, z)` appears exactly.

## 5. Side effects
- Writes or replaces `filename`.
- Mutates `self.triangles`.
- Inputs are not mutated.

## 6. Branches and edge cases
- `lens is None` with both radii curved: spherical profiles plus a cylinder, `8N²` triangles.
- `r1=0` or `inf`: the front is a flat fan at z=0. `r2=0` gives a flat fan at z=`thickness`.
- `lens` given: surfaces and edge come from `LensGeometry`. Only assert `n > 0`, `size == 84+50n`, and that the header count equals the return value.

## 7. Rules from comments
- `is_front` does nothing in the profile calculation; only the sign of R matters.
- The back cap uses `is_front=False`, which reverses the winding.
- The last segment wraps to angle 0 exactly; `sin(2π)≠0` would otherwise leave a seam.
- A radius too small for the diameter must raise instead of silently producing a flat disc.

## 8. Testing hints
- Use pytest `tmp_path` for the output file.
- Parse the file:
  ```python
  b = p.read_bytes()
  n = struct.unpack_from('<I', b, 80)[0]
  # triangle k: struct.unpack_from('<12fH', b, 84 + 50*k)
  ```
- The values are float32, so compare with `pytest.approx(..., abs=1e-4)`.
- Assert exact counts with small resolutions, e.g. N=2:
  - (50, -50) gives 32.
  - (0, 0) gives 16.
  - (50, 0) gives 24.
- Check the extent of z: the minimum is 0 and the maximum equals `thickness` for biconvex. Check the rim z values 1.5877 and 3.4123 (with D=25).
- Check the sign of the front normal on triangle 0: its z component is less than 0.
- Check that no file exists after a `ValueError`.
- Check that two consecutive calls return the same count.
- Pitfalls:
  - Do not use `thickness` equal to the rim-sag difference, or a zero thickness with flat caps. That makes the edge triangles degenerate, and normalizing a zero vector depends on `Vector3` (not shown).
  - Do not assume that `r1`/`r2` matter when `lens` is passed.
  - Exact float formatting in the error message: `{radius}` uses `str()`, while `y` and `h` use `.4f`.

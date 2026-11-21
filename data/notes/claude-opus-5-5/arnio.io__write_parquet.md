# Notes: `arnio.io.write_parquet`

## 1. Purpose
- Writes an `ArFrame` to a Parquet file.
- It validates its arguments, converts with `arnio.convert.to_pandas(frame)`, then calls `df.to_parquet(path, engine="pyarrow", compression=..., index=False[, row_group_size=...])`.
- Returns `None`.

## 2. Inputs
- **`frame`**: must be an `ArFrame` instance. A pandas DataFrame is rejected.
- **`path`**: `str`, `bytes` or `os.PathLike`.
  - It is normalised with `os.fsdecode(os.fspath(path))`, so `b"out.parquet"` and `pathlib.Path` both work.
  - The extension check is case-insensitive: the name must end with `.parquet` or `.pq`. `"X.PARQUET"` and `"x.Pq"` are OK.
- **`compression`** (keyword-only, default `"snappy"`):
  - Must be a `str` from exactly `{"snappy","gzip","brotli","zstd","none"}`.
  - The check is case-sensitive, so `"SNAPPY"` is rejected.
  - `"none"` is passed to pandas as `compression=None`.
- **`row_group_size`** (default `None`):
  - When given, it must be an `int` that is not a `bool`, and it must be `> 0`.
  - It is forwarded to pandas only when not `None`.
- **`preserve_attrs`** (default `True`): must be a real `bool`. `1`, `0` and `None` are rejected.

## 3. Raises (in this exact order)
1. `TypeError("frame must be an ArFrame")`
   - Trigger: `write_parquet(pd.DataFrame(), "a.parquet")`.
   - This check comes before the path check: `write_parquet("x", 123)` raises the frame error.
2. `TypeError(f"path must be a string, bytes, or os.PathLike object, got {type(path).__name__!r}")`
   - Trigger: `path=123` gives `...got 'int'`.
3. `ValueError(f"Unsupported file format: {path}. write_parquet only supports .parquet and .pq files.")`
   - Trigger: `path="out.csv"`.
   - `{path}` is the decoded string, not its repr.
4. `TypeError("compression must be a string")`
   - Trigger: `compression=None`.
5. `ValueError(f"Unknown compression codec: {compression!r}. Valid options are: ['brotli', 'gzip', 'none', 'snappy', 'zstd']")`
   - Trigger: `compression="lz4"` gives `'lz4'` in the message.
6. `TypeError("row_group_size must be an integer")`
   - Triggers: `row_group_size=True`, `2.0`, `"5"`.
7. `ValueError("row_group_size must be a positive integer")`
   - Triggers: `0`, `-1`.
8. `TypeError("preserve_attrs must be a bool")`
   - Trigger: `preserve_attrs=1`.
9. `ImportError("pyarrow is required for Parquet export. Install it with: pip install arnio[parquet]")`
   - Raised only if pyarrow is missing. Not testable normally.
10. `ValueError(f"Cannot write a zero-column ArFrame with {rows} rows to Parquet: the current export path cannot preserve row count without columns.")`
    - Raised when `frame.shape[1] == 0 and frame.shape[0] > 0`.
    - It comes after all argument validation.
    - A `(0, 0)` frame does NOT hit this check.
11. `TypeError` when `preserve_attrs=True` and `json.dumps(df.attrs)` fails.
    - The message starts with `"write_parquet() requires that DataFrame.attrs contain only JSON-serializable values (str, int, float, bool, list, dict, None). Serialization failed: "`.
    - It ends with `". To export without metadata, pass preserve_attrs=False."`.
    - The original exception is chained as `__cause__`.
12. Errors from pandas or pyarrow while writing are NOT wrapped. Example: the parent directory does not exist.

All checks 1–10 run before anything touches disk. A failing call never creates the file.

## 4. Returns and guarantees
- Returns `None`.
- The file is written with `index=False`. No `__index_level_0__` or extra index column appears; the columns are exactly `frame.columns`, in the same order.
- The compression codec is recorded in the Parquet metadata:
  - `"none"` gives `UNCOMPRESSED`.
  - Otherwise you get `SNAPPY`, `GZIP`, `BROTLI` or `ZSTD`.
- `row_group_size=k` with n rows gives `ceil(n/k)` row groups. Example: 10 rows with `k=3` gives 4.
- Attrs handling:
  - If `df.attrs` is non-empty and `preserve_attrs=False`, `df.attrs` is set to `{}` before writing, so no attrs are written.
  - If `preserve_attrs=True`, attrs pass through to pandas. Whether pandas stores them in Parquet metadata depends on the pandas version (≥2.1 does). Do not assert attrs round-trip strictly.
- Whether `ArFrame` attrs reach `df.attrs` depends on `to_pandas` in `convert.py`, which is not shown here.

## 5. Side effects
- Creates or overwrites the file at `path`. This is done by pandas, not atomically.
- `df.attrs = {}` only affects the temporary pandas object. No mutation of `frame` is visible in this code; `frame.shape` and `frame.columns` stay unchanged.
- No global state is changed.

## 6. Branches and edge cases
- **Default call**: `write_parquet(ar.from_pandas(pd.DataFrame({"a":[1,2,3],"b":["x","y",None]})), tmp/"o.parquet")` writes the file. `pd.read_parquet` returns 3 rows and columns `["a","b"]`.
- **`compression="none"`**: the file is still written, and the column metadata compression is `"UNCOMPRESSED"`.
- **`row_group_size`**:
  - `None`: the key is omitted and pyarrow uses its default (1 row group for small data).
  - `int`: pyarrow splits the rows into groups of that size.
- **Zero columns with rows**: raises the ValueError from item 10.
  - A frame of that shape could be built with `ar.from_pandas(pd.DataFrame(index=range(3)))`. Whether it really has shape `(3, 0)` depends on the C++ core.
  - So guard the test: `if f.shape != (3, 0): pytest.skip(...)`.
- **Zero rows with ≥1 column**: allowed, and an empty table is written.
- **`df.attrs` empty**: the attrs branch is skipped entirely. So even unserialisable values cannot raise when attrs are empty.
- **What `json.dumps` accepts**:
  - `float("nan")` passes, because `allow_nan` defaults to True.
  - A `set`, an `object()` or tuple keys raise TypeError, but only if attrs actually reach `df.attrs`.

## 7. Rules from docstrings and comments
- With `preserve_attrs=True`, all attrs must be JSON-serializable or a clear TypeError is raised. `preserve_attrs=False` silently drops them.
- The zero-column check exists because the export path cannot preserve the row count without columns.

## 8. Testing hints
- Use the pytest `tmp_path` fixture. Build frames with `ar.from_pandas(pd.DataFrame(...))`.
- Read files back with `pyarrow.parquet`:
  - `pq.ParquetFile(p).metadata.num_row_groups`
  - `.metadata.row_group(0).column(0).compression`, which is an uppercase string such as `"SNAPPY"`.
  - `pq.read_table(p).column_names`
  - `pd.read_parquet(p)`
- You can also round-trip with `ar.read_parquet(p)` and compare `.columns`, `.shape` and `ar.to_pandas(...)["a"].tolist()`.
- Compare values with `.tolist()`. Avoid exact dtype checks on string columns, because object vs. string dtype can differ across the round-trip. For nulls, check `pd.isna` rather than `== None`.
- Use `pytest.raises(..., match=re.escape(msg))` with the exact messages above.
  - Remember the f-string formatting: compression uses `!r`, the path does not.
- Check-order tests are good regression catchers:
  - Bad extension plus bad compression should report the extension.
  - `row_group_size=True` should raise TypeError, not ValueError.
  - `compression="Snappy"` should raise ValueError.
- After a validation error, assert that `not p.exists()`.
- Do not rely on attrs-related behaviour unless you first confirm that `ar.to_pandas(frame).attrs` is non-empty. Otherwise skip the test.

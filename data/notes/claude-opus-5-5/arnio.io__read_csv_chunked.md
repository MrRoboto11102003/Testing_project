# Notes: `arnio.read_csv_chunked`

## 1. Purpose
- Reads a CSV in chunks and yields one `ArFrame` per chunk, with at most `chunksize` data rows each.
- **It is a generator function.** Calling it runs no code at all. All validation, file access and errors happen on the first `next()` or `list()`.

## 2. Inputs
- **`path`**
  - Accepted: `str`, `os.PathLike`, or a text file-like object (anything with a callable `.read()` that returns `str`).
  - A string with scheme `http`/`https` is fetched to a temporary UTF-8 file.
  - A path ending in `.gz` (case-insensitive) is gunzipped to a temporary **`.csv`** file. So `x.tsv.gz` does **not** get automatic tab detection.
  - File-like objects are copied to a temporary `.csv` file. For these, `encoding` is ignored and the data is treated as UTF-8.
- **Extension check** (only for `str`/PathLike input, checked on the original path, case-insensitive): it must end in `.csv`, `.txt`, `.tsv` or `.gz`.
- **`chunksize=10_000`**: must be an `int` and not a `bool`; must be `> 0`.
- **`delimiter=None`**: `None` becomes `"\t"` if the materialized path ends with `.tsv`, otherwise `","`.
  - It must then be a single character.
  - It must not be `\n`, `\r` or `"`.
  - It must not be a control character, except tab, which is allowed.
- **`has_header=True`, `trim_headers=True`**: must be real `bool` values. `1` is rejected.
- **`usecols`**: a non-empty Sequence of unique `str` values. A tuple is accepted. A bare `str` is rejected.
- **`nrows`**: an `int`, not a `bool`, `>= 0`. There is **no** short-circuit for `nrows=0`; the value is passed to C++, whose behaviour is unknown.
- **`skip_rows=0` / `skiprows=None`**: the two are aliases. Both skip *data rows after the header*.
  - If `skiprows` is given it replaces `skip_rows`.
  - They conflict only when `skip_rows != 0` and `skip_rows != skiprows`.
- **`encoding="utf-8"`**:
  - Must be a `str`.
  - `"utf-8"`, `"UTF8"` and `"utf_8"` all count as UTF-8.
  - Any other encoding is transcoded to a temporary file using `errors="strict"`.
  - There is **no `encoding_errors` parameter**; passing one gives Python's normal `TypeError`.
- **`decimal_separator="."`, `thousands_separator=None`**: see the Raises table.
- **`null_values`**: must be a `list` of `str` (a tuple is rejected).
- **`dtype`**: a `dict` mapping `str` to one of `"string"`, `"int64"`, `"float64"`, `"bool"`.
- **`mode`**: `"strict"` or `"permissive"`.
- **`on_bad_lines`**: `"error"`, `"warn"` or `"skip"`.

## 3. Raises (in check order; every one raises on first `next()`)
1. **Input-type and URL problems**, raised while the input is materialized:
   - `TypeError("read_csv_chunked expected a filesystem path or text file-like object")`, e.g. for `path=123`.
   - `ValueError("Cloud scheme 's3' is not yet supported by arnio. Install the optional extra when available: pip install \"arnio[s3]\"")` for `"s3://b/x.csv"`. The same pattern applies to `gs`, `gcs`, `az`, `abfs` and `abfss`.
   - `TypeError("read_csv file-like objects must return text, not bytes")` for `io.BytesIO(b"a\n1\n")`.
   - A missing `.gz` file gives gzip's own `FileNotFoundError`.
2. **Bad extension**: `ValueError(f"Unsupported file format: {path}. Only .csv, .txt, .tsv, and compressed .csv.gz are supported.")`.
   - Raised even if the file does not exist, e.g. `"data.dat"`.
3. **Encoding not a string**: `TypeError(f"encoding must be a string, got {type(encoding).__name__!r}")`, e.g. `"... got 'int'"`.
4. **Empty file** (0 bytes): `CsvReadError(f"CSV file is empty: {path!r}")`.
   - An empty `StringIO("")` also raises this, with the temporary file's path in the message.
   - A missing file is *not* reported here.
5. **`decimal_separator`**:
   - `TypeError("decimal_separator must be a string")`.
   - `ValueError("decimal_separator must be a single character")`.
   - `ValueError("decimal_separator must be a single non-alphanumeric character")` for `"a"`, `"1"`, `'"'`, `"\n"`.
   - `ValueError("Invalid decimal_separator: '+' and '-' are not allowed")`.
6. **`thousands_separator`**:
   - `TypeError("thousands_separator must be a string or None")`.
   - `ValueError` with the same wording as above, with the prefix `thousands_separator` instead of `decimal_separator`.
   - `ValueError("thousands_separator must differ from decimal_separator")` for `thousands_separator="."`.
7. **`delimiter`**:
   - `TypeError("delimiter must be a string")`.
   - `ValueError("delimiter must be a single character; delimiter must be exactly one character")`.
   - `"delimiter must not be a newline character"`.
   - `"delimiter must not be the CSV quote character"`.
   - `"delimiter must not be a control character"`, e.g. `"\x01"`.
8. **`mode`**: `TypeError("mode must be a string")`; `ValueError("mode must be either 'strict' or 'permissive'")`.
9. **`chunksize`**: `TypeError("chunksize must be an integer")` (also for `True` or `2.0`); `ValueError("chunksize must be a positive integer")` for `0` or `-1`.
10. **`skiprows`**:
    - `TypeError("skiprows must be an integer")`.
    - `ValueError("skiprows must be non-negative")`.
    - `ValueError("Conflicting values: skiprows=3 and skip_rows=2. Pass only one of them.")` for `skip_rows=2, skiprows=3`.
11. **`skip_rows`**: `TypeError("skip_rows must be an integer")`; `ValueError("skip_rows must be non-negative")`.
12. **`on_bad_lines`**: `TypeError("on_bad_lines must be a string")`; `ValueError("on_bad_lines must be either 'error', 'warn', 'skip'")`.
13. **`has_header` / `trim_headers`**: `TypeError("has_header must be True or False, got int: 1")`. The template is `f"{name} must be True or False, got {type(value).__name__}: {value!r}"`.
14. **`null_values`**:
    - `TypeError("null_values must be a list of strings, not a bare string")`.
    - `TypeError("null_values must be a list of strings")` (for a tuple).
    - `TypeError("null_values must contain only strings")`.
15. **`dtype`**:
    - `TypeError("dtype must be a dictionary mapping column names to dtype strings")`.
    - `TypeError("dtype column names must be strings")`.
    - `TypeError("dtype values must be strings")`.
    - `ValueError("Unsupported dtype 'int32'. Expected one of: ['bool', 'float64', 'int64', 'string']")`.
16. **`usecols`**:
    - `TypeError("usecols must be a sequence of column names, not a string")`.
    - `TypeError("usecols must be a sequence of strings")`.
    - `ValueError("usecols must not be empty")`.
    - `TypeError("usecols must contain only strings")`.
    - `ValueError("usecols must not contain duplicate column names")`.
17. **`nrows`**: `TypeError("nrows must be an integer")`; `ValueError("nrows must be non-negative")`.
18. **Transcoding** (non-UTF-8 encodings only):
    - `ValueError(f"Unknown encoding: {encoding}")`, with **no** quotes around the name, e.g. `encoding="nope"`.
    - `CsvReadError("Could not decode '<path>' using encoding 'ascii': invalid byte(s) at position N (byte value: 0x..). Try a different encoding or use encoding_errors='replace'.")`.
    - A missing file with a non-UTF-8 encoding gives `CsvReadError("Could not access CSV file '<path>': <strerror>")`.
19. **Native parse errors**: any `RuntimeError` from C++ becomes `CsvReadError(str(e))`, with the message unchanged.
    - Unlike `read_csv`, there is **no** "Malformed CSV: ..." enrichment.
    - With the default UTF-8 encoding, a missing file is handled by C++ (exact exception and message not visible).

## 4. Returns and guarantees
- Yields `ArFrame` objects.
  - Each chunk has at most `chunksize` rows.
  - The total row count respects `nrows`.
  - Dropped bad rows count toward `nrows` (per the docstring).
- Column types are inferred from the first chunk and applied to all later chunks (per the docstring; done in C++).
- Concatenating the chunks should equal the whole file. For example, a 5-row file with `chunksize=2` gives row counts `[2, 2, 1]`.
- **Not done here (unlike `read_csv`):**
  - No NUL-byte rejection.
  - No delimiter-mismatch warning.

## 5. Side effects
- Temporary files are created for URL, gzip, file-like and non-UTF-8 inputs.
  - They are deleted when the generator finishes, or on validation error.
- `on_bad_lines="warn"` emits a `UserWarning` for each chunk that has bad rows:
  - Format: `"{n} malformed CSV row(s):\n  CSV row R has A fields; expected E"`.
  - At most 10 rows are listed, followed by `"  (+k more)"` if there are more.

## 6. Branches and edge cases
- If a chunk has 0 rows **and** `bad_rows` is non-empty, it is skipped only if a non-empty chunk was already yielded. Otherwise it is yielded.
- An explicit `delimiter=","` on a `.tsv` path is honoured.
- A `.tsv` file with `delimiter=None` uses tab.
- `skip_rows=2, skiprows=2` is accepted.
- `skip_rows=0, skiprows=3` gives an effective skip of 3.
- `pathlib.Path` input works, and the extension check applies to it.
- A `StringIO` input skips the extension check entirely.

## 7. Rules from comments and docstrings
- Explicit `delimiter` always wins over `.tsv` auto-detection.
- `skip_rows` and `skiprows` both skip data rows after the header.
- Validation runs on the decompressed temporary file, not on the `.gz` bytes.

## 8. Testing hints
- Build inputs with `p = tmp_path / "d.csv"; p.write_text("a,b\n1,x\n2,y\n3,z\n")`.
- Collect results with `chunks = list(ar.read_csv_chunked(p, chunksize=2))`.
- Check `[c.shape for c in chunks] == [(2, 2), (1, 2)]` and `ar.to_pandas(c)["a"].tolist()`.
- **Pitfall:** errors only appear when the generator is consumed. Write `with pytest.raises(ValueError, match="..."): next(ar.read_csv_chunked(p, chunksize=0))`. Calling the function without iterating never raises.
- `match=` is a regex. Use `re.escape` for messages containing `(`, `[`, `.` or `+`.
  - Avoid matching full `repr(path)` strings: on Windows the backslashes are doubled.
- To check ordering, combine two bad arguments: `decimal_separator="ab", chunksize=0` raises the decimal error. A file `x.dat` with `chunksize=0` raises the format error.
- **Avoid asserting:**
  - Results for header-only files.
  - `nrows=0`.
  - Results with `has_header=False` (the column names are decided in C++).
  - Exact native error texts.
- To test TSV: write tab-separated data to `d.tsv` and assert 2 columns.
- To test gzip: write the file with `gzip.open(tmp_path/"d.csv.gz", "wt")`.

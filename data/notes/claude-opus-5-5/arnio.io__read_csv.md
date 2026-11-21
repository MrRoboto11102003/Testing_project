# Notes: `arnio.io.read_csv`

## 1. Purpose
- Reads CSV text (local path, http(s) URL, `.gz` file, or text file-like object) into an `ArFrame`.
- Parsing is done by the C++ `_CsvReader`. The Python layer validates options, materialises inputs to a UTF-8 temp file, and wraps errors and warnings.

## 2. Inputs
All parameters after `path` are keyword-only.

- **`path`**
  - `str` / `os.PathLike`:
    - For `str` only, the URL scheme is checked (lower-cased).
    - Cloud schemes `s3, gs, gcs, az, abfs, abfss` → ValueError.
    - `http`/`https` are fetched via urllib (network; avoid in tests).
  - Paths ending `.gz` (case-insensitive) are gunzipped to a temp file with suffix **`.csv`**.
  - Any object with a callable `.read()` (e.g. `io.StringIO`) is copied to a temp `.csv`. Each `read()` must return `str`.
  - Any file extension is accepted (`.dat`, `.txt`, and so on).
- **`delimiter=None`**
  - `None` resolves to `"\t"` if the *materialised* path lower-cased ends with `.tsv`, otherwise `","`.
  - So `x.TSV` → tab. But `x.tsv.gz` → `","` and StringIO → `","`, because their temp files end `.csv`.
  - An explicit value always wins.
  - Validated rules:
    - Must be a `str` of length 1.
    - Not `\n` or `\r`.
    - Not `"`.
    - Not a control char (code point ≤0x1F except tab 0x09, or 0x7F).
- **`has_header=True`, `trim_headers=True`**: must be real `bool` (an `int` such as `1` is rejected).
- **`encoding="utf-8"`**
  - Must be `str`.
  - `"utf-8"`, `"UTF8"`, `"utf_8"` are treated as UTF-8 (lower-cased, `_`→`-`).
  - Any other value transcodes the file through a temp UTF-8 file.
  - For file-like/URL inputs, transcoding is skipped (they are already text).
- **`decimal_separator="."`**: one char, not alphanumeric, not `"`, `\n`, `\r`, `+`, `-`.
- **`thousands_separator=None`**: same rules as `decimal_separator`, and must differ from `decimal_separator`.
  - Not checked against `delimiter`.
- **`mode`**: `"strict"` | `"permissive"`.
- **`encoding_errors`**: `"strict"` | `"replace"` | `"ignore"`.
- **`on_bad_lines`**: `"error"` | `"warn"` | `"skip"`.
- **`null_values`**: `list` of `str` only (a tuple is rejected).
- **`dtype`**: `dict[str, str]`; values must be in `{"string", "int64", "float64", "bool"}`.
- **`usecols`**: non-string Sequence of `str`, non-empty, no duplicates.
- **`nrows`**: int ≥ 0 (`bool` rejected; `0` allowed).
- **`skiprows`**: int ≥ 0; skips lines *before* the header.

## 3. Raises
`CsvReadError` is in `arnio.exceptions`. Checks run in the order listed.

1. **Input type / scheme** (before any cleanup block):
   - `read_csv(123)` → `TypeError("read_csv expected a filesystem path, a URL, or a text file-like object")`.
   - A file-like whose `read()` returns bytes (e.g. `io.BytesIO(b"a\n1\n")`) → `TypeError("read_csv file-like objects must return text, not bytes")`.
   - `read_csv("s3://b/k.csv")` → `ValueError("Cloud scheme 's3' is not yet supported by arnio. Install the optional extra when available: pip install \"arnio[s3]\"")`.
     - The message literally contains `pip install "arnio[s3]"`.
     - `gs`/`gcs` → `arnio[gcs]`; `az`/`abfs`/`abfss` → `arnio[azure]`.
2. **Encoding type**: `encoding=5` → `TypeError("encoding must be a string, got 'int'")`.
3. **NUL bytes** (only when encoding is UTF-8): a file containing `b"\0"` → `CsvReadError("CSV input contains NUL bytes and appears to be binary or corrupted")`.
4. **Zero-byte file** → `CsvReadError(f"CSV file is empty: {native_path!r}")`.
   - Happens BEFORE option validation, so `read_csv(empty, delimiter=";;")` raises the empty-file error.
   - An empty `StringIO("")` raises the same error, but with the temp path in the message; use `match="CSV file is empty"`.
5. **Option validation**, in this order:
   - `decimal_separator`:
     - non-str → `TypeError("decimal_separator must be a string")`
     - `",,"` → `ValueError("decimal_separator must be a single character")`
     - `"a"` → `"decimal_separator must be a single non-alphanumeric character"`
     - `"+"` → `"Invalid decimal_separator: '+' and '-' are not allowed"`
   - `thousands_separator`: same messages with `thousands_separator` in place of `decimal_separator`. The TypeError reads `"thousands_separator must be a string or None"`.
     - `thousands_separator="."` (with default decimal) → `ValueError("thousands_separator must differ from decimal_separator")`.
   - `delimiter`:
     - `1` → `TypeError("delimiter must be a string")`
     - `";;"` → `ValueError("delimiter must be a single character; delimiter must be exactly one character")`
     - `"\n"` → `"delimiter must not be a newline character"`
     - `'"'` → `"delimiter must not be the CSV quote character"`
     - `"\x01"` → `"delimiter must not be a control character"`
   - `mode`:
     - non-str → `TypeError("mode must be a string")`
     - `"x"` → `ValueError("mode must be either 'strict' or 'permissive'")`
   - `encoding_errors`:
     - non-str → `TypeError("encoding_errors must be a string")`
     - bad value → `ValueError("encoding_errors must be one of 'strict', 'replace', or 'ignore'")`
   - `on_bad_lines`:
     - non-str → `TypeError("on_bad_lines must be a string")`
     - bad value → `ValueError("on_bad_lines must be either 'error', 'warn', 'skip'")`
   - `has_header=1` → `TypeError("has_header must be True or False, got int: 1")`. `trim_headers` uses the same template.
   - `null_values`:
     - `"NA"` → `TypeError("null_values must be a list of strings, not a bare string")`
     - `("NA",)` → `"null_values must be a list of strings"`
     - `[1]` → `"null_values must contain only strings"`
   - `dtype`:
     - `[]` → `TypeError("dtype must be a dictionary mapping column names to dtype strings")`
     - `{1: "int64"}` → `"dtype column names must be strings"`
     - `{"a": 1}` → `"dtype values must be strings"`
     - `{"a": "int32"}` → `ValueError("Unsupported dtype 'int32'. Expected one of: ['bool', 'float64', 'int64', 'string']")`
   - `usecols`:
     - `"a"` → `TypeError("usecols must be a sequence of column names, not a string")`
     - `{"a"}` → `"usecols must be a sequence of strings"`
     - `[]` → `ValueError("usecols must not be empty")`
     - `[1]` → `TypeError("usecols must contain only strings")`
     - `["a", "a"]` → `ValueError("usecols must not contain duplicate column names")`
   - `nrows`:
     - `True` / `1.0` → `TypeError("nrows must be an integer")`
     - `-1` → `ValueError("nrows must be non-negative")`
   - `skiprows` (the message says **skip_rows**):
     - `"1"` → `TypeError("skip_rows must be an integer")`
     - `-1` → `ValueError("skip_rows must be non-negative")`
6. **Non-UTF-8 encoding paths**:
   - `encoding="bogus"` on a real file → `ValueError("Unknown encoding: bogus")` (no quotes).
   - Undecodable bytes with e.g. `encoding="ascii"` and file bytes `b"a\n\xe9\n"` → `CsvReadError("Could not decode '<path>' using encoding 'ascii': invalid byte(s) at position 2 (byte value: 0xe9). Try a different encoding or use encoding_errors='replace'.")`.
   - Missing file with a non-UTF-8 encoding → `CsvReadError("Could not access CSV file '<path>': No such file or directory")`.
7. **Native parse errors**: a `RuntimeError` from C++ becomes `CsvReadError`.
   - Row-width messages are enriched to `"Malformed CSV: row N has too many|few fields (A found, E expected). To skip bad rows silently use on_bad_lines='skip', or on_bad_lines='warn' to collect them. For rows with missing trailing fields, try mode='permissive'. (<orig>)"`.
   - Whether C++ raises `RuntimeError` or `CsvReadError` directly is not visible. Only assert `pytest.raises(CsvReadError)` for `"a,b\n1,2,3\n"`.
   - A missing UTF-8 file is passed to C++. The result is most likely `CsvReadError` (exact type and message depend on C++).

## 4. Returns and guarantees
- Returns a new `ArFrame(cpp_frame)` with no attrs.
- Column names, order, inference, and null handling all come from C++ and are not visible here.
  - Per the docs, `has_header=False` gives `col_0`, `col_1`, …
  - `dtype` forces the given column types; `frame.dtypes` uses strings like `"int64"` and `"string"`.
  - With `trim_headers=True`, a header `" a "` becomes `"a"`.
  - Headers differing only by surrounding whitespace raise `CsvReadError` (doc claim, C++).

## 5. Side effects
- The input file is never modified.
- Temp files (gz, file-like, URL, transcoding) are deleted in `finally`, including on errors.
- May emit `UserWarning`s (see §6).

## 6. Branches and edge cases
- **`on_bad_lines="warn"` with bad rows**: one `UserWarning` with text `f"{n} malformed CSV row(s):\n  CSV row {r} has {a} fields; expected {e}"`.
  - At most 10 lines are listed, then `"  (+k more)"`.
  - Row numbering comes from C++; use `match="malformed CSV row"`.
- **`on_bad_lines="skip"`**: no warning; bad rows are dropped. In permissive mode, narrow rows are padded rather than dropped.
- **Delimiter-mismatch warning**: fires only if `usecols is None`, `has_header` is True, and the result has exactly 1 column.
  - A candidate in `{",", ";", "\t", "|"}` minus the delimiter must appear unquoted in both the header and the first data record.
  - Example: file `"a;b\n1;2\n"`, default delimiter → UserWarning: `"Parsed as a single column — possible delimiter mismatch. The file appears to contain ';' characters but delimiter=',' was used. Try ar.read_csv(path, delimiter=';') or ar.sniff_delimiter(path) to auto-detect."`
  - No warning for:
    - header-only files;
    - `usecols=[...]`;
    - `has_header=False`;
    - `"a;b\n1\n"` (no `;` in the data record);
    - `'"a;b"\n"1;2"\n'` (quoted).
- **`.gz` input**: `gzip.open` failures (e.g. a non-gzip file named `x.csv.gz` → `gzip.BadGzipFile`; a missing `.gz` → `FileNotFoundError`) propagate unwrapped.

## 7. Rules from comments and docstrings
- An explicit delimiter is always honoured, even for `.tsv` files.
- Validation runs on the decompressed file, not the gzip bytes.
- Dropped bad rows count toward `nrows`.
- Both modes reject extra fields.

## 8. Testing hints
- **Building inputs**: write files with `tmp_path / "x.csv"` and `.write_text(..., newline="")` or `.write_bytes(...)`. `io.StringIO("a,b\n1,2\n")` also works.
- **Observing results**: use `frame.shape`, `frame.columns`, `frame.dtypes`, `ar.to_pandas(frame)`.
  - Prefer string data or `dtype={...}` when asserting values; inferred dtypes are C++-defined.
- **Gzip and TSV**:
  - gz round-trip: `gzip.open(p, "wt").write("a,b\n1,2\n")`, then `shape == (1, 2)`.
  - `.tsv` auto-tab: `"a\tb\n1\t2\n"` → 2 columns. With `delimiter=","` → 1 column.
- **Warnings**:
  - Use `pytest.warns(UserWarning, match="possible delimiter mismatch")`.
  - Avoid tab as the mismatch candidate (its repr formatting is messy).
  - Avoid files containing two candidate characters (set iteration order decides which is named).
- **Error messages**: use `re.escape` with `match=` for messages containing `(`, `[`, `+`.
- **URLs**: don't test http(s); it needs the network.

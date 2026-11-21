# Notes: `arnio.io.write_csv(frame, path, *, delimiter=",", write_header=True, line_terminator="\n", escape_formulas=False, encoding="utf-8", encoding_errors="strict") -> None`

## 1. Purpose
- Writes an `ArFrame` to a CSV file using the C++ `_CsvWriter`.
- Non-UTF-8 encodings work in three steps: the C++ writer emits UTF-8 to a temp file, Python transcodes it into a second temp file in the destination directory, then `os.replace` moves it onto `path`.

## 2. Inputs
- **frame**: must be an `ArFrame` instance. A pandas DataFrame is rejected.
- **path**: `str`, `bytes` or `os.PathLike`.
  - Normalised with `os.fsdecode(os.fspath(path))`, so `b"out.csv"` and `pathlib.Path` both work.
  - The lowercased path must end with `.csv`, `.txt` or `.tsv`. `"OUT.CSV"` is accepted.
  - A `.tsv` path does **not** change the delimiter. It stays `","` unless you pass one.
- **delimiter**: must be exactly one character.
  - Rejected: `"\n"`, `"\r"`, `'"'`, and control characters (codepoint <= 0x1F or 0x7F).
  - Tab `"\t"` is allowed.
- **write_header**, **escape_formulas**: must be real `bool`. `1` and `0` are rejected.
- **line_terminator**: must be one of `"\n"`, `"\r\n"`, `"\r"`.
- **encoding**: must be a `str` accepted by `codecs.lookup`.
  - It is treated as UTF-8 (fast path) if `encoding.lower().replace("_","-")` is in `{"utf-8","utf8"}`. So `"UTF-8"`, `"utf_8"` and `"UTF8"` take the fast path.
  - `"utf-8-sig"` is **not** the fast path; it goes through transcoding.
- **encoding_errors**: one of `"strict"`, `"replace"`, `"ignore"`. It is validated always but only used on the non-UTF-8 path.

## 3. Raises (in this order; all validation happens before any file is written)
1. `TypeError("frame must be an ArFrame")`
   - Trigger: `write_csv(pd.DataFrame(), "a.csv")`
2. `TypeError(f"path must be a string, bytes, or os.PathLike object, got {type(path).__name__!r}")`
   - Trigger: `path=123` gives `...got 'int'`.
3. `ValueError(f"Unsupported file format: {path}. Only .csv, .txt, and .tsv are supported.")`
   - The path appears without quotes. Trigger: `"out.json"`.
4. Delimiter checks:
   - `TypeError("delimiter must be a string")`
   - `ValueError("delimiter must be a single character; delimiter must be exactly one character")` for `""` or `";;"`
   - `ValueError("delimiter must not be a newline character")`
   - `ValueError("delimiter must not be the CSV quote character")` for `'"'`
   - `ValueError("delimiter must not be a control character")` for `"\x01"` or `"\x7f"`
5. Line terminator checks:
   - `TypeError("line_terminator must be a string")`
   - `ValueError(f"line_terminator must be one of '\\n', '\\r\\n', or '\\r', got {line_terminator!r}")`
   - The message contains literal backslash sequences, e.g. `line_terminator must be one of '\n', '\r\n', or '\r', got ';'`. Use `pytest.raises(..., match=re.escape(...))` or match only the prefix.
6. Encoding checks:
   - `TypeError(f"encoding must be a string, got {type(encoding).__name__!r}")`
   - `ValueError(f"Unknown encoding: {encoding!r}")`, e.g. `encoding="nope"` gives `Unknown encoding: 'nope'`.
7. Encoding-errors checks:
   - `TypeError("encoding_errors must be a string")`
   - `ValueError("encoding_errors must be one of 'strict', 'replace', or 'ignore'")`
8. `TypeError(f"write_header must be True or False, got {type(v).__name__}: {v!r}")`
   - Trigger: `write_header=1` gives `write_header must be True or False, got int: 1`.
9. Same message pattern for `escape_formulas`, e.g. `escape_formulas="yes"` gives `...got str: 'yes'`.
10. Write-time errors:
    - **UTF-8 path:** a C++ `RuntimeError` is re-raised as `RuntimeError(str(e))`, e.g. for a missing parent directory. The exact message comes from C++ and is not visible.
    - **Non-UTF-8, unencodable character with `strict`:** `ValueError(f"write_csv: character cannot be encoded in {encoding!r}: {exc}")`. Trigger: `"é"` with `encoding="ascii"`.
    - **Non-UTF-8, `OSError`** (e.g. destination directory does not exist): `RuntimeError(str(exc))`.

## 4. Returns and guarantees
- Returns `None`.
- The file content (quoting rules, float formatting, null representation, header layout) is produced by the C++ writer and is **not visible** here.
- Per the docstring, `escape_formulas=True` prefixes `'` to string cells starting with `=`, `+`, `-`, `@`, tab or CR. Numeric columns are unchanged. This is implemented in C++.
- On the non-UTF-8 path, files are opened with `newline=""` on both sides, so the line terminator bytes are preserved exactly (no Windows `\n` to `\r\n` translation).

## 5. Side effects
- Creates or overwrites `path`. The frame is not mutated.
- Non-UTF-8 path, in more detail:
  - Uses a system temp `.csv` file and a `.{basename}.XXXX.tmp` file in the destination directory.
  - Both are deleted in `finally`.
  - On an encode failure, the destination is not replaced: an existing file keeps its old content, and a new path is not created.

## 6. Branches and edge cases
- **Validation error:** no file is created, e.g. `write_csv(f, tmp/"x.json")` leaves `x.json` absent.
- **UTF-8 fast path:** the writer writes directly to `path`; `encoding_errors` is ignored.
- **Non-UTF-8 path** (e.g. `"latin-1"`) with a column value `"café"`: the file bytes contain `b"caf\xe9"`.
- **`encoding="ascii"` with value `"café"`:**
  - `"strict"` raises the `ValueError` above.
  - `"replace"` writes `caf?`.
  - `"ignore"` writes `caf`.
- **`encoding="utf-8-sig"`:** the file starts with `b"\xef\xbb\xbf"` (Python codec behaviour).
- **`line_terminator="\r\n"`:** use bytes containing `b"\r\n"`. Use `"\r"` for old-Mac style.

## 7. Rules from comments/docstring
- Validate `encoding` and `encoding_errors` "before any file I/O".
- Transcode in bounded 1 MiB chunks.
- `newline=""` preserves the `line_terminator` exactly.

## 8. Testing hints
- Build frames with `ar.from_pandas(pd.DataFrame({"a": [1, 2], "s": ["x", "y"]}))`, and use pytest's `tmp_path`.
- **Round trip:** `ar.read_csv(p)`, then compare via `ar.to_pandas`. Prefer ints and plain strings; avoid floats, NaN and exact quoting assertions.
- Check raw output with `open(p, "rb").read()`:
  - Safe assertion: `b"\r\n" in data` for `line_terminator="\r\n"`.
  - Safe assertion: `b"\r\n" not in data` with the default terminator.
- **Header:**
  - `write_header=True`: the first decoded line should contain `a` and `s`.
  - `write_header=False`: the column names should not appear; read back with `has_header=False`.
- **Delimiter:** with `delimiter=";"`, read back with `ar.read_csv(p, delimiter=";")`, or check that `b";"` appears in the header line.
- **Formula escaping:** a string value `"=1+1"` with `escape_formulas=True` should read back as `"'=1+1"`. With the default (`False`) it reads back as `"=1+1"`.
- **Validation order:** e.g. `write_csv(frame, "x.json", delimiter=";;")` raises the file-format `ValueError`, not the delimiter one.
- After an ascii-strict failure, assert that no `*.tmp` files remain in `tmp_path` and that the destination does not exist.
- **Pitfalls:**
  - Do not expect TSV auto-tab.
  - Do not expect `bool` coercion of `1` and `0`.
  - The error for an unknown encoding is `ValueError`, not `LookupError`.

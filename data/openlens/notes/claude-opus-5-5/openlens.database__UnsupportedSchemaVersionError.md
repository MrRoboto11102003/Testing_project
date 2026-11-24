# Notes: `openlens.database.UnsupportedSchemaVersionError`

## 1. Purpose
- An exception class, a subclass of `RuntimeError`.
- `DatabaseManager.__init__` raises it, through `_initialize_db`, when the SQLite file's `PRAGMA user_version` is greater than the module constant `SCHEMA_VERSION = 7`. In other words, the file was written by a newer OpenLens.
- The point is to refuse the file instead of downgrading it.

## 2. Inputs (`__init__(self, found: int, supported: int, path: str)`)
- All three are positional or keyword. There are no defaults.
- There is **no validation or conversion**. The values are stored exactly as passed:
  - `self.found = found`
  - `self.supported = supported`
  - `self.path = path`
- Any type works, for example `found="x"` or `path=None`. Each value is simply interpolated into the f-string.

## 3. Raises
- The constructor itself never raises for any argument values.
- Where it gets raised: in `DatabaseManager._initialize_db`, the first step inside the `BEGIN IMMEDIATE` transaction is:
  ```python
  version = PRAGMA user_version
  if version > SCHEMA_VERSION:   # 7
      raise UnsupportedSchemaVersionError(version, SCHEMA_VERSION, self.db_path)
  ```
- This check runs **before** any CREATE TABLE or migration statement.
- The `_transaction` helper then runs `ROLLBACK` and re-raises. The connection is closed in a `finally` block.

## 4. Returns and guarantees
- `str(e)` equals `e.args[0]`, and `e.args == (message,)`. The message is exactly this (two adjacent f-strings plus plain strings, joined with **single spaces**):
  ```
  f"Database '{path}' has schema version {found}, but this OpenLens only supports up to version {supported}. It was written by a newer OpenLens; upgrade OpenLens rather than opening this file with an older build, which would downgrade the library and lose data."
  ```
- Example: `UnsupportedSchemaVersionError(9, 7, "a.db")` gives:
  `"Database 'a.db' has schema version 9, but this OpenLens only supports up to version 7. It was written by a newer OpenLens; upgrade OpenLens rather than opening this file with an older build, which would downgrade the library and lose data."`
- Type facts:
  - `isinstance(e, RuntimeError)` is True.
  - It is not a subclass of `LensInUseError`, and the reverse is also false.
  - `e.found`, `e.supported` and `e.path` keep their identity (the same objects that were passed in).
- When raised from `DatabaseManager`:
  - `e.supported == 7`.
  - `e.found` is the int read from the file.
  - `e.path` is `resolve_db_path(<arg>)`. `resolve_db_path` lives in `openlens.paths`, whose code is not shown. Do not assume it returns the argument unchanged; compare against `openlens.paths.resolve_db_path(p)` instead.

## 5. Side effects
- Constructing the exception has no side effects.
- When raised by `DatabaseManager`, the file is left unchanged:
  - No tables are created.
  - `user_version` stays at the value it had (for example 8).
  - The connection has already run `PRAGMA journal_mode=WAL`, so `-wal`/`-shm` files may appear next to the db.

## 6. Branches and edge cases (via `DatabaseManager`)
| `user_version` in file | Result |
|---|---|
| 0 (new file) | Tables are created and the file is stamped 7. No error. |
| 7 | Opens normally. No error. This is the boundary: the check is strict `>`. |
| 8 | Raises, with `found=8, supported=7`. |
| 100 | Raises, with `found=100`. |

- Retrying is deterministic. Calling `DatabaseManager(p)` again raises the same error again, because nothing was modified.

## 7. Rules from docstrings and comments
- A newer database must be **refused, not opened**. A downgrade would "drop columns the newer build relies on and corrupt the library".
- The documented attributes are `found`, `supported` and `path`.

## 8. Testing hints
- Direct construction is the easiest way to pin the exact message and the attributes:
  ```python
  from openlens.database import UnsupportedSchemaVersionError
  e = UnsupportedSchemaVersionError(9, 7, "a.db")
  assert str(e) == ("Database 'a.db' has schema version 9, but this OpenLens only supports "
                    "up to version 7. It was written by a newer OpenLens; upgrade OpenLens "
                    "rather than opening this file with an older build, which would downgrade "
                    "the library and lose data.")
  ```
- Integration test. Create the file with plain sqlite3, then open it with the manager:
  ```python
  import sqlite3
  p = str(tmp_path / "new.db")
  c = sqlite3.connect(p); c.execute("PRAGMA user_version = 8"); c.commit(); c.close()
  with pytest.raises(UnsupportedSchemaVersionError) as ei:
      DatabaseManager(p)
  assert ei.value.found == 8 and ei.value.supported == 7
  ```
  - Afterwards, reopen with sqlite3 and check two things:
    - `PRAGMA user_version` is still 8.
    - `SELECT name FROM sqlite_master WHERE type='table'` returns no rows.
  - Close each connection, or Windows file locks may cause trouble.
- Boundary test: a file stamped 7 constructs fine. You can also check `openlens.database.SCHEMA_VERSION == 7`.
- Pitfalls:
  - The path is wrapped in single quotes in the message.
  - "version(s)" in the message is plain text, with no pluralisation logic.
  - Use `pytest.raises(..., match=...)` carefully: `match` takes a regex, so the message's `.`, `(` and `'` need `re.escape`.
  - Do not assert `e.path == p` without passing it through `resolve_db_path`; its behaviour is not visible here.

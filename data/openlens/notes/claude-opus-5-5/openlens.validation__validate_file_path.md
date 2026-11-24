# Notes: `openlens.validation.validate_file_path`

## 1. Purpose
- Checks a file path's type, resolves it to an absolute `Path`, and checks that it exists when asked.
- Checks that the parent directory exists, and creates it when asked.
- Checks that the parent directory is writable, but only for a new file.
- Returns the resolved `Path`. It never creates or opens the file itself.

## 2. Inputs
- `file_path`: must be an instance of `str` or `pathlib.Path`. `WindowsPath` and `PosixPath` are accepted because they subclass `Path`. These are rejected: `bytes`, `int`, `None`, `PurePath`/`PurePosixPath` (they are not `Path`), and `os.PathLike` objects that are not `Path`.
- `must_exist: bool = False`: when true, the resolved path must exist. A directory also counts, because only `path.exists()` is checked.
- `create_parent: bool = False`: when true, a missing parent is created with `parent.mkdir(parents=True, exist_ok=True)`.
- `param_name: str = "file_path"`: this string is placed into every error message.
- Normalisation: `path = Path(file_path).resolve()`. This is non-strict, so a nonexistent path is fine. A relative path is resolved against the current working directory, and `""` resolves to the cwd.

## 3. Raises
Every error is `openlens.validation.ValidationError`. It subclasses `Exception` only, not `ValueError`. The checks run in this order:

1. **Type check**: `f"{param_name} must be a string or Path object"`.
   - Trigger: `validate_file_path(123)` gives `"file_path must be a string or Path object"`.
   - Trigger: `validate_file_path(b"x")` raises the same error.
2. **Resolve failure**: `f"Invalid {param_name}: {e}"`. It fires only if `resolve()` raises `ValueError` or `RuntimeError`. Behaviour here is platform-dependent (for example a path with an embedded NUL), so do not test it.
3. **Missing file**: `must_exist and not path.exists()` gives `f"{param_name} does not exist: {path}"`. Here `path` is the *resolved* path.
   - Trigger: `validate_file_path(tmp_path/"nope.txt", must_exist=True)`.
4. **Missing parent, not allowed to create it**: `f"Parent directory does not exist for {param_name}: {parent}"`.
   - Trigger: `validate_file_path(tmp_path/"missing"/"f.txt")`.
5. **Missing parent, creation fails**: `mkdir` raises `OSError`, giving `f"Cannot create directory for {param_name}: {e}"`.
   - Trigger: create a regular file `tmp_path/"blk"`, then call `validate_file_path(tmp_path/"blk"/"sub"/"f.txt", create_parent=True)`.
6. **Parent not writable**: `not must_exist and not os.access(parent, os.W_OK)` gives `f"Parent directory is not writable for {param_name}: {parent}"`.

Ordering facts worth testing:
- `must_exist` is checked before the parent check. For a missing file whose parent is also missing, with `must_exist=True`, you get the "does not exist" message, not the "Parent directory" one.
- The type check comes first. `validate_file_path(None, must_exist=True)` therefore raises the type error.

## 4. Returns and guarantees
- Returns a new `pathlib.Path` equal to `Path(file_path).resolve()`. It is absolute (`.is_absolute()` is True).
- Returns the same value whether the input was a `str` or a `Path`.
- It does not check that the target is a regular file. An existing directory passes, including with `must_exist=True`.
- The extension is not checked; that is done by `validate_json_file_path`.

## 5. Side effects
- The only side effect: with `create_parent=True` and a missing parent, the whole parent chain is created on disk. This happens before the writability check, and the directories stay even if that check later fails.
- The file itself is never created.
- Nothing global changes and there is no logging.

## 6. Branches and edge cases
- **New file, parent exists**: `validate_file_path(tmp_path/"a.txt")` returns `(tmp_path/"a.txt").resolve()`, and the file still does not exist afterwards.
- **Existing file**: `validate_file_path(f, must_exist=True)` returns the resolved path. The writability check is skipped in this branch.
- **Nested missing parent**: `validate_file_path(tmp_path/"x"/"y"/"f.txt", create_parent=True)` creates `tmp_path/x/y` as a directory. The file is not created.
- **Missing parent, no creation**: the same call without `create_parent` raises, and nothing is created (`(tmp_path/"x").exists()` stays False).
- **Custom name**: `param_name="output"` gives messages such as `"output does not exist: ..."` and `"output must be a string or Path object"`.
- **Relative path**: after `monkeypatch.chdir(tmp_path)`, `validate_file_path("rel.txt") == (tmp_path/"rel.txt").resolve()`.

## 7. Rules from docstring
- "Validate and sanitize file path".
- Returns a "Validated and resolved Path object".
- The writability check applies only "for new files", i.e. only when `must_exist` is False.

## 8. Testing hints
- Import with `from openlens.validation import validate_file_path, ValidationError`.
- Use `tmp_path` for every filesystem case.
- Compare results against `(tmp_path/...).resolve()`, not the raw `tmp_path`. On Windows, `tmp_path` may contain 8.3 short names or a different case, and `resolve()` normalises them.
- Messages embed Windows paths containing backslashes. Use `pytest.raises(ValidationError, match=re.escape(expected))`, or assert `str(exc.value) == expected`, with `expected` built from the resolved path.
  - Example: `f"Parent directory does not exist for file_path: {(tmp_path/'missing').resolve()}"`.
- **Do not try to trigger the not-writable error with chmod.** On Windows, `os.access(dir, W_OK)` is effectively always True for directories, and root ignores the bits on POSIX. Instead, monkeypatch it: `monkeypatch.setattr(openlens.validation.os, "access", lambda p, m: False)`.
  - Then `validate_file_path(tmp_path/"a.txt")` raises the "not writable" error.
  - With the same patch, `validate_file_path(existing_file, must_exist=True)` still succeeds, because the check is skipped.
- For the mkdir-failure case, assert only the message prefix. The `{e}` part is an OS-specific error string.
  - Example: `match=r"^Cannot create directory for file_path: "`.
- Do not assert that `ValidationError` is a `ValueError`; it is not.

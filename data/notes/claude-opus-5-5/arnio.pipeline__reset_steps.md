# Notes for `arnio.pipeline.reset_steps`

## 1. Purpose
- Restores the global Python step registry (`_PYTHON_STEP_REGISTRY`) to the built-in Python steps captured at import time.
- Removes every user-registered custom step.
- Re-adds any built-in Python step that was removed from the private dict.

## 2. Inputs
- Takes no parameters. `reset_steps(1)` raises a plain Python `TypeError`, because the signature is `reset_steps() -> None`.
- Full body:
  ```python
  with _REGISTRY_LOCK:
      _PYTHON_STEP_REGISTRY.clear()
      _PYTHON_STEP_REGISTRY.update(_BUILTIN_PYTHON_STEP_REGISTRY)
  ```
- `_BUILTIN_PYTHON_STEP_REGISTRY` is filled once, at module load, by `_BUILTIN_PYTHON_STEP_REGISTRY.update(_PYTHON_STEP_REGISTRY)`. At that point the registry holds exactly these 8 entries, in this insertion order:
  - `standardize_missing_tokens` → `cleaning.standardize_missing_tokens`
  - `coalesce_columns` → `cleaning.coalesce_columns`
  - `normalize_whitespace` → `cleaning.normalize_whitespace`
  - `filter_rows` → `cleaning.filter_rows`
  - `drop_columns_matching` → `cleaning.drop_columns_matching`
  - `rename_columns_matching` → `cleaning.rename_columns_matching`
  - `safe_divide_columns` → `cleaning.safe_divide_columns`
  - `replace_values` → `cleaning.replace_values`

## 3. Raises
- It raises nothing itself; the only failure is a `TypeError` if you pass arguments.

## 4. Returns and guarantees
- Returns `None`.
- After the call, `_PYTHON_STEP_REGISTRY` has exactly the 8 names above, each mapped to the original `arnio.cleaning` function object.
- It is the same dict object as before, because the code calls `clear()` then `update()` rather than rebinding the name.
- It does not touch:
  - `_STEP_REGISTRY`, the 24 C++ steps;
  - `_DEPRECATED_STEP_ALIASES`;
  - `_BUILTIN_PYTHON_STEP_REGISTRY` itself.
- After a reset, `list_steps()` returns `sorted(24 C++ names + 8 Python names)`. That is 32 names with no overlap. The C++ names are:
  - drop_nulls, drop_columns, select_columns, keep_rows_with_nulls, fill_nulls, validate_columns_exist
  - drop_duplicates, drop_constant_columns, drop_empty_columns
  - clip_numeric, winsorize_outliers, normalize_minmax
  - strip_whitespace, hash_columns, parse_bool_strings, normalize_case, normalize_unicode
  - rename_columns, cast_types, round_numeric_columns, combine_columns
  - slugify_column_names, trim_column_names, clean_column_names

## 5. Side effects
- It mutates process-global registry state under the non-reentrant `threading.Lock`.
- `pipeline()` releases the lock before running steps, so calling `reset_steps()` from inside a step does not deadlock. Avoid it anyway.
- It writes no files and mutates no frames.

## 6. Branches and edge cases
There is no branching in the function. The scenarios worth testing are:
- **Custom step removed.** Before reset, `register_step("my_step", f)` succeeds and `"my_step" in list_steps()` is True. After reset, `"my_step" not in list_steps()`.
- **Custom step no longer usable in a pipeline.** After reset, `ar.pipeline(frame, [("my_step",)])` raises `UnknownStepError`. The error comes from `_validate_pipeline_steps`, before any step runs. `UnknownStepError` subclasses `ArnioError`.
  - Its message format lives in `arnio/exceptions.py`, which is not shown. Do not assert the message text.
- **Re-registering after reset.** Before reset, a second `register_step("my_step", f)` without `overwrite` raises `ValueError` with exactly: `"Step 'my_step' is already registered as a custom Python step. To intentionally overwrite it, set 'overwrite=True'."`. After reset, the same call succeeds with no error.
- **Overwritten custom step.** `register_step("my_step", f)` then `register_step("my_step", g, overwrite=True)`, then reset: the name is gone.
- **Built-in Python step restored.** You cannot remove one through the public API:
  - `unregister_step("filter_rows")` raises `ValueError("Cannot unregister built-in step 'filter_rows'. Only custom user-registered steps can be unregistered.")`.
  - `register_step("filter_rows", f)` raises a `ValueError` starting `"Cannot register 'filter_rows': conflicts with built-in Python step."`.
  - To test restoration, delete it from the private dict: `del pm._PYTHON_STEP_REGISTRY["filter_rows"]`, then reset. Afterwards `"filter_rows" in list_steps()` and `pm._PYTHON_STEP_REGISTRY["filter_rows"] is ar.cleaning.filter_rows` (or `pm.cleaning.filter_rows`).
- **Idempotent.** Calling it twice in a row, or on an already clean registry, leaves the same 8 entries.
- **Multiple custom steps.** Register `a`, `b`, `c`, then reset: all are removed and all 8 built-ins remain.

## 7. Rules from comments/docstrings
- Docstring: "Restore the Python pipeline registry to built-in steps only."
- So the result is built-ins only: no leftover custom steps, and no missing built-ins.

## 8. Testing hints
- Get the module object safely, because `ar.pipeline` is probably the *function*, which shadows the submodule:
  ```python
  import importlib
  pm = importlib.import_module("arnio.pipeline")
  reset_steps, register_step, list_steps = pm.reset_steps, pm.register_step, pm.list_steps
  ```
  Avoid `import arnio.pipeline as pm`; it may bind the function.
- Use an autouse fixture that calls `reset_steps()` before and after each test, so global state does not leak between tests.
- A minimal custom step: `def my_step(df): return df`. Register it under a unique name such as `"my_custom_step_xyz"`.
- Build frames with `ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))`.
- Assert each part separately, so a regression that only clears, or only restores, is caught:
  - (a) the custom name is absent;
  - (b) all 8 built-in Python names are present, e.g. `set(BUILTIN_PY) <= set(list_steps())`;
  - (c) `len(list_steps()) == 32` after reset;
  - (d) `pm._PYTHON_STEP_REGISTRY.keys() == set(BUILTIN_PY)`.
- Assert `reset_steps() is None`.
- Use the registry-identity check `pm._PYTHON_STEP_REGISTRY is` (the same object as before the call) only if you accept testing a private detail. It is true on the current code.
- Do not run the built-in Python steps (`filter_rows` etc.) to prove they are restored. Their kwargs and semantics live in `cleaning.py`, which is not shown. Check membership and identity instead.
- Do not assert on the text of `UnknownStepError`; its constructor is not shown.
- `list_steps()` output is sorted, so compare with `sorted(...)` or as sets.

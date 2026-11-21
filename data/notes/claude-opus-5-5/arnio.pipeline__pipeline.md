# Notes: `arnio.pipeline.pipeline(frame, steps, *, return_metadata=False, dry_run=False, verbose=False, track_lineage=False)`

## 1. Purpose
- Applies a list of steps to an `ArFrame` in order and returns the final frame.
- Steps can be C++-backed cleaning functions, built-in Python steps, or user-registered Python steps.
- Optionally returns timing/row metadata (`return_metadata`) or row-drop lineage (`track_lineage`).

## 2. Inputs
- `frame`: must be an `ArFrame`. A pandas DataFrame is rejected.
- `steps`: must be a `list`. Each element is `(name,)` or `(name, kwargs_dict)`.
- `name` can be:
  - a C++ registry name: `drop_nulls`, `drop_columns`, `select_columns`, `keep_rows_with_nulls`, `fill_nulls`, `validate_columns_exist`, `drop_duplicates`, `drop_constant_columns`, `drop_empty_columns`, `clip_numeric`, `winsorize_outliers`, `normalize_minmax`, `strip_whitespace`, `hash_columns`, `parse_bool_strings`, `normalize_case`, `normalize_unicode`, `rename_columns`, `cast_types`, `round_numeric_columns`, `combine_columns`, `slugify_column_names`, `trim_column_names`, `clean_column_names`;
  - a built-in Python step: `standardize_missing_tokens`, `coalesce_columns`, `normalize_whitespace`, `filter_rows`, `drop_columns_matching`, `rename_columns_matching`, `safe_divide_columns`, `replace_values`;
  - any name added with `ar.register_step`;
  - `"builtin:<name>"` for any built-in name above. It resolves to the plain name, which is the name used in metadata and lineage keys;
  - a deprecated alias. The table is empty by default and only the private `_register_deprecated_step_alias` fills it. Using an alias emits `DeprecationWarning("Pipeline step 'old' is deprecated; use 'new' instead.")`.
- The flags must be real `bool`. `1` and `None` are rejected (`isinstance(x, bool)`).
- A user step registered as `"mystep"` is **not** reachable as `"builtin:mystep"`; that raises `UnknownStepError`.

## 3. Raises (checks run in this order)
1. `TypeError("frame must be an ArFrame")`, e.g. `pipeline(pd.DataFrame(), [])`.
2. `TypeError("return_metadata must be a bool, got 'int'")` for `return_metadata=1`.
   - The same template applies, in this order, to `dry_run`, `verbose` and `track_lineage`, e.g. `"dry_run must be a bool, got 'str'"`.
3. `ValueError` when both `track_lineage=True` and `return_metadata=True`. The message starts with `"track_lineage=True and return_metadata=True cannot be used together."`
4. Container checks (all `TypeError`):
   - `None` → `"steps must be a list of step tuples, got None"`.
   - `str`/`bytes`: `"steps must be a list of step tuples, got str. Each step must be (name,) or (name, kwargs), for example [(\"drop_nulls\",)]."` (the example uses literal double quotes).
   - A tuple: `steps=("drop_nulls",)` → `"steps must be a list of step tuples, got bare step tuple ('drop_nulls',). Use [('drop_nulls',)] instead."`
   - Other types: `{}` → `"steps must be a list of step tuples, got dict. "` (note the trailing space).
5. Per-step validation. **Every step is checked before any step runs**, in list order, so the first bad step wins. All are `ValueError` except the last:
   - Not a tuple, or length not 1–2: `"Invalid step format: ['drop_nulls']. Expected (name,) or (name, kwargs)"`. Applies to `["drop_nulls"]` and to `()`.
   - Name not a string: `(1,)` → `"Invalid pipeline step name: 1. Expected a string"`.
   - Kwargs not a dict: `("drop_nulls", [1])` → `"Invalid step kwargs for 'drop_nulls': [1]. Expected a dict"`.
   - Unknown name: `UnknownStepError(name, sorted(available))`. Its message format lives in `arnio.exceptions` (not shown), so assert only the type, or that the name appears in `str(e)`.
   - Example: `[("drop_nulls",), ("bogus",)]` raises before `drop_nulls` executes.
6. `ValueError` when `track_lineage=True` and the input already has column `__arnio_lineage_id__`. This happens after step validation. Message: `"track_lineage=True requires that the input frame does not already contain a column named '__arnio_lineage_id__'.  Please rename or drop that column before calling pipeline()."` (two spaces before "Please").
7. `cast_types` with a reserved key but no dict `mapping`: `("cast_types", {"a": "int64", "errors": "raise"})` → `ValueError("cast_types shorthand mapping cannot be mixed with reserved keyword arguments. Use {'mapping': {...}, ...} form instead.")`.
8. Exceptions inside steps:
   - C++ steps and built-in Python steps: the exception propagates unchanged (e.g. whatever `cleaning.select_columns` raises for a missing column).
   - Custom (user-registered) steps: the exception is wrapped as `PipelineStepError(name, e)`, with `__cause__ is e`.
9. Bad custom-step return values (`TypeError`, **not** wrapped):
   - `None` → `"Custom pipeline step 'x' returned None. Steps must return a pandas DataFrame."`
   - A list → `"Custom pipeline step 'x' returned 'list' instead of a pandas DataFrame. Steps must return a pandas DataFrame."`
10. With `track_lineage=True`, any step that removes or renames the sentinel raises `PipelineStepError(name, KeyError(...))`. This includes C++ steps such as `select_columns(["a"])` or `drop_columns(["__arnio_lineage_id__"])`.

## 4. Returns and guarantees
- **Default:** returns the last step's result.
  - With `steps=[]`, it returns **the input object itself** (`pipeline(f, []) is f`).
- **`return_metadata=True`:** returns `(frame, meta)` where:
  - `meta["applied_steps"]` is a list of canonical names.
  - `meta["row_counts"]` is a list of `{"step", "before", "after", "dry_run"}`.
  - `meta["step_timings"]` is a list of `{"step", "seconds", "dry_run"}`.
  - `meta["execution_summary"]` is `{"total_runtime_ms": float, "steps": [...]}`. Each entry in `steps` is `{"name", "status": "success", "runtime_ms", "rows_before", "rows_after", "rows_affected", "columns_before", "columns_after"}`.
  - `rows_affected = before - after`.
  - Field values:
    - `before` is the row count of the current *result* frame.
    - `after` is the step output's row count.
    - `columns_*` come from the working frame and the step output.
  - With `steps=[]`, all lists are empty.
- **`track_lineage=True`:** returns `(frame, LineageReport)`.
  - `dropped_by_step` has one key per canonical step name, in order of first appearance. Every step gets a key, with `[]` if it dropped nothing.
  - Values are sorted original 0-based positions. Repeated names are merged and re-sorted.
  - `total_dropped` is the sum of list lengths.
  - The returned frame is new (rebuilt via pandas), without the sentinel, with the original column order.

## 5. Side effects
- The input frame is not mutated. Custom steps receive a deep-copied, writable pandas `DataFrame` subclass, so in-place edits do not leak back.
- The step registry is not changed.
- `verbose=True` logs one INFO record per step on logger `"arnio"`, in this form: `"[1/2] drop_nulls | path=arnio.cleaning.drop_nulls | rows: 3 -> 2 | 0.01ms"`.
  - For custom steps, `path` is `f"{fn.__module__}.{fn.__name__}"`.

## 6. Branches and edge cases
- **`dry_run=True`:**
  - Steps still execute and chain on each other's output (the working frame advances), but the returned frame is the **input object itself** (`res is f`).
  - In metadata, `before` is always the input row count, and `after == before`, `rows_affected == 0`, `"dry_run": True`.
  - With `track_lineage`, drops are **still recorded** (computed from the working frame), and the returned frame equals the input data.
- **`rename_columns` shorthand:** `("rename_columns", {"a": "b"})` uses the kwargs dict as the mapping. `{"mapping": {"a": "b"}}` also works.
- **`cast_types` forms:**
  - `{"mapping": {...}, "errors": ...}` is passed as kwargs.
  - Otherwise the whole dict is the mapping, e.g. `("cast_types", {"a": "float64"})`.
  - Avoid `errors="report"`: it returns a `CastReport`, which has no `.shape`, so the call fails.
- **`drop_duplicates` with lineage:** the subset is auto-restricted to non-sentinel columns, or the sentinel is filtered out of a user-given subset. Duplicates are therefore detected normally.
- **`context` injection:** if a step function has a `context` parameter and kwargs contain no `"context"`, it receives `PipelineContext(step_name=<canonical name>, step_index=<0-based>, total_steps=len(steps), dry_run=dry_run)`.

## 7. Rules from comments and docstrings
- Lineage and metadata are mutually exclusive.
- The sentinel is never visible in the output.
- Custom steps must not drop or rename the sentinel.
- Drops from repeated step names are merged under one key, in sorted order.

## 8. Testing hints
- Build inputs with `f = ar.from_pandas(pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]}))`.
  - `("drop_duplicates",)` gives 2 rows. Lineage is then `{"drop_duplicates": [1]}`, `total_dropped == 1`.
- Inspect results with `ar.to_pandas(res)`, `res.shape` and `res.columns`.
- For nulls, use `None` in a string column. The exact null semantics of `drop_nulls` live in C++, so prefer `drop_duplicates` and `drop_columns` for exact counts.
- Register custom steps in a fixture and clean up with `ar.reset_steps()` (or `ar.unregister_step`). Global registry state leaks between tests otherwise.
- Import `PipelineStepError` and `UnknownStepError` from `arnio.exceptions`.
- Never assert exact timing values. Only check that they exist and are `>= 0` floats.
- Use `pytest.raises(..., match=re.escape(msg))`, because the messages contain `(`, `[`, `.` and `{`.
- Verbose logs: use `caplog.set_level(logging.INFO, logger="arnio")`, then check that `"[1/1] drop_duplicates"` and `"rows: 3 -> 2"` appear in `caplog.text`.
- With `track_lineage`, avoid steps that rename or select columns (`select_columns`, `clean_column_names`, `slugify_column_names`). They touch the sentinel and raise `PipelineStepError`.

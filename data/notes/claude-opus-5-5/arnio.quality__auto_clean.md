# Notes: `arnio.quality.auto_clean`

## 1. Purpose
- Profiles the frame with `profile(frame)`, then applies the cleaning suggestions from that profile.
- `mode="safe"` applies only `strip_whitespace`.
- `mode="strict"` also applies `cast_types` (gated by two flags) and `drop_duplicates`, then runs a final dedup pass.
- It can also return the pre-cleaning report and/or an audit trail (`CleanExplanation`).

## 2. Inputs
- **`frame`**: an ArFrame, checked by `_validate_arframe(frame)`. That helper lives in `arnio.frame`; its exception type and message are not visible (probably `TypeError`).
- **`mode`**: must be a `str` in `{"safe", "strict"}`. The match is case-sensitive, so `"SAFE"` fails, and so does `None`. Both fail with `ValueError`, not `TypeError`.
- **`return_report`, `dry_run`, `allow_lossy_casts`, `explain`**: must be real `bool`. `0` and `1` raise `TypeError`.
- **`confirmed_casts`**: `None` or a `dict[str, str]`. It is copied into a new dict. In strict mode it is compared to the proposed mapping with `==`, so key order does not matter.

## 3. Raises (checks run in this order)
1. `_validate_arframe` fails (message unknown).
2. `ValueError("mode must be 'safe' or 'strict'")`, e.g. `mode="fast"`.
3. `TypeError("return_report must be a bool")`.
4. `TypeError("dry_run must be a bool")`.
5. `TypeError("allow_lossy_casts must be a bool")`.
6. `confirmed_casts` checks:
   - Not a dict: `TypeError("confirmed_casts must be a dict[str, str] or None")`, e.g. `confirmed_casts=[("a","int64")]`.
   - Non-str key: `TypeError(f"confirmed_casts keys must be str column names, got {type(column).__name__}")`, e.g. `{1:"int64"}` gives `...got int`.
   - Non-str value: `TypeError(f"confirmed_casts values must be str dtype names, got {type(dtype).__name__}")`, e.g. `{"a":1}` gives `...got int`.
7. `TypeError("explain must be a bool")`.
8. `ValueError("explain=True cannot be used with dry_run=True")`.
9. `ValueError("return_report=True cannot be used with dry_run=True")`. Check 8 runs first if all three flags are set.

The remaining errors fire only in strict mode, and only when a `cast_types` suggestion exists. They are raised while the steps are being applied. `{kwargs}` below is the proposed mapping dict.
- **`allow_lossy_casts=False`** gives `ValueError`:
  `"auto_clean(mode='strict') would apply type casts. Proposed mapping: {kwargs}. Run with dry_run=True to inspect the report, then pass allow_lossy_casts=True and confirmed_casts=<proposed mapping> to apply them."`
- **`allow_lossy_casts=True` with `confirmed_casts=None`** gives `ValueError`:
  `"auto_clean(mode='strict') requires confirmed_casts before applying type casts. Run with dry_run=True to inspect the proposed mapping, then pass confirmed_casts={kwargs!r}."`
- **Mapping differs** gives `ValueError`:
  `"confirmed_casts must match the proposed cast mapping exactly. Proposed mapping: {kwargs}. Confirmed mapping: {confirmed_casts_map}."`
- These errors are never raised in safe mode, even with `allow_lossy_casts=False`.
- If no cast is proposed, `confirmed_casts` is ignored, even if it has wrong content.

## 4. Returns and guarantees
| Flags | Return value |
|---|---|
| none | `ArFrame` |
| `dry_run=True` | `DataQualityReport` (pre-cleaning `profile(frame)`); nothing applied |
| `return_report=True` | `(ArFrame, DataQualityReport)` |
| `explain=True` | `(ArFrame, CleanExplanation)` |
| both | `(ArFrame, DataQualityReport, CleanExplanation)` |

- The report always describes the **original** frame: `row_count` is the original row count, and its `suggestions` are the ones that were applied.
- **Suggestion order** (from `suggest_cleaning`):
  1. `("strip_whitespace", {"subset": [string cols with whitespace_count>0, in column order]})`
  2. `("cast_types", {col: dtype})`
  3. `("drop_duplicates", {"keep": "first"})`, only if the original data has full-row duplicates.
- **Identity:** if no step is applied (and in strict mode there are no duplicates left), the result is the input object itself, so `auto_clean(f) is f`.
- **`CleanExplanation` fields:**
  - `mode` is the mode string.
  - `rows_before` and `rows_after` are the original and final row counts.
  - `rows_removed = rows_before - rows_after`.
  - `steps` lists a `CleanStepRecord` for each applied step, in order: `step`, `kwargs` (the suggestion dict), `rows_before`, `rows_after`, `rows_removed`, `reason`.
- **Reasons**, taken from the suggestion's `confidence_reason`:
  - **strip_whitespace:** `"Trimming leading/trailing whitespace is a safe and highly recommended operation for columns with formatting anomalies."`
  - **cast_types** (per column, joined with `"; "`):
    - Default: `f"Column '{c}' conforms perfectly to {dtype} structure."`
    - Non-null ratio < 0.2 or < 0.5: a "very low" or "moderate non-null support (xx.x%)" variant.
  - **drop_duplicates:**
    - `duplicate_ratio <= 0.5`: `f"Low duplicate ratio ({r:.1%}) suggests standard redundant records."`
    - `duplicate_ratio > 0.5`: `f"High duplicate ratio ({r:.1%}) suggests potential schema or merge anomalies; review before dropping."`
    - Example: 1 duplicate in 3 rows gives `"33.3%"`.
  - **Strict final dedup:** `"Remove duplicate rows introduced by earlier strict-mode normalization steps."` with kwargs `{"keep": "first"}`.

## 5. Side effects
- No global state is changed and no files are written.
- Steps return new frames; the input frame is not modified (based on the visible code).

## 6. Branches and edge cases
- **When a cast is suggested** (only for columns whose ArFrame dtype is `"string"`), based on the non-null values after stripping:
  - All values in {true, false, 1, 0} (case-insensitive) give `"bool"`. So `["1","0","1"]` gives **bool, not int64**.
  - All values match `[+-]?\d+` give `"int64"`.
  - All values are numeric and finite give `"float64"`.
  - Example: `[" 1","2","3"]` gives int64, because values are stripped first.
- **Identifier columns** never get int64/float64 casts, but bool casts are still allowed. A column is an identifier if its lowercased name is in `id, uuid, zip, zipcode, zip_code` or ends in `_id`.
- **Safe mode** applies only strip; duplicates and casts are skipped and do not appear in `explain.steps`.
- **Strict final dedup:** in strict mode, if no `drop_duplicates` step ran but duplicates now exist, a dedup step is added. Example: `{"a": ["x", " x"]}`:
  - Strict: strip makes the rows equal, they are deduped to 1 row, and the steps are `["strip_whitespace", "drop_duplicates"]` with the final reason above.
  - Safe: 2 rows remain.
- **Zero-row frame:** no suggestions, so the input object is returned; explanation `steps == []`.
- **Numeric pandas columns** (int64/float64) never get cast suggestions or whitespace stripping.

## 7. Rules from the docstring
- `safe` is low-risk only.
- `strict` adds deterministic casts and exact duplicate removal.
- Casts need both `allow_lossy_casts=True` and an exact `confirmed_casts` previewed via `dry_run`.
- `dict(report.suggestions)["cast_types"]` works because a `CleaningSuggestion` is a 2-tuple.
- The cleaned frame is always element 0 of a returned tuple.

## 8. Testing hints
Example setup:
```python
df = pd.DataFrame({"name": [" alice", "bob", "carol"], "n": ["1", "2", "3"]})
f = ar.from_pandas(df)

ar.auto_clean(f)                     # name stripped; n stays string
rep = ar.auto_clean(f, mode="strict", dry_run=True)
casts = dict(rep.suggestions)["cast_types"]   # {'n': 'int64'}
out = ar.auto_clean(f, mode="strict", allow_lossy_casts=True, confirmed_casts=casts)
```
- Read values back with `ar.to_pandas(out)["name"].tolist()` and check `out.shape`.
- Check `f.dtypes["n"] == "string"` before casting.
- The exact dtype and values produced by `cast_types` come from `arnio.cleaning` and the C++ core, so assert only `out.dtypes["n"] == "int64"` and values like `[1, 2, 3]` with care.
- For error messages use `pytest.raises(ValueError, match=re.escape(...))`. Dict repr is `{'n': 'int64'}`.
- Pitfalls:
  - Avoid `"0"/"1"`-only columns unless you are testing bool.
  - Avoid id-like column names when testing numeric casts.
  - Duplicates are detected on the **raw**, unstripped data.
  - Whitespace counts only for string columns.
- Do not assert on report scores or the exact text of reasons beyond the strings given above.

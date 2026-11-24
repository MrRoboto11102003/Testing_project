# Notes: `openlens.validation.safe_float_conversion`

## 1. Purpose
- Converts a value to `float` without ever raising.
- Returns a `(success, value)` tuple: `(True, float_value)` on success, `(False, default)` on any failure.

## 2. Inputs
- `value` is checked in this exact order:
  1. `isinstance(value, bool)` gives `(False, default)` immediately. This covers `True` and `False`.
  2. `isinstance(value, (int, float))` tries `float(value)`, then the finite check.
  3. `isinstance(value, str)` tries `float(value)` with Python's normal string parsing, then the finite check.
  4. Anything else gives `(False, default)`.
- `default` (default `0.0`) is returned **as is** on failure, with no conversion and no validation. `default=None` gives `(False, None)`, and an object passed as `default` comes back as the identical object (`is`).
- `param_name` (default `"value"`) is **unused**. Nothing is logged or raised, and no message is produced.

## 3. Raises
- Nothing. The docstring states the contract "never raises".
- `float()` errors (`ValueError`, `OverflowError`, `TypeError`) are caught in both the numeric and string branches.
- Assert no exception for odd inputs: `None`, `[]`, `{}`, `object()`, `b"1"`, `10**400`.

## 4. Returns and guarantees
- The result is always a 2-element `tuple` (not a list), shaped `(bool, ...)`.
- On success the second item has type `float`, even for int input: `safe_float_conversion(5)` returns `(True, 5.0)` and `type(r[1]) is float`.
- On success the value is `float(value)` exactly.
- On failure the second item is `default` itself.
- The value is never NaN or inf on success. The finite check runs on both the numeric and string paths.

## 5. Side effects
- None. No logging, no mutation, no global state.

## 6. Branches and edge cases (concrete results)
- `(3.5)` → `(True, 3.5)`
- `(7)` → `(True, 7.0)`
- `(-0.25)` → `(True, -0.25)`
- `(0)` → `(True, 0.0)`. Zero is valid.
- `(True)` / `(False)` → `(False, 0.0)`
- `(True, default=9.0)` → `(False, 9.0)`
- `(float('nan'))` → `(False, 0.0)`
- `(float('inf'), 1.5)` → `(False, 1.5)`
- `(float('-inf'))` → `(False, 0.0)`
- `(10**400)` → `(False, 0.0)`. The `OverflowError` is caught.
- `(10**300)` → `(True, 1e300)`
- String cases:
  - `"3.5"` → `(True, 3.5)`
  - `"  42  "` → `(True, 42.0)`. Surrounding whitespace is allowed by `float()`.
  - `"1e3"` → `(True, 1000.0)`
  - `"-2"` → `(True, -2.0)`
  - `"1_000"` → `(True, 1000.0)`. Python's `float` accepts underscores.
- `"abc"`, `""`, `"   "`, `"1,5"` → `(False, default)`
- `"nan"`, `"inf"`, `"-Infinity"`, `"1e400"` → `(False, default)`. They parse to a non-finite value and are then rejected.
- `"True"` → `(False, default)`. It is a string, and `float("True")` fails.
- Other types:
  - `None` → `(False, 0.0)`
  - `[1.0]` → `(False, 0.0)`
  - `b"1"` → `(False, 0.0)`. Bytes are not handled.
  - `Decimal("1.5")` and `Fraction(1, 2)` → `(False, 0.0)`. They are not int, float or str subclasses, unlike `_validate_number`, which accepts `numbers.Real`.
- numpy values:
  - `numpy.float64(2.5)` → `(True, 2.5)`. It subclasses `float`.
  - `numpy.float32(2.5)` and `numpy.int64(3)` → `(False, 0.0)`. They are not int or float subclasses.
  - Testing numpy is optional and riskier. Prefer the stdlib cases.

## 7. Rules from the docstring and comments
- Booleans are rejected explicitly, because `bool` subclasses `int`.
- It must never raise. Large ints such as `10**400` used to escape through `OverflowError`, as did anything `TypeError` could reach.

## 8. Testing hints
- Import with `from openlens.validation import safe_float_conversion`.
- Compare whole tuples, e.g. `assert safe_float_conversion("2.5") == (True, 2.5)`. Also check `isinstance(result, tuple)` and `len(result) == 2`.
- Check `result[0] is True` / `is False`. This catches a regression that returns a truthy non-bool.
- Pass a non-zero default on every failure case (e.g. `default=-1.0`) so a regression that returns `0.0` or a converted value is caught.
- Pass a sentinel default and check identity with `res[1] is sentinel` to confirm it is returned unconverted.
- Check the success type: `type(safe_float_conversion(5)[1]) is float`.
- Pitfalls:
  - Do not expect a `ValidationError` or any exception.
  - Do not expect `" 42 "` to fail; it succeeds.
  - Do not use `pytest.approx` with NaN.
  - `param_name` has no observable effect, so do not assert logging.

# Notes: `openlens.validation.validate_conic_constant`

## 1. Purpose
- Validates a conic constant K. It must be a real, finite number in the closed range [-100, 100].
- Returns the value converted to a Python `float`.
- The docstring says the bound is generous on purpose: hyperbolic surfaces often have K < -1, but a typo should still fail loudly.

## 2. Inputs
- `conic_constant`: any `numbers.Real` except `bool`.
  - Accepted: `int`, `float`, `fractions.Fraction`, `numpy.float32`, `numpy.float64`, `numpy.int64`.
  - Rejected: `str`, `None`, `complex`, `decimal.Decimal`, `bool`. `Decimal` is not registered as `numbers.Real`.
- `param_name: str = "conic constant"`: used verbatim as the prefix of every error message.
- Step 1 is the private helper `_validate_number(value, param_name)`, which runs these checks in order:
  1. Is it a `bool`?
  2. Is it a `numbers.Real`?
  3. Does `float(value)` succeed? An `OverflowError` or `ValueError` here is caught.
  4. Is the result NaN or ±inf?
- Step 2 is the range check: `value < -100.0 or value > 100.0` raises. Both ends are inclusive and valid.

## 3. Raises
All errors are `openlens.validation.ValidationError`. It subclasses `Exception` directly, not `ValueError`. With the default `param_name`, checks run in this order:

| Call | Exact message |
|---|---|
| `validate_conic_constant(True)` | `conic constant must be a number, got bool` |
| `validate_conic_constant("1")` | `conic constant must be a number, got str` |
| `validate_conic_constant(None)` | `conic constant must be a number, got NoneType` |
| `validate_conic_constant(1j)` | `conic constant must be a number, got complex` |
| `validate_conic_constant(10**400)` | `conic constant must be a finite number, got a value too large to represent (int)` |
| `validate_conic_constant(float('nan'))` | `conic constant must be a finite number, got nan` |
| `validate_conic_constant(float('inf'))` | `conic constant must be a finite number, got inf` |
| `validate_conic_constant(-float('inf'))` | `conic constant must be a finite number, got -inf` |
| `validate_conic_constant(150)` | `conic constant must be between -100 and 100, got 150.0` |
| `validate_conic_constant(-100.5)` | `conic constant must be between -100 and 100, got -100.5` |

- Message templates:
  - Type: `f"{param_name} must be a number, got {type(value).__name__}"`
  - Range: `f"{param_name} must be between -100 and 100, got {value}"`
- In the range message, `{value}` is the **float-converted** value. An int input of `150` therefore prints `150.0`.
- The bounds print as `-100` and `100`, because they are int literals inside the f-string text.
- In the NaN/inf message, `{value}` is the **original** input.
- A custom name changes the prefix: `validate_conic_constant(200, "K1")` gives `K1 must be between -100 and 100, got 200.0`.
- The `10**400` case chains the original error: `__cause__` is an `OverflowError`.

## 4. Returns and guarantees
- Always returns a built-in `float`, never the input object:
  - `validate_conic_constant(-1)` returns `-1.0` with `type(...) is float`.
  - `validate_conic_constant(np.float32(0.5))` returns `0.5` with `type(...) is float`.
  - `validate_conic_constant(Fraction(1, 4))` returns `0.25`.
- The numeric value is unchanged. Values are never clamped or rounded.
- Boundary values are accepted:
  - `-100` returns `-100.0`.
  - `100` returns `100.0`.
  - `100.0` returns `100.0`.
  - `0` returns `0.0`.
  - `-0.0` returns `-0.0`.

## 5. Side effects
- None: no logging, no global state, no I/O.

## 6. Branches and edge cases
- Bool is rejected before the Real check (`False` is an int subclass): `validate_conic_constant(False)` raises "got bool".
- Valid typical values pass through:
  - Sphere: `0` returns `0.0`.
  - Parabola: `-1` returns `-1.0`.
  - Hyperbola: `-2.5` returns `-2.5`.
  - Oblate: `0.3` returns `0.3`.
- Values just outside the range fail:
  - `100.0000001` and `-100.000001` raise the range error.
  - `1e300` raises the range error, not the finiteness error.
- Values just inside the range pass: `99.999` and `-99.999` are returned unchanged.

## 7. Rules stated in comments or docstrings
- The docstring's "Raises" section says: "If the value is not finite or outside [-100, 100]". Both bounds are inclusive.
- `_validate_number` deliberately accepts numpy scalars (e.g. `float32`) because they are `numbers.Real`.
- `float()` overflow on a huge int is reported as a `ValidationError`, never as a raw `OverflowError`.
- Inside the module, `validate_lens_parameters` calls this function with `param_name` set to `"conic constant 1"` and `"conic constant 2"`. For example, `validate_lens_parameters(50, -50, 5, 25, 1.5, conic_constant_1=101)` raises `conic constant 1 must be between -100 and 100, got 101.0`.

## 8. Testing hints
- Import with `from openlens.validation import validate_conic_constant, ValidationError`.
- Match messages with `pytest.raises(ValidationError, match=re.escape(msg))`. Messages contain `.` and `(`/`)`, so escape them.
- Test both boundaries for acceptance (`-100`, `100`) and a value just past each for rejection. These cases catch `<`/`<=` regressions and bound changes (e.g. ±10, ±1000).
- Assert `type(result) is float` for an int input. This catches a regression that returns the raw input.
- Assert the exact `got 150.0` text for an int input of 150. If the code formatted the original input instead, the message would read `got 150`.
- Do not use `pytest.raises(ValueError)`: `ValidationError` is not a `ValueError`.
- Do not expect `Decimal` to be accepted; it raises `... got Decimal`.

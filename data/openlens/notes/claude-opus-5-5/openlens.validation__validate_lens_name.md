# Notes: `openlens.validation.validate_lens_name(name)`

## 1. Purpose
- Checks that a lens name is a string, strips surrounding whitespace, turns an empty or blank name into `"Untitled"`, and rejects names longer than 100 characters.

## 2. Inputs
- `name`: must pass `isinstance(name, str)`. There is no default value.
  - `str` subclasses are accepted.
  - Nothing is converted. `int`, `None`, `bytes`, `list` and other types are all rejected.
- Normalisation is `name.strip()`, which removes all leading and trailing whitespace:
  - spaces, `\t`, `\n`, `\r`
  - Unicode whitespace such as `\u00a0`
- Inner whitespace is kept.

## 3. Raises
Import the exception with `from openlens.validation import ValidationError`. It subclasses `Exception` directly, **not** `ValueError`.

The checks run in this order:

1. **Type check, before anything else.**
   - Message template: `f"Name must be a string, got {type(name)}"`.
   - It uses `type(...)`, not `__name__`, so the text contains the class repr.
   - `validate_lens_name(123)` gives `"Name must be a string, got <class 'int'>"`.
   - `validate_lens_name(None)` gives `"Name must be a string, got <class 'NoneType'>"`.
   - `validate_lens_name(b"abc")` gives `"Name must be a string, got <class 'bytes'>"`.
   - `validate_lens_name(True)` gives `"... got <class 'bool'>"`.
2. **Strip.**
3. **Empty check.** A name that is empty after stripping returns `"Untitled"`; it does not raise.
4. **Length check.** If `len(stripped) > 100`:
   - Exact message: `"Name must be 100 characters or less"`.
   - `validate_lens_name("a" * 101)` triggers it.

## 4. Returns and guarantees
- Returns a `str`: the stripped name, or the literal `"Untitled"`.
- `validate_lens_name("  Lens A  ")` returns `"Lens A"`.
- `validate_lens_name("Lens A")` returns `"Lens A"`, which is equal to the input.
  - Do not assert identity with `is`; only check `==`.
- For a `str` subclass input, `.strip()` returns a plain `str`.

## 5. Side effects
- None. Strings are immutable, there is no logging and no global state.

## 6. Branches and edge cases
| Input | Result |
|---|---|
| `""` | `"Untitled"` |
| `"   "`, `"\t\n"` | `"Untitled"` |
| `"Untitled"` | `"Untitled"` |
| `"  x  "` | `"x"` |
| `"a b  c"` | `"a b  c"` (inner spaces kept) |
| `"a" * 100` | returned unchanged (exactly 100 is allowed) |
| `"a" * 101` | raises "Name must be 100 characters or less" |
| `"  " + "a"*100 + "  "` | `"a"*100` (length is measured **after** stripping, so no error) |
| `" " + "a"*101` | raises (101 characters after stripping) |
| `"é" * 100` | OK, because length counts characters, not bytes |
| `123`, `None`, `["x"]` | raises the type message |

## 7. Rules from the docstring
- Returns the validated name, or `"Untitled"` if it is empty.
- Raises `ValidationError` only for an invalid type. The 100-character limit is also enforced, although the docstring does not mention it.

## 8. Testing hints
- Use `pytest.raises(ValidationError, match=...)`.
  - Escape regex characters, or compare `str(excinfo.value)` exactly.
  - Example: `str(e.value) == "Name must be a string, got <class 'int'>"`.
  - Do **not** use `pytest.raises(ValueError)`; it will not catch this exception.
- Pin the boundaries: 100 characters passes and 101 raises.
- Pin that stripping happens before both the empty check and the length check, using the padded 100-character case.
- Pin the exact fallback string `"Untitled"`: capital U, no trailing space.
- A `bool` is rejected because it is not a `str`, which differs from the numeric validators in this module. The message shows `<class 'bool'>`.

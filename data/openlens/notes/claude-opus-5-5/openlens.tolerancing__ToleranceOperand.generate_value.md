# Notes: `openlens.tolerancing.ToleranceOperand.generate_value`

## 1. Purpose
- Draws one random deviation for a tolerance operand.
- Uses a uniform draw in `[min_val, max_val]`, or a zero-mean Gaussian draw clamped to `[min_val, max_val]`.

## 2. Inputs
- `self` is a `ToleranceOperand` dataclass, built positionally or by keyword:
  `ToleranceOperand(element_index, param_type, min_val, max_val, distribution="uniform", std_dev=0.0, surface=1)`.
  - Only `min_val`, `max_val`, `distribution` and `std_dev` are used here.
  - `element_index`, `param_type` and `surface` are ignored. Any `ToleranceType` works, e.g. `ToleranceType.THICKNESS`.
- `rng` (optional, default `None`) is any object with `.uniform(a, b)` and `.gauss(mu, sigma)` methods.
  - If it is `None`, the stdlib module `random` is used: `source = rng if rng is not None else random`.
  - The check is `is not None`, so any non-None object is used as the source.
- Nothing is validated. Distribution names, `std_dev` sign and `min_val <= max_val` are never checked.

## 3. Raises
- The function raises nothing itself.
- Errors come only from the source object. For example, an `rng` without a `.gauss` method raises `AttributeError` on the gaussian path.

## 4. Returns and guarantees
**Gaussian branch** (only when `distribution == "gaussian"`, exact, case-sensitive string compare):
- `sigma = self.std_dev`.
- If `sigma == 0` (so `0` or `0.0`), then `sigma = (max_val - min_val) / 6.0`.
- It draws `val = source.gauss(0, sigma)`. The mean is always **0**, not the midpoint of the range.
- It returns `max(self.min_val, min(self.max_val, val))`.
- When clamped, the returned object **is** `min_val` or `max_val` itself. If you passed ints, you get the int back (e.g. `-1`, not `-1.0`).

**Every other distribution value** (`"uniform"`, `"Gaussian"`, `"normal"`, `""`, …):
- Returns `source.uniform(self.min_val, self.max_val)`.
- Python's `uniform(a, b)` is `a + (b-a)*random()`, so it does not error when `min_val > max_val`.

Common to both branches:
- The result is not stored. The operand object is unchanged.

## 5. Side effects
- The only side effect is consuming the RNG state: `rng`'s state, or the global `random` state when `rng` is `None`.
- No fields of `self` are modified.

## 6. Branches / edge cases
| Input | Result |
|---|---|
| `ToleranceOperand(0, T, -0.1, 0.1)` with `rng=random.Random(42)` | equals `random.Random(42).uniform(-0.1, 0.1)` exactly |
| uniform, `min=max=0.5` | `0.5` |
| `distribution="gaussian", std_dev=0.0, min=-0.6, max=0.6` | sigma = 0.2; equals `max(-0.6, min(0.6, random.Random(s).gauss(0, 0.2)))` |
| gaussian, `std_dev=0.05` (nonzero) | uses sigma 0.05, ignoring the range |
| gaussian, `std_dev=100, min=-0.1, max=0.1` | essentially always exactly `-0.1` or `0.1` (clamped) |
| gaussian, `min=1.0, max=2.0` | mean 0, so almost always exactly `1.0` (clamped to min) |
| gaussian, `min=max=0, std_dev=0` | sigma 0, so `gauss(0, 0)` gives `0.0`; returns `0` (the int `min_val` if ints were passed) |
| gaussian, `min_val > max_val`, e.g. `min=1, max=-1` | always returns `min_val` (`1`), because `max(min_val, …)` wins |
| `distribution="Gaussian"` (capital G) | takes the **uniform** path |

## 7. Rules from comments / docstring
- When `std_dev` is not provided (it is 0), the range is assumed to be ±3 sigma, so sigma = (max - min)/6.
- Gaussian draws are clamped to the limits ("typically yes for manufacturing").
- `MonteCarloAnalyzer` always passes its own `random.Random(seed)`, so seeded runs are reproducible. The module-level `random` is only a fallback.

## 8. Testing hints
- For deterministic checks, compare against a fresh, identically seeded generator.
  - Example: `op.generate_value(random.Random(7)) == random.Random(7).uniform(lo, hi)`.
  - Use a **new** `Random` for each side, because `gauss` caches a second value internally.
- A fake rng pins the arguments exactly. This is the strongest test.
  ```python
  class Fake:
      def __init__(s, ret): s.ret, s.calls = ret, []
      def gauss(s, mu, sigma): s.calls.append(("gauss", mu, sigma)); return s.ret
      def uniform(s, a, b): s.calls.append(("uniform", a, b)); return s.ret
  op = ToleranceOperand(0, ToleranceType.THICKNESS, -0.3, 0.3, "gaussian")
  f = Fake(5.0)
  assert op.generate_value(f) == 0.3
  assert f.calls == [("gauss", 0, pytest.approx(0.1))]
  ```
  - With the fake: gaussian returning `-5.0` gives `-0.3`; returning `0.01` gives `0.01` (inside the range, not clamped).
  - On the uniform path, `calls == [("uniform", -0.3, 0.3)]` and the return value is passed through unchanged, even when it is outside the range (uniform results are not clamped).
- The `rng=None` path can be checked in two ways:
  - `random.seed(3)`, call the function, then compare with `random.seed(3); random.uniform(lo, hi)`.
  - Or `monkeypatch.setattr(random, "uniform", lambda a, b: 123.0)`. The module uses stdlib `random` directly.
- Check that a gaussian draw stays inside `[min_val, max_val]` over many draws (e.g. 1000 with a seeded rng).
- Check that the default `"uniform"` path also stays inside the range for `min <= max`.
- Pitfalls:
  - Do not assume the Gaussian mean is the midpoint of the range.
  - Do not assume a nonzero `std_dev` is overridden by the range.
  - Do not expect errors for bad distribution names.
  - Use `pytest.approx` for sigma, since (max - min)/6 is float arithmetic.

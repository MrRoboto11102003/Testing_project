# Notes: `GlobalOptimizer.optimize_simulated_annealing`

## 1. Purpose
- Runs a seeded simulated-annealing search over `self.variables`.
- Then runs a local simplex refinement (`optimize_simplex(max_iterations=50, start=best_values, callback=callback)`) from the best point found.
- Returns the result built by `self._finalize(...)`.

**Not visible here:** `_evaluate_design`, `optimize_simplex`, `_finalize` and `OptimizationVariable.clamp` live in `openlens.optimizer`. Anything about merit values, `success`, and which message is chosen depends on that module.

## 2. Inputs
- **`max_iterations`** (int, default `1000`)
  - Upper bound on SA loop iterations (`range(max_iterations)`).
  - `<= 0` means the loop never runs.
- **`initial_temperature`** (float, default `100.0`)
  - If `<= 0`: logs a WARNING on logger `openlens.global_optimizer` (`"initial_temperature=%r is not positive; clamping ..."`) and sets it to `1e-6`.
  - It is not rejected.
- **`cooling_rate`** (float or None, default `None`)
  - `None` → `_derive_cooling_rate(T0, max_iterations)`:
    - returns `1.0` if `max_iterations <= 0`;
    - returns `1.0` if `ratio = 1e-6 / T0` is `>= 1.0` (i.e. `T0 <= 1e-6`);
    - otherwise returns `ratio ** (1/max_iterations)`.
  - An explicit value is used as-is, with no validation.
- **`restart_threshold`**: unused.
- **`callback(iteration, best_merit, best_values)`**
  - Called at the start of every SA iteration.
  - Also forwarded to `optimize_simplex`, so it is called more times after the SA loop.
  - Exceptions it raises propagate (this is how the GUI cancels).
- **Seed**: comes from `GlobalOptimizer(..., seed=...)` → private `random.Random(seed)`. The module-level `random` is never touched.

## 3. Raises
- **No explicit raises.**
- **Empty `variables` with `max_iterations >= 1`**: `self._rng.randint(0, -1)` raises `ValueError`.
  - The message varies by Python version, so assert the type only.
- **Exceptions from `callback`** propagate unchanged. For example, `def cb(*a): raise RuntimeError("stop")` → `pytest.raises(RuntimeError)`.

## 4. Returns / guarantees
- Returns an `OptimizationResult` from `_finalize`.
- **Arguments passed to `_finalize`** (n = number of SA iterations actually run, m = `local_result.iterations`):
  - `initial_merit` = `_evaluate_design([v.current_value for v in variables])`. This is the start design's merit, computed before any perturbation.
  - `iterations` = `n + m`.
  - `best_values` and `final_merit` are taken from the simplex result.
  - `variable_history` = SA history (n dicts `{var.name: value}` of the *current*, accepted point after each step) + simplex history.
  - `merit_history` = `[starting_merit]` + n current merits + simplex `merit_history`.
- **Messages:**
  - Converged: `f"SA Converged after {n} iterations + {m} local steps"`.
  - Out of bounds: `f"SA finished after {n} iterations + {m} local steps outside variable bounds"`.
  - Which one is used, and `success`, is decided by `_finalize` (unseen). The likely rule is the converged message when inside the bounds; verify loosely with `startswith("SA ")` or `"iterations + " in msg`.
- **Value of n:**
  - Derived rate and T0 > 1e-6: n == `max_iterations` (the floor is reached at the last step).
  - `max_iterations <= 0`: n = 0, because `iteration` starts at -1. SA `variable_history` is empty and the SA part of `merit_history` is `[starting_merit]`.
  - Explicit rate: the loop breaks after the cooling step once `temperature < 1e-6`. Example: `initial_temperature=1.0, cooling_rate=0.5, max_iterations=100` → n = 20 (2^-20 ≈ 9.5e-7; 2^-19 ≈ 1.9e-6).
  - `cooling_rate >= 1` with T0 > 1e-6: never cools, so n = `max_iterations`.

## 5. Side effects
- Does **not** write to `var.current_value` of the caller's `OptimizationVariable` objects (the comment says this explicitly).
- Consumes the instance's private RNG, so a second call on the same instance draws different numbers.
- Logs a warning only for `initial_temperature <= 0`.
- Mutation of `self.system` by simplex/_finalize is not visible here; the comment says `_apply_variables` returns a deepcopy.

## 6. Branches / edge cases
- **Each step:**
  - Picks one random index, `delta = step_size * clamp(T/T0, 0.01, 1.0) * gauss(0,1)`.
  - The neighbour value is `var.clamp(old + delta)`.
  - Only ONE variable changes per step.
- **Acceptance:**
  - `delta_E < 0` → accept.
  - Otherwise, if `T < 1e-9` → prob 0.
  - Otherwise prob = `exp(-delta_E/T)`, accepted if `rng.random() < prob`.
  - The best point is updated only on strict improvement.
- **`initial_temperature <= 0`** (e.g. `-5` or `0`):
  - Clamped to `1e-6`; the derived rate is `1.0`.
  - Temperature stays at `1e-6`, which is not `< 1e-6`, so the run lasts the full `max_iterations` and a warning is logged.
- **`0 < initial_temperature < 1e-6`** (e.g. `1e-12`), not clamped:
  - Derived rate is `1.0`, and the temperature is already below the floor.
  - Breaks after exactly **1** SA iteration.
  - Since `T < 1e-9`, worse neighbours are never accepted.
- **`max_iterations=0`**: SA is skipped; the simplex still runs from the start values.

## 7. Rules from comments
- `seed=` must make runs reproducible, using a private generator.
- The default schedule must last exactly `max_iterations`.
- `initial_merit` is the starting design's merit, not `merit_history[0]` after a perturbation.
- Caller variables must not be mutated.
- The callback is also consulted during the final refinement.

## 8. Testing hints
- **Building inputs:**
  - Build a system with `sys = OpticalSystem(); sys.add_lens(Lens())`.
  - Build variables like `OptimizationVariable("r1", 0, "radius_of_curvature_1", 50.0, 20.0, 200.0, step_size=5.0)`.
  - Use a target like `OptimizationTarget("focal_length", 100.0)`.
  - Valid parameter and target names are defined in `openlens.optimizer` (unseen). Use the obvious ones above and avoid asserting merit values.
- **Speed:** keep `max_iterations` small (5–30), since the simplex adds up to 50 iterations.
- **Robust assertions:**
  - Same seed with two freshly built optimizers gives equal `best_values`, `final_merit` and `merit_history`.
  - `result.initial_merit == GlobalOptimizer(...)._evaluate_design([v.current_value for v in vars])`, or equivalently equals `merit_history[0]`.
  - `variables[i].current_value` is unchanged after the run.
  - Record callback args; the first call is `(0, initial_merit, start_values)`.
  - Count the calls whose first arg comes from the SA phase: `callback` is called at least n times, and the first n calls have iteration = 0..n-1 in order.
  - `max_iterations=0` → `"after 0 iterations"` appears in the message.
  - Explicit `T0=1.0, rate=0.5` → message contains `"after 20 iterations"`.
  - `T0=1e-12` → `"after 1 iterations"`.
  - With the derived rate and `max_iterations=10` → `"after 10 iterations"`.
  - `T0=-1` emits a WARNING (`caplog`) and runs the full `max_iterations`.
  - SA part of `variable_history`: the first n entries have keys = variable names, all values within [min, max], and consecutive entries differ in at most one key.
- **Pitfalls:**
  - `best_values` passed to the callback may be the same list object later rebound; copy it when recording.
  - Do not assert the exact `success` value or the exact simplex iteration count; both come from unseen code.
  - The total `iterations` field is `>= n`, not `== n`.

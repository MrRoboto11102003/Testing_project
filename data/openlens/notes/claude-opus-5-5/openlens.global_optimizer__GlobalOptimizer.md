# Notes: `openlens.global_optimizer.GlobalOptimizer`

## 1. Purpose
- Subclass of `LensOptimizer` that adds two global searches: simulated annealing (`optimize_simulated_annealing`) and a real-valued genetic algorithm (`optimize_genetic`).
- Each search ends with a local `optimize_simplex` run, then returns `self._finalize(...)`.
- `_evaluate_design`, `optimize_simplex`, `_finalize` and `OptimizationVariable.clamp` live in `openlens.optimizer`, which is **not shown**. Do not assume how they behave internally.

## 2. Inputs
- `__init__(system, variables, targets, constraints=None, seed=None)`: calls `super().__init__(system, variables, targets, constraints)`, then sets `self._rng = random.Random(seed)`. This is a private RNG; it does not reseed the global `random` module.
- Class constant `ANNEALING_FLOOR = 1e-6`.
- `_derive_cooling_rate(T0, n)` is a classmethod. Its checks run in this order:
  - `n <= 0` → `1.0`
  - `ratio = 1e-6/T0`
  - `ratio <= 0` (negative T0) → `1.0`
  - `ratio >= 1` (`T0 <= 1e-6`) → `1.0`
  - otherwise `float(ratio ** (1/n))`
  - Example: `(100, 1000)` → `pytest.approx(1e-8 ** 0.001)` ≈ 0.981748.
  - `T0 == 0` raises `ZeroDivisionError` when the helper is called directly. The public method clamps T0 first, so it never hits this.
- SA parameters:
  - `max_iterations=1000`
  - `initial_temperature=100.0`: a value `<= 0` logs a warning and is clamped to `1e-6`.
  - `cooling_rate=None`: `None` means the rate is derived with `_derive_cooling_rate(clamped T0, max_iterations)`.
  - `restart_threshold`: unused.
  - `callback(iter, best_merit, best_values)`
- GA parameters: `population_size=50`, `generations=50`, `mutation_rate=0.1`, `crossover_rate=0.7`, `callback`.

## 3. Raises
- **SA with an empty `variables` list and `max_iterations >= 1`:** `self._rng.randint(0, -1)` raises `ValueError`. The message depends on the Python version, so assert the type only.
- **GA with `population_size=0`:** `population[0] = current_design` raises `IndexError` ("list assignment index out of range").
- **Exceptions from `callback`:** these propagate unchanged. The GUI uses this for cancel. The callback is also forwarded to `optimize_simplex`.
- **`_tournament_select` on an empty population:** raises `ValueError("cannot select from an empty population")`. This cannot be reached through the public API.

## 4. Returns and guarantees
The method returns whatever `_finalize(**kw)` returns, presumably an `OptimizationResult`; how `_finalize` maps the keywords onto fields is unseen. The keywords are listed below.

**Simulated annealing.** Let `k` = number of SA loop iterations (`iteration + 1`) and `L = local_result`.
- `best_values = L.best_values`
- `final_merit = L.final_merit`
- `initial_merit` = merit of the starting `[v.current_value ...]`, evaluated before any perturbation.
- `iterations = k + L.iterations`
- `variable_history`: `k` dicts `{var.name: value}` holding the current (accepted) values, followed by `L.variable_history`.
- `merit_history = [starting_merit] + [current_merit after each iter] + L.merit_history`, so the SA part has length `k + 1`.
- `converged_message = f"SA Converged after {k} iterations + {L.iterations} local steps"`
- `out_of_bounds_message = f"SA finished after {k} iterations + {L.iterations} local steps outside variable bounds"`
- `optimize_simplex` is called as `(max_iterations=50, start=best_values, callback=callback)`.

**Genetic algorithm.**
- `iterations = generations + L.iterations`
- `merit_history = history_merit`: one entry per generation (the best merit so far), non-increasing. It does **not** include `L`'s history.
- `variable_history = L.variable_history` only.
- `converged_message = f"GA Completed {generations} generations"`
- `out_of_bounds_message = f"GA finished {generations} generations outside variable bounds"`
- `initial_merit` = the starting design's merit.
- `optimize_simplex` is called with `start=best_overall_design`.

## 5. Side effects
- The `OptimizationVariable` objects are **not mutated**: their `current_value` is unchanged after either method, as the comments state explicitly.
- No global RNG reseeding happens in this module.
- SA logs a warning (logger `openlens.global_optimizer`) when `initial_temperature <= 0`.

## 6. Branches and edge cases
**Simulated annealing**
- **Number of iterations `k`:**
  - With a derived rate and `T0 > 1e-6`, `k == max_iterations`.
  - With an explicit rate, the loop breaks once `temperature < 1e-6` after cooling. Example: `T0=1, cooling_rate=0.5` gives 0.5**20 < 1e-6, so `k = 20` (even with `max_iterations=1000`).
- **`max_iterations=0`:** no loop runs and `iteration=-1`, so `k=0`. The SA `merit_history` part is `[starting]`, `variable_history` is empty, and the message reads "SA Converged after 0 iterations + ...".
- **`initial_temperature=0` or `-5`:** clamped to `1e-6` and the rate becomes `1.0`. The temperature stays at `1e-6`, which is not `< 1e-6`, so the full `max_iterations` run.
- **`initial_temperature=1e-7` (positive but at or below the floor):** the rate is `1.0` and the loop breaks after the first iteration, so `k = 1`.
- **Acceptance rule:**
  - `delta_E < 0` → always accepted.
  - Otherwise, the acceptance probability is `exp(-dE/T)`, or `0` if `T < 1e-9`.
  - `best_merit` and `best_values` update only when the new merit is strictly lower.
- **Perturbation:** `scale = clamp(T/T0, 0.01, 1)`. One random index is changed by `step_size * scale * gauss(0,1)`, then passed through `var.clamp`.
- **Callback calls:** exactly `k` times from SA, with iteration indices `0..k-1`, before each perturbation. Further calls come from simplex (unseen).
- **Number of `_evaluate_design` calls in SA:** `1 + k`, plus any simplex calls.

**Genetic algorithm**
- **Initial population:** `population_size` individuals drawn with `uniform(min, max)`; individual 0 is replaced by the current design.
- **Evaluations:** `1 + generations * population_size` calls to `_evaluate_design` before simplex.
- **Callback calls per generation:** `1 + population_size`, all with the same `gen` index. The callback is skipped entirely when `best_overall_design` is empty, i.e. when there are no variables.
- **`generations=0`:** `merit_history == []`, `iterations == L.iterations`, message "GA Completed 0 generations".
- **Updating the best:** only on a finite `min_merit` strictly lower than the current best; NaN is ignored. Elitism always keeps the best design as `new_population[0]`.
- **Crossover:** `alpha*p1 + (1-alpha)*p2`.
- **Mutation:** `gauss(0, 0.1*(max-min))`, then `clamp`.
- **Tournament:** `k = min(3, len(pop))`, so `population_size` of 1 or 2 works.

## 7. Rules from comments
- `seed=` must make runs reproducible: same seed and same evaluate function give an identical result.
- `max_iterations` must be honoured by the default schedule; it must not stop early around ~360 iterations.
- `initial_merit` is the starting design's merit, not `merit_history[0]` of the perturbed run, and not the best of GA generation 0.
- GA `iterations` counts generations, not evaluations.
- `success` comes from `_finalize`'s bounds check, never from a hard-coded `True`.

## 8. Testing hints
**Make runs deterministic and cheap by overriding on the instance.** Instance attributes override the methods because the class calls them via `self`:
```python
opt = GlobalOptimizer(system, [var], [target], seed=1)
opt._evaluate_design = lambda vals: (vals[0] - 3.0) ** 2
calls = {}
opt.optimize_simplex = lambda **kw: (calls.update(kw), SimpleNamespace(best_values=list(kw["start"]),
    final_merit=0.0, iterations=0, variable_history=[], merit_history=[]))[1]
opt._finalize = lambda **kw: kw        # result is then the kwargs dict
```
- Build `system` with `s = OpticalSystem(); s.add_lens(Lens())`.
- Build `var` with `OptimizationVariable("v", 0, "radius_of_curvature_1", 0.0, -10.0, 10.0, step_size=1.0)`. Valid parameter strings and target names are defined in `optimizer.py` (unseen), so keep them simple; with stubs they barely matter.

**Checks that pin behaviour down:**
- `len(kw["variable_history"]) == max_iterations` with the derived rate.
- `kw["iterations"] == k`.
- `kw["merit_history"][0] == 9.0` (start at 0 against target 3).
- `kw["initial_merit"] == 9.0`.
- The exact message strings.
- `calls["max_iterations"] == 50`.
- `var.current_value == 0.0` after the run.
- Callback count and indices.
- Seeds: same seed gives an equal `best_values` / `merit_history`.
- `_derive_cooling_rate` values, compared with `pytest.approx`.
- The GA `merit_history` length is `generations` and the list is non-increasing.
- SA `best_values` passed to simplex has merit `<=` the starting merit.

**Pitfalls:**
- Do not assert exact random values or exact real-optics merits.
- Do not assert `result.message` or `success` without stubbing `_finalize`, because that mapping is unseen.
- Keep runs small: the real simplex runs up to 50 steps.
- Use `approx` for floats.

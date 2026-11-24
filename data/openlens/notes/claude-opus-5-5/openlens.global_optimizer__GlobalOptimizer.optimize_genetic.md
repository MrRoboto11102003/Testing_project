# Notes: `GlobalOptimizer.optimize_genetic` (openlens/global_optimizer.py)

## 1. Purpose
- A real-valued genetic algorithm over `self.variables`. It uses tournament selection, arithmetic crossover, Gaussian mutation and elitism.
- It then runs a local `optimize_simplex` from the best design and returns `self._finalize(...)`.

## 2. Inputs
- `population_size: int = 50`: number of individuals. Not validated.
- `generations: int = 50`: number of GA loops (`range(generations)`). Values ≤ 0 mean no loop.
- `mutation_rate: float = 0.1`: chance per gene of a Gaussian mutation (`rng.random() < rate`).
- `crossover_rate: float = 0.7`: chance per child of crossover (`rng.random() < crossover_rate`). Otherwise the child is a copy of parent1.
- `callback(gen, best_merit, best_design)`: optional. Also passed on to `optimize_simplex(callback=callback)`.
- Instance state used:
  - `self.variables`: list of `OptimizationVariable` with `current_value`, `min_value`, `max_value`, `clamp()`.
  - `self._rng = random.Random(seed)`, built in `__init__`.
- The following are inherited from `LensOptimizer` in `openlens/optimizer.py`, which is NOT shown, so their behaviour is unknown:
  - `self._evaluate_design(values)`
  - `self.optimize_simplex(max_iterations=50, start=..., callback=...)`
  - `self._finalize(**kw)`

## 3. Raises
- `population_size <= 0` raises `IndexError: list assignment index out of range`.
  - Cause: `population[0] = current_design` on an empty list.
  - This happens before any evaluation and even when `generations=0`.
  - Example: `opt.optimize_genetic(population_size=0)`.
- An exception raised inside `callback` propagates unchanged. This is the GUI's cancel mechanism.
  - Example: `callback=lambda *a: (_ for _ in ()).throw(RuntimeError("stop"))` raises `RuntimeError("stop")` on the first call.
  - The first call is `callback(0, start_merit, design)`, but only if `generations >= 1` and there is at least one variable.
- `_tournament_select`'s `ValueError("cannot select from an empty population")` cannot be reached through this method.
- Any error from `_evaluate_design`, `optimize_simplex` or `_finalize` comes from unseen code.

## 4. Returns / guarantees
Returns whatever `self._finalize(...)` returns, normally an `OptimizationResult`. It is called with exactly these keyword arguments:
- `best_values = local_result.best_values`
- `final_merit = local_result.final_merit`
- `initial_merit = starting_merit`
  - This is `_evaluate_design([v.current_value for v in variables])`, computed once before the loop.
- `iterations = generations + local_result.iterations`
  - This counts generations, not evaluations.
- `variable_history = local_result.variable_history`
  - Only the simplex history; the GA adds nothing.
- `merit_history = history_merit`
  - One entry per generation, so `len == max(generations, 0)`.
  - Each entry is the best-so-far merit, so the list is non-increasing.
  - It does NOT include the starting merit.
- `converged_message = f"GA Completed {generations} generations"`
- `out_of_bounds_message = f"GA finished {generations} generations outside variable bounds"`

How `_finalize` maps these to `success`, `message`, `improvement` and `optimized_system` is in the unseen module.

## 5. Side effects
- The `OptimizationVariable` objects are not mutated (`current_value` is unchanged). The source comment says this is deliberate.
- `self._rng` state advances.
- Whether `self.system` changes depends on the unseen `_evaluate_design` and `_finalize`. A comment says `_apply_variables` returns a deepcopy.

## 6. Branches and edge cases
**Initial population**
- `population_size` individuals, each with one `rng.uniform(min, max)` per variable.
- Individual 0 is then replaced by the current design. That design is not clamped, so it may lie outside the bounds.

**Each generation `gen`**
- The callback fires once at the start of the generation, then once before each individual's evaluation.
  - That is `1 + population_size` calls per generation, all with the same `gen`.
  - Condition: `callback and best_overall_design`. With zero variables the design is `[]` (falsy), so the callback is never called in the GA loop.
- The best design is updated only if `math.isfinite(min_merit) and min_merit < best_overall_merit`.
  - Ties keep the old design.
  - NaN or inf minima are ignored.
- The next population is built as follows:
  - Slot 0 is the elite: a copy of the best design.
  - Every other slot is a child of two tournament winners. Tournament size is `k = min(3, len(population))`, sampled without replacement, and the lowest merit wins.
- `population_size=1`: only the elite survives, so no selection or crossover happens.

**Crossover and mutation**
- Crossover gives `child[i] = a*p1[i] + (1-a)*p2[i]`, with one `a = rng.random()` per child. This is not clamped.
- Mutation adds `gauss(0, 0.1*(max-min))` to a gene and then clamps it with `var.clamp`.
  - With `min == max`, sigma is 0.

**Other cases**
- `generations=0`:
  - There is no GA loop and no GA callback.
  - `merit_history == []` and `iterations == 0 + local.iterations`.
  - The message is `"GA Completed 0 generations"`.
  - The simplex starts from the current design.
- The simplex `start` is the best design list. With 1 generation and a deterministic merit, it is a design whose merit is ≤ `starting_merit`, because the current design is in generation 0.

## 7. Rules from comments
- `best_overall_design` is seeded with the starting design, so it is never `None`. Elitism must not raise `TypeError` on NaN or on `generations=0`.
- `initial_merit` must be the starting design's merit, not `history_merit[0]`.
- `iterations` counts generations, not `generations * population_size`.
- `success` comes from `_finalize`'s bounds check, not a hard-coded `True`.

## 8. Testing hints
Isolate the method from the unseen code by stubbing on the instance:
```python
opt = GlobalOptimizer(system, [var], [target], seed=1)
opt._evaluate_design = lambda v: (v[0] - 50.0) ** 2
calls = {}
def fake_simplex(**kw):
    calls.update(kw)
    return types.SimpleNamespace(best_values=list(kw["start"]), final_merit=0.5,
                                 iterations=3, variable_history=[{"x": 1}], merit_history=[9])
opt.optimize_simplex = fake_simplex
opt._finalize = lambda **kw: kw   # returns the kwargs dict for exact checks
```

Building the inputs:
- System: `OpticalSystem(); system.add_lens(Lens())`.
- Variable: `OptimizationVariable("x", 0, "radius_of_curvature_1", 50.0, 20.0, 100.0)`.
  - Which parameter and target names are valid is defined in `optimizer.py` (unseen). With the stubs above they do not matter, assuming `LensOptimizer.__init__` accepts them.
- Target: `OptimizationTarget("focal_length", 100.0)`.

What to check:
- `out["iterations"] == generations + 3`.
- `out["merit_history"]` has length `generations` and is non-increasing.
- `out["variable_history"] == [{"x": 1}]`.
- The exact message strings.
- `out["initial_merit"] == (start - 50)**2`.
- `calls["max_iterations"] == 50`.
- `calls["callback"] is cb`.
- Callback count `== generations * (population_size + 1)`.
- The first callback is `(0, starting_merit, [current])`.
- `var.current_value` is unchanged afterwards.

Reproducibility:
- The same seed with deterministic stubs gives identical outputs.
- Do not assert specific random values.

Pitfalls:
- The callback receives the live `best_overall_design` list. Copy it if you record it.
- Crossover children can leave the bounds when the start design is out of bounds. Mutated genes cannot.
- Don't assert `success`, `improvement` or `message` from the real `_finalize`. Its logic is unseen.

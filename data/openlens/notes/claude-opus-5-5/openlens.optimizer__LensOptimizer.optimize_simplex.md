# Notes: `LensOptimizer.optimize_simplex(max_iterations=100, tolerance=1e-6, callback=None, start=None)`

## 1. Purpose
- Runs a Nelder–Mead simplex search over `self.variables` to minimise `self._evaluate_design(values)`.
- Returns an `OptimizationResult`.
- `optimize(max_iterations, tolerance, callback)` simply calls `optimize_simplex(max_iterations, tolerance, callback)`.

## 2. Inputs
- **`max_iterations`** (int): upper bound for `for iteration in range(max_iterations)`. A value of `0` means the loop never runs.
- **`tolerance`** (float): a *relative* convergence bound. The loop converges when either:
  - `merit_range <= 0.0`, or
  - `merit_range <= max(tolerance * max(|m| for m in merit_values), 1e-12)`.
- **`callback(iteration, best_merit, best_values_list)`**:
  - Called once at the top of every loop iteration, after sorting.
  - The arguments are the 0-based iteration index, `merit_values[0]` and `simplex[0]`.
  - It is called even on the iteration that then converges or aborts.
- **`start`**: an optional sequence with one value per variable. It is copied with `list(start)`.
  - If omitted, the start is `[v.current_value for v in self.variables]`.
  - The variables' `current_value` is never written.
- Initial simplex:
  - Vertex 0 is the start.
  - Vertex `i+1` is the start with `values[i] += variables[i].step_size`.
  - The vertices are **not clamped** to bounds.
  - Exactly `n_vars + 1` calls to `_evaluate_design` happen before the loop.
- How `_evaluate_design(values)` scores a point:
  - It returns `merit_function.evaluate(self._apply_variables(values)) + bound_penalty`.
  - `bound_penalty = Σ 1e6 * (v - clamp(v))**2` over out-of-bounds variables.
  - Results are cached on `tuple(round(v, 10))`, a FIFO cache capped at 4096 entries.
  - A NaN merit is replaced by `INFEASIBLE_MERIT = 1e9`.

## 3. Raises
Both checks run before any evaluation, in this order:
1. **No variables.** `ValueError("optimize_simplex needs at least one variable; a design with nothing to optimise is not an optimisation problem")`.
   - Trigger: `LensOptimizer(OpticalSystem(), [], []).optimize_simplex()`.
2. **Wrong `start` length.** `ValueError(f"start has {len(start)} value(s) but there are {n_vars} variable(s)")`.
   - Example: 1 variable with `start=[1.0, 2.0]` gives `"start has 2 value(s) but there are 1 variable(s)"`.
   - `start=[]` with 1 variable also raises, because `[]` is not `None`.
- An empty variable list with a bad `start` raises error 1, not error 2.
- `optimize()` raises the same errors.

## 4. Returns: `OptimizationResult`
- **`initial_merit`**: the merit of the start vertex.
- **`variable_history`**:
  - Starts as `[{var.name: start_value, ...}]`.
  - One dict is appended per iteration that neither converged nor aborted.
- **`merit_history`**:
  - Starts as `[initial_merit]` and is appended in step with `variable_history`, so the two always have the same length.
  - It is the best-so-far value, so it is non-increasing.
- **`iterations`**: `last_iteration + 1`.
  - Equals `1` when `max_iterations=0`.
  - Equals `k+1` if the loop breaks at iteration `k`.
  - Equals `max_iterations` if the loop runs to the end.
- **Normal exit**, via `_finalize`:
  - `best_values = list(simplex[0])` and `final_merit = merit_values[0]`.
  - `optimized_system = self._apply_variables(best_values)`, a deep copy; `self.system` is untouched.
  - `best_valid` is true when every variable satisfies `min <= v <= max`.
  - `feasible` is `not merit_function.has_hard_violation(optimized_system)`, i.e. the physical penalty is below `1e8`.
  - `success = best_valid and feasible`.
  - Message:
    - success: `f"Converged after {N} iterations"`. This is printed even when the iteration budget ran out without converging.
    - out of bounds: `f"Finished after {N} iterations outside variable bounds"`.
    - in bounds but a hard violation: `"Converged, but the design breaks a physical constraint (it cannot be made as it is)"`.
- **`improvement`**: `(initial - final) / initial * 100` if `initial_merit > 0`, else the int `0`.
- **Abort exit ("asphyxiation")**:
  - Checked inside the loop, before the convergence test.
  - Condition: `n_clamped > n_vars/2 and merit_range < tolerance`, where `n_clamped` counts vertices with any variable out of bounds.
  - Result: `success=False`, `iterations=iteration+1`, `best_values=list(simplex[0])`.
  - Message: `f"Aborted after {iteration+1} iterations: {n_clamped} of {n_vars+1} simplex vertices stuck outside variable bounds"`.
  - `has_hard_violation` is not checked on this path.

## 5. Side effects
- Does not mutate `self.system`, `self.variables` or the `start` list.
- Fills `self._merit_cache`.
- Sorting by merit is stable, so tied vertices keep their order: the start vertex stays `simplex[0]` when tied.

## 6. Branches with concrete inputs
Fixture for all cases:
- `sys = OpticalSystem()` (empty, so no physical penalty: no elements and no gaps).
- `targets=[]`, so `merit_function.evaluate` is `0.0` and the merit equals the bound penalty.
- `parameter="dummy"`: unknown names are a no-op in `_apply_single_variable`.

```python
v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=1.0)
opt = LensOptimizer(OpticalSystem(), [v], [])
```

- **Converge at iteration 0**: `opt.optimize_simplex()`.
  - Both vertices (5, 6) have merit 0, so the range is 0.
  - Result: `success True`, `iterations 1`, `best_values [5.0]`, `final_merit 0.0`, `improvement 0`.
  - Message: `"Converged after 1 iterations"`.
  - History lengths are 1. The callback is called exactly once, with `(0, 0.0, [5.0])`.
- **`max_iterations=0`, in bounds**:
  - Same values as above, but the callback is never called.
  - `_evaluate_design` is called exactly twice, with `[5.0]` then `[6.0]`.
- **`max_iterations=0`, `start=[5.0]`, bounds [0, 1]**:
  - `initial_merit = final_merit = 1e6*16 = 1.6e7`, `improvement 0.0`.
  - `success False`, message `"Finished after 1 iterations outside variable bounds"`, `best_values [5.0]`.
- **Abort**: the same out-of-bounds variable with `opt._evaluate_design = lambda vals: 0.0` (instance override).
  - Vertices 5 and 6 are both out of bounds, and the range 0 is `< 1e-6`.
  - Result: `success False`, `iterations 1`, `best_values [5.0]`.
  - Message: `"Aborted after 1 iterations: 2 of 2 simplex vertices stuck outside variable bounds"`.
- **Real search**:
  - Setup: `opt._evaluate_design = lambda vals: (vals[0]-3.0)**2`, bounds [-100, 100], start 0, `max_iterations=200`.
  - Expect `best_values[0] ≈ 3` (abs tolerance 1e-3), `success True` and `final_merit <= initial_merit (=9.0)`.
  - `merit_history` is non-increasing.
- **Budget exhausted**: the quadratic above with `max_iterations=2`.
  - Expect `iterations == 2` and `len(merit_history) == 3`, provided it does not converge within 2 iterations.

## 7. Rules from comments
- Bounds act as penalties, not walls; nothing is clamped during the search.
- A best point outside the box is never reported as a success.
- A hard physical violation is never reported as a success.
- `start` exists so that GlobalOptimizer does not need to write into the caller's variables.

## 8. Testing hints / pitfalls
- Prefer the empty `OpticalSystem()` plus `"dummy"` parameter setup. Real lens targets go through ray tracing and aberration code (other modules), whose numbers you cannot predict.
- Overriding `opt._evaluate_design` on the instance works because the method calls `self._evaluate_design`. `_finalize` still uses the real `_apply_variables` and `has_hard_violation`, which is fine with an empty system.
- Do not assert exact iteration counts or vertex sequences for a real search; assert bounds or approximate optima instead.
- The int `0` compares equal to `0.0`, so `improvement == 0` is safe.
- After the call, assert that `v.current_value` is unchanged and that `result.optimized_system is not opt.system`.

# Notes: `LensOptimizer.optimize_gradient_descent(max_iterations=100, learning_rate=0.1, tolerance=1e-6)`

## 1. Purpose
- Minimises the merit by gradient descent with backtracking.
- Gradients come from forward finite differences (eps = `1e-5`).
- Starts from each variable's `current_value`.
- Returns an `OptimizationResult`.

## 2. Inputs
- `self.variables`: list of `OptimizationVariable`. The start point is `[v.current_value for v in variables]`.
- `max_iterations`: int, default 100. Used as `range(max_iterations)`, so 0 or a negative value means the loop never runs.
- `learning_rate`: float, default 0.1. It is the initial `step_size`.
  - `step_size` is **never reset or increased**. It only halves during backtracking and carries over between iterations.
- `tolerance`: float, default 1e-6. This is an **absolute** test: the loop stops when one accepted step lowers the merit by less than `tolerance`.
- No validation of any argument. There is no `callback` parameter (unlike `optimize_simplex`).
- Zero variables is allowed here. It does **not** raise; `optimize_simplex` does.

## 3. Raises
- The function raises nothing itself. Errors from `self.merit_function.evaluate` propagate.
- With the real `MeritFunction`, an unknown target name raises `ValueError` at the first (initial) evaluation. Exact message:
  `"Unknown optimization target 'foo'; known targets: astigmatism, chromatic_aberration, coma, focal_length, mtf, rms_spot_radius, spherical_aberration, system_length"`
  - Trigger: `LensOptimizer(sys, vars, [OptimizationTarget("foo", 0.0)]).optimize_gradient_descent()`

## 4. Algorithm (exact)
1. `initial_merit = _evaluate_design(x0)`.
   - `_evaluate_design(vals)` returns `merit_function.evaluate(_apply_variables(vals)) + bound_penalty`.
   - `bound_penalty = Σ 1e6*(v - clamp(v))**2` over variables outside `[min_value, max_value]`.
   - Results are cached by `tuple(round(v,10))` (FIFO, max 4096 entries).
   - A NaN merit is returned as `INFEASIBLE_MERIT = 1e9`.
2. Histories start as:
   - `variable_history = [dict(name -> x0)]`
   - `merit_history = [initial_merit]`
3. For each iteration:
   - Compute the gradient: `g_i = (f(x + 1e-5·e_i) - f(x)) / 1e-5`. With more than 4 variables this runs in a thread pool; the values are the same.
   - Backtracking, up to 12 tries:
     - `new = x - step_size*g`.
     - Accept if `new_merit <= current_merit` (**ties are accepted**).
     - Otherwise `step_size *= 0.5`. Stop trying early if `step_size < 1e-12`.
   - If no try was accepted: break. Nothing is appended for this iteration.
   - If accepted: `x = new`, `current_merit = new_merit`, and both histories get one entry appended.
     - Then, if `old_merit - new_merit < tolerance`, break.
4. Build the result:
   - `iterations = last_iteration + 1`. This counts the stalled or converging iteration. It is **1 even when `max_iterations=0`**.
   - `final_merit = current_merit`, which is the last accepted value, or `initial_merit` if nothing was accepted.
   - `improvement = (initial-final)/initial*100` if `initial_merit > 0`, else `0`.
   - `success = all(v.is_valid(x_i))`. Only the variable bounds are checked: there is **no hard-violation check** (unlike simplex `_finalize`).
   - `message`:
     - valid: `f"Completed {iterations} iterations"`
     - otherwise: `f"Finished {iterations} iterations outside variable bounds"`
   - `optimized_system` is a fresh deep copy with the final values applied.
   - `best_values` is left at its default `[]`. It is **not** filled in.

## 5. Side effects
- `self.system` is not mutated; every evaluation deep-copies it.
- `OptimizationVariable.current_value` is **not** updated.
- `self._merit_cache` fills up across calls. A second call with the same values reuses cached merits.

## 6. Branches / edge cases (using a fake merit, see §8)
- **Smooth bowl**, `f = (x-3)**2`, start `x=5`, lr 0.1:
  - The first step goes to `x ≈ 4.6` (merit ≈ 2.56).
  - Each step shrinks `x-3` by about 0.8, so it converges in roughly 30 iterations.
  - Expect:
    - `final_merit < 1e-4`
    - `success` is True; message starts with `"Completed "`
    - `merit_history` is non-increasing, and `len(merit_history) == len(variable_history)`
    - `variable_history[0] == {name: 5.0}`
    - `improvement ≈ 100`
  - Do not assert the exact iteration count.
- **Constant merit** `c > 0`:
  - Gradient is 0 and `new == x`, so the cached tie is accepted. Improvement 0 < tol, so it breaks.
  - Result: `iterations == 1`, `len(merit_history) == 2`, both entries equal `c`, `improvement == 0`.
- **Stall**, `f = |x-3|` starting exactly at `x=3`:
  - The forward gradient is `+1`, and every trial step raises the merit, so all 12 tries are rejected and it breaks.
  - Result: `iterations == 1`, `merit_history == [0.0]`, `len(variable_history) == 1`, `final_merit == 0`, `improvement == 0` (because initial is 0).
- **`max_iterations=0`**: no loop.
  - Result: `iterations == 1`, histories have length 1, `final_merit == initial_merit`, `improvement == 0`.
  - If `current_value` is in bounds: `"Completed 1 iterations"`, `success` is True.
- **Out of bounds**, `max_iterations=0`, `current_value=5` with bounds `[0,4]`:
  - `success` is False.
  - `message == "Finished 1 iterations outside variable bounds"`.
  - `initial_merit == fake(5) + 1e6`.
- **No variables**, `variables=[]`:
  - `iterations == 1`, `success` is True, `"Completed 1 iterations"`.
  - `variable_history == [{}, {}]`, `merit_history == [m, m]`.

## 7. Rules from comments
- A step that does not reduce the merit is never taken (backtracking).
- Variables are not clamped; bounds act as penalties only.
- `final_merit` is the current merit, not `merit_history[-1]` (after a stall, the two can differ in meaning).

## 8. Testing hints
- Build the system with: `s = OpticalSystem(); s.add_lens(Lens(thickness=5.0))`.
- Variable example: `OptimizationVariable("t", 0, "thickness", 5.0, 1.0, 20.0)`. This writes `system.elements[0].lens.thickness`. The `Lens` setter behaviour is in `lens.py`, which is not shown, so stay within ordinary values (about 1–20).
- Use a fake merit for exact, cheap behaviour, and set it before calling (the cache is empty after construction):
  ```python
  class Fake:
      def evaluate(self, system): return (system.elements[0].lens.thickness - 3.0)**2
  opt = LensOptimizer(s, [var], [OptimizationTarget("system_length", 0.0)])
  opt._set_merit_function(Fake())   # or opt.merit_function = Fake()
  ```
  Keep bounds wide (e.g. `[1, 20]`) so no bound penalty applies. Note that start `x=3` is below the example variable's `min_value` of 1? No: 3 is inside `[1, 20]`, so the `|x-3|` stall test is in bounds.
- Things that should hold after a call:
  - `opt.system.elements[0].lens.thickness == 5.0`
  - `var.current_value == 5.0`
  - `res.optimized_system is not opt.system`
  - `res.best_values == []`
- The real `system_length` / `focal_length` targets depend on `OpticalSystem` code (not shown). Avoid asserting exact numbers from them.
- Use `pytest.approx` for floats. The finite-difference gradient carries a bias of about 1e-5.

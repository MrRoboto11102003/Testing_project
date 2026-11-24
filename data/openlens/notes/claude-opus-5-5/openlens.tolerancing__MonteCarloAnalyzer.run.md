# Notes: `openlens.tolerancing.MonteCarloAnalyzer.run`

## 1. Purpose
- Runs `num_trials` randomly perturbed copies of the design, all in place on the same system object.
- For each trial it reapplies compensators, measures the RMS spot radius, and restores the nominal state afterwards.
- Returns a statistics dict, and stores per-trial records in `self.results`.

## 2. Inputs
- `self` is built with `MonteCarloAnalyzer(system, tolerances, seed=None, compensators=None, comp_sweeps=1)`.
  - `nominal_system` is the **same object** as `system`, not a copy.
  - `self._rng = random.Random(seed)` is a private generator. The global `random` state is never touched.
  - `compensators` is converted to a list, or `[]` if falsy.
  - `comp_sweeps` becomes `max(1, comp_sweeps)`.
- `num_trials: int = 100`. Must be `> 0`.
- `criterion: str = "rms_spot_radius"`. This is the only accepted value (`SUPPORTED_CRITERIA == "rms_spot_radius"`, a plain string).
- `criterion_limit: float = 0.05`. A trial passes when `value <= criterion_limit` (inclusive).

## 3. Raises
The checks run in this order, before any tracing:
1. `ValueError(f"num_trials must be positive, got {num_trials}")`
   - `run(0)` gives `"num_trials must be positive, got 0"`.
   - `run(-3)` gives `"... got -3"`.
   - `run(0, criterion="x")` also raises the num_trials message, because this check comes first.
2. `ValueError(f"Unsupported criterion {criterion!r}; supported: 'rms_spot_radius'")`
   - `run(1, criterion="mtf")` gives `"Unsupported criterion 'mtf'; supported: 'rms_spot_radius'"`.

Nothing else propagates:
- Spot evaluation failures become `inf`.
- A trial that raises is logged as a warning and skipped.

## 4. Returns and guarantees
The result is a new dict with exactly these 12 keys:

| Key | Value |
|---|---|
| `"nominal"` | RMS spot of the unperturbed system. `inf` if the trace fails or `rms_radius` is None. |
| `"mean"` | `statistics.mean(finite)`, or `inf` if there are no finite values. |
| `"std_dev"` | `statistics.stdev(finite)` if `len(finite) > 1`, else the int `0`. |
| `"min"` | `min(finite)`, or `inf`. |
| `"max"` | `max(values)`. Includes `inf` values. The int `0` if no trial was recorded. |
| `"yield"` | `pass_count / num_trials * 100`. The denominator is always `num_trials`, even when trials failed. |
| `"trials"` | `num_trials` |
| `"criterion"` | `criterion` |
| `"limit"` | `criterion_limit` (echoed) |
| `"90th_percentile"` | Linear-interpolated percentile (numpy "linear") of `finite`, or `inf`. |
| `"vignetted_trials"` | Count of recorded trials whose value is non-finite. |
| `"skipped_operands"` | List of the operands that could not be applied (see below). |

Details:
- `finite` is the list of finite `value`s; `values` is all recorded `value`s.
- `"skipped_operands"` holds sorted unique strings `f"Element {element_index+1} {param_type.value}"`, e.g. `"Element 1 Focus"`.
  - It is taken from the **last** trial's apply step.
  - It is `[]` when everything applied.

`self.results` is reset to `[]` at the start of each run. Each successful trial appends:
```python
{"trial": i, "perturbations": {...}, "compensators": {...}, "value": float, "passed": bool}
```
- `perturbations`:
  - Keys are `f"El_{element_index}_{param_type.name}"`, e.g. `"El_0_RADIUS_1"`.
  - Values are the drawn deltas.
  - The value is recorded **even if the operand was skipped**.
  - Duplicate keys overwrite each other (for example, IRREGULARITY on surface 1 and surface 2 share one key).
- `compensators` is `{"focus_shift_mm": 0.0}` when no compensators are given. Otherwise it also holds `El_..` keys for mechanical compensators.

## 5. Side effects
- The system is perturbed in place during each trial. In a `finally` block it is restored to the state captured **after** the nominal evaluation:
  - node positions and rotations;
  - r1, r2, thickness;
  - `model_glass_mode`, `model_nd`, `model_vd`, `refractive_index`;
  - air-gap thicknesses.
- After `run` returns, these lens and gap values equal their values before the call.
- Sets `self.results` and `self._skipped_operands`.
- Logs warnings for failed trials and skipped operands.
- Writes no files.

## 6. Branches and edge cases

**Empty `tolerances=[]`, no compensators.**
- Every trial is the nominal system, which is deterministic.
- So `mean == min == max == 90th_percentile == nominal` (when finite).
- `std_dev == 0`.
- `skipped_operands == []`.
- `yield` is `100.0` if `nominal <= limit`, else `0.0`.

**`criterion_limit` extremes.**
- `criterion_limit=float("inf")` gives `yield == 100.0`, because `inf <= inf` is True.
- `criterion_limit=-1.0` gives `yield == 0.0` and every `passed` is False.

**`num_trials=1`.**
- `std_dev == 0` (int).
- `len(results) == 1`.
- `90th_percentile` equals the single value.

**Operands that cannot be applied.** `_apply_value` returns False for these, so they are reported in `skipped_operands`:
- FOCUS in the tolerances list gives `"Element 1 Focus"`.
- An `element_index` outside the element range gives, e.g. for index 5, `"Element 6 Radius 1"`.
- DECENTER_X on element 0 gives `"Element 1 Decenter X"`.
- AIR_GAP with `element_index >= len(air_gaps)`.
- RADIUS on a non-finite (flat) radius.
- IRREGULARITY when the fringe radius cannot be found.

Skipped operands still consume an RNG draw, still appear in `perturbations`, and leave the system unchanged.

**Random draws.**
- Uniform operands draw `rng.uniform(min_val, max_val)`.
- With `min_val == max_val == v`, the draw is exactly `v`.
- Gaussian operands draw `rng.gauss(0, sigma)` clamped to `[min, max]`, where `sigma = std_dev`, or `(max - min) / 6` if `std_dev == 0`.

**Exceptions inside a trial.** The trial is omitted from `results` (so `trial` indices can have gaps) but still counts in the yield denominator.

**FOCUS compensators.** A golden-section search over `[min(min_val), max(max_val)]` of the focus ops. This does not use the RNG. If `lo >= hi`, the shift is `lo`.

## 7. Rules from comments and docstrings
- Unsupported criteria must raise, not be silently echoed back.
- The restore must happen even when a trial raises, so a perturbed design is never left behind.
- Vignetted trials (`inf`) are excluded from mean/stdev/min/percentile but kept in `max` and counted in `vignetted_trials`.
- Operands that are not applied must be reported, because silently ignoring them inflates the yield.
- A seeded run must be reproducible and isolated from the global `random` module.

## 8. Testing hints

Build a system:
```python
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem, create_doublet
sys_ = OpticalSystem(); sys_.add_lens(Lens())   # or create_doublet()
```

Pitfalls (these depend on code outside this module):
- Absolute RMS values come from `SpotDiagram.trace_spot`, whose code is not shown here. **Do not hard-code spot values.** Instead:
  - compare values against `stats["nominal"]`;
  - use `math.isfinite`;
  - use limit extremes (`inf` / `-1`).
- Whether `Lens()` has finite radii, and how many air gaps a system has, are also defined in modules not shown. FOCUS and an out-of-range `element_index` are the most reliable ways to trigger skips.

Reproducibility:
- Two analyzers with the same `seed` produce identical `results[*]["perturbations"]` and identical stats.
- The perturbation sequence is predictable:
  ```python
  r = random.Random(7)
  expected = [r.uniform(-0.1, 0.1) for _ in range(n_trials)]
  ```
  This matches `results[i]["perturbations"]["El_0_THICKNESS"]` for a single uniform operand and no failures.
- `random.seed(...)` or global `random` calls between runs must not change the results.

Restoration check: save `lens.thickness`, `radius_of_curvature_1` and gap thicknesses before `run`, then assert they are equal afterwards.

Pass/fail checks:
- `stats["yield"] == 100 * sum(r["passed"] for r in an.results) / num_trials`.
- `r["passed"] == (r["value"] <= limit)` for every record.

Use `pytest.raises(ValueError, match=...)` with the exact messages, escaping regex characters as needed. `re.escape` is safest.

Type pitfalls:
- `std_dev` can be the int `0`. Compare with `== 0`, not `is 0.0`.
- `"criterion"` is the string, and `"trials"` is the int passed in.

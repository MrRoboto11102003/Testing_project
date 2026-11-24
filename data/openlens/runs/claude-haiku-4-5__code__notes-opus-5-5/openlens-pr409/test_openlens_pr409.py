import pytest
from openlens import OpticalSystem, LensOptimizer, OptimizationVariable


def test_no_variables_raises_error():
    """optimize_simplex should raise ValueError when there are no variables."""
    opt = LensOptimizer(OpticalSystem(), [], [])
    with pytest.raises(ValueError, match="optimize_simplex needs at least one variable"):
        opt.optimize_simplex()


def test_start_wrong_length_raises_error():
    """optimize_simplex should raise ValueError when start has wrong length."""
    v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=1.0)
    opt = LensOptimizer(OpticalSystem(), [v], [])
    with pytest.raises(ValueError, match="start has 2 value.*but there are 1 variable"):
        opt.optimize_simplex(start=[1.0, 2.0])


def test_start_empty_raises_error():
    """optimize_simplex should raise ValueError when start is empty list."""
    v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=1.0)
    opt = LensOptimizer(OpticalSystem(), [v], [])
    with pytest.raises(ValueError, match="start has 0 value"):
        opt.optimize_simplex(start=[])


def test_converge_at_iteration_zero():
    """Simplex should converge immediately when merit range is zero."""
    v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=1.0)
    opt = LensOptimizer(OpticalSystem(), [v], [])
    result = opt.optimize_simplex()
    
    assert result.success is True
    assert result.iterations == 1
    assert result.best_values == [5.0]
    assert result.final_merit == 0.0
    assert result.improvement == 0
    assert "Converged" in result.message
    assert len(result.variable_history) == 1
    assert len(result.merit_history) == 1


def test_callback_called_on_each_iteration():
    """Callback should be called once per iteration."""
    v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=1.0)
    opt = LensOptimizer(OpticalSystem(), [v], [])
    callback_calls = []
    
    def callback(iteration, best_merit, best_values):
        callback_calls.append((iteration, best_merit, list(best_values)))
    
    result = opt.optimize_simplex(callback=callback)
    
    assert len(callback_calls) == 1
    assert callback_calls[0] == (0, 0.0, [5.0])


def test_callback_not_called_with_zero_iterations():
    """Callback should not be called when max_iterations is 0."""
    v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=1.0)
    opt = LensOptimizer(OpticalSystem(), [v], [])
    callback_calls = []
    
    def callback(iteration, best_merit, best_values):
        callback_calls.append(iteration)
    
    result = opt.optimize_simplex(max_iterations=0, callback=callback)
    
    assert len(callback_calls) == 0
    assert result.iterations == 1


def test_start_parameter_overrides_current_value():
    """start parameter should override variables' current_value."""
    v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=1.0)
    opt = LensOptimizer(OpticalSystem(), [v], [])
    result = opt.optimize_simplex(start=[7.0])
    
    assert result.best_values[0] == 7.0


def test_start_parameter_not_modified():
    """start list should not be modified by optimize_simplex."""
    v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=1.0)
    opt = LensOptimizer(OpticalSystem(), [v], [])
    start_list = [3.5]
    result = opt.optimize_simplex(start=start_list)
    
    assert start_list == [3.5]


def test_variables_current_value_not_modified():
    """Variables' current_value should not be modified by optimize_simplex."""
    v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=1.0)
    original_value = v.current_value
    opt = LensOptimizer(OpticalSystem(), [v], [])
    result = opt.optimize_simplex(start=[8.0])
    
    assert v.current_value == original_value


def test_out_of_bounds_values_incur_penalty():
    """Out-of-bounds values should incur penalty; success should be False."""
    v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 1.0, step_size=1.0)
    opt = LensOptimizer(OpticalSystem(), [v], [])
    result = opt.optimize_simplex()
    
    assert result.final_merit > 0.0
    assert result.best_valid is False
    assert result.success is False
    assert "outside variable bounds" in result.message


def test_asphyxiation_abort_when_vertices_stuck():
    """Should abort when majority of vertices stuck outside bounds."""
    v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 1.0, step_size=1.0)
    opt = LensOptimizer(OpticalSystem(), [v], [])
    # Override to give zero merit, allowing asphyxiation to trigger
    opt._evaluate_design = lambda vals: 0.0
    
    result = opt.optimize_simplex()
    
    assert result.success is False
    assert "Aborted" in result.message
    assert "stuck outside variable bounds" in result.message


def test_history_tracking_consistency():
    """History lengths should match; merit_history should be non-increasing."""
    v = OptimizationVariable("x", 0, "dummy", 0.0, -100.0, 100.0, step_size=1.0)
    opt = LensOptimizer(OpticalSystem(), [v], [])
    opt._evaluate_design = lambda vals: (vals[0] - 10.0)**2
    
    result = opt.optimize_simplex(max_iterations=20)
    
    assert len(result.variable_history) == len(result.merit_history)
    assert result.merit_history[0] == result.initial_merit
    
    for i in range(1, len(result.merit_history)):
        assert result.merit_history[i] <= result.merit_history[i-1]


def test_multiple_variables_optimization():
    """optimize_simplex should work with multiple variables."""
    v1 = OptimizationVariable("x1", 0, "dummy", 0.0, -10.0, 10.0, step_size=1.0)
    v2 = OptimizationVariable("x2", 0, "dummy", 0.0, -10.0, 10.0, step_size=1.0)
    opt = LensOptimizer(OpticalSystem(), [v1, v2], [])
    
    opt._evaluate_design = lambda vals: vals[0]**2 + vals[1]**2
    result = opt.optimize_simplex(max_iterations=50)
    
    assert len(result.best_values) == 2
    assert result.final_merit <= result.initial_merit
    assert len(result.variable_history[0]) == 2
    assert "x1" in result.variable_history[0]
    assert "x2" in result.variable_history[0]


def test_optimized_system_is_separate_copy():
    """optimized_system should be a different object than the original system."""
    v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=1.0)
    sys = OpticalSystem()
    opt = LensOptimizer(sys, [v], [])
    result = opt.optimize_simplex()
    
    assert result.optimized_system is not sys


def test_max_iterations_zero_behavior():
    """With max_iterations=0, loop never runs but initial simplex evaluated."""
    v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=1.0)
    opt = LensOptimizer(OpticalSystem(), [v], [])
    result = opt.optimize_simplex(max_iterations=0)
    
    assert result.iterations == 1
    assert len(result.variable_history) == 1
    assert len(result.merit_history) == 1


def test_improvement_calculation_positive_merit():
    """improvement should be (initial - final) / initial * 100 when initial > 0."""
    v = OptimizationVariable("x", 0, "dummy", 0.0, -10.0, 10.0, step_size=1.0)
    opt = LensOptimizer(OpticalSystem(), [v], [])
    
    opt._evaluate_design = lambda vals: (vals[0])**2 + 100.0
    result = opt.optimize_simplex(max_iterations=50)
    
    if result.initial_merit > 0:
        expected = (result.initial_merit - result.final_merit) / result.initial_merit * 100
        assert result.improvement == pytest.approx(expected, rel=0.01)


def test_zero_merit_range_converges_regardless_of_tolerance():
    """Should converge when merit_range is zero, even with strict tolerance."""
    v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=1.0)
    opt = LensOptimizer(OpticalSystem(), [v], [])
    result = opt.optimize_simplex(tolerance=1e-15)
    
    assert result.success is True
    assert result.iterations == 1

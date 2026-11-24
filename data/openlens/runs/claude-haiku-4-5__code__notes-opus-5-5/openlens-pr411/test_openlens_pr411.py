import pytest
from openlens import (
    OpticalSystem,
    LensOptimizer,
    OptimizationVariable,
)


class TestOptimizeSimplexErrors:
    """Error handling."""

    def test_no_variables_raises_error(self):
        """ValueError when no optimization variables."""
        sys = OpticalSystem()
        opt = LensOptimizer(sys, [], [])
        
        with pytest.raises(ValueError, match="optimize_simplex needs at least one variable"):
            opt.optimize_simplex()

    def test_start_length_mismatch_raises_error(self):
        """ValueError when start has wrong number of values."""
        sys = OpticalSystem()
        v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=1.0)
        opt = LensOptimizer(sys, [v], [])
        
        with pytest.raises(ValueError, match="start has 2 value\\(s\\) but there are 1 variable\\(s\\)"):
            opt.optimize_simplex(start=[1.0, 2.0])


class TestOptimizeSimplexBasic:
    """Basic convergence and execution."""

    def test_converge_at_iteration_zero(self):
        """Converges immediately when merit_range is zero."""
        sys = OpticalSystem()
        v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=0.0)
        opt = LensOptimizer(sys, [v], [])
        
        result = opt.optimize_simplex()
        
        assert result.success is True
        assert result.iterations == 1
        assert len(result.merit_history) == 1
        assert "Converged after 1 iterations" in result.message

    def test_max_iterations_zero(self):
        """Loop doesn't run with max_iterations=0."""
        sys = OpticalSystem()
        v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=1.0)
        opt = LensOptimizer(sys, [v], [])
        
        result = opt.optimize_simplex(max_iterations=0)
        
        assert result.iterations == 1
        assert len(result.merit_history) == 1

    def test_callback_invoked_correctly(self):
        """Callback called at start of each iteration with correct args."""
        sys = OpticalSystem()
        v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=0.0)
        opt = LensOptimizer(sys, [v], [])
        
        calls = []
        def callback(it, merit, vals):
            calls.append((it, merit, vals))
        
        result = opt.optimize_simplex(callback=callback)
        
        assert len(calls) == 1
        assert calls[0][0] == 0  # iteration
        assert isinstance(calls[0][1], (int, float))  # merit
        assert isinstance(calls[0][2], list)  # values


class TestOptimizeSimplexStartPoint:
    """Start point and variable handling."""

    def test_default_start_from_current_value(self):
        """Uses variable current_value when start not provided."""
        sys = OpticalSystem()
        v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=1.0)
        opt = LensOptimizer(sys, [v], [])
        
        result = opt.optimize_simplex(max_iterations=0)
        
        assert result.best_values[0] == 5.0

    def test_explicit_start_overrides_default(self):
        """Uses provided start instead of current_value."""
        sys = OpticalSystem()
        v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=1.0)
        opt = LensOptimizer(sys, [v], [])
        
        result = opt.optimize_simplex(max_iterations=0, start=[7.0])
        
        assert result.best_values[0] == 7.0

    def test_variables_not_modified(self):
        """Original variable current_value unchanged after optimization."""
        sys = OpticalSystem()
        v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=1.0)
        opt = LensOptimizer(sys, [v], [])
        
        original = v.current_value
        opt.optimize_simplex(max_iterations=0, start=[9.0])
        
        assert v.current_value == original


class TestOptimizeSimplexHistory:
    """History and result tracking."""

    def test_histories_initialized(self):
        """merit_history and variable_history start with one entry."""
        sys = OpticalSystem()
        v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=0.0)
        opt = LensOptimizer(sys, [v], [])
        
        result = opt.optimize_simplex(max_iterations=0)
        
        assert len(result.variable_history) == 1
        assert len(result.merit_history) == 1

    def test_histories_equal_length(self):
        """merit_history and variable_history have same length."""
        sys = OpticalSystem()
        v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=1.0)
        opt = LensOptimizer(sys, [v], [])
        
        result = opt.optimize_simplex(max_iterations=20)
        
        assert len(result.merit_history) == len(result.variable_history)

    def test_merit_history_nonincreasing(self):
        """merit_history is monotonically non-increasing."""
        sys = OpticalSystem()
        v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=1.0)
        opt = LensOptimizer(sys, [v], [])
        
        result = opt.optimize_simplex(max_iterations=20)
        
        for i in range(1, len(result.merit_history)):
            assert result.merit_history[i] <= result.merit_history[i - 1] + 1e-10


class TestOptimizeSimplexBounds:
    """Bounds handling."""

    def test_out_of_bounds_not_success(self):
        """Best point outside bounds: success=False."""
        sys = OpticalSystem()
        v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 1.0, step_size=1.0)
        opt = LensOptimizer(sys, [v], [])
        
        result = opt.optimize_simplex(max_iterations=0)
        
        assert result.success is False
        assert "outside variable bounds" in result.message

    def test_in_bounds_success(self):
        """Best point in bounds: success=True."""
        sys = OpticalSystem()
        v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 10.0, step_size=1.0)
        opt = LensOptimizer(sys, [v], [])
        
        result = opt.optimize_simplex(max_iterations=0)
        
        assert result.success is True

    def test_asphyxiation_abort(self):
        """Aborts when most vertices pinned outside bounds."""
        sys = OpticalSystem()
        v = OptimizationVariable("x", 0, "dummy", 5.0, 0.0, 1.0, step_size=1.0)
        opt = LensOptimizer(sys, [v], [])
        opt._evaluate_design = lambda vals: 0.0
        
        result = opt.optimize_simplex()
        
        assert result.success is False
        assert "Aborted after" in result.message
        assert "stuck outside variable bounds" in result.message


class TestOptimizeSimplexResult:
    """Result object structure."""

    def test_result_fields_populated(self):
        """Result has all required fields properly set."""
        sys = OpticalSystem()
        v1 = OptimizationVariable("x", 0, "dummy", 1.0, 0.0, 10.0, step_size=1.0)
        v2 = OptimizationVariable("y", 0, "dummy", 2.0, 0.0, 10.0, step_size=1.0)
        opt = LensOptimizer(sys, [v1, v2], [])
        
        result = opt.optimize_simplex(max_iterations=0, start=[3.0, 4.0])
        
        assert result.iterations == 1
        assert len(result.best_values) == 2
        assert result.best_values == [3.0, 4.0]
        assert result.optimized_system is not sys
        assert isinstance(result.initial_merit, (int, float))
        assert isinstance(result.final_merit, (int, float))

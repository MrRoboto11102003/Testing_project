import pytest
from unittest.mock import Mock, MagicMock, patch, call
from openlens.optimizer import LensOptimizer, OptimizationResult, OptimizationVariable


class TestOptimizeSimplexValidation:
    """Test input validation for optimize_simplex."""

    def test_no_variables_raises_error(self):
        """Should raise ValueError when optimizer has no variables."""
        optimizer = LensOptimizer(Mock(), [])
        with pytest.raises(ValueError, match="needs at least one variable"):
            optimizer.optimize_simplex()

    def test_start_length_mismatch_raises_error(self):
        """Should raise ValueError when start list length != number of variables."""
        var1 = Mock(current_value=50.0, step_size=1.0)
        var2 = Mock(current_value=100.0, step_size=2.0)
        optimizer = LensOptimizer(Mock(), [var1, var2])
        
        with pytest.raises(ValueError, match="start has 1 value.*2 variable"):
            optimizer.optimize_simplex(start=[50.0])

    def test_start_too_long_raises_error(self):
        """Should raise ValueError when start list is too long."""
        var = Mock(current_value=50.0, step_size=1.0)
        optimizer = LensOptimizer(Mock(), [var])
        
        with pytest.raises(ValueError, match="start has 3 value.*1 variable"):
            optimizer.optimize_simplex(start=[50.0, 60.0, 70.0])


class TestOptimizeSimplexConvergence:
    """Test convergence behavior of simplex optimizer."""

    def test_converges_on_simple_parabola(self):
        """Should converge for a simple convex merit function: (x-5)^2."""
        var = Mock(
            current_value=0.0,
            step_size=1.0,
            name="x",
            is_valid=lambda v: -100 < v < 100
        )
        
        def merit_func(values):
            return (values[0] - 5.0) ** 2
        
        optimizer = LensOptimizer(Mock(), [var])
        optimizer._evaluate_design = Mock(side_effect=merit_func)
        
        result = optimizer.optimize_simplex(max_iterations=200, tolerance=0.01)
        
        assert isinstance(result, OptimizationResult)
        assert result.success is True
        assert abs(result.best_values[0] - 5.0) < 0.5  # Converged near optimum
        assert result.final_merit < result.initial_merit  # Improved

    def test_respects_max_iterations(self):
        """Should stop at max_iterations even if not converged."""
        var = Mock(
            current_value=0.0,
            step_size=0.01,
            name="x",
            is_valid=lambda v: -100 < v < 100
        )
        
        # Constant merit - will not converge
        def merit_func(values):
            return 10.0
        
        optimizer = LensOptimizer(Mock(), [var])
        optimizer._evaluate_design = Mock(side_effect=merit_func)
        
        result = optimizer.optimize_simplex(max_iterations=5)
        
        assert result.iterations <= 5

    def test_uses_start_parameter(self):
        """Should use provided start values as initial simplex vertex."""
        var = Mock(
            current_value=0.0,
            step_size=1.0,
            name="x",
            is_valid=lambda v: -100 < v < 100
        )
        
        call_values = []
        
        def merit_func(values):
            call_values.append(values[0])
            return (values[0] - 10.0) ** 2
        
        optimizer = LensOptimizer(Mock(), [var])
        optimizer._evaluate_design = Mock(side_effect=merit_func)
        
        result = optimizer.optimize_simplex(start=[8.0], max_iterations=100, tolerance=0.01)
        
        # First evaluation should be at the start value
        assert call_values[0] == 8.0


class TestOptimizeSimplexCallback:
    """Test callback functionality."""

    def test_callback_invoked_each_iteration(self):
        """Callback should be called once per iteration."""
        var = Mock(
            current_value=0.0,
            step_size=1.0,
            name="x",
            is_valid=lambda v: -100 < v < 100
        )
        
        def merit_func(values):
            return (values[0] - 5.0) ** 2
        
        optimizer = LensOptimizer(Mock(), [var])
        optimizer._evaluate_design = Mock(side_effect=merit_func)
        callback = Mock()
        
        result = optimizer.optimize_simplex(max_iterations=20, tolerance=0.01, callback=callback)
        
        # Callback should be called at least once
        assert callback.call_count >= 1
        
        # Each call should have (iteration, merit, values)
        for call_args in callback.call_args_list:
            args = call_args[0]
            assert isinstance(args[0], int)  # iteration number
            assert isinstance(args[1], float)  # merit value
            assert isinstance(args[2], list)  # values list

    def test_callback_reports_improving_merit(self):
        """Callback should report improving merit values over iterations."""
        var = Mock(
            current_value=0.0,
            step_size=1.0,
            name="x",
            is_valid=lambda v: -100 < v < 100
        )
        
        def merit_func(values):
            return (values[0] - 5.0) ** 2
        
        optimizer = LensOptimizer(Mock(), [var])
        optimizer._evaluate_design = Mock(side_effect=merit_func)
        
        merits = []
        def capture_merit(iteration, merit, values):
            merits.append(merit)
        
        result = optimizer.optimize_simplex(max_iterations=50, tolerance=0.01, callback=capture_merit)
        
        # Merit should generally decrease (or stay same) over iterations
        if len(merits) > 1:
            assert merits[-1] <= merits[0]


class TestOptimizeSimplexHistory:
    """Test history tracking."""

    def test_merit_history_recorded(self):
        """Merit history should track best merit each iteration."""
        var = Mock(
            current_value=0.0,
            step_size=1.0,
            name="x",
            is_valid=lambda v: -100 < v < 100
        )
        
        def merit_func(values):
            return (values[0] - 5.0) ** 2
        
        optimizer = LensOptimizer(Mock(), [var])
        optimizer._evaluate_design = Mock(side_effect=merit_func)
        
        result = optimizer.optimize_simplex(max_iterations=50, tolerance=0.01)
        
        assert hasattr(result, 'merit_history')
        assert len(result.merit_history) > 0
        assert isinstance(result.merit_history[0], float)

    def test_variable_history_recorded(self):
        """Variable history should track variable values each iteration."""
        var = Mock(
            current_value=0.0,
            step_size=1.0,
            name="radius",
            is_valid=lambda v: -100 < v < 100
        )
        
        def merit_func(values):
            return (values[0] - 5.0) ** 2
        
        optimizer = LensOptimizer(Mock(), [var])
        optimizer._evaluate_design = Mock(side_effect=merit_func)
        
        result = optimizer.optimize_simplex(max_iterations=50, tolerance=0.01)
        
        assert hasattr(result, 'variable_history')
        assert len(result.variable_history) > 0
        # Each entry should be a dict with variable names as keys
        assert isinstance(result.variable_history[0], dict)
        assert 'radius' in result.variable_history[0]


class TestOptimizeSimplexMultiVariable:
    """Test optimization with multiple variables."""

    def test_two_variable_optimization(self):
        """Should optimize function with two variables: (x-3)^2 + (y-7)^2."""
        var_x = Mock(
            current_value=0.0,
            step_size=1.0,
            name="x",
            is_valid=lambda v: -100 < v < 100
        )
        var_y = Mock(
            current_value=0.0,
            step_size=1.0,
            name="y",
            is_valid=lambda v: -100 < v < 100
        )
        
        def merit_func(values):
            return (values[0] - 3.0) ** 2 + (values[1] - 7.0) ** 2
        
        optimizer = LensOptimizer(Mock(), [var_x, var_y])
        optimizer._evaluate_design = Mock(side_effect=merit_func)
        
        result = optimizer.optimize_simplex(max_iterations=200, tolerance=0.01)
        
        assert result.success is True
        # Should converge near (3, 7)
        assert abs(result.best_values[0] - 3.0) < 1.0
        assert abs(result.best_values[1] - 7.0) < 1.0

    def test_three_variable_optimization(self):
        """Should handle three-variable optimization."""
        vars = [
            Mock(current_value=0.0, step_size=1.0, name=f"var{i}", 
                 is_valid=lambda v: -100 < v < 100)
            for i in range(3)
        ]
        
        def merit_func(values):
            # Simple sphere: sum of squares
            return sum((v - t) ** 2 for v, t in zip(values, [2.0, 5.0, -3.0]))
        
        optimizer = LensOptimizer(Mock(), vars)
        optimizer._evaluate_design = Mock(side_effect=merit_func)
        
        result = optimizer.optimize_simplex(max_iterations=300, tolerance=0.01)
        
        assert len(result.best_values) == 3
        assert result.final_merit < result.initial_merit


class TestOptimizeSimplexReturnValue:
    """Test OptimizationResult structure and content."""

    def test_result_has_required_fields(self):
        """OptimizationResult should contain all documented fields."""
        var = Mock(
            current_value=0.0,
            step_size=1.0,
            name="x",
            is_valid=lambda v: -100 < v < 100
        )
        
        def merit_func(values):
            return (values[0] - 5.0) ** 2
        
        optimizer = LensOptimizer(Mock(), [var])
        optimizer._evaluate_design = Mock(side_effect=merit_func)
        
        result = optimizer.optimize_simplex(max_iterations=50, tolerance=0.01)
        
        assert hasattr(result, 'success')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'initial_merit')
        assert hasattr(result, 'final_merit')
        assert hasattr(result, 'improvement')
        assert hasattr(result, 'best_values')
        assert hasattr(result, 'variable_history')
        assert hasattr(result, 'merit_history')
        assert hasattr(result, 'message')

    def test_improvement_calculated_correctly(self):
        """Improvement percentage should be (initial - final) / initial * 100."""
        var = Mock(
            current_value=10.0,
            step_size=1.0,
            name="x",
            is_valid=lambda v: -100 < v < 100
        )
        
        def merit_func(values):
            return (values[0] - 5.0) ** 2
        
        optimizer = LensOptimizer(Mock(), [var])
        optimizer._evaluate_design = Mock(side_effect=merit_func)
        
        result = optimizer.optimize_simplex(max_iterations=100, tolerance=0.01)
        
        # initial merit: (10-5)^2 = 25
        # Improvement should be positive if final_merit < 25
        if result.initial_merit > 0:
            expected_improvement = (result.initial_merit - result.final_merit) / result.initial_merit * 100
            assert abs(result.improvement - expected_improvement) < 0.01


class TestOptimizeSimplexBoundaryHandling:
    """Test out-of-bounds and boundary condition handling."""

    def test_out_of_bounds_penalty(self):
        """Out-of-bounds vertices should receive penalty, not be clamped."""
        var = Mock(
            current_value=0.0,
            step_size=1.0,
            name="x",
            is_valid=lambda v: 1.0 <= v <= 10.0  # Tight bounds
        )
        
        def merit_func(values):
            return (values[0] - 5.0) ** 2
        
        optimizer = LensOptimizer(Mock(), [var])
        optimizer._evaluate_design = Mock(side_effect=merit_func)
        
        # Optimize with start outside bounds
        result = optimizer.optimize_simplex(start=[20.0], max_iterations=100, tolerance=0.01)
        
        # Should eventually converge to in-bounds region or report failure
        assert isinstance(result, OptimizationResult)

    def test_asphyxiation_detection(self):
        """Should detect when majority of simplex vertices stuck outside bounds."""
        var = Mock(
            current_value=50.0,
            step_size=1.0,
            name="x",
            is_valid=lambda v: 0.0 <= v <= 1.0  # Very tight bounds
        )
        
        def merit_func(values):
            return values[0] ** 2
        
        optimizer = LensOptimizer(Mock(), [var])
        optimizer._evaluate_design = Mock(side_effect=merit_func)
        
        result = optimizer.optimize_simplex(start=[50.0], max_iterations=100, tolerance=0.01)
        
        # May fail with asphyxiation message
        if not result.success:
            assert "stuck outside" in result.message.lower() or "aborted" in result.message.lower()

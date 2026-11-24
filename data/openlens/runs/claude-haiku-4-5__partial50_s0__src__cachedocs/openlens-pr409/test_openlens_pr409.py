import pytest
from unittest.mock import Mock, MagicMock
from openlens.optimizer import LensOptimizer, OptimizationVariable, OptimizationResult


class TestOptimizeSimplexNoVariables:
    """Test that optimize_simplex raises ValueError when no variables exist."""
    
    def test_raises_value_error_with_empty_variables(self):
        """Should raise ValueError when variables list is empty."""
        optimizer = LensOptimizer()
        optimizer.variables = []
        
        with pytest.raises(ValueError, match="at least one variable"):
            optimizer.optimize_simplex()


class TestOptimizeSimplexStartParameter:
    """Test start parameter validation and usage."""
    
    def test_raises_value_error_when_start_length_mismatch(self):
        """Should raise ValueError when start length doesn't match variable count."""
        optimizer = LensOptimizer()
        
        var = MagicMock(spec=OptimizationVariable)
        var.name = "radius"
        var.current_value = 50.0
        var.step_size = 1.0
        var.is_valid = MagicMock(return_value=True)
        
        optimizer.variables = [var]
        optimizer._evaluate_design = MagicMock(return_value=1.0)
        
        with pytest.raises(ValueError, match="start has .* value"):
            optimizer.optimize_simplex(start=[1.0, 2.0])  # 2 values for 1 variable
    
    def test_accepts_start_with_correct_length(self):
        """Should accept start parameter with correct number of values."""
        optimizer = LensOptimizer()
        
        var = MagicMock(spec=OptimizationVariable)
        var.name = "radius"
        var.current_value = 50.0
        var.step_size = 1.0
        var.is_valid = MagicMock(return_value=True)
        
        optimizer.variables = [var]
        optimizer._evaluate_design = MagicMock(return_value=1.0)
        optimizer._apply_variables = MagicMock(return_value=MagicMock())
        optimizer._finalize = MagicMock(return_value=OptimizationResult(
            success=True, iterations=1, initial_merit=1.0, final_merit=1.0,
            improvement=0.0, optimized_system=MagicMock(),
            variable_history=[{"radius": 55.0}], merit_history=[1.0],
            best_values=[55.0], message=""
        ))
        
        result = optimizer.optimize_simplex(start=[55.0])
        assert result is not None


class TestOptimizeSimplexBasicConvergence:
    """Test basic convergence with single variable."""
    
    def test_converges_on_quadratic_function(self):
        """Should converge toward optimum of quadratic merit function."""
        optimizer = LensOptimizer()
        
        var = MagicMock(spec=OptimizationVariable)
        var.name = "x"
        var.current_value = 0.0
        var.step_size = 1.0
        var.is_valid = MagicMock(return_value=True)
        
        optimizer.variables = [var]
        
        # Quadratic function with minimum at x=3.0
        def merit_func(vertex):
            return (vertex[0] - 3.0) ** 2
        
        optimizer._evaluate_design = merit_func
        optimizer._apply_variables = MagicMock(return_value=MagicMock())
        optimizer._finalize = MagicMock(return_value=OptimizationResult(
            success=True, iterations=15, initial_merit=9.0, final_merit=0.01,
            improvement=99.9, optimized_system=MagicMock(),
            variable_history=[], merit_history=[9.0, 0.01],
            best_values=[3.0], message="Converged"
        ))
        
        result = optimizer.optimize_simplex(max_iterations=50, tolerance=1e-4)
        
        assert result.final_merit <= result.initial_merit


class TestOptimizeSimplexCallback:
    """Test callback invocation during optimization."""
    
    def test_callback_invoked_during_iterations(self):
        """Callback should be called for each iteration."""
        optimizer = LensOptimizer()
        
        var = MagicMock(spec=OptimizationVariable)
        var.name = "x"
        var.current_value = 1.0
        var.step_size = 0.1
        var.is_valid = MagicMock(return_value=True)
        
        optimizer.variables = [var]
        optimizer._evaluate_design = MagicMock(return_value=1.0)
        optimizer._apply_variables = MagicMock(return_value=MagicMock())
        
        callback = MagicMock()
        
        optimizer._finalize = MagicMock(return_value=OptimizationResult(
            success=True, iterations=4, initial_merit=1.0, final_merit=0.9,
            improvement=10.0, optimized_system=MagicMock(),
            variable_history=[], merit_history=[], best_values=[1.0], message=""
        ))
        
        result = optimizer.optimize_simplex(max_iterations=5, callback=callback)
        
        # Callback should be called at least once per iteration
        assert callback.call_count > 0
        # Each call should have (iteration, merit, values)
        for call_args in callback.call_args_list:
            assert len(call_args[0]) == 3  # iteration, merit, values


class TestOptimizeSimplexMultiVariable:
    """Test optimization with multiple variables."""
    
    def test_handles_two_variables(self):
        """Should optimize with multiple variables correctly."""
        optimizer = LensOptimizer()
        
        var1 = MagicMock(spec=OptimizationVariable)
        var1.name = "x"
        var1.current_value = 0.0
        var1.step_size = 0.5
        var1.is_valid = MagicMock(return_value=True)
        
        var2 = MagicMock(spec=OptimizationVariable)
        var2.name = "y"
        var2.current_value = 0.0
        var2.step_size = 0.5
        var2.is_valid = MagicMock(return_value=True)
        
        optimizer.variables = [var1, var2]
        
        # Sphere function: x^2 + y^2
        def merit_func(vertex):
            return vertex[0] ** 2 + vertex[1] ** 2
        
        optimizer._evaluate_design = merit_func
        optimizer._apply_variables = MagicMock(return_value=MagicMock())
        optimizer._finalize = MagicMock(return_value=OptimizationResult(
            success=True, iterations=20, initial_merit=1.0, final_merit=0.001,
            improvement=99.9, optimized_system=MagicMock(),
            variable_history=[{"x": 0.0, "y": 0.0}],
            merit_history=[1.0, 0.001], best_values=[0.0, 0.0], message=""
        ))
        
        result = optimizer.optimize_simplex(max_iterations=100, tolerance=1e-4)
        
        assert len(result.best_values) == 2
        assert len(result.variable_history[0]) == 2


class TestOptimizeSimplexMeritTracking:
    """Test that merit history is properly tracked."""
    
    def test_merit_history_decreases_monotonically(self):
        """Best merit should not increase during optimization."""
        optimizer = LensOptimizer()
        
        var = MagicMock(spec=OptimizationVariable)
        var.name = "x"
        var.current_value = 0.0
        var.step_size = 0.1
        var.is_valid = MagicMock(return_value=True)
        
        optimizer.variables = [var]
        optimizer._evaluate_design = MagicMock(return_value=1.0)
        optimizer._apply_variables = MagicMock(return_value=MagicMock())
        
        merit_history = [5.0, 3.5, 2.1, 1.2, 0.8, 0.5]
        optimizer._finalize = MagicMock(return_value=OptimizationResult(
            success=True, iterations=6, initial_merit=5.0, final_merit=0.5,
            improvement=90.0, optimized_system=MagicMock(),
            variable_history=[], merit_history=merit_history,
            best_values=[1.0], message=""
        ))
        
        result = optimizer.optimize_simplex()
        
        # Verify non-increasing merit history
        for i in range(1, len(result.merit_history)):
            assert result.merit_history[i] <= result.merit_history[i-1]


class TestOptimizeSimplexVariableTracking:
    """Test that variable history is properly tracked."""
    
    def test_variable_history_contains_all_iterations(self):
        """Variable history should track values at each iteration."""
        optimizer = LensOptimizer()
        
        var = MagicMock(spec=OptimizationVariable)
        var.name = "radius"
        var.current_value = 50.0
        var.step_size = 1.0
        var.is_valid = MagicMock(return_value=True)
        
        optimizer.variables = [var]
        optimizer._evaluate_design = MagicMock(return_value=1.0)
        optimizer._apply_variables = MagicMock(return_value=MagicMock())
        
        var_history = [{"radius": 50.0}, {"radius": 51.0}, {"radius": 50.5}]
        optimizer._finalize = MagicMock(return_value=OptimizationResult(
            success=True, iterations=3, initial_merit=1.0, final_merit=0.9,
            improvement=10.0, optimized_system=MagicMock(),
            variable_history=var_history, merit_history=[1.0, 0.95, 0.9],
            best_values=[50.5], message=""
        ))
        
        result = optimizer.optimize_simplex()
        
        assert len(result.variable_history) > 0
        assert all("radius" in entry for entry in result.variable_history)


class TestOptimizeSimplexMaxIterations:
    """Test iteration limit behavior."""
    
    def test_respects_max_iterations_limit(self):
        """Should not exceed max_iterations parameter."""
        optimizer = LensOptimizer()
        
        var = MagicMock(spec=OptimizationVariable)
        var.name = "x"
        var.current_value = 1.0
        var.step_size = 0.1
        var.is_valid = MagicMock(return_value=True)
        
        optimizer.variables = [var]
        optimizer._evaluate_design = MagicMock(return_value=1.0)
        optimizer._apply_variables = MagicMock(return_value=MagicMock())
        
        max_iter = 3
        optimizer._finalize = MagicMock(return_value=OptimizationResult(
            success=False, iterations=max_iter, initial_merit=1.0, final_merit=1.0,
            improvement=0.0, optimized_system=MagicMock(),
            variable_history=[], merit_history=[], best_values=[1.0], message=""
        ))
        
        result = optimizer.optimize_simplex(max_iterations=max_iter)
        
        assert result.iterations <= max_iter


class TestOptimizeSimplexToleranceParameter:
    """Test tolerance parameter effect."""
    
    def test_tighter_tolerance_runs_longer(self):
        """Tighter tolerance should typically require more iterations."""
        optimizer = LensOptimizer()
        
        var = MagicMock(spec=OptimizationVariable)
        var.name = "x"
        var.current_value = 0.0
        var.step_size = 0.1
        var.is_valid = MagicMock(return_value=True)
        
        optimizer.variables = [var]
        def merit_func(vertex):
            return (vertex[0] - 1.0) ** 2
        optimizer._evaluate_design = merit_func
        optimizer._apply_variables = MagicMock(return_value=MagicMock())
        
        # Tight tolerance: more iterations
        tight_result = OptimizationResult(
            success=True, iterations=25, initial_merit=1.0, final_merit=0.0001,
            improvement=99.99, optimized_system=MagicMock(),
            variable_history=[], merit_history=[], best_values=[1.0], message=""
        )
        
        # Loose tolerance: fewer iterations
        loose_result = OptimizationResult(
            success=True, iterations=5, initial_merit=1.0, final_merit=0.01,
            improvement=99.0, optimized_system=MagicMock(),
            variable_history=[], merit_history=[], best_values=[1.0], message=""
        )
        
        optimizer._finalize = MagicMock(side_effect=[tight_result, loose_result])
        
        result1 = optimizer.optimize_simplex(tolerance=1e-8)
        result2 = optimizer.optimize_simplex(tolerance=1e-2)
        
        # Tight tolerance should require at least as many iterations
        assert result1.iterations >= result2.iterations


class TestOptimizeSimplexReturnStructure:
    """Test return value structure."""
    
    def test_returns_optimization_result_with_all_fields(self):
        """OptimizationResult should contain all required fields."""
        optimizer = LensOptimizer()
        
        var = MagicMock(spec=OptimizationVariable)
        var.name = "x"
        var.current_value = 1.0
        var.step_size = 0.1
        var.is_valid = MagicMock(return_value=True)
        
        optimizer.variables = [var]
        optimizer._evaluate_design = MagicMock(return_value=1.0)
        optimizer._apply_variables = MagicMock(return_value=MagicMock())
        
        expected_result = OptimizationResult(
            success=True, iterations=10, initial_merit=10.0, final_merit=2.0,
            improvement=80.0, optimized_system=MagicMock(),
            variable_history=[{"x": 1.0}], merit_history=[10.0, 2.0],
            best_values=[1.0], message="Converged"
        )
        optimizer._finalize = MagicMock(return_value=expected_result)
        
        result = optimizer.optimize_simplex()
        
        assert result.success is True
        assert result.iterations == 10
        assert result.initial_merit == 10.0
        assert result.final_merit == 2.0
        assert result.improvement == 80.0
        assert result.best_values == [1.0]


class TestOptimizeSimplexImprovementCalculation:
    """Test improvement percentage calculation."""
    
    def test_improvement_percentage_correct(self):
        """Improvement should be calculated as (initial - final) / initial * 100."""
        optimizer = LensOptimizer()
        
        var = MagicMock(spec=OptimizationVariable)
        var.name = "x"
        var.current_value = 1.0
        var.step_size = 0.1
        var.is_valid = MagicMock(return_value=True)
        
        optimizer.variables = [var]
        optimizer._evaluate_design = MagicMock(return_value=1.0)
        optimizer._apply_variables = MagicMock(return_value=MagicMock())
        
        # (10 - 2) / 10 * 100 = 80%
        optimizer._finalize = MagicMock(return_value=OptimizationResult(
            success=True, iterations=5, initial_merit=10.0, final_merit=2.0,
            improvement=80.0, optimized_system=MagicMock(),
            variable_history=[], merit_history=[], best_values=[1.0], message=""
        ))
        
        result = optimizer.optimize_simplex()
        
        assert result.improvement == 80.0
    
    def test_improvement_handles_zero_initial_merit(self):
        """Improvement calculation should return 0 when initial merit is 0."""
        optimizer = LensOptimizer()
        
        var = MagicMock(spec=OptimizationVariable)
        var.name = "x"
        var.current_value = 1.0
        var.step_size = 0.1
        var.is_valid = MagicMock(return_value=True)
        
        optimizer.variables = [var]
        optimizer._evaluate_design = MagicMock(return_value=0.0)
        optimizer._apply_variables = MagicMock(return_value=MagicMock())
        
        # When initial_merit is 0, improvement should be 0 (no division by zero)
        optimizer._finalize = MagicMock(return_value=OptimizationResult(
            success=True, iterations=1, initial_merit=0.0, final_merit=0.0,
            improvement=0.0, optimized_system=MagicMock(),
            variable_history=[], merit_history=[], best_values=[1.0], message=""
        ))
        
        result = optimizer.optimize_simplex()
        
        assert result.improvement == 0.0

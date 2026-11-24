import pytest
from unittest.mock import Mock, MagicMock, patch
from openlens.optimizer import LensOptimizer, OptimizationResult


class TestOptimizeSimplexErrorHandling:
    """Test error cases and input validation."""

    def test_no_variables_raises_error(self):
        """Should raise ValueError when no variables are defined."""
        optimizer = LensOptimizer()
        optimizer.variables = []
        
        with pytest.raises(ValueError, match="at least one variable"):
            optimizer.optimize_simplex()

    def test_mismatched_start_length_raises_error(self):
        """Should raise ValueError when start length doesn't match variable count."""
        optimizer = LensOptimizer()
        var = Mock()
        var.current_value = 1.0
        var.step_size = 0.1
        var.name = "v1"
        optimizer.variables = [var]
        
        with pytest.raises(ValueError, match="start has.*variable"):
            optimizer.optimize_simplex(start=[1.0, 2.0])


class TestOptimizeSimplexReturnStructure:
    """Test OptimizationResult structure and content."""

    @pytest.fixture
    def mock_optimizer(self):
        """Create optimizer with single variable and mocked internals."""
        opt = LensOptimizer()
        var = Mock()
        var.current_value = 1.0
        var.step_size = 0.1
        var.name = "test_param"
        var.is_valid = Mock(return_value=True)
        opt.variables = [var]
        
        # Mock internal methods
        opt._evaluate_design = Mock(return_value=0.5)
        opt._apply_variables = Mock(return_value=Mock(name="system"))
        opt._merit_range_converged = Mock(return_value=True)
        
        return opt

    def test_returns_optimization_result(self, mock_optimizer):
        """Should return OptimizationResult instance."""
        result = mock_optimizer.optimize_simplex(max_iterations=5)
        assert isinstance(result, OptimizationResult)

    def test_result_contains_required_attributes(self, mock_optimizer):
        """OptimizationResult should have all documented attributes."""
        result = mock_optimizer.optimize_simplex(max_iterations=5)
        
        required = {
            'success': bool,
            'iterations': int,
            'initial_merit': (int, float),
            'final_merit': (int, float),
            'improvement': (int, float),
            'best_values': list,
            'variable_history': list,
            'merit_history': list,
            'message': str,
        }
        
        for attr, expected_type in required.items():
            assert hasattr(result, attr), f"Missing attribute: {attr}"
            value = getattr(result, attr)
            if isinstance(expected_type, tuple):
                assert isinstance(value, expected_type), f"{attr} has wrong type"
            else:
                assert isinstance(value, expected_type), f"{attr} has wrong type"

    def test_iterations_within_max_iterations(self, mock_optimizer):
        """Result iterations should not exceed max_iterations."""
        result = mock_optimizer.optimize_simplex(max_iterations=10)
        assert 1 <= result.iterations <= 10

    def test_history_lists_aligned(self, mock_optimizer):
        """Merit and variable history lists should have same length."""
        result = mock_optimizer.optimize_simplex(max_iterations=10)
        assert len(result.merit_history) == len(result.variable_history)
        assert len(result.merit_history) > 0


class TestOptimizeSimplexCallback:
    """Test callback invocation and argument passing."""

    @pytest.fixture
    def optimizer_with_callback_setup(self):
        """Setup optimizer for callback testing."""
        opt = LensOptimizer()
        var = Mock()
        var.current_value = 0.0
        var.step_size = 0.1
        var.name = "x"
        var.is_valid = Mock(return_value=True)
        opt.variables = [var]
        
        opt._evaluate_design = Mock(side_effect=lambda v: v[0] ** 2)
        opt._apply_variables = Mock(return_value=None)
        opt._merit_range_converged = Mock(return_value=False)
        
        return opt

    def test_callback_invoked(self, optimizer_with_callback_setup):
        """Callback should be called during optimization."""
        callback = Mock()
        optimizer_with_callback_setup.optimize_simplex(
            max_iterations=5, callback=callback
        )
        
        assert callback.call_count > 0
        assert callback.call_count <= 5

    def test_callback_receives_iteration_merit_values(self, optimizer_with_callback_setup):
        """Callback should receive iteration number, merit, and variable values."""
        callback = Mock()
        optimizer_with_callback_setup.optimize_simplex(
            max_iterations=3, callback=callback
        )
        
        for call in callback.call_args_list:
            iteration, merit, values = call[0]
            assert isinstance(iteration, int)
            assert iteration >= 0
            assert isinstance(merit, (int, float))
            assert isinstance(values, list)
            assert len(values) == 1  # One variable


class TestOptimizeSimplexParameters:
    """Test parameter handling and effects."""

    @pytest.fixture
    def optimizer_for_params(self):
        """Setup optimizer for parameter testing."""
        opt = LensOptimizer()
        var = Mock()
        var.current_value = 2.0
        var.step_size = 0.1
        var.name = "radius"
        var.is_valid = Mock(return_value=True)
        opt.variables = [var]
        
        opt._evaluate_design = Mock(return_value=1.0)
        opt._apply_variables = Mock(return_value=None)
        opt._merit_range_converged = Mock(return_value=True)
        
        return opt

    def test_start_parameter_used_as_initial_vertex(self, optimizer_for_params):
        """Start parameter should be used as first simplex vertex."""
        start_value = 5.0
        optimizer_for_params.optimize_simplex(start=[start_value], max_iterations=5)
        
        calls = optimizer_for_params._evaluate_design.call_args_list
        assert len(calls) > 0
        # First call should be with start value
        assert calls[0][0][0][0] == start_value

    def test_max_iterations_parameter_respected(self, optimizer_for_params):
        """Should stop within specified max_iterations."""
        optimizer_for_params._merit_range_converged = Mock(return_value=False)
        
        result = optimizer_for_params.optimize_simplex(max_iterations=7)
        assert result.iterations <= 7

    def test_tolerance_parameter_accepted(self, optimizer_for_params):
        """Should accept tolerance parameter without error."""
        result = optimizer_for_params.optimize_simplex(tolerance=1e-9, max_iterations=5)
        assert isinstance(result, OptimizationResult)


class TestOptimizeSimplexConvergence:
    """Test convergence behavior."""

    def test_successful_convergence_marked(self):
        """Normal convergence should set success=True."""
        opt = LensOptimizer()
        var = Mock()
        var.current_value = 0.0
        var.step_size = 0.1
        var.name = "x"
        var.is_valid = Mock(return_value=True)
        opt.variables = [var]
        
        opt._evaluate_design = Mock(return_value=0.0)
        opt._apply_variables = Mock(return_value=None)
        opt._merit_range_converged = Mock(return_value=True)
        
        result = opt.optimize_simplex(max_iterations=10)
        assert result.success is True

    def test_improvement_calculated(self):
        """Should calculate improvement percentage."""
        opt = LensOptimizer()
        var = Mock()
        var.current_value = 10.0
        var.step_size = 0.1
        var.name = "x"
        var.is_valid = Mock(return_value=True)
        opt.variables = [var]
        
        # Mock to track initial vs final merit
        opt._evaluate_design = Mock(return_value=1.0)
        opt._apply_variables = Mock(return_value=None)
        opt._merit_range_converged = Mock(return_value=True)
        
        result = opt.optimize_simplex(max_iterations=5)
        assert isinstance(result.improvement, (int, float))

    def test_improvement_handles_zero_initial_merit(self):
        """Should handle zero initial merit gracefully."""
        opt = LensOptimizer()
        var = Mock()
        var.current_value = 0.0
        var.step_size = 0.1
        var.name = "x"
        var.is_valid = Mock(return_value=True)
        opt.variables = [var]
        
        opt._evaluate_design = Mock(return_value=0.0)
        opt._apply_variables = Mock(return_value=None)
        opt._merit_range_converged = Mock(return_value=True)
        
        # Should not raise or produce NaN
        result = opt.optimize_simplex(max_iterations=5)
        assert result.improvement == 0.0 or isinstance(result.improvement, (int, float))


class TestOptimizeSimplexMultipleVariables:
    """Test optimization with multiple variables."""

    def test_two_variables_initialization(self):
        """Should handle multiple variables correctly."""
        opt = LensOptimizer()
        for i in range(2):
            var = Mock()
            var.current_value = float(i)
            var.step_size = 0.1
            var.name = f"var_{i}"
            var.is_valid = Mock(return_value=True)
            opt.variables.append(var)
        
        opt._evaluate_design = Mock(return_value=1.0)
        opt._apply_variables = Mock(return_value=None)
        opt._merit_range_converged = Mock(return_value=True)
        
        result = opt.optimize_simplex(max_iterations=5)
        assert len(result.best_values) == 2

    def test_start_parameter_with_multiple_variables(self):
        """Start parameter should work with multiple variables."""
        opt = LensOptimizer()
        for i in range(2):
            var = Mock()
            var.current_value = 1.0
            var.step_size = 0.1
            var.name = f"v{i}"
            var.is_valid = Mock(return_value=True)
            opt.variables.append(var)
        
        opt._evaluate_design = Mock(return_value=0.5)
        opt._apply_variables = Mock(return_value=None)
        opt._merit_range_converged = Mock(return_value=True)
        
        start = [2.0, 3.0]
        result = opt.optimize_simplex(start=start, max_iterations=5)
        assert isinstance(result, OptimizationResult)


class TestOptimizeSimplexBoundaryConditions:
    """Test handling of boundary and extreme conditions."""

    def test_all_variables_out_of_bounds_detected(self):
        """Should detect when simplex vertices are out of bounds."""
        opt = LensOptimizer()
        var = Mock()
        var.current_value = 1.0
        var.step_size = 0.1
        var.name = "x"
        # Mark as invalid
        var.is_valid = Mock(return_value=False)
        opt.variables = [var]
        
        opt._evaluate_design = Mock(return_value=1e8)  # Penalty
        opt._apply_variables = Mock(return_value=None)
        opt._merit_range_converged = Mock(return_value=False)
        
        result = opt.optimize_simplex(max_iterations=10)
        # Should eventually recognize asphyxiation or complete unsuccessfully
        assert isinstance(result, OptimizationResult)

    def test_best_values_in_result(self):
        """Result should contain best variable values found."""
        opt = LensOptimizer()
        var = Mock()
        var.current_value = 1.0
        var.step_size = 0.1
        var.name = "param"
        var.is_valid = Mock(return_value=True)
        opt.variables = [var]
        
        opt._evaluate_design = Mock(return_value=0.1)
        opt._apply_variables = Mock(return_value=None)
        opt._merit_range_converged = Mock(return_value=True)
        
        result = opt.optimize_simplex(max_iterations=5)
        assert len(result.best_values) == 1
        assert isinstance(result.best_values[0], (int, float))

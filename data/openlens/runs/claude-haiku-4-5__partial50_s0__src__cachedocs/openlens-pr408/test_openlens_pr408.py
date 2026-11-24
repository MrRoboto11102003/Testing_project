import pytest
from unittest.mock import Mock, MagicMock
from openlens.optimizer import LensOptimizer, OptimizationResult


class TestLensOptimizerGradientDescent:
    """Test suite for LensOptimizer.optimize_gradient_descent method."""

    @pytest.fixture
    def optimizer_with_mocks(self):
        """Create LensOptimizer with mocked internal methods."""
        mock_lens = Mock()
        mock_lens.name = "test_lens"
        
        optimizer = LensOptimizer(mock_lens)
        
        # Mock internal methods
        optimizer._evaluate_design = Mock(return_value=1.0)
        optimizer._calculate_gradient = Mock(return_value=[0.1])
        optimizer._apply_variables = Mock(return_value=Mock())
        
        # Create mock variable
        var = Mock()
        var.name = "test_var"
        var.current_value = 1.0
        var.is_valid = Mock(return_value=True)
        optimizer.variables = [var]
        
        return optimizer

    def test_returns_optimization_result_instance(self, optimizer_with_mocks):
        """Test that method returns OptimizationResult."""
        result = optimizer_with_mocks.optimize_gradient_descent()
        assert isinstance(result, OptimizationResult)

    def test_result_contains_required_fields(self, optimizer_with_mocks):
        """Test that OptimizationResult contains all required fields."""
        result = optimizer_with_mocks.optimize_gradient_descent()
        
        required_fields = {
            'success', 'iterations', 'initial_merit', 'final_merit',
            'improvement', 'optimized_system', 'variable_history',
            'merit_history', 'message'
        }
        result_fields = set(dir(result)) - set(dir(object))
        
        for field in required_fields:
            assert hasattr(result, field), f"Missing required field: {field}"

    def test_converges_when_improvement_below_tolerance(self, optimizer_with_mocks):
        """Test convergence stops when improvement < tolerance."""
        optimizer_with_mocks._evaluate_design.side_effect = [
            1.0,      # initial
            0.99,     # improvement = 0.01
            0.985,    # improvement = 0.005 -> stops here
        ]
        
        result = optimizer_with_mocks.optimize_gradient_descent(
            max_iterations=100,
            tolerance=0.01
        )
        
        assert result.iterations <= 3
        assert result.final_merit <= result.initial_merit

    def test_stops_at_max_iterations_limit(self, optimizer_with_mocks):
        """Test that optimization stops at max_iterations."""
        max_iters = 5
        optimizer_with_mocks._evaluate_design.side_effect = [
            float(10 - i) for i in range(max_iters + 2)
        ]
        
        result = optimizer_with_mocks.optimize_gradient_descent(
            max_iterations=max_iters,
            tolerance=1e-9
        )
        
        assert result.iterations <= max_iters

    def test_merit_never_increases_in_history(self, optimizer_with_mocks):
        """Test that final merit <= initial merit (backtracking prevents worsening)."""
        optimizer_with_mocks._evaluate_design.side_effect = [10.0, 9.0, 8.5, 8.3]
        
        result = optimizer_with_mocks.optimize_gradient_descent(tolerance=1e-9)
        
        assert result.final_merit <= result.initial_merit
        assert result.improvement >= 0

    def test_improvement_percentage_calculated_correctly(self, optimizer_with_mocks):
        """Test that improvement percentage matches formula: (initial - final) / initial * 100."""
        initial_merit = 100.0
        final_merit = 50.0
        optimizer_with_mocks._evaluate_design.side_effect = [initial_merit, final_merit]
        
        result = optimizer_with_mocks.optimize_gradient_descent(tolerance=1e-9)
        
        expected_improvement = ((initial_merit - final_merit) / initial_merit) * 100
        assert abs(result.improvement - expected_improvement) < 0.1

    def test_improvement_zero_when_initial_merit_zero(self, optimizer_with_mocks):
        """Test that improvement is 0 when initial_merit is 0 (division protection)."""
        optimizer_with_mocks._evaluate_design.side_effect = [0.0, 0.0]
        optimizer_with_mocks._calculate_gradient.return_value = [0.0]
        
        result = optimizer_with_mocks.optimize_gradient_descent()
        
        assert result.improvement == 0

    def test_variable_history_initialization_and_updates(self, optimizer_with_mocks):
        """Test that variable_history starts with initial values and grows with accepted steps."""
        optimizer_with_mocks._evaluate_design.side_effect = [1.0, 0.99, 0.98]
        
        result = optimizer_with_mocks.optimize_gradient_descent()
        
        assert isinstance(result.variable_history, list)
        assert len(result.variable_history) >= 1
        assert isinstance(result.variable_history[0], dict)
        assert "test_var" in result.variable_history[0]

    def test_merit_history_starts_with_initial_merit(self, optimizer_with_mocks):
        """Test that merit_history[0] equals initial_merit evaluation."""
        merits = [10.0, 9.5, 9.0, 8.8]
        optimizer_with_mocks._evaluate_design.side_effect = merits
        
        result = optimizer_with_mocks.optimize_gradient_descent()
        
        assert result.merit_history[0] == merits[0]
        assert result.merit_history[0] == result.initial_merit

    def test_success_true_when_all_variables_valid(self, optimizer_with_mocks):
        """Test that success=True when is_valid returns True for all variables."""
        optimizer_with_mocks._evaluate_design.side_effect = [1.0, 0.99]
        
        result = optimizer_with_mocks.optimize_gradient_descent()
        
        assert result.success is True

    def test_success_false_when_any_variable_invalid(self, optimizer_with_mocks):
        """Test that success=False when is_valid returns False for any variable."""
        optimizer_with_mocks._evaluate_design.side_effect = [1.0, 0.99]
        optimizer_with_mocks.variables[0].is_valid.return_value = False
        
        result = optimizer_with_mocks.optimize_gradient_descent()
        
        assert result.success is False

    def test_message_format_on_convergence(self, optimizer_with_mocks):
        """Test that message contains iteration count and indicates completion."""
        optimizer_with_mocks._evaluate_design.side_effect = [1.0, 0.99, 0.98]
        
        result = optimizer_with_mocks.optimize_gradient_descent()
        
        assert isinstance(result.message, str)
        assert "iteration" in result.message.lower()
        assert str(result.iterations) in result.message

    def test_multiple_variables_all_tracked_in_history(self, optimizer_with_mocks):
        """Test that all variables are tracked in variable_history when multiple exist."""
        var1 = Mock(name="radius1", current_value=1.0, is_valid=Mock(return_value=True))
        var2 = Mock(name="thickness", current_value=2.0, is_valid=Mock(return_value=True))
        var3 = Mock(name="radius2", current_value=3.0, is_valid=Mock(return_value=True))
        optimizer_with_mocks.variables = [var1, var2, var3]
        
        optimizer_with_mocks._evaluate_design.side_effect = [5.0, 4.5]
        optimizer_with_mocks._calculate_gradient.return_value = [0.1, 0.1, 0.1]
        
        result = optimizer_with_mocks.optimize_gradient_descent()
        
        assert len(result.variable_history[0]) == 3
        assert "radius1" in result.variable_history[0]
        assert "thickness" in result.variable_history[0]
        assert "radius2" in result.variable_history[0]

    def test_stops_on_stationary_point_or_stall(self, optimizer_with_mocks):
        """Test that optimization stops when backtracking cannot find improvement (stall)."""
        # Return constant merit to trigger backtracking exhaustion
        optimizer_with_mocks._evaluate_design.return_value = 1.0
        optimizer_with_mocks._calculate_gradient.return_value = [0.0]
        
        result = optimizer_with_mocks.optimize_gradient_descent(max_iterations=100)
        
        # Should stop much earlier due to stall detection
        assert result.iterations < 100
        assert result.final_merit == result.initial_merit

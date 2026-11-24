import pytest
import numpy as np
from unittest.mock import Mock
from openlens.optimizer import LensOptimizer


class TestOptimizeGradientDescent:
    """Test suite for LensOptimizer.optimize_gradient_descent method"""

    def create_optimizer_with_quadratic(self, num_vars=2, var_names=None):
        """Helper: create optimizer with quadratic merit function"""
        if var_names is None:
            var_names = [f"var{i}" for i in range(num_vars)]
        
        optimizer = LensOptimizer()
        
        variables = []
        for name in var_names:
            var = Mock()
            var.name = name
            var.current_value = 0.0
            var.is_valid = Mock(return_value=True)
            variables.append(var)
        
        optimizer.variables = variables
        optimizer.MAX_BACKTRACKING_STEPS = 10
        optimizer.MIN_LEARNING_RATE = 1e-8
        
        # Quadratic: sum of (v - offset)^2
        def merit(values):
            return sum((v - 1.0)**2 for v in values)
        
        def gradient(values):
            return [2.0 * (v - 1.0) for v in values]
        
        optimizer._evaluate_design = Mock(side_effect=merit)
        optimizer._calculate_gradient = Mock(side_effect=gradient)
        optimizer._apply_variables = Mock(return_value=Mock())
        
        return optimizer

    def test_basic_convergence(self):
        """Test that optimization reduces merit on simple quadratic"""
        optimizer = self.create_optimizer_with_quadratic(num_vars=2)
        
        result = optimizer.optimize_gradient_descent(
            max_iterations=100,
            learning_rate=0.1,
            tolerance=1e-6
        )
        
        assert result.success is True
        assert result.final_merit < result.initial_merit
        assert result.iterations > 0

    def test_merit_monotonically_decreases(self):
        """Verify merit history never increases"""
        optimizer = self.create_optimizer_with_quadratic(num_vars=3)
        
        result = optimizer.optimize_gradient_descent(
            max_iterations=50,
            learning_rate=0.1
        )
        
        for i in range(1, len(result.merit_history)):
            assert result.merit_history[i] <= result.merit_history[i-1]

    def test_history_consistency(self):
        """Test variable and merit histories have matching lengths"""
        optimizer = self.create_optimizer_with_quadratic(num_vars=2)
        
        result = optimizer.optimize_gradient_descent(max_iterations=20)
        
        assert len(result.variable_history) == len(result.merit_history)
        assert len(result.variable_history) >= 1

    def test_convergence_with_large_tolerance(self):
        """Test that large tolerance causes early stopping"""
        optimizer = self.create_optimizer_with_quadratic(num_vars=2)
        
        result = optimizer.optimize_gradient_descent(
            max_iterations=1000,
            learning_rate=0.1,
            tolerance=0.1
        )
        
        assert result.iterations < 1000

    def test_max_iterations_respected(self):
        """Test that optimization respects max_iterations limit"""
        optimizer = self.create_optimizer_with_quadratic(num_vars=2)
        
        result = optimizer.optimize_gradient_descent(
            max_iterations=5,
            learning_rate=0.01,
            tolerance=1e-12
        )
        
        assert result.iterations <= 5

    def test_stalling_on_zero_gradient(self):
        """Test detection when no progress possible (plateau)"""
        optimizer = LensOptimizer()
        
        var = Mock()
        var.name = "x"
        var.current_value = 0.0
        var.is_valid = Mock(return_value=True)
        
        optimizer.variables = [var]
        optimizer.MAX_BACKTRACKING_STEPS = 10
        optimizer.MIN_LEARNING_RATE = 1e-8
        
        optimizer._evaluate_design = Mock(return_value=100.0)
        optimizer._calculate_gradient = Mock(return_value=[0.0])
        optimizer._apply_variables = Mock(return_value=Mock())
        
        result = optimizer.optimize_gradient_descent(
            max_iterations=100,
            learning_rate=0.1
        )
        
        # On zero gradient, should stop after one failed backtracking attempt
        assert result.final_merit == result.initial_merit

    def test_backtracking_handles_large_learning_rate(self):
        """Test that backtracking corrects oversized steps"""
        optimizer = self.create_optimizer_with_quadratic(num_vars=2)
        
        result = optimizer.optimize_gradient_descent(
            max_iterations=50,
            learning_rate=5.0,  # Very large step
            tolerance=1e-6
        )
        
        # Should still converge via backtracking
        assert result.success is True
        assert result.final_merit < result.initial_merit

    def test_improvement_calculation_with_positive_merit(self):
        """Test improvement percentage calculation"""
        optimizer = self.create_optimizer_with_quadratic(num_vars=2)
        
        result = optimizer.optimize_gradient_descent(
            max_iterations=20,
            learning_rate=0.2
        )
        
        if result.initial_merit > 0:
            expected = (result.initial_merit - result.final_merit) / result.initial_merit * 100
            assert abs(result.improvement - expected) < 1e-6

    def test_improvement_zero_when_initial_merit_zero(self):
        """Test improvement calculation when initial merit is zero"""
        optimizer = LensOptimizer()
        
        var = Mock()
        var.name = "x"
        var.current_value = 0.0
        var.is_valid = Mock(return_value=True)
        
        optimizer.variables = [var]
        optimizer.MAX_BACKTRACKING_STEPS = 10
        optimizer.MIN_LEARNING_RATE = 1e-8
        
        optimizer._evaluate_design = Mock(return_value=0.0)
        optimizer._calculate_gradient = Mock(return_value=[0.0])
        optimizer._apply_variables = Mock(return_value=Mock())
        
        result = optimizer.optimize_gradient_descent(max_iterations=10)
        
        assert result.improvement == 0
        assert not np.isnan(result.improvement)

    def test_invalid_variables_in_result(self):
        """Test that invalid variables are reported in success flag"""
        optimizer = LensOptimizer()
        
        var = Mock()
        var.name = "x"
        var.current_value = 0.0
        var.is_valid = Mock(return_value=False)
        
        optimizer.variables = [var]
        optimizer.MAX_BACKTRACKING_STEPS = 10
        optimizer.MIN_LEARNING_RATE = 1e-8
        
        optimizer._evaluate_design = Mock(return_value=1.0)
        optimizer._calculate_gradient = Mock(return_value=[0.0])
        optimizer._apply_variables = Mock(return_value=Mock())
        
        result = optimizer.optimize_gradient_descent(max_iterations=10)
        
        assert result.success is False

    def test_variable_history_structure(self):
        """Test variable history contains correct variable names"""
        optimizer = self.create_optimizer_with_quadratic(
            num_vars=2, var_names=["alpha", "beta"]
        )
        
        result = optimizer.optimize_gradient_descent(max_iterations=5)
        
        for entry in result.variable_history:
            assert isinstance(entry, dict)
            assert "alpha" in entry
            assert "beta" in entry

    def test_result_object_has_all_fields(self):
        """Test that OptimizationResult contains all required fields"""
        optimizer = self.create_optimizer_with_quadratic(num_vars=1)
        
        result = optimizer.optimize_gradient_descent(max_iterations=10)
        
        required = ['success', 'iterations', 'initial_merit', 'final_merit',
                   'improvement', 'optimized_system', 'variable_history',
                   'merit_history', 'message']
        for field in required:
            assert hasattr(result, field)
        
        assert isinstance(result.success, bool)
        assert isinstance(result.iterations, int) and result.iterations > 0
        assert isinstance(result.variable_history, list)
        assert isinstance(result.merit_history, list)
        assert isinstance(result.message, str)

    def test_learning_rate_affects_convergence_speed(self):
        """Test that larger learning rate can converge faster"""
        opt_small = self.create_optimizer_with_quadratic(num_vars=2)
        result_small = opt_small.optimize_gradient_descent(
            max_iterations=100, learning_rate=0.01, tolerance=1e-6
        )
        
        opt_large = self.create_optimizer_with_quadratic(num_vars=2)
        result_large = opt_large.optimize_gradient_descent(
            max_iterations=100, learning_rate=0.3, tolerance=1e-6
        )
        
        # Larger learning rate typically converges faster (fewer iterations)
        assert result_large.iterations <= result_small.iterations + 20

    def test_tolerance_affects_final_merit(self):
        """Test that stricter tolerance achieves better merit"""
        opt_loose = self.create_optimizer_with_quadratic(num_vars=2)
        result_loose = opt_loose.optimize_gradient_descent(
            max_iterations=1000, learning_rate=0.1, tolerance=0.5
        )
        
        opt_strict = self.create_optimizer_with_quadratic(num_vars=2)
        result_strict = opt_strict.optimize_gradient_descent(
            max_iterations=1000, learning_rate=0.1, tolerance=1e-9
        )
        
        # Strict tolerance should achieve better (lower) merit
        assert result_strict.final_merit <= result_loose.final_merit

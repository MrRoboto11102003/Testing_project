import pytest
import numpy as np
from openlens.lens import Lens
from openlens.optimizer import LensOptimizer, OptimizationResult


class TestOptimizeGradientDescent:
    """Test suite for LensOptimizer.optimize_gradient_descent"""

    @pytest.fixture
    def simple_lens(self):
        """Create a basic biconvex lens for testing"""
        return Lens(
            name="test_lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )

    @pytest.fixture
    def optimizer_with_focal_target(self, simple_lens):
        """Create optimizer targeting focal length around 100mm"""
        optimizer = LensOptimizer(simple_lens)
        optimizer.add_variable(
            name="radius_1",
            current_value=50.0,
            min_value=20.0,
            max_value=200.0,
        )
        optimizer.add_target(
            name="focal_length",
            target_value=100.0,
            weight=1.0,
        )
        return optimizer

    def test_gradient_descent_returns_optimization_result(self, optimizer_with_focal_target):
        """Test that method returns OptimizationResult object"""
        result = optimizer_with_focal_target.optimize_gradient_descent()
        assert isinstance(result, OptimizationResult)

    def test_gradient_descent_result_has_required_fields(self, optimizer_with_focal_target):
        """Test that result contains all documented fields"""
        result = optimizer_with_focal_target.optimize_gradient_descent()

        required_fields = {
            "success": bool,
            "iterations": int,
            "initial_merit": (int, float),
            "final_merit": (int, float),
            "improvement": (int, float),
            "optimized_system": object,
            "variable_history": list,
            "merit_history": list,
            "message": str,
        }

        for field, expected_type in required_fields.items():
            assert hasattr(result, field), f"Missing field: {field}"
            value = getattr(result, field)
            assert isinstance(value, expected_type), (
                f"Field {field} has type {type(value)}, expected {expected_type}"
            )

    def test_gradient_descent_merit_decreases(self, optimizer_with_focal_target):
        """Test that final merit is less than or equal to initial merit"""
        result = optimizer_with_focal_target.optimize_gradient_descent()
        assert result.final_merit <= result.initial_merit

    def test_gradient_descent_merit_history_monotonic(self, optimizer_with_focal_target):
        """Test that merit history is monotonically non-increasing"""
        result = optimizer_with_focal_target.optimize_gradient_descent()

        for i in range(1, len(result.merit_history)):
            # Allow small numerical tolerance
            assert result.merit_history[i] <= result.merit_history[i - 1] + 1e-10

    def test_gradient_descent_history_lengths_match(self, optimizer_with_focal_target):
        """Test that variable and merit history have same length"""
        result = optimizer_with_focal_target.optimize_gradient_descent()

        assert len(result.variable_history) == len(result.merit_history)
        assert len(result.variable_history) > 0

    def test_gradient_descent_accepts_max_iterations(self, optimizer_with_focal_target):
        """Test that max_iterations parameter is respected"""
        result = optimizer_with_focal_target.optimize_gradient_descent(
            max_iterations=5,
            learning_rate=0.01,
            tolerance=1e-12,
        )

        assert result.iterations <= 5

    def test_gradient_descent_accepts_learning_rate(self, optimizer_with_focal_target):
        """Test that learning_rate parameter affects optimization"""
        result_small = optimizer_with_focal_target.optimize_gradient_descent(
            learning_rate=0.01, max_iterations=50
        )
        result_large = optimizer_with_focal_target.optimize_gradient_descent(
            learning_rate=0.5, max_iterations=50
        )

        # Both should complete
        assert result_small.iterations >= 1
        assert result_large.iterations >= 1

    def test_gradient_descent_accepts_tolerance(self, optimizer_with_focal_target):
        """Test that tolerance parameter affects convergence"""
        result_tight = optimizer_with_focal_target.optimize_gradient_descent(
            tolerance=1e-10, max_iterations=100
        )
        result_loose = optimizer_with_focal_target.optimize_gradient_descent(
            tolerance=1e-2, max_iterations=100
        )

        # Loose tolerance typically converges faster
        assert result_loose.iterations <= result_tight.iterations

    def test_gradient_descent_improvement_calculation(self, optimizer_with_focal_target):
        """Test that improvement percentage is calculated correctly"""
        result = optimizer_with_focal_target.optimize_gradient_descent()

        if result.initial_merit > 0:
            expected_improvement = (
                (result.initial_merit - result.final_merit)
                / result.initial_merit
                * 100
            )
            assert abs(result.improvement - expected_improvement) < 1e-6

    def test_gradient_descent_improvement_non_negative(self, optimizer_with_focal_target):
        """Test that improvement is non-negative percentage"""
        result = optimizer_with_focal_target.optimize_gradient_descent()
        assert 0 <= result.improvement <= 100

    def test_gradient_descent_multiple_variables(self, simple_lens):
        """Test optimization with multiple variables"""
        optimizer = LensOptimizer(simple_lens)
        optimizer.add_variable("radius_1", 50.0, 20.0, 200.0)
        optimizer.add_variable("radius_2", -50.0, -200.0, -20.0)
        optimizer.add_target("focal_length", 50.0, 1.0)

        result = optimizer.optimize_gradient_descent(max_iterations=50)

        assert result.iterations > 0
        # Should have entries for both variables
        if result.variable_history:
            assert len(result.variable_history[0]) == 2

    def test_gradient_descent_message_contains_info(self, optimizer_with_focal_target):
        """Test that message field contains optimization information"""
        result = optimizer_with_focal_target.optimize_gradient_descent()

        assert isinstance(result.message, str)
        assert "iteration" in result.message.lower()

    def test_gradient_descent_success_indicates_validity(self, optimizer_with_focal_target):
        """Test that success field reflects variable bound validity"""
        result = optimizer_with_focal_target.optimize_gradient_descent()

        assert isinstance(result.success, bool)
        # Message should indicate the reason
        if not result.success:
            assert "outside variable bounds" in result.message.lower() or "failed" in result.message.lower()

    def test_gradient_descent_handles_tight_tolerance(self, optimizer_with_focal_target):
        """Test that extremely tight tolerance does not cause errors"""
        result = optimizer_with_focal_target.optimize_gradient_descent(
            max_iterations=100, tolerance=1e-15
        )

        assert result.iterations > 0
        assert result.iterations <= 100

    def test_gradient_descent_backtracking_handles_large_learning_rate(self, optimizer_with_focal_target):
        """Test that backtracking line search handles large learning rates"""
        result = optimizer_with_focal_target.optimize_gradient_descent(
            learning_rate=5.0, max_iterations=50
        )

        # Should complete without diverging due to backtracking
        assert result.iterations >= 1
        assert result.final_merit >= 0

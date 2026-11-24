import pytest
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.optimizer import (
    LensOptimizer,
    OptimizationVariable,
    OptimizationTarget,
    OptimizationResult,
)


class FakeMerit:
    """Deterministic merit function for testing."""

    def __init__(self, func):
        """
        Args:
            func: callable(thickness) -> float, where thickness is the first
                  lens thickness from the system.
        """
        self.func = func

    def evaluate(self, system):
        """Evaluate merit at current system state."""
        if not system.elements:
            return 0.0
        thickness = system.elements[0].lens.thickness
        return float(self.func(thickness))


@pytest.fixture
def simple_system():
    """Create a simple optical system with one lens."""
    sys = OpticalSystem()
    lens = Lens(
        name="test_lens",
        radius_of_curvature_1=50.0,
        radius_of_curvature_2=-50.0,
        thickness=5.0,
        diameter=25.0,
        material="BK7",
    )
    sys.add_lens(lens)
    return sys


@pytest.fixture
def thickness_variable():
    """OptimizationVariable for the first lens thickness."""
    return OptimizationVariable(
        name="thickness",
        index=0,
        field="thickness",
        current_value=5.0,
        min_value=1.0,
        max_value=20.0,
    )


class TestGradientDescentConvergence:
    """Test convergence on smooth merit functions."""

    def test_smooth_quadratic_convergence(self, simple_system, thickness_variable):
        """Test convergence on quadratic merit f=(x-3)^2 starting from x=5."""
        opt = LensOptimizer(
            simple_system,
            [thickness_variable],
            [OptimizationTarget("focal_length", 50.0)],
        )
        opt.merit_function = FakeMerit(lambda x: (x - 3.0) ** 2)

        result = opt.optimize_gradient_descent(
            max_iterations=100, learning_rate=0.1, tolerance=1e-6
        )

        assert isinstance(result, OptimizationResult)
        assert result.success is True
        assert result.final_merit < 0.01
        assert result.improvement > 50
        assert result.iterations > 1
        assert len(result.merit_history) == len(result.variable_history)
        assert result.merit_history[-1] <= result.merit_history[0]

    def test_constant_merit_accepts_tie_and_stops(self, simple_system, thickness_variable):
        """Test on constant merit: gradient 0, ties accepted, improvement 0 stops."""
        opt = LensOptimizer(
            simple_system,
            [thickness_variable],
            [OptimizationTarget("focal_length", 50.0)],
        )
        opt.merit_function = FakeMerit(lambda x: 5.0)

        result = opt.optimize_gradient_descent(max_iterations=100)

        assert result.iterations == 1
        assert len(result.merit_history) == 2
        assert len(result.variable_history) == 2
        assert result.merit_history[0] == result.merit_history[1] == 5.0
        assert result.improvement == 0.0
        assert result.success is True


class TestGradientDescentEdgeCases:
    """Test boundary conditions and special cases."""

    def test_max_iterations_zero(self, simple_system, thickness_variable):
        """Test that max_iterations=0 prevents loop from running."""
        opt = LensOptimizer(
            simple_system,
            [thickness_variable],
            [OptimizationTarget("focal_length", 50.0)],
        )
        opt.merit_function = FakeMerit(lambda x: (x - 3.0) ** 2)

        result = opt.optimize_gradient_descent(max_iterations=0)

        assert result.iterations == 1
        assert len(result.merit_history) == 1
        assert len(result.variable_history) == 1
        assert result.final_merit == result.initial_merit

    def test_no_variables(self, simple_system):
        """Test optimization with zero variables."""
        opt = LensOptimizer(
            simple_system, [], [OptimizationTarget("focal_length", 50.0)]
        )
        opt.merit_function = FakeMerit(lambda x: 10.0)

        result = opt.optimize_gradient_descent(max_iterations=100)

        assert result.iterations == 1
        assert result.success is True
        assert len(result.variable_history) == 1
        assert result.variable_history[0] == {}

    def test_improvement_zero_when_initial_merit_zero(
        self, simple_system, thickness_variable
    ):
        """Test improvement calculation when initial_merit is 0."""
        opt = LensOptimizer(
            simple_system,
            [thickness_variable],
            [OptimizationTarget("focal_length", 50.0)],
        )
        opt.merit_function = FakeMerit(lambda x: 0.0)

        result = opt.optimize_gradient_descent(max_iterations=50)

        assert result.improvement == 0.0
        assert result.initial_merit == 0.0


class TestGradientDescentBehavior:
    """Test specific algorithm behaviors and mechanics."""

    def test_ties_accepted(self, simple_system, thickness_variable):
        """Test that merit ties (new==current) are accepted."""
        opt = LensOptimizer(
            simple_system,
            [thickness_variable],
            [OptimizationTarget("focal_length", 50.0)],
        )
        opt.merit_function = FakeMerit(lambda x: 7.0)

        result = opt.optimize_gradient_descent(max_iterations=100, learning_rate=0.1)

        assert result.iterations == 1
        assert len(result.merit_history) == 2

    def test_tolerance_causes_convergence(self, simple_system, thickness_variable):
        """Test that small improvement < tolerance causes early exit."""
        opt = LensOptimizer(
            simple_system,
            [thickness_variable],
            [OptimizationTarget("focal_length", 50.0)],
        )
        opt.merit_function = FakeMerit(lambda x: 1e-8 * (x - 5.0) ** 2)

        result = opt.optimize_gradient_descent(
            max_iterations=100, tolerance=1e-9, learning_rate=0.001
        )

        assert result.iterations < 100

    def test_history_lengths_equal(self, simple_system, thickness_variable):
        """Test that variable_history and merit_history have same length."""
        opt = LensOptimizer(
            simple_system,
            [thickness_variable],
            [OptimizationTarget("focal_length", 50.0)],
        )
        opt.merit_function = FakeMerit(lambda x: (x - 5.5) ** 2)

        result = opt.optimize_gradient_descent(max_iterations=50)

        assert len(result.variable_history) == len(result.merit_history)

    def test_improvement_formula(self, simple_system, thickness_variable):
        """Test improvement = (initial - final) / initial * 100."""
        opt = LensOptimizer(
            simple_system,
            [thickness_variable],
            [OptimizationTarget("focal_length", 50.0)],
        )
        opt.merit_function = FakeMerit(lambda x: (x - 3.0) ** 2)

        result = opt.optimize_gradient_descent(max_iterations=100)

        expected = (result.initial_merit - result.final_merit) / result.initial_merit * 100
        assert result.improvement == pytest.approx(expected, rel=1e-6)


class TestGradientDescentSideEffects:
    """Test that original objects are not mutated."""

    def test_original_system_unchanged(self, simple_system, thickness_variable):
        """Test that input system is not modified."""
        original_thickness = simple_system.elements[0].lens.thickness
        opt = LensOptimizer(
            simple_system,
            [thickness_variable],
            [OptimizationTarget("focal_length", 50.0)],
        )
        opt.merit_function = FakeMerit(lambda x: (x - 10.0) ** 2)

        result = opt.optimize_gradient_descent(max_iterations=50)

        assert simple_system.elements[0].lens.thickness == original_thickness
        assert result.optimized_system is not simple_system

    def test_variable_current_value_unchanged(self, simple_system, thickness_variable):
        """Test that OptimizationVariable.current_value is not updated."""
        original_value = thickness_variable.current_value
        opt = LensOptimizer(
            simple_system,
            [thickness_variable],
            [OptimizationTarget("focal_length", 50.0)],
        )
        opt.merit_function = FakeMerit(lambda x: (x - 10.0) ** 2)

        result = opt.optimize_gradient_descent(max_iterations=50)

        assert thickness_variable.current_value == original_value


class TestGradientDescentResultFields:
    """Test OptimizationResult field correctness."""

    def test_success_message_format(self, simple_system, thickness_variable):
        """Test success message includes 'Completed' and iteration count."""
        opt = LensOptimizer(
            simple_system,
            [thickness_variable],
            [OptimizationTarget("focal_length", 50.0)],
        )
        opt.merit_function = FakeMerit(lambda x: (x - 3.0) ** 2)

        result = opt.optimize_gradient_descent(max_iterations=100)

        assert result.success is True
        assert "Completed" in result.message
        assert str(result.iterations) in result.message

    def test_best_values_not_populated(self, simple_system, thickness_variable):
        """Test that best_values field is not filled."""
        opt = LensOptimizer(
            simple_system,
            [thickness_variable],
            [OptimizationTarget("focal_length", 50.0)],
        )
        opt.merit_function = FakeMerit(lambda x: (x - 3.0) ** 2)

        result = opt.optimize_gradient_descent(max_iterations=50)

        assert result.best_values == []

    def test_initial_merit_matches_history_start(self, simple_system, thickness_variable):
        """Test that initial_merit equals first merit_history entry."""
        opt = LensOptimizer(
            simple_system,
            [thickness_variable],
            [OptimizationTarget("focal_length", 50.0)],
        )
        opt.merit_function = FakeMerit(lambda x: (x - 5.0) ** 2)

        result = opt.optimize_gradient_descent(max_iterations=100)

        assert result.initial_merit == pytest.approx(result.merit_history[0])


class TestGradientDescentOutOfBounds:
    """Test out-of-bounds variable handling."""

    def test_out_of_bounds_causes_failure(self, simple_system):
        """Test that starting outside bounds causes success=False."""
        var = OptimizationVariable(
            name="thickness",
            index=0,
            field="thickness",
            current_value=5.0,
            min_value=10.0,
            max_value=20.0,
        )
        opt = LensOptimizer(
            simple_system, [var], [OptimizationTarget("focal_length", 50.0)]
        )
        opt.merit_function = FakeMerit(lambda x: (x - 15.0) ** 2)

        result = opt.optimize_gradient_descent(max_iterations=100)

        assert result.success is False
        assert "outside variable bounds" in result.message


class TestGradientDescentErrors:
    """Test error handling and exceptions."""

    def test_unknown_target_raises_value_error(self, simple_system, thickness_variable):
        """Test that unknown target name raises ValueError."""
        opt = LensOptimizer(
            simple_system,
            [thickness_variable],
            [OptimizationTarget("unknown_target_xyz", 0.0)],
        )

        with pytest.raises(ValueError, match="Unknown optimization target"):
            opt.optimize_gradient_descent(max_iterations=10)


class TestGradientDescentNumerical:
    """Test numerical correctness."""

    def test_merit_non_increasing(self, simple_system, thickness_variable):
        """Test that merit_history is monotonically non-increasing."""
        opt = LensOptimizer(
            simple_system,
            [thickness_variable],
            [OptimizationTarget("focal_length", 50.0)],
        )
        opt.merit_function = FakeMerit(lambda x: abs(x - 7.0) ** 1.5)

        result = opt.optimize_gradient_descent(max_iterations=100)

        for i in range(1, len(result.merit_history)):
            assert result.merit_history[i] <= result.merit_history[i - 1] + 1e-10

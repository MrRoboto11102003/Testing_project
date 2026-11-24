import pytest
from unittest.mock import MagicMock
from openlens.optimizer import LensOptimizer, OptimizationVariable
from openlens.optical_system import OpticalSystem


class TestOptimizeSimplex:
    """Tests for LensOptimizer.optimize_simplex()"""

    @pytest.fixture
    def empty_system(self):
        """Empty optical system with no merit penalty."""
        return OpticalSystem()

    @pytest.fixture
    def simple_variable(self):
        """Single variable for basic testing."""
        return OptimizationVariable(
            name="x",
            index=0,
            parameter="dummy",
            current_value=5.0,
            min_value=0.0,
            max_value=10.0,
            step_size=1.0,
        )

    def test_no_variables_raises_error(self, empty_system):
        """Should raise ValueError when no variables provided."""
        opt = LensOptimizer(empty_system, [], [])
        with pytest.raises(
            ValueError,
            match="optimize_simplex needs at least one variable",
        ):
            opt.optimize_simplex()

    def test_wrong_start_length_raises_error(self, empty_system, simple_variable):
        """Should raise ValueError when start length != num variables."""
        opt = LensOptimizer(empty_system, [simple_variable], [])
        with pytest.raises(
            ValueError,
            match=r"start has 2 value\(s\) but there are 1 variable\(s\)",
        ):
            opt.optimize_simplex(start=[1.0, 2.0])

    def test_empty_start_list_raises_error(self, empty_system, simple_variable):
        """Should raise ValueError when start is empty list (not None)."""
        opt = LensOptimizer(empty_system, [simple_variable], [])
        with pytest.raises(
            ValueError,
            match=r"start has 0 value\(s\) but there are 1 variable\(s\)",
        ):
            opt.optimize_simplex(start=[])

    def test_converge_at_iteration_zero(self, empty_system, simple_variable):
        """Should converge immediately when simplex has zero merit range."""
        opt = LensOptimizer(empty_system, [simple_variable], [])
        result = opt.optimize_simplex()

        assert result.success is True
        assert result.iterations == 1
        assert result.best_values == [5.0]
        assert result.final_merit == 0.0
        assert result.improvement == 0
        assert "Converged after 1 iterations" in result.message
        assert len(result.variable_history) == 1
        assert len(result.merit_history) == 1

    def test_callback_invoked_at_iteration_zero(self, empty_system, simple_variable):
        """Callback should be called once at iteration 0 before convergence."""
        opt = LensOptimizer(empty_system, [simple_variable], [])
        callback = MagicMock()

        opt.optimize_simplex(callback=callback)

        callback.assert_called_once()
        iteration, merit, values = callback.call_args[0]
        assert iteration == 0
        assert merit == 0.0
        assert values == [5.0]

    def test_max_iterations_zero(self, empty_system, simple_variable):
        """With max_iterations=0, no loop iterations occur."""
        opt = LensOptimizer(empty_system, [simple_variable], [])
        callback = MagicMock()

        result = opt.optimize_simplex(max_iterations=0, callback=callback)

        # Initial simplex evaluated but loop never runs
        callback.assert_not_called()
        assert result.iterations == 1
        assert len(result.merit_history) == 1

    def test_start_parameter_overrides_current_value(
        self, empty_system, simple_variable
    ):
        """When start is provided, it should override current_value."""
        opt = LensOptimizer(empty_system, [simple_variable], [])
        result = opt.optimize_simplex(start=[7.0])

        assert result.best_values[0] == 7.0
        assert result.initial_merit == 0.0

    def test_variables_current_value_unchanged(
        self, empty_system, simple_variable
    ):
        """Variables' current_value should not be modified by optimize_simplex."""
        original_value = simple_variable.current_value
        opt = LensOptimizer(empty_system, [simple_variable], [])
        opt.optimize_simplex()

        assert simple_variable.current_value == original_value

    def test_optimized_system_is_copy(self, empty_system, simple_variable):
        """Returned optimized_system should be a different object from input."""
        opt = LensOptimizer(empty_system, [simple_variable], [])
        result = opt.optimize_simplex()

        assert result.optimized_system is not empty_system

    def test_history_lengths_match(self, empty_system, simple_variable):
        """variable_history and merit_history should always have same length."""
        opt = LensOptimizer(empty_system, [simple_variable], [])
        result = opt.optimize_simplex()

        assert len(result.variable_history) == len(result.merit_history)

    def test_merit_history_nonincreasing(self, empty_system):
        """merit_history should be non-increasing (monotonic)."""
        var = OptimizationVariable(
            name="x",
            index=0,
            parameter="dummy",
            current_value=0.0,
            min_value=-100.0,
            max_value=100.0,
            step_size=1.0,
        )
        opt = LensOptimizer(empty_system, [var], [])

        # Simple quadratic merit function
        opt._evaluate_design = lambda vals: (vals[0] - 3.0) ** 2

        result = opt.optimize_simplex(max_iterations=20)

        # Verify non-increasing (allow tiny float tolerance)
        for i in range(1, len(result.merit_history)):
            assert (
                result.merit_history[i]
                <= result.merit_history[i - 1] + 1e-10
            )

    def test_improvement_calculation(self, empty_system):
        """Improvement = (initial - final) / initial * 100 when initial > 0."""
        var = OptimizationVariable(
            name="x",
            index=0,
            parameter="dummy",
            current_value=0.0,
            min_value=-100.0,
            max_value=100.0,
            step_size=1.0,
        )
        opt = LensOptimizer(empty_system, [var], [])

        # Quadratic with minimum at 3.0
        opt._evaluate_design = lambda vals: (vals[0] - 3.0) ** 2

        result = opt.optimize_simplex(max_iterations=50)

        if result.initial_merit > 0:
            expected_improvement = (
                (result.initial_merit - result.final_merit)
                / result.initial_merit
                * 100
            )
            assert result.improvement == pytest.approx(expected_improvement)

    def test_improvement_zero_when_initial_merit_zero(
        self, empty_system, simple_variable
    ):
        """When initial_merit=0, improvement should be 0."""
        opt = LensOptimizer(empty_system, [simple_variable], [])
        result = opt.optimize_simplex()

        assert result.initial_merit == 0.0
        assert result.improvement == 0

    def test_multiple_variables(self, empty_system):
        """Simplex should handle multiple variables."""
        variables = [
            OptimizationVariable(
                name=f"x{i}",
                index=i,
                parameter="dummy",
                current_value=0.0,
                min_value=-100.0,
                max_value=100.0,
                step_size=1.0,
            )
            for i in range(3)
        ]
        opt = LensOptimizer(empty_system, variables, [])

        result = opt.optimize_simplex(max_iterations=0)

        assert len(result.best_values) == 3
        assert result.iterations == 1

    def test_start_with_multiple_variables(self, empty_system):
        """start list should correctly initialize multiple variables."""
        variables = [
            OptimizationVariable(
                name=f"x{i}",
                index=i,
                parameter="dummy",
                current_value=0.0,
                min_value=-100.0,
                max_value=100.0,
                step_size=1.0,
            )
            for i in range(3)
        ]
        opt = LensOptimizer(empty_system, variables, [])
        start = [1.5, 2.5, 3.5]

        result = opt.optimize_simplex(start=start, max_iterations=0)

        assert result.best_values == pytest.approx(start, abs=0.1)

    def test_out_of_bounds_success_false(self, empty_system):
        """Out-of-bounds best point should result in success=False."""
        var = OptimizationVariable(
            name="x",
            index=0,
            parameter="dummy",
            current_value=5.0,
            min_value=0.0,
            max_value=1.0,  # start is out of bounds
            step_size=1.0,
        )
        opt = LensOptimizer(empty_system, [var], [])

        result = opt.optimize_simplex(max_iterations=0)

        # Out of bounds should incur penalty
        assert result.final_merit > 0
        assert result.success is False

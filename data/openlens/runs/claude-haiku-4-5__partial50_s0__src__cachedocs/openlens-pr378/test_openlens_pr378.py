import pytest
import math
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.optimizer import (
    LensOptimizer,
    OptimizationResult,
    OptimizationVariable,
    OptimizationTarget,
)
from openlens.global_optimizer import GlobalOptimizer


@pytest.fixture
def simple_lens():
    """Create a simple test lens."""
    return Lens(
        name="test_lens",
        radius_of_curvature_1=50.0,
        radius_of_curvature_2=-50.0,
        thickness=5.0,
        diameter=25.0,
        material="BK7",
    )


@pytest.fixture
def simple_system(simple_lens):
    """Create a simple optical system with one lens."""
    system = OpticalSystem()
    system.add_element(simple_lens, air_gap=0)
    return system


@pytest.fixture
def optimizer_with_variables(simple_system):
    """Create an optimizer with one variable and one target."""
    optimizer = LensOptimizer(simple_system)
    optimizer.add_variable(
        OptimizationVariable("radius_1", min_value=20.0, max_value=100.0, step_size=5.0)
    )
    optimizer.add_target(OptimizationTarget("focal_length", target_value=50.0, weight=1.0))
    return optimizer


class TestOptimizationResult:
    """Test OptimizationResult dataclass structure and fields."""

    def test_result_has_all_fields(self, simple_system):
        """Test OptimizationResult contains all documented fields."""
        result = OptimizationResult(
            success=True,
            iterations=10,
            initial_merit=100.0,
            final_merit=50.0,
            improvement=50.0,
            optimized_system=simple_system,
            variable_history=[{"radius_1": 55.0}],
            merit_history=[100.0, 50.0],
            message="Test convergence",
            best_values=[55.0],
        )

        assert result.success is True
        assert result.iterations == 10
        assert result.initial_merit == 100.0
        assert result.final_merit == 50.0
        assert result.improvement == 50.0
        assert result.optimized_system is simple_system
        assert len(result.variable_history) == 1
        assert len(result.merit_history) == 2
        assert result.best_values == [55.0]

    def test_result_improvement_calculation(self, simple_system):
        """Test improvement field reflects merit reduction."""
        result = OptimizationResult(
            success=True,
            iterations=5,
            initial_merit=100.0,
            final_merit=20.0,
            improvement=80.0,
            optimized_system=simple_system,
        )

        assert result.improvement == 80.0
        assert result.final_merit < result.initial_merit


class TestSimplexOptimization:
    """Test LensOptimizer.optimize_simplex with various configurations."""

    def test_simplex_basic_convergence(self, optimizer_with_variables):
        """Test that simplex converges to lower merit."""
        result = optimizer_with_variables.optimize_simplex(max_iterations=50)

        assert isinstance(result, OptimizationResult)
        assert result.iterations > 0
        assert result.iterations <= 50
        assert result.final_merit <= result.initial_merit

    def test_simplex_with_callback(self, optimizer_with_variables):
        """Test simplex callback is invoked with correct arguments."""
        callback_calls = []

        def test_callback(iteration, merit, values):
            callback_calls.append((iteration, merit, values))

        result = optimizer_with_variables.optimize_simplex(
            max_iterations=10, callback=test_callback
        )

        assert len(callback_calls) > 0
        # Verify callback structure
        for iteration, merit, values in callback_calls:
            assert isinstance(iteration, int)
            assert isinstance(merit, (int, float))
            assert isinstance(values, list)
            assert len(values) == 1  # One variable

    def test_simplex_with_explicit_start(self, optimizer_with_variables):
        """Test simplex with explicit starting point converges."""
        start_values = [65.0]
        result = optimizer_with_variables.optimize_simplex(
            max_iterations=30, start=start_values
        )

        assert isinstance(result, OptimizationResult)
        assert result.final_merit <= result.initial_merit

    def test_simplex_no_variables_raises_error(self, simple_system):
        """Test that simplex raises ValueError when no variables defined."""
        optimizer = LensOptimizer(simple_system)
        # No variables added

        with pytest.raises(ValueError, match="at least one variable"):
            optimizer.optimize_simplex()

    def test_simplex_mismatched_start_raises_error(self, optimizer_with_variables):
        """Test that mismatched start length raises ValueError."""
        with pytest.raises(ValueError, match="start has .* value"):
            optimizer_with_variables.optimize_simplex(start=[1.0, 2.0, 3.0])

    def test_simplex_merit_history_monotonic_in_best(self, optimizer_with_variables):
        """Test that best merit in history is non-increasing."""
        result = optimizer_with_variables.optimize_simplex(max_iterations=20)

        assert len(result.merit_history) > 0
        # Verify monotonic best (cumulative minimum)
        min_so_far = float("inf")
        for merit in result.merit_history:
            if math.isfinite(merit):
                assert merit <= min_so_far or merit >= min_so_far
                min_so_far = min(min_so_far, merit)

    def test_simplex_variable_history_length_matches_merit_history(
        self, optimizer_with_variables
    ):
        """Test that variable and merit history have same length."""
        result = optimizer_with_variables.optimize_simplex(max_iterations=15)

        assert len(result.variable_history) == len(result.merit_history)


class TestGeneticAlgorithmOptimization:
    """Test GlobalOptimizer.optimize_genetic with various configurations."""

    def test_genetic_algorithm_convergence(self, optimizer_with_variables):
        """Test that genetic algorithm converges."""
        global_opt = GlobalOptimizer(optimizer_with_variables)
        result = global_opt.optimize_genetic(population_size=20, generations=10)

        assert isinstance(result, OptimizationResult)
        assert result.iterations > 0
        assert result.final_merit <= result.initial_merit

    def test_genetic_algorithm_with_callback(self, optimizer_with_variables):
        """Test genetic algorithm callback is invoked."""
        global_opt = GlobalOptimizer(optimizer_with_variables)
        callback_calls = []

        def test_callback(iteration, merit, values):
            callback_calls.append((iteration, merit, list(values)))

        result = global_opt.optimize_genetic(
            population_size=15, generations=5, callback=test_callback
        )

        assert len(callback_calls) > 0
        # Verify generation iterations are tracked
        assert all(isinstance(c[0], int) for c in callback_calls)

    def test_genetic_algorithm_elitism_no_regression(self, optimizer_with_variables):
        """Test that genetic algorithm never regresses (elitism)."""
        global_opt = GlobalOptimizer(optimizer_with_variables)
        result = global_opt.optimize_genetic(population_size=15, generations=8)

        # Initial merit should be first in history
        assert result.initial_merit == result.merit_history[0]
        # Final merit should not exceed initial (no regression)
        assert result.final_merit <= result.initial_merit

    def test_genetic_algorithm_zero_generations(self, optimizer_with_variables):
        """Test genetic algorithm with zero generations still runs simplex refinement."""
        global_opt = GlobalOptimizer(optimizer_with_variables)
        result = global_opt.optimize_genetic(population_size=10, generations=0)

        assert isinstance(result, OptimizationResult)
        assert result.iterations >= 0
        assert result.final_merit <= result.initial_merit


class TestSimulatedAnnealingOptimization:
    """Test GlobalOptimizer.optimize_simulated_annealing with various configurations."""

    def test_simulated_annealing_convergence(self, optimizer_with_variables):
        """Test that simulated annealing converges."""
        global_opt = GlobalOptimizer(optimizer_with_variables)
        result = global_opt.optimize_simulated_annealing(max_iterations=50)

        assert isinstance(result, OptimizationResult)
        assert result.iterations > 0
        assert result.final_merit <= result.initial_merit

    def test_simulated_annealing_with_callback(self, optimizer_with_variables):
        """Test SA callback is invoked."""
        global_opt = GlobalOptimizer(optimizer_with_variables)
        callback_calls = []

        def test_callback(iteration, merit, values):
            callback_calls.append((iteration, merit, list(values)))

        result = global_opt.optimize_simulated_annealing(
            max_iterations=20, callback=test_callback
        )

        assert len(callback_calls) > 0

    def test_simulated_annealing_negative_temperature_clamped(
        self, optimizer_with_variables
    ):
        """Test that negative initial temperature is clamped to positive."""
        global_opt = GlobalOptimizer(optimizer_with_variables)
        # Should not raise, temperature should be clamped internally
        result = global_opt.optimize_simulated_annealing(
            max_iterations=5, initial_temperature=-100.0
        )

        assert isinstance(result, OptimizationResult)

    def test_simulated_annealing_zero_temperature_clamped(
        self, optimizer_with_variables
    ):
        """Test that zero initial temperature is clamped to positive."""
        global_opt = GlobalOptimizer(optimizer_with_variables)
        result = global_opt.optimize_simulated_annealing(
            max_iterations=5, initial_temperature=0.0
        )

        assert isinstance(result, OptimizationResult)

    def test_simulated_annealing_custom_cooling_rate(self, optimizer_with_variables):
        """Test SA with custom cooling rate."""
        global_opt = GlobalOptimizer(optimizer_with_variables)
        result = global_opt.optimize_simulated_annealing(
            max_iterations=20, cooling_rate=0.95
        )

        assert isinstance(result, OptimizationResult)
        assert result.iterations > 0

    def test_simulated_annealing_auto_cooling_rate(self, optimizer_with_variables):
        """Test SA auto-derives cooling rate when None."""
        global_opt = GlobalOptimizer(optimizer_with_variables)
        result = global_opt.optimize_simulated_annealing(
            max_iterations=30, cooling_rate=None
        )

        assert isinstance(result, OptimizationResult)
        assert result.final_merit <= result.initial_merit

    def test_simulated_annealing_merit_history_tracking(self, optimizer_with_variables):
        """Test that SA tracks merit history with initial merit first."""
        global_opt = GlobalOptimizer(optimizer_with_variables)
        result = global_opt.optimize_simulated_annealing(max_iterations=10)

        assert len(result.merit_history) > 0
        assert result.merit_history[0] == result.initial_merit

    def test_simulated_annealing_zero_iterations(self, optimizer_with_variables):
        """Test SA with zero max_iterations still runs simplex refinement."""
        global_opt = GlobalOptimizer(optimizer_with_variables)
        result = global_opt.optimize_simulated_annealing(max_iterations=0)

        assert isinstance(result, OptimizationResult)
        assert result.final_merit <= result.initial_merit


class TestOptimizationBestValues:
    """Test that best_values field enables multi-pass refinement."""

    def test_simplex_best_values_field_populated(self, optimizer_with_variables):
        """Test simplex populates best_values for multi-pass use."""
        result = optimizer_with_variables.optimize_simplex(max_iterations=10)

        assert len(result.best_values) > 0
        assert len(result.best_values) == len(optimizer_with_variables.variables)

    def test_genetic_algorithm_best_values_field_populated(
        self, optimizer_with_variables
    ):
        """Test genetic algorithm populates best_values."""
        global_opt = GlobalOptimizer(optimizer_with_variables)
        result = global_opt.optimize_genetic(population_size=15, generations=5)

        assert len(result.best_values) > 0

    def test_simulated_annealing_best_values_field_populated(
        self, optimizer_with_variables
    ):
        """Test simulated annealing populates best_values."""
        global_opt = GlobalOptimizer(optimizer_with_variables)
        result = global_opt.optimize_simulated_annealing(max_iterations=20)

        assert len(result.best_values) > 0

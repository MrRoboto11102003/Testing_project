import pytest
import numpy as np
from openlens.global_optimizer import GlobalOptimizer, OptimizationResult
from openlens.lens import Lens


@pytest.fixture
def test_lens():
    """Create a simple test lens for optimization tests."""
    return Lens(
        name="test_lens",
        radius_of_curvature_1=50.0,
        radius_of_curvature_2=-50.0,
        thickness=5.0,
        diameter=25.0,
        material="BK7",
    )


@pytest.fixture
def optimizer(test_lens):
    """Create an optimizer with two simple variables."""
    opt = GlobalOptimizer(test_lens)
    opt.add_variable("radius_1", min_value=20.0, max_value=100.0, step_size=5.0)
    opt.add_variable("radius_2", min_value=-100.0, max_value=-20.0, step_size=5.0)
    return opt


class TestOptimizeGenetic:
    """Test suite for optimize_genetic method."""

    def test_genetic_returns_valid_result(self, optimizer):
        """GA should return OptimizationResult with required fields."""
        result = optimizer.optimize_genetic(generations=1)
        assert isinstance(result, OptimizationResult)
        assert result.best_values is not None
        assert result.final_merit is not None
        assert result.iterations >= 0

    def test_genetic_zero_generations_valid(self, optimizer):
        """GA with zero generations should still return valid result."""
        result = optimizer.optimize_genetic(generations=0)
        assert isinstance(result, OptimizationResult)
        assert len(result.best_values) == 2

    def test_genetic_no_regression(self, optimizer):
        """GA should not produce worse result than starting design (no regression)."""
        result = optimizer.optimize_genetic(population_size=10, generations=3)
        # In minimization: final_merit <= initial_merit
        is_improvement = result.final_merit <= result.initial_merit
        is_nan = np.isnan(result.final_merit)
        assert is_improvement or is_nan

    def test_genetic_callback_invoked(self, optimizer):
        """Callback should be invoked during optimization."""
        calls = []
        def callback(gen, merit, design):
            calls.append((gen, merit, len(design)))
        
        optimizer.optimize_genetic(
            population_size=5,
            generations=2,
            callback=callback,
        )
        assert len(calls) > 0
        # Check callback receives correct parameters
        assert all(isinstance(gen, int) for gen, _, _ in calls)
        assert all(isinstance(merit, (int, float)) for _, merit, _ in calls)
        assert all(length == 2 for _, _, length in calls)

    def test_genetic_population_size_parameter(self, optimizer):
        """GA should accept and use population_size parameter."""
        result = optimizer.optimize_genetic(population_size=20, generations=1)
        assert result.final_merit is not None
        assert len(result.best_values) == 2

    def test_genetic_mutation_rate_parameter(self, optimizer):
        """GA should accept mutation_rate parameter without error."""
        result = optimizer.optimize_genetic(
            generations=1,
            population_size=5,
            mutation_rate=0.3,
        )
        assert result.final_merit is not None

    def test_genetic_crossover_rate_parameter(self, optimizer):
        """GA should accept crossover_rate parameter without error."""
        result = optimizer.optimize_genetic(
            generations=1,
            population_size=5,
            crossover_rate=0.5,
        )
        assert result.final_merit is not None

    def test_genetic_best_design_respects_bounds(self, optimizer):
        """Final best design should be within variable bounds."""
        result = optimizer.optimize_genetic(population_size=10, generations=2)
        for i, var in enumerate(optimizer.variables):
            assert var.min_value <= result.best_values[i] <= var.max_value

    def test_genetic_merit_history_tracked(self, optimizer):
        """GA should record merit history throughout optimization."""
        result = optimizer.optimize_genetic(population_size=8, generations=4)
        assert hasattr(result, 'merit_history')
        assert result.merit_history is not None
        assert len(result.merit_history) > 0


class TestOptimizeSimulatedAnnealing:
    """Test suite for optimize_simulated_annealing method."""

    def test_sa_returns_valid_result(self, optimizer):
        """SA should return OptimizationResult with required fields."""
        result = optimizer.optimize_simulated_annealing(max_iterations=5)
        assert isinstance(result, OptimizationResult)
        assert result.best_values is not None
        assert result.final_merit is not None
        assert result.iterations >= 0

    def test_sa_zero_iterations_valid(self, optimizer):
        """SA with zero max_iterations should return valid result."""
        result = optimizer.optimize_simulated_annealing(max_iterations=0)
        assert isinstance(result, OptimizationResult)
        assert result.best_values is not None

    def test_sa_no_regression(self, optimizer):
        """SA should not produce worse result than starting design."""
        result = optimizer.optimize_simulated_annealing(max_iterations=10)
        is_improvement = result.final_merit <= result.initial_merit
        is_nan = np.isnan(result.final_merit)
        assert is_improvement or is_nan

    def test_sa_callback_invoked(self, optimizer):
        """Callback should be invoked during optimization."""
        calls = []
        def callback(iteration, merit, design):
            calls.append((iteration, merit, len(design)))
        
        optimizer.optimize_simulated_annealing(
            max_iterations=5,
            callback=callback,
        )
        assert len(calls) > 0

    def test_sa_zero_temperature_clamped(self, optimizer):
        """Zero initial_temperature should be clamped without error."""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=2,
            initial_temperature=0.0,
        )
        assert result.final_merit is not None

    def test_sa_negative_temperature_clamped(self, optimizer):
        """Negative initial_temperature should be clamped without error."""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=2,
            initial_temperature=-100.0,
        )
        assert result.final_merit is not None

    def test_sa_custom_cooling_rate(self, optimizer):
        """SA should accept custom cooling_rate parameter."""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=10,
            initial_temperature=50.0,
            cooling_rate=0.95,
        )
        assert result.final_merit is not None

    def test_sa_derived_cooling_rate(self, optimizer):
        """SA should derive cooling_rate from max_iterations when None."""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=20,
            initial_temperature=100.0,
            cooling_rate=None,
        )
        assert result.final_merit is not None

    def test_sa_best_design_respects_bounds(self, optimizer):
        """Final best design should be within variable bounds."""
        result = optimizer.optimize_simulated_annealing(max_iterations=10)
        for i, var in enumerate(optimizer.variables):
            assert var.min_value <= result.best_values[i] <= var.max_value

    def test_sa_merit_history_tracked(self, optimizer):
        """SA should record merit history throughout optimization."""
        result = optimizer.optimize_simulated_annealing(max_iterations=10)
        assert hasattr(result, 'merit_history')
        assert result.merit_history is not None
        assert len(result.merit_history) > 0

    def test_sa_variable_history_tracked(self, optimizer):
        """SA should record variable history throughout optimization."""
        result = optimizer.optimize_simulated_annealing(max_iterations=10)
        assert hasattr(result, 'variable_history')
        assert result.variable_history is not None
        assert len(result.variable_history) > 0

    def test_sa_restart_threshold_api_compatibility(self, optimizer):
        """restart_threshold parameter should be accepted for API compatibility."""
        # Should not raise error or warning, parameter is ignored per docstring
        result = optimizer.optimize_simulated_annealing(
            max_iterations=5,
            restart_threshold=0.0001,
        )
        assert result.final_merit is not None

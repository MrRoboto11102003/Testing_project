"""
Pytest tests for openlens.global_optimizer.GlobalOptimizer.

Tests cover the public API and edge cases mentioned in docstrings.
Requires: pytest, numpy, pandas, and openlens with its dependencies.
"""
import pytest
import math
from unittest.mock import Mock, patch
import random
from typing import List

from openlens.global_optimizer import GlobalOptimizer
from openlens.optimizer import OptimizationResult


class MockOpticalSystem:
    """Mock OpticalSystem for testing."""
    pass


class MockOptimizationVariable:
    """Mock OptimizationVariable for testing."""
    def __init__(self, name, min_val=0.0, max_val=10.0, current=5.0, step=0.1):
        self.name = name
        self.min_value = min_val
        self.max_value = max_val
        self.current_value = current
        self.step_size = step

    def clamp(self, value):
        return max(self.min_value, min(self.max_value, value))


class MockOptimizationTarget:
    """Mock OptimizationTarget for testing."""
    def __init__(self, name, target_value):
        self.name = name
        self.target_value = target_value


@pytest.fixture
def optimizer_base():
    """Create a base GlobalOptimizer with mock components."""
    system = MockOpticalSystem()
    variables = [
        MockOptimizationVariable("r1", 10.0, 100.0, 50.0, 1.0),
        MockOptimizationVariable("r2", 10.0, 100.0, 50.0, 1.0),
        MockOptimizationVariable("thick", 1.0, 10.0, 5.0, 0.1),
    ]
    targets = [MockOptimizationTarget("focal_length", 50.0)]

    opt = GlobalOptimizer(
        system=system,
        variables=variables,
        targets=targets,
        constraints=None,
        seed=42,
    )

    # Mock parent class methods
    opt._evaluate_design = Mock(return_value=1.5)
    opt.optimize_simplex = Mock(
        return_value=Mock(
            best_values=[50.0, 50.0, 5.0],
            final_merit=1.0,
            iterations=10,
            variable_history=[],
            merit_history=[],
        )
    )
    opt._finalize = Mock(
        side_effect=lambda **kwargs: Mock(
            best_values=kwargs.get("best_values"),
            final_merit=kwargs.get("final_merit"),
            iterations=kwargs.get("iterations", 0),
            success=True,
        )
    )

    return opt


class TestGlobalOptimizerInitialization:
    """Test initialization and RNG setup."""

    def test_init_creates_private_rng(self):
        """Test that __init__ creates a private Random instance."""
        system = MockOpticalSystem()
        variables = [MockOptimizationVariable("r", 0, 10)]
        targets = [MockOptimizationTarget("f", 50)]

        opt = GlobalOptimizer(system, variables, targets, seed=42)

        assert hasattr(opt, "_rng")
        assert isinstance(opt._rng, random.Random)

    def test_rng_reproducibility_with_seed(self):
        """Test that same seed produces reproducible random sequences."""
        system = MockOpticalSystem()
        variables = [MockOptimizationVariable("r", 0, 10)]
        targets = [MockOptimizationTarget("f", 50)]

        opt1 = GlobalOptimizer(system, variables, targets, seed=12345)
        opt2 = GlobalOptimizer(system, variables, targets, seed=12345)

        seq1 = [opt1._rng.random() for _ in range(10)]
        seq2 = [opt2._rng.random() for _ in range(10)]

        assert seq1 == seq2

    def test_different_seeds_produce_different_sequences(self):
        """Test that different seeds produce different sequences."""
        system = MockOpticalSystem()
        variables = [MockOptimizationVariable("r", 0, 10)]
        targets = [MockOptimizationTarget("f", 50)]

        opt1 = GlobalOptimizer(system, variables, targets, seed=111)
        opt2 = GlobalOptimizer(system, variables, targets, seed=222)

        seq1 = [opt1._rng.random() for _ in range(10)]
        seq2 = [opt2._rng.random() for _ in range(10)]

        assert seq1 != seq2


class TestDeriveCoolingRate:
    """Test the _derive_cooling_rate classmethod."""

    def test_cooling_rate_reaches_floor_at_max_iterations(self):
        """Verify cooling schedule reaches floor at exactly max_iterations."""
        initial_temp = 100.0
        max_iters = 1000
        floor = GlobalOptimizer.ANNEALING_FLOOR

        rate = GlobalOptimizer._derive_cooling_rate(initial_temp, max_iters)
        final_temp = initial_temp * (rate ** max_iters)

        assert math.isclose(final_temp, floor, rel_tol=1e-5)

    def test_cooling_rate_zero_iterations(self):
        """Test that max_iterations=0 returns no cooling (rate=1.0)."""
        rate = GlobalOptimizer._derive_cooling_rate(100.0, 0)
        assert rate == 1.0

    def test_cooling_rate_negative_iterations(self):
        """Test that negative max_iterations returns no cooling."""
        rate = GlobalOptimizer._derive_cooling_rate(100.0, -5)
        assert rate == 1.0

    def test_cooling_rate_non_positive_initial_temperature(self):
        """Test that non-positive initial_temperature returns no cooling."""
        rate_zero = GlobalOptimizer._derive_cooling_rate(0.0, 500)
        rate_neg = GlobalOptimizer._derive_cooling_rate(-50.0, 500)

        assert rate_zero == 1.0
        assert rate_neg == 1.0

    def test_cooling_rate_initial_at_floor(self):
        """Test when initial temperature is already at or below floor."""
        floor = GlobalOptimizer.ANNEALING_FLOOR
        rate = GlobalOptimizer._derive_cooling_rate(floor, 1000)

        assert rate == 1.0

    def test_cooling_rate_valid_range(self):
        """Test that cooling rate is strictly between 0 and 1 for valid inputs."""
        rate = GlobalOptimizer._derive_cooling_rate(100.0, 500)
        assert 0 < rate < 1


class TestSimulatedAnnealing:
    """Test optimize_simulated_annealing method."""

    def test_simulated_annealing_basic_execution(self, optimizer_base):
        """Test basic simulated annealing run completes."""
        result = optimizer_base.optimize_simulated_annealing(
            max_iterations=10,
            initial_temperature=100.0,
        )

        assert result is not None
        optimizer_base.optimize_simplex.assert_called_once()
        optimizer_base._finalize.assert_called_once()

    def test_simulated_annealing_custom_cooling_rate(self, optimizer_base):
        """Test SA with explicitly provided cooling rate."""
        result = optimizer_base.optimize_simulated_annealing(
            max_iterations=10,
            initial_temperature=100.0,
            cooling_rate=0.95,
        )

        assert result is not None

    def test_simulated_annealing_non_positive_temperature_clamped(self, optimizer_base):
        """Test that non-positive initial_temperature is clamped to positive."""
        # Should not raise an exception
        result = optimizer_base.optimize_simulated_annealing(
            max_iterations=5,
            initial_temperature=-100.0,
        )

        assert result is not None

    def test_simulated_annealing_zero_max_iterations(self, optimizer_base):
        """Test SA with zero iterations (should run local refinement only)."""
        result = optimizer_base.optimize_simulated_annealing(
            max_iterations=0,
            initial_temperature=100.0,
        )

        # Local refinement should still execute
        optimizer_base.optimize_simplex.assert_called_once()

    def test_simulated_annealing_callback_invoked(self, optimizer_base):
        """Test that callback is called during SA execution."""
        callback = Mock()
        result = optimizer_base.optimize_simulated_annealing(
            max_iterations=5,
            initial_temperature=100.0,
            callback=callback,
        )

        # Callback should be invoked at least once per iteration
        assert callback.call_count >= 1

    def test_simulated_annealing_starting_merit_captured(self, optimizer_base):
        """Test that starting merit is captured before annealing starts."""
        starting_merit_value = 3.5
        optimizer_base._evaluate_design.return_value = starting_merit_value

        result = optimizer_base.optimize_simulated_annealing(
            max_iterations=3,
            initial_temperature=100.0,
        )

        # Verify _finalize received correct starting merit
        call_kwargs = optimizer_base._finalize.call_args[1]
        assert call_kwargs["initial_merit"] == starting_merit_value


class TestGeneticAlgorithm:
    """Test optimize_genetic method."""

    def test_genetic_algorithm_basic_execution(self, optimizer_base):
        """Test basic genetic algorithm run completes."""
        result = optimizer_base.optimize_genetic(
            population_size=10,
            generations=3,
        )

        assert result is not None
        optimizer_base.optimize_simplex.assert_called_once()

    def test_genetic_algorithm_zero_generations(self, optimizer_base):
        """Test GA with zero generations (local refinement only)."""
        result = optimizer_base.optimize_genetic(
            population_size=10,
            generations=0,
        )

        optimizer_base.optimize_simplex.assert_called_once()

    def test_genetic_algorithm_handles_nan_merits(self, optimizer_base):
        """Test that GA correctly handles NaN merit values."""
        # Mix of NaN and valid merits
        merits = [float("nan"), 1.5, 1.4, float("nan"), 1.3] + [1.5] * 100
        optimizer_base._evaluate_design.side_effect = merits

        result = optimizer_base.optimize_genetic(
            population_size=5,
            generations=2,
        )

        assert result is not None

    def test_genetic_algorithm_callback_invoked(self, optimizer_base):
        """Test that callback is called during GA execution."""
        callback = Mock()
        result = optimizer_base.optimize_genetic(
            population_size=5,
            generations=2,
            callback=callback,
        )

        assert callback.call_count >= 1

    def test_genetic_algorithm_crossover_produces_valid_offspring(self, optimizer_base):
        """Test that crossover produces offspring between parents."""
        parent1 = [10.0, 20.0, 30.0]
        parent2 = [40.0, 50.0, 60.0]

        child = optimizer_base._crossover(parent1, parent2)

        assert len(child) == 3
        for i, (p1, p2) in enumerate(zip(parent1, parent2)):
            # Arithmetic crossover produces values between parents
            assert min(p1, p2) <= child[i] <= max(p1, p2)

    def test_genetic_algorithm_mutation_respects_bounds(self, optimizer_base):
        """Test that mutation respects variable bounds."""
        individual = [50.0, 50.0, 5.0]

        # Apply mutation with high rate multiple times
        for _ in range(100):
            optimizer_base._mutate(individual, mutation_rate=0.5)

            # Verify all values stay within bounds
            assert 10.0 <= individual[0] <= 100.0
            assert 10.0 <= individual[1] <= 100.0
            assert 1.0 <= individual[2] <= 10.0

    def test_genetic_algorithm_tournament_select(self, optimizer_base):
        """Test that tournament selection picks from population."""
        population = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        merits = [2.0, 1.0, 3.0]  # Second is best

        selected = optimizer_base._tournament_select(population, merits, k=2)

        assert selected in population

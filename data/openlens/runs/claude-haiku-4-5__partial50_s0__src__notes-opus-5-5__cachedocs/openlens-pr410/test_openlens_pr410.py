import pytest
import math
import random
import logging
from unittest.mock import Mock
from types import SimpleNamespace

from openlens.global_optimizer import GlobalOptimizer
from openlens.optical_system import OpticalSystem
from openlens.lens import Lens
from openlens.optimizer import OptimizationVariable, OptimizationTarget


@pytest.fixture
def simple_system():
    """Create a minimal optical system for testing."""
    system = OpticalSystem()
    lens = Lens(
        name="test_lens",
        radius_of_curvature_1=50.0,
        radius_of_curvature_2=-50.0,
        thickness=5.0,
        diameter=25.0,
        material="BK7",
    )
    system.add_lens(lens)
    return system


@pytest.fixture
def simple_variable():
    """Create a single optimization variable."""
    return OptimizationVariable(
        name="r1",
        index=0,
        parameter="radius_of_curvature_1",
        current_value=0.0,
        min_value=-10.0,
        max_value=10.0,
        step_size=1.0,
    )


@pytest.fixture
def simple_target():
    """Create a single optimization target."""
    return OptimizationTarget(
        name="f",
        parameter="focal_length",
        target_value=50.0,
        weight=1.0,
    )


def create_test_optimizer(system, variables, targets, seed=None):
    """Factory to create an optimizer with stubbed evaluation methods."""
    opt = GlobalOptimizer(system, variables, targets, seed=seed)
    
    # Stub _evaluate_design: quadratic around 3, so (val - 3)^2
    opt._evaluate_design = lambda vals: (vals[0] - 3.0) ** 2 if vals else 0.0
    
    # Stub optimize_simplex to return a minimal result
    def simplex_stub(**kwargs):
        return SimpleNamespace(
            best_values=list(kwargs.get("start", [0.0])),
            final_merit=0.0,
            iterations=0,
            variable_history=[],
            merit_history=[],
        )
    opt.optimize_simplex = simplex_stub
    
    # Stub _finalize to return kwargs dict for inspection
    opt._finalize = lambda **kw: kw
    
    return opt


class TestGlobalOptimizerInit:
    def test_init_creates_private_rng(self, simple_system, simple_variable, simple_target):
        """Test that __init__ creates a private RNG instance."""
        opt = GlobalOptimizer(simple_system, [simple_variable], [simple_target], seed=42)
        assert opt._rng is not None
        assert isinstance(opt._rng, random.Random)

    def test_init_without_seed(self, simple_system, simple_variable, simple_target):
        """Test initialization without explicit seed."""
        opt = GlobalOptimizer(simple_system, [simple_variable], [simple_target])
        assert isinstance(opt._rng, random.Random)


class TestDeriveCoolingRate:
    def test_cooling_rate_basic_calculation(self):
        """Test cooling rate calculation with typical inputs."""
        rate = GlobalOptimizer._derive_cooling_rate(100.0, 1000)
        assert 0 < rate < 1
        # Verify it reaches floor in max_iterations
        expected = (1e-6 / 100.0) ** (1.0 / 1000)
        assert rate == pytest.approx(expected)

    def test_cooling_rate_zero_or_negative_iterations(self):
        """Test that non-positive max_iterations returns 1.0 (no cooling)."""
        assert GlobalOptimizer._derive_cooling_rate(100.0, 0) == 1.0
        assert GlobalOptimizer._derive_cooling_rate(100.0, -1) == 1.0

    def test_cooling_rate_negative_temperature(self):
        """Test that negative initial temperature returns 1.0."""
        assert GlobalOptimizer._derive_cooling_rate(-50.0, 100) == 1.0

    def test_cooling_rate_temperature_at_floor(self):
        """Test that temperature at or below floor returns 1.0."""
        assert GlobalOptimizer._derive_cooling_rate(1e-7, 100) == 1.0
        assert GlobalOptimizer._derive_cooling_rate(1e-6, 100) == 1.0


class TestOptimizeSimulatedAnnealing:
    def test_sa_basic_run_completes(self, simple_system, simple_variable, simple_target):
        """Test that simulated annealing completes and returns result dict."""
        opt = create_test_optimizer(simple_system, [simple_variable], [simple_target])
        result = opt.optimize_simulated_annealing(max_iterations=5, initial_temperature=100.0)
        
        assert result is not None
        assert "best_values" in result
        assert "iterations" in result
        assert "merit_history" in result
        assert result["iterations"] == 5

    def test_sa_zero_iterations(self, simple_system, simple_variable, simple_target):
        """Test SA with max_iterations=0."""
        opt = create_test_optimizer(simple_system, [simple_variable], [simple_target])
        result = opt.optimize_simulated_annealing(max_iterations=0, initial_temperature=100.0)
        
        assert result["iterations"] == 0
        assert "SA Converged after 0 iterations" in result["converged_message"]

    def test_sa_merit_history_starts_with_initial_merit(self, simple_system, simple_variable, simple_target):
        """Test that merit_history[0] is the starting design's merit."""
        opt = create_test_optimizer(simple_system, [simple_variable], [simple_target])
        # Variable starts at 0, so merit = (0 - 3)^2 = 9
        result = opt.optimize_simulated_annealing(max_iterations=3, initial_temperature=100.0)
        
        assert result["merit_history"][0] == pytest.approx(9.0)
        assert result["initial_merit"] == pytest.approx(9.0)

    def test_sa_negative_temperature_is_clamped(self, simple_system, simple_variable, simple_target, caplog):
        """Test that negative initial_temperature is clamped to 1e-6 and logged."""
        opt = create_test_optimizer(simple_system, [simple_variable], [simple_target])
        
        with caplog.at_level(logging.WARNING):
            result = opt.optimize_simulated_annealing(
                max_iterations=2, initial_temperature=-100.0
            )
        
        # Should complete successfully despite negative temperature
        assert result["iterations"] >= 0
        # Warning should be logged
        assert any("not positive" in record.message for record in caplog.records)

    def test_sa_variables_not_mutated_after_run(self, simple_system, simple_variable, simple_target):
        """Test that optimization does not mutate the original variable objects."""
        original_value = simple_variable.current_value
        opt = create_test_optimizer(simple_system, [simple_variable], [simple_target])
        opt.optimize_simulated_annealing(max_iterations=5, initial_temperature=100.0)
        
        assert simple_variable.current_value == original_value

    def test_sa_callback_propagates_exceptions(self, simple_system, simple_variable, simple_target):
        """Test that exceptions from callback propagate."""
        opt = create_test_optimizer(simple_system, [simple_variable], [simple_target])
        
        def failing_callback(iteration, merit, values):
            raise ValueError("Callback error")
        
        with pytest.raises(ValueError, match="Callback error"):
            opt.optimize_simulated_annealing(
                max_iterations=2,
                initial_temperature=100.0,
                callback=failing_callback,
            )

    def test_sa_empty_variables_raises_error(self, simple_system, simple_target):
        """Test that SA with empty variables list raises ValueError."""
        opt = create_test_optimizer(simple_system, [], [simple_target])
        
        with pytest.raises(ValueError):
            opt.optimize_simulated_annealing(max_iterations=1, initial_temperature=100.0)

    def test_sa_deterministic_with_seed(self, simple_system, simple_variable, simple_target):
        """Test that same seed produces deterministic results."""
        opt1 = create_test_optimizer(simple_system, [simple_variable], [simple_target], seed=12345)
        result1 = opt1.optimize_simulated_annealing(max_iterations=10, initial_temperature=100.0)
        
        opt2 = create_test_optimizer(simple_system, [simple_variable], [simple_target], seed=12345)
        result2 = opt2.optimize_simulated_annealing(max_iterations=10, initial_temperature=100.0)
        
        # Same seed should give same iteration count and structure
        assert result1["iterations"] == result2["iterations"]
        assert len(result1["merit_history"]) == len(result2["merit_history"])


class TestOptimizeGenetic:
    def test_ga_basic_run_completes(self, simple_system, simple_variable, simple_target):
        """Test that genetic algorithm completes and returns result dict."""
        opt = create_test_optimizer(simple_system, [simple_variable], [simple_target])
        result = opt.optimize_genetic(population_size=10, generations=5)
        
        assert result is not None
        assert "best_values" in result
        assert "iterations" in result
        assert result["iterations"] == 5

    def test_ga_merit_history_is_non_increasing(self, simple_system, simple_variable, simple_target):
        """Test that GA merit_history is monotonically non-increasing."""
        opt = create_test_optimizer(simple_system, [simple_variable], [simple_target])
        result = opt.optimize_genetic(population_size=10, generations=10)
        
        history = result["merit_history"]
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1] or math.isnan(history[i])

    def test_ga_zero_generations(self, simple_system, simple_variable, simple_target):
        """Test GA with zero generations."""
        opt = create_test_optimizer(simple_system, [simple_variable], [simple_target])
        result = opt.optimize_genetic(population_size=10, generations=0)
        
        assert result["iterations"] == 0
        assert "GA Completed 0 generations" in result["converged_message"]

    def test_ga_variables_not_mutated_after_run(self, simple_system, simple_variable, simple_target):
        """Test that optimization does not mutate the original variable objects."""
        original_value = simple_variable.current_value
        opt = create_test_optimizer(simple_system, [simple_variable], [simple_target])
        opt.optimize_genetic(population_size=5, generations=5)
        
        assert simple_variable.current_value == original_value

    def test_ga_zero_population_raises_error(self, simple_system, simple_variable, simple_target):
        """Test that GA with population_size=0 raises IndexError."""
        opt = create_test_optimizer(simple_system, [simple_variable], [simple_target])
        
        with pytest.raises(IndexError):
            opt.optimize_genetic(population_size=0, generations=1)

    def test_ga_deterministic_with_seed(self, simple_system, simple_variable, simple_target):
        """Test that same seed produces deterministic merit histories."""
        opt1 = create_test_optimizer(simple_system, [simple_variable], [simple_target], seed=54321)
        result1 = opt1.optimize_genetic(population_size=5, generations=5)
        
        opt2 = create_test_optimizer(simple_system, [simple_variable], [simple_target], seed=54321)
        result2 = opt2.optimize_genetic(population_size=5, generations=5)
        
        # Same seed should give identical merit histories
        assert len(result1["merit_history"]) == len(result2["merit_history"])
        for m1, m2 in zip(result1["merit_history"], result2["merit_history"]):
            assert m1 == pytest.approx(m2) or (math.isnan(m1) and math.isnan(m2))


class TestTournamentSelection:
    def test_tournament_select_returns_best_candidate(self, simple_system, simple_variable, simple_target):
        """Test that tournament selection returns the best of k random candidates."""
        opt = GlobalOptimizer(simple_system, [simple_variable], [simple_target], seed=99)
        population = [[1.0], [5.0], [3.0], [2.0]]
        merits = [4.0, 10.0, 6.0, 1.0]  # Last one is best
        
        selected = opt._tournament_select(population, merits)
        assert selected == [2.0]

    def test_tournament_select_empty_population_raises(self, simple_system, simple_variable, simple_target):
        """Test that tournament selection on empty population raises ValueError."""
        opt = GlobalOptimizer(simple_system, [simple_variable], [simple_target])
        
        with pytest.raises(ValueError, match="cannot select from an empty population"):
            opt._tournament_select([], [])

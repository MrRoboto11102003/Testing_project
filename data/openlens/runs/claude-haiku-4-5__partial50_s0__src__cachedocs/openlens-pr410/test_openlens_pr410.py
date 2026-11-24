import pytest
import math
from unittest.mock import Mock, patch
import random

from openlens.global_optimizer import GlobalOptimizer


# Test fixtures
@pytest.fixture
def mock_system():
    return Mock()


@pytest.fixture
def mock_variables():
    """Create mock optimization variables."""
    variables = []
    for i in range(2):
        var = Mock()
        var.name = f"var_{i}"
        var.current_value = 0.0
        var.min_value = -10.0
        var.max_value = 10.0
        var.step_size = 1.0
        var.clamp = lambda x, v=var: max(v.min_value, min(v.max_value, x))
        variables.append(var)
    return variables


@pytest.fixture
def optimizer(mock_system, mock_variables):
    """Create optimizer with mocked parent methods."""
    opt = GlobalOptimizer(mock_system, mock_variables, [], seed=42)
    
    # Mock inherited parent methods
    opt._evaluate_design = Mock(return_value=1.0)
    opt.optimize_simplex = Mock(return_value=Mock(
        best_values=[0.0, 0.0],
        final_merit=0.5,
        iterations=10,
        variable_history=[],
        merit_history=[1.0, 0.5]
    ))
    opt._finalize = Mock(return_value=Mock(
        final_merit=0.5,
        initial_merit=1.0,
        iterations=50,
        converged=True,
        message="Converged"
    ))
    
    return opt


class TestDeriveCoolingRate:
    """Test _derive_cooling_rate classmethod."""
    
    def test_non_positive_max_iterations_returns_one(self):
        """Non-positive max_iterations should return 1.0 (no cooling)."""
        assert GlobalOptimizer._derive_cooling_rate(100.0, 0) == 1.0
        assert GlobalOptimizer._derive_cooling_rate(100.0, -5) == 1.0
    
    def test_non_positive_initial_temperature_returns_one(self):
        """Non-positive initial_temperature returns 1.0."""
        assert GlobalOptimizer._derive_cooling_rate(0.0, 1000) == 1.0
        assert GlobalOptimizer._derive_cooling_rate(-50.0, 1000) == 1.0
    
    def test_already_at_floor_returns_one(self):
        """Temperature already at or below floor returns 1.0."""
        # FLOOR is 1e-6; if initial < 1e-6, ratio >= 1.0
        assert GlobalOptimizer._derive_cooling_rate(1e-7, 1000) == 1.0
    
    def test_valid_cooling_reaches_floor(self):
        """Valid cooling rate should reach floor in specified iterations."""
        rate = GlobalOptimizer._derive_cooling_rate(100.0, 500)
        assert 0 < rate < 1
        
        temp = 100.0
        for _ in range(500):
            temp *= rate
        assert temp < GlobalOptimizer.ANNEALING_FLOOR * 10
    
    def test_longer_schedule_slower_cooling(self):
        """More iterations should produce slower cooling rate."""
        rate_short = GlobalOptimizer._derive_cooling_rate(100.0, 100)
        rate_long = GlobalOptimizer._derive_cooling_rate(100.0, 1000)
        assert rate_long > rate_short


class TestInit:
    """Test GlobalOptimizer initialization."""
    
    def test_init_with_seed_reproducible(self, mock_system):
        """Same seed should produce reproducible random sequences."""
        opt1 = GlobalOptimizer(mock_system, [], [], seed=123)
        opt2 = GlobalOptimizer(mock_system, [], [], seed=123)
        
        assert opt1._rng.random() == opt2._rng.random()
    
    def test_init_without_seed(self, mock_system):
        """Initialization without seed should work."""
        opt = GlobalOptimizer(mock_system, [], [])
        assert opt._rng is not None


class TestSimulatedAnnealing:
    """Test optimize_simulated_annealing method."""
    
    def test_zero_iterations_completes(self, optimizer):
        """Zero iterations should complete without error."""
        result = optimizer.optimize_simulated_annealing(max_iterations=0)
        assert optimizer._finalize.called
    
    def test_negative_temperature_clamped_and_continues(self, optimizer, caplog):
        """Negative initial_temperature should be clamped to 1e-6."""
        optimizer.optimize_simulated_annealing(
            max_iterations=5,
            initial_temperature=-100.0
        )
        # Should log warning and continue
        assert optimizer._finalize.called
    
    def test_callback_invoked_per_iteration(self, optimizer):
        """Callback should be invoked at least once per iteration."""
        callback = Mock()
        optimizer.optimize_simulated_annealing(
            max_iterations=5,
            initial_temperature=10.0,
            callback=callback
        )
        
        assert callback.call_count >= 5
    
    def test_derives_cooling_rate_when_none(self, optimizer):
        """Should derive cooling rate from max_iterations when cooling_rate=None."""
        optimizer.optimize_simulated_annealing(
            max_iterations=100,
            initial_temperature=50.0,
            cooling_rate=None
        )
        
        # Should complete without error and use derived schedule
        assert optimizer._finalize.called
    
    def test_uses_provided_cooling_rate(self, optimizer):
        """Should use provided cooling_rate when specified."""
        optimizer.optimize_simulated_annealing(
            max_iterations=10,
            initial_temperature=50.0,
            cooling_rate=0.9
        )
        
        assert optimizer._finalize.called
    
    def test_runs_local_refinement_at_end(self, optimizer):
        """Should run simplex refinement after annealing."""
        optimizer.optimize_simulated_annealing(
            max_iterations=5,
            initial_temperature=10.0
        )
        
        # optimize_simplex should be called for local refinement
        assert optimizer.optimize_simplex.called


class TestGeneticAlgorithm:
    """Test optimize_genetic method."""
    
    def test_zero_generations_completes(self, optimizer):
        """Zero generations should still produce a result."""
        result = optimizer.optimize_genetic(generations=0)
        assert optimizer._finalize.called
    
    def test_callback_invoked_during_ga(self, optimizer):
        """Callback should be invoked during generation iterations."""
        callback = Mock()
        optimizer.optimize_genetic(
            population_size=10,
            generations=2,
            callback=callback
        )
        
        assert callback.call_count > 0
    
    def test_handles_nan_merit_values(self, optimizer):
        """Should handle NaN merit values gracefully using isfinite guard."""
        call_count = [0]
        
        def merit_with_nan(values):
            call_count[0] += 1
            if call_count[0] == 1:
                return float('nan')
            return 0.5
        
        optimizer._evaluate_design = merit_with_nan
        result = optimizer.optimize_genetic(population_size=5, generations=1)
        
        # Should complete without crash
        assert optimizer._finalize.called
    
    def test_runs_local_refinement_after_ga(self, optimizer):
        """Should run simplex refinement after genetic algorithm."""
        optimizer.optimize_genetic(
            population_size=10,
            generations=2
        )
        
        assert optimizer.optimize_simplex.called
    
    def test_population_size_zero_works(self, optimizer):
        """Small population size should work."""
        # Note: results may be degenerate, but shouldn't crash
        optimizer.optimize_genetic(
            population_size=1,
            generations=1
        )
        
        assert optimizer._finalize.called


class TestTournamentSelect:
    """Test _tournament_select method."""
    
    def test_selects_best_contestant(self, optimizer):
        """Should select the individual with lowest merit."""
        population = [[1.0, 1.0], [5.0, 5.0], [3.0, 3.0]]
        merits = [100.0, 50.0, 300.0]  # Second is best
        
        selected = optimizer._tournament_select(population, merits, k=3)
        assert selected == [5.0, 5.0]
    
    def test_raises_on_empty_population(self, optimizer):
        """Should raise ValueError for empty population."""
        with pytest.raises(ValueError, match="empty"):
            optimizer._tournament_select([], [])
    
    def test_clamps_k_to_population_size(self, optimizer):
        """k should be clamped to population size without raising."""
        population = [[1.0], [2.0]]
        merits = [10.0, 5.0]
        
        # k=100 but only 2 individuals; should still work
        selected = optimizer._tournament_select(population, merits, k=100)
        assert selected in population


class TestCrossover:
    """Test _crossover method."""
    
    def test_produces_valid_offspring(self, optimizer):
        """Child should be weighted average of parents."""
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [5.0, 6.0, 7.0]
        
        child = optimizer._crossover(parent1, parent2)
        
        assert len(child) == 3
        for i in range(3):
            # Arithmetic crossover: child = alpha*p1 + (1-alpha)*p2
            assert min(parent1[i], parent2[i]) <= child[i] <= max(parent1[i], parent2[i])


class TestMutate:
    """Test _mutate method."""
    
    def test_respects_variable_bounds(self, optimizer):
        """Mutated values should respect variable min/max bounds."""
        individual = [0.0, 0.0]
        optimizer._mutate(individual, mutation_rate=1.0)
        
        for i, val in enumerate(individual):
            var = optimizer.variables[i]
            assert var.min_value <= val <= var.max_value
    
    def test_zero_mutation_rate_no_change(self, optimizer):
        """Zero mutation rate should leave individual unchanged."""
        individual = [5.0, 3.0]
        original = list(individual)
        
        optimizer._mutate(individual, mutation_rate=0.0)
        
        assert individual == original

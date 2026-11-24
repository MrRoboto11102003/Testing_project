import pytest
import math
import random
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Optional

import openlens
from openlens.global_optimizer import GlobalOptimizer
from openlens.lens_optimizer import OptimizationVariable, OptimizationTarget


class TestDeriveCoolingRate:
    """Test GlobalOptimizer._derive_cooling_rate class method."""

    def test_non_positive_max_iterations_returns_one(self):
        """Non-positive max_iterations should return 1.0 (no cooling)."""
        assert GlobalOptimizer._derive_cooling_rate(100.0, 0) == 1.0
        assert GlobalOptimizer._derive_cooling_rate(100.0, -5) == 1.0

    def test_non_positive_initial_temperature_returns_one(self):
        """Non-positive initial_temperature should return 1.0 (no cooling)."""
        assert GlobalOptimizer._derive_cooling_rate(0.0, 100) == 1.0
        assert GlobalOptimizer._derive_cooling_rate(-50.0, 100) == 1.0

    def test_already_at_or_below_floor_returns_one(self):
        """When ratio >= 1.0 (already at floor), return 1.0."""
        # ANNEALING_FLOOR = 1e-6, so if initial_temp = 1e-7, ratio > 1
        assert GlobalOptimizer._derive_cooling_rate(1e-7, 1000) == 1.0
        assert GlobalOptimizer._derive_cooling_rate(5e-7, 1000) == 1.0

    def test_valid_cooling_rate_in_valid_range(self):
        """Valid cooling rate should be between 0 and 1 (exclusive)."""
        rate = GlobalOptimizer._derive_cooling_rate(100.0, 1000)
        assert 0.0 < rate < 1.0

    def test_cooling_rate_reaches_floor_at_max_iterations(self):
        """Verify cooling schedule reaches ANNEALING_FLOOR near max_iterations."""
        initial_temp = 100.0
        max_iters = 500
        rate = GlobalOptimizer._derive_cooling_rate(initial_temp, max_iters)
        
        # Simulate cooling schedule
        temperature = initial_temp
        for _ in range(max_iters):
            temperature *= rate
        
        # Should reach floor within reasonable tolerance
        assert temperature <= GlobalOptimizer.ANNEALING_FLOOR * 10

    def test_cooling_rate_formula_correctness(self):
        """Test the geometric cooling formula: T0 * rate^n = FLOOR."""
        initial_temp = 50.0
        max_iters = 100
        rate = GlobalOptimizer._derive_cooling_rate(initial_temp, max_iters)
        
        # Verify formula: initial_temp * rate^max_iters ≈ FLOOR
        final_temp = initial_temp * (rate ** max_iters)
        assert math.isclose(final_temp, GlobalOptimizer.ANNEALING_FLOOR, rel_tol=1e-5)


class TestGlobalOptimizerInit:
    """Test GlobalOptimizer initialization."""

    def _make_mock_system_and_vars(self):
        """Helper to create mock optical system and variables."""
        system = Mock()
        variables = [
            Mock(spec=OptimizationVariable, current_value=1.0, name="var1")
        ]
        targets = [Mock(spec=OptimizationTarget)]
        return system, variables, targets

    def test_init_with_seed_uses_private_rng(self):
        """Initialization with seed should create private Random generator."""
        system, variables, targets = self._make_mock_system_and_vars()
        opt = GlobalOptimizer(system, variables, targets, seed=42)
        
        assert hasattr(opt, '_rng')
        assert isinstance(opt._rng, random.Random)

    def test_init_without_seed_still_creates_rng(self):
        """Initialization without seed should still create Random generator."""
        system, variables, targets = self._make_mock_system_and_vars()
        opt = GlobalOptimizer(system, variables, targets)
        
        assert hasattr(opt, '_rng')
        assert isinstance(opt._rng, random.Random)

    def test_init_with_constraints(self):
        """Initialization should accept constraints dict."""
        system, variables, targets = self._make_mock_system_and_vars()
        constraints = {"constraint1": 10.0}
        opt = GlobalOptimizer(system, variables, targets, constraints=constraints)
        
        assert opt is not None


class TestSimulatedAnnealing:
    """Test optimize_simulated_annealing method."""

    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock GlobalOptimizer for testing."""
        system = Mock()
        var = Mock(spec=OptimizationVariable)
        var.current_value = 1.0
        var.step_size = 0.1
        var.min_value = 0.0
        var.max_value = 10.0
        var.name = "var1"
        var.clamp = lambda x: max(0.0, min(10.0, x))
        
        variables = [var]
        targets = [Mock(spec=OptimizationTarget)]
        
        opt = GlobalOptimizer(system, variables, targets, seed=42)
        
        # Mock the internal methods
        opt._evaluate_design = Mock(return_value=1.0)
        opt.optimize_simplex = Mock(return_value=Mock(
            best_values=[1.0],
            final_merit=0.9,
            iterations=5,
            variable_history=[],
            merit_history=[0.9]
        ))
        opt._finalize = Mock(return_value=Mock(
            success=True, best_values=[1.0], final_merit=0.9
        ))
        
        return opt

    def test_sa_zero_iterations_completes(self, mock_optimizer):
        """SA with max_iterations=0 should not enter loop but complete."""
        result = mock_optimizer.optimize_simulated_annealing(max_iterations=0)
        assert result is not None
        mock_optimizer.optimize_simplex.assert_called_once()

    def test_sa_clamps_negative_initial_temperature(self, mock_optimizer):
        """Negative initial_temperature should be clamped to 1e-6."""
        with patch('openlens.global_optimizer.logger'):
            mock_optimizer.optimize_simulated_annealing(
                max_iterations=5,
                initial_temperature=-10.0
            )
        # Should complete without error (temperature was clamped internally)
        mock_optimizer.optimize_simplex.assert_called_once()

    def test_sa_uses_derived_cooling_rate_when_none(self, mock_optimizer):
        """When cooling_rate=None, should derive from max_iterations."""
        mock_optimizer.optimize_simulated_annealing(
            max_iterations=100,
            initial_temperature=100.0,
            cooling_rate=None
        )
        mock_optimizer.optimize_simplex.assert_called_once()

    def test_sa_uses_custom_cooling_rate(self, mock_optimizer):
        """When cooling_rate provided, should use custom rate."""
        mock_optimizer.optimize_simulated_annealing(
            max_iterations=10,
            initial_temperature=100.0,
            cooling_rate=0.95
        )
        mock_optimizer.optimize_simplex.assert_called_once()

    def test_sa_callback_invoked_per_iteration(self, mock_optimizer):
        """Callback should be invoked during optimization iterations."""
        callback = Mock()
        mock_optimizer.optimize_simulated_annealing(
            max_iterations=5,
            callback=callback
        )
        # Callback called at least once per iteration before simplex refinement
        assert callback.call_count >= 1

    def test_sa_simplex_refinement_called(self, mock_optimizer):
        """Final local refinement with simplex should be called."""
        mock_optimizer.optimize_simulated_annealing(max_iterations=10)
        
        # Verify simplex was called with start parameter
        mock_optimizer.optimize_simplex.assert_called_once()
        call_kwargs = mock_optimizer.optimize_simplex.call_args[1]
        assert 'start' in call_kwargs
        assert 'max_iterations' in call_kwargs


class TestGeneticAlgorithm:
    """Test optimize_genetic method."""

    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock GlobalOptimizer for GA testing."""
        system = Mock()
        var = Mock(spec=OptimizationVariable)
        var.current_value = 1.0
        var.min_value = 0.0
        var.max_value = 10.0
        var.name = "var1"
        var.clamp = lambda x: max(0.0, min(10.0, x))
        
        variables = [var]
        targets = [Mock(spec=OptimizationTarget)]
        
        opt = GlobalOptimizer(system, variables, targets, seed=42)
        
        opt._evaluate_design = Mock(return_value=1.0)
        opt.optimize_simplex = Mock(return_value=Mock(
            best_values=[1.0],
            final_merit=0.9,
            iterations=5,
            variable_history=[],
            merit_history=[0.9]
        ))
        opt._finalize = Mock(return_value=Mock(
            success=True, best_values=[1.0], final_merit=0.9
        ))
        
        return opt

    def test_ga_zero_generations_completes(self, mock_optimizer):
        """GA with generations=0 should complete without error."""
        result = mock_optimizer.optimize_genetic(
            population_size=10,
            generations=0
        )
        assert result is not None
        mock_optimizer.optimize_simplex.assert_called_once()

    def test_ga_callback_invoked(self, mock_optimizer):
        """Callback should be invoked during GA optimization."""
        callback = Mock()
        mock_optimizer.optimize_genetic(
            population_size=5,
            generations=3,
            callback=callback
        )
        # Callback invoked during population evaluation
        assert callback.call_count > 0

    def test_ga_simplex_refinement_called(self, mock_optimizer):
        """Final local refinement with simplex should be called."""
        mock_optimizer.optimize_genetic(
            population_size=5,
            generations=2
        )
        
        mock_optimizer.optimize_simplex.assert_called_once()
        call_kwargs = mock_optimizer.optimize_simplex.call_args[1]
        assert 'start' in call_kwargs

    def test_ga_finite_merit_check_with_nan(self, mock_optimizer):
        """GA should use isfinite to skip NaN merits."""
        # Mock _evaluate_design to return both NaN and valid values
        mock_optimizer._evaluate_design = Mock(
            side_effect=[1.0] + [float('nan'), 0.8, 0.9, 1.1] * 10
        )
        
        result = mock_optimizer.optimize_genetic(
            population_size=5,
            generations=2
        )
        assert result is not None


class TestTournamentSelect:
    """Test _tournament_select helper method."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer for selection tests."""
        system = Mock()
        variables = [Mock(spec=OptimizationVariable, current_value=1.0)]
        targets = [Mock(spec=OptimizationTarget)]
        return GlobalOptimizer(system, variables, targets, seed=42)

    def test_tournament_select_returns_population_member(self, optimizer):
        """Tournament select should return one of the population members."""
        population = [[1.0], [2.0], [3.0]]
        merits = [3.0, 2.0, 1.0]
        
        selected = optimizer._tournament_select(population, merits, k=2)
        assert selected in population

    def test_tournament_select_empty_population_raises_error(self, optimizer):
        """Empty population should raise ValueError."""
        with pytest.raises(ValueError, match="cannot select from an empty population"):
            optimizer._tournament_select([], [], k=3)

    def test_tournament_select_k_clamped_to_population_size(self, optimizer):
        """k should be clamped to population size (k=100 → k=2)."""
        population = [[1.0], [2.0]]
        merits = [2.0, 1.0]
        
        # This should not raise "Sample larger than population"
        selected = optimizer._tournament_select(population, merits, k=100)
        assert selected in population

    def test_tournament_select_k_minimum_one(self, optimizer):
        """k should be clamped to minimum of 1."""
        population = [[1.0], [2.0], [3.0]]
        merits = [3.0, 2.0, 1.0]
        
        # k=0 should be treated as k=1
        selected = optimizer._tournament_select(population, merits, k=0)
        assert selected in population

    def test_tournament_select_selects_best_candidate(self, optimizer):
        """Tournament should select the best among k candidates."""
        population = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        merits = [5.0, 4.0, 3.0, 2.0, 1.0]  # Last is best
        
        # With high k and seeded RNG, should bias toward best
        results = []
        for _ in range(10):
            selected = optimizer._tournament_select(population, merits, k=5)
            results.append(selected)
        
        # At least some should select the best candidate
        best_candidate = population[-1]
        assert any(r == best_candidate for r in results)


class TestCrossover:
    """Test _crossover helper method."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer for crossover tests."""
        system = Mock()
        variables = [Mock(spec=OptimizationVariable)]
        targets = [Mock(spec=OptimizationTarget)]
        return GlobalOptimizer(system, variables, targets, seed=42)

    def test_crossover_produces_valid_child(self, optimizer):
        """Crossover should produce child with values between parents."""
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        child = optimizer._crossover(parent1, parent2)
        
        assert len(child) == 3
        for i in range(3):
            assert min(parent1[i], parent2[i]) <= child[i] <= max(parent1[i], parent2[i])

    def test_crossover_deterministic_with_seed(self):
        """Crossover with same seed should be reproducible."""
        system = Mock()
        variables = [Mock(spec=OptimizationVariable)]
        targets = [Mock(spec=OptimizationTarget)]
        
        opt1 = GlobalOptimizer(system, variables, targets, seed=99)
        opt2 = GlobalOptimizer(system, variables, targets, seed=99)
        
        parent1 = [1.0, 2.0, 3.0, 4.0]
        parent2 = [5.0, 6.0, 7.0, 8.0]
        
        child1 = opt1._crossover(parent1, parent2)
        child2 = opt2._crossover(parent1, parent2)
        
        assert child1 == child2


class TestMutate:
    """Test _mutate helper method."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer for mutation tests."""
        system = Mock()
        var = Mock(spec=OptimizationVariable)
        var.min_value = 0.0
        var.max_value = 10.0
        var.clamp = lambda x: max(0.0, min(10.0, x))
        
        variables = [var]
        targets = [Mock(spec=OptimizationTarget)]
        return GlobalOptimizer(system, variables, targets, seed=42)

    def test_mutate_zero_rate_no_change(self, optimizer):
        """With mutation_rate=0.0, individual should not change."""
        individual = [5.0]
        original = individual[0]
        
        for _ in range(100):
            optimizer._mutate(individual, 0.0)
        
        assert individual[0] == original

    def test_mutate_respects_bounds(self, optimizer):
        """Mutations should stay within variable bounds."""
        individual = [5.0]
        
        for _ in range(100):
            optimizer._mutate(individual, 1.0)
            assert 0.0 <= individual[0] <= 10.0

    def test_mutate_high_rate_produces_changes(self, optimizer):
        """With mutation_rate=1.0, changes should occur."""
        changes = 0
        for trial in range(20):
            individual = [5.0]
            original = individual[0]
            optimizer._mutate(individual, 1.0)
            if individual[0] != original:
                changes += 1
        
        # With high rate, should see mutations in most trials
        assert changes > 10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_annealing_floor_constant_defined(self):
        """ANNEALING_FLOOR should be defined as 1e-6."""
        assert GlobalOptimizer.ANNEALING_FLOOR == 1e-6

    def test_sa_low_initial_temperature_clamp_logged(self):
        """SA should log warning when initial_temperature is clamped."""
        system = Mock()
        variables = [
            Mock(spec=OptimizationVariable, current_value=1.0, 
                 step_size=0.1, min_value=0.0, max_value=10.0, 
                 name="v", clamp=lambda x: x)
        ]
        targets = [Mock(spec=OptimizationTarget)]
        
        opt = GlobalOptimizer(system, variables, targets)
        opt._evaluate_design = Mock(return_value=1.0)
        opt.optimize_simplex = Mock(return_value=Mock(
            best_values=[1.0], final_merit=1.0, iterations=0,
            variable_history=[], merit_history=[]
        ))
        opt._finalize = Mock(return_value=Mock(success=True))
        
        with patch('openlens.global_optimizer.logger') as mock_logger:
            opt.optimize_simulated_annealing(max_iterations=1, initial_temperature=-1.0)
            mock_logger.warning.assert_called()

    def test_iteration_count_unbound_error_prevented(self):
        """With max_iterations=0, iteration variable should be accessible."""
        system = Mock()
        variables = [
            Mock(spec=OptimizationVariable, current_value=1.0,
                 step_size=0.1, min_value=0.0, max_value=10.0,
                 name="v", clamp=lambda x: x)
        ]
        targets = [Mock(spec=OptimizationTarget)]
        
        opt = GlobalOptimizer(system, variables, targets)
        opt._evaluate_design = Mock(return_value=1.0)
        opt.optimize_simplex = Mock(return_value=Mock(
            best_values=[1.0], final_merit=1.0, iterations=0,
            variable_history=[], merit_history=[]
        ))
        opt._finalize = Mock(return_value=Mock(success=True))
        
        # This should not raise UnboundLocalError
        result = opt.optimize_simulated_annealing(max_iterations=0)
        assert result is not None

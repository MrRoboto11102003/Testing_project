import pytest
import random
import math
from unittest.mock import Mock, patch
from types import SimpleNamespace

from openlens import GlobalOptimizer, OpticalSystem, Lens, OptimizationVariable


@pytest.fixture
def optical_system():
    """Create a minimal optical system for testing."""
    system = OpticalSystem()
    system.add_lens(Lens())
    return system


@pytest.fixture
def optimization_variable():
    """Create a simple optimization variable."""
    return OptimizationVariable(
        name="test_var",
        index=0,
        param_name="radius_of_curvature_1",
        current_value=0.0,
        min_value=-10.0,
        max_value=10.0,
        step_size=1.0
    )


@pytest.fixture
def optimizer(optical_system, optimization_variable):
    """Create GlobalOptimizer with mocked parent methods."""
    target = Mock(name="test_target")
    opt = GlobalOptimizer(
        system=optical_system,
        variables=[optimization_variable],
        targets=[target],
        seed=42
    )
    
    # Mock inherited methods from LensOptimizer
    opt._evaluate_design = Mock(return_value=1.0)
    opt.optimize_simplex = Mock(return_value=SimpleNamespace(
        best_values=[1.0],
        final_merit=0.5,
        iterations=5,
        variable_history=[],
        merit_history=[]
    ))
    opt._finalize = Mock(side_effect=lambda **kw: SimpleNamespace(**kw))
    
    return opt


class TestDeriveCooolingRate:
    """Test _derive_cooling_rate classmethod."""
    
    def test_non_positive_max_iterations_returns_one(self):
        """max_iterations <= 0 returns 1.0 (no cooling)."""
        assert GlobalOptimizer._derive_cooling_rate(100.0, 0) == 1.0
        assert GlobalOptimizer._derive_cooling_rate(100.0, -5) == 1.0
    
    def test_non_positive_temperature_returns_one(self):
        """Non-positive initial_temperature returns 1.0."""
        assert GlobalOptimizer._derive_cooling_rate(0.0, 1000) == 1.0
        assert GlobalOptimizer._derive_cooling_rate(-5.0, 1000) == 1.0
    
    def test_temperature_at_floor_returns_one(self):
        """initial_temperature <= ANNEALING_FLOOR returns 1.0."""
        assert GlobalOptimizer._derive_cooling_rate(1e-6, 1000) == 1.0
        assert GlobalOptimizer._derive_cooling_rate(1e-7, 1000) == 1.0
    
    def test_valid_rate_converges_to_floor(self):
        """Valid cooling rate reaches floor at max_iterations."""
        rate = GlobalOptimizer._derive_cooling_rate(100.0, 1000)
        assert 0.98 < rate < 0.99
        
        # Verify convergence to floor
        T = 100.0
        for _ in range(1000):
            T *= rate
        assert T < 1e-6


class TestInit:
    """Test GlobalOptimizer initialization."""
    
    def test_init_creates_rng_with_seed(self, optical_system, optimization_variable):
        """__init__ sets up private RNG with given seed."""
        target = Mock()
        opt = GlobalOptimizer(
            system=optical_system,
            variables=[optimization_variable],
            targets=[target],
            seed=42
        )
        assert opt._rng is not None
        assert isinstance(opt._rng, random.Random)
    
    def test_seed_enables_reproducibility(self, optical_system, optimization_variable):
        """Same seed produces same random sequence."""
        target = Mock()
        opt1 = GlobalOptimizer(
            system=optical_system,
            variables=[optimization_variable],
            targets=[target],
            seed=42
        )
        opt2 = GlobalOptimizer(
            system=optical_system,
            variables=[optimization_variable],
            targets=[target],
            seed=42
        )
        
        # Both RNGs should produce same sequence
        assert opt1._rng.random() == opt2._rng.random()


class TestSimulatedAnnealing:
    """Test optimize_simulated_annealing method."""
    
    def test_sa_basic_run_completes(self, optimizer):
        """Basic SA run completes and calls _finalize."""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=10,
            initial_temperature=100.0
        )
        assert result is not None
        optimizer._finalize.assert_called_once()
    
    def test_sa_zero_iterations(self, optimizer):
        """max_iterations=0 produces valid output."""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=0,
            initial_temperature=100.0
        )
        call_kw = optimizer._finalize.call_args[1]
        # 0 SA iterations + 5 from simplex
        assert call_kw['iterations'] == 5
        # merit_history should have starting merit
        assert len(call_kw['merit_history']) >= 1
    
    def test_sa_clamped_temperature_logs_warning(self, optimizer):
        """Non-positive initial_temperature is clamped with warning."""
        with patch('openlens.global_optimizer.logger') as mock_logger:
            result = optimizer.optimize_simulated_annealing(
                max_iterations=5,
                initial_temperature=-1.0
            )
            mock_logger.warning.assert_called()
    
    def test_sa_callback_invoked_per_iteration(self, optimizer):
        """Callback is invoked for each SA iteration."""
        callback = Mock()
        result = optimizer.optimize_simulated_annealing(
            max_iterations=5,
            initial_temperature=100.0,
            callback=callback
        )
        # Callback called at least 5 times from SA loop
        assert callback.call_count >= 5
    
    def test_sa_variables_not_mutated(self, optical_system, optimization_variable):
        """Optimization variables remain unchanged after SA."""
        original_value = optimization_variable.current_value
        target = Mock()
        opt = GlobalOptimizer(
            system=optical_system,
            variables=[optimization_variable],
            targets=[target],
            seed=42
        )
        opt._evaluate_design = Mock(return_value=1.0)
        opt.optimize_simplex = Mock(return_value=SimpleNamespace(
            best_values=[5.0],
            final_merit=0.5,
            iterations=5,
            variable_history=[],
            merit_history=[]
        ))
        opt._finalize = Mock(side_effect=lambda **kw: SimpleNamespace(**kw))
        
        opt.optimize_simulated_annealing(max_iterations=5, initial_temperature=100.0)
        
        assert optimization_variable.current_value == original_value
    
    def test_sa_empty_variables_raises_error(self, optical_system):
        """Empty variables list raises ValueError from randint."""
        target = Mock()
        opt = GlobalOptimizer(
            system=optical_system,
            variables=[],
            targets=[target],
            seed=42
        )
        opt._evaluate_design = Mock(return_value=1.0)
        opt.optimize_simplex = Mock(return_value=SimpleNamespace(
            best_values=[],
            final_merit=0.5,
            iterations=5,
            variable_history=[],
            merit_history=[]
        ))
        opt._finalize = Mock()
        
        with pytest.raises(ValueError):
            opt.optimize_simulated_annealing(max_iterations=5, initial_temperature=100.0)
    
    def test_sa_callback_exception_propagates(self, optimizer):
        """Exceptions from callback propagate unchanged."""
        def bad_callback(iteration, merit, values):
            raise RuntimeError("Test callback error")
        
        with pytest.raises(RuntimeError, match="Test callback error"):
            optimizer.optimize_simulated_annealing(
                max_iterations=5,
                initial_temperature=100.0,
                callback=bad_callback
            )


class TestGeneticAlgorithm:
    """Test optimize_genetic method."""
    
    def test_ga_basic_run_completes(self, optimizer):
        """Basic GA run completes and calls _finalize."""
        result = optimizer.optimize_genetic(
            population_size=10,
            generations=5
        )
        assert result is not None
        optimizer._finalize.assert_called_once()
    
    def test_ga_zero_generations(self, optimizer):
        """generations=0 produces valid output."""
        result = optimizer.optimize_genetic(
            population_size=10,
            generations=0
        )
        call_kw = optimizer._finalize.call_args[1]
        assert call_kw['merit_history'] == []
        assert call_kw['iterations'] == 5  # Only simplex iterations
    
    def test_ga_zero_population_raises_error(self, optical_system, optimization_variable):
        """population_size=0 raises IndexError."""
        target = Mock()
        opt = GlobalOptimizer(
            system=optical_system,
            variables=[optimization_variable],
            targets=[target],
            seed=42
        )
        opt._evaluate_design = Mock(return_value=1.0)
        opt.optimize_simplex = Mock(return_value=SimpleNamespace(
            best_values=[1.0],
            final_merit=0.5,
            iterations=5,
            variable_history=[],
            merit_history=[]
        ))
        opt._finalize = Mock()
        
        with pytest.raises(IndexError):
            opt.optimize_genetic(population_size=0, generations=5)
    
    def test_ga_callback_invoked(self, optimizer):
        """Callback is invoked during GA."""
        callback = Mock()
        result = optimizer.optimize_genetic(
            population_size=5,
            generations=2,
            callback=callback
        )
        # Callback called multiple times per generation
        assert callback.call_count > 0
    
    def test_ga_simplex_called_with_correct_params(self, optimizer):
        """optimize_simplex is called with max_iterations=50."""
        optimizer.optimize_genetic(population_size=5, generations=2)
        
        call_kw = optimizer.optimize_simplex.call_args[1]
        assert call_kw['max_iterations'] == 50
        assert 'start' in call_kw
        assert isinstance(call_kw['start'], list)
    
    def test_ga_callback_exception_propagates(self, optimizer):
        """Exceptions from callback propagate unchanged."""
        def bad_callback(gen, merit, values):
            raise RuntimeError("Test callback error")
        
        with pytest.raises(RuntimeError, match="Test callback error"):
            optimizer.optimize_genetic(
                population_size=5,
                generations=2,
                callback=bad_callback
            )


class TestTournamentSelect:
    """Test _tournament_select helper method."""
    
    def test_empty_population_raises_error(self, optimizer):
        """Empty population raises ValueError with specific message."""
        with pytest.raises(ValueError, match="cannot select from an empty population"):
            optimizer._tournament_select([], [])
    
    def test_tournament_selects_best_individual(self, optimizer):
        """Tournament select returns individual with best merit."""
        population = [[1.0], [2.0], [3.0]]
        merits = [3.0, 1.0, 2.0]  # [2.0] has best (lowest) merit
        
        # With k=3, all are in tournament
        selected = optimizer._tournament_select(population, merits, k=3)
        assert selected == [2.0]

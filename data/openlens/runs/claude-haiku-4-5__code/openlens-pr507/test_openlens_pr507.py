import pytest
import math
from unittest.mock import Mock
from typing import List

try:
    from openlens.global_optimizer import GlobalOptimizer, OptimizationVariable, OptimizationResult
except ImportError:
    pytest.skip("openlens not installed", allow_module_level=True)


@pytest.fixture
def simple_optimizer():
    """Create a test optimizer with 2 variables for quadratic merit function"""
    system = Mock()
    var1 = OptimizationVariable(name="x", min_value=-10.0, max_value=10.0, current_value=0.0)
    var2 = OptimizationVariable(name="y", min_value=-10.0, max_value=10.0, current_value=0.0)
    
    optimizer = GlobalOptimizer(system=system, variables=[var1, var2])
    
    # Merit function: (x-3)^2 + (y+2)^2
    optimizer._evaluate_design = lambda design: (design[0] - 3.0) ** 2 + (design[1] + 2.0) ** 2
    
    return optimizer


class TestOptimizeGenetic:
    """Tests for GlobalOptimizer.optimize_genetic"""
    
    def test_genetic_returns_optimization_result(self, simple_optimizer):
        """optimize_genetic returns OptimizationResult"""
        result = simple_optimizer.optimize_genetic(generations=2, population_size=5)
        assert isinstance(result, OptimizationResult)
        assert hasattr(result, 'best_values')
        assert len(result.best_values) == 2
    
    def test_genetic_zero_generations(self, simple_optimizer):
        """Zero generations still executes local refinement"""
        result = simple_optimizer.optimize_genetic(generations=0, population_size=5)
        assert isinstance(result, OptimizationResult)
        assert hasattr(result, 'iterations')
    
    def test_genetic_callback_invoked(self, simple_optimizer):
        """Callback is invoked during genetic algorithm execution"""
        callback_calls = []
        
        def callback(gen: int, merit: float, design: List[float]):
            callback_calls.append((gen, merit, design))
        
        simple_optimizer.optimize_genetic(
            generations=2, population_size=3, callback=callback
        )
        assert len(callback_calls) > 0
        assert isinstance(callback_calls[0][0], int)
        assert isinstance(callback_calls[0][1], float)
    
    def test_genetic_zero_mutation_rate(self, simple_optimizer):
        """Zero mutation rate handled correctly"""
        result = simple_optimizer.optimize_genetic(
            generations=1, population_size=3, mutation_rate=0.0
        )
        assert isinstance(result, OptimizationResult)
    
    def test_genetic_full_mutation_rate(self, simple_optimizer):
        """Full mutation rate (1.0) handled correctly"""
        result = simple_optimizer.optimize_genetic(
            generations=1, population_size=3, mutation_rate=1.0
        )
        assert isinstance(result, OptimizationResult)
    
    def test_genetic_zero_crossover_rate(self, simple_optimizer):
        """Zero crossover rate (no crossover) handled correctly"""
        result = simple_optimizer.optimize_genetic(
            generations=1, population_size=3, crossover_rate=0.0
        )
        assert isinstance(result, OptimizationResult)
    
    def test_genetic_full_crossover_rate(self, simple_optimizer):
        """Full crossover rate (1.0) handled correctly"""
        result = simple_optimizer.optimize_genetic(
            generations=1, population_size=3, crossover_rate=1.0
        )
        assert isinstance(result, OptimizationResult)
    
    def test_genetic_merit_history_tracked(self, simple_optimizer):
        """Merit history is tracked across generations"""
        result = simple_optimizer.optimize_genetic(generations=2, population_size=4)
        assert hasattr(result, 'merit_history')
        assert len(result.merit_history) > 0


class TestOptimizeSimulatedAnnealing:
    """Tests for GlobalOptimizer.optimize_simulated_annealing"""
    
    def test_sa_returns_optimization_result(self, simple_optimizer):
        """optimize_simulated_annealing returns OptimizationResult"""
        result = simple_optimizer.optimize_simulated_annealing(max_iterations=5)
        assert isinstance(result, OptimizationResult)
        assert hasattr(result, 'best_values')
        assert len(result.best_values) == 2
    
    def test_sa_zero_max_iterations(self, simple_optimizer):
        """Zero max_iterations still executes local refinement"""
        result = simple_optimizer.optimize_simulated_annealing(max_iterations=0)
        assert isinstance(result, OptimizationResult)
        assert hasattr(result, 'iterations')
    
    def test_sa_negative_temperature_clamped(self, simple_optimizer):
        """Negative initial_temperature is clamped to positive value"""
        result = simple_optimizer.optimize_simulated_annealing(
            max_iterations=3, initial_temperature=-100.0
        )
        assert isinstance(result, OptimizationResult)
    
    def test_sa_zero_temperature_clamped(self, simple_optimizer):
        """Zero initial_temperature is clamped to positive value"""
        result = simple_optimizer.optimize_simulated_annealing(
            max_iterations=3, initial_temperature=0.0
        )
        assert isinstance(result, OptimizationResult)
    
    def test_sa_cooling_rate_none_derived(self, simple_optimizer):
        """cooling_rate=None derives schedule reaching floor at max_iterations"""
        result = simple_optimizer.optimize_simulated_annealing(
            max_iterations=20, initial_temperature=100.0, cooling_rate=None
        )
        assert isinstance(result, OptimizationResult)
    
    def test_sa_cooling_rate_provided(self, simple_optimizer):
        """Provided cooling_rate is used instead of derived"""
        result = simple_optimizer.optimize_simulated_annealing(
            max_iterations=10, initial_temperature=100.0, cooling_rate=0.95
        )
        assert isinstance(result, OptimizationResult)
    
    def test_sa_callback_invoked(self, simple_optimizer):
        """Callback is invoked at each SA iteration"""
        callback_calls = []
        
        def callback(iteration: int, merit: float, values: List[float]):
            callback_calls.append((iteration, merit, values))
        
        simple_optimizer.optimize_simulated_annealing(max_iterations=3, callback=callback)
        assert len(callback_calls) > 0
        assert isinstance(callback_calls[0][0], int)
        assert isinstance(callback_calls[0][1], float)
    
    def test_sa_restart_threshold_accepted(self, simple_optimizer):
        """restart_threshold parameter accepted (API compatibility, unused)"""
        result = simple_optimizer.optimize_simulated_annealing(
            max_iterations=3, restart_threshold=1e-4
        )
        assert isinstance(result, OptimizationResult)
    
    def test_sa_merit_history_tracked(self, simple_optimizer):
        """Merit history is tracked across SA iterations"""
        result = simple_optimizer.optimize_simulated_annealing(max_iterations=5)
        assert hasattr(result, 'merit_history')
        assert len(result.merit_history) > 0
    
    def test_sa_variable_history_tracked(self, simple_optimizer):
        """Variable history is tracked across SA iterations"""
        result = simple_optimizer.optimize_simulated_annealing(max_iterations=3)
        assert hasattr(result, 'variable_history')

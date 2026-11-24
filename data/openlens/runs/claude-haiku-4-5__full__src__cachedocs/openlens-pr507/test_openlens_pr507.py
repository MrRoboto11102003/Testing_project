import pytest
import math
from unittest.mock import Mock


class TestOptimizeGenetic:
    """Tests for GlobalOptimizer.optimize_genetic"""

    @pytest.fixture
    def optimizer(self):
        """Create a minimal optimizer for genetic algorithm testing"""
        from openlens.global_optimizer import GlobalOptimizer

        system = Mock()

        # Create variable mocks with necessary attributes
        var1 = Mock()
        var1.name = "x"
        var1.current_value = 0.5
        var1.min_value = 0.0
        var1.max_value = 1.0
        var1.step_size = 0.1
        var1.clamp = lambda x: max(0.0, min(1.0, x))

        var2 = Mock()
        var2.name = "y"
        var2.current_value = 0.5
        var2.min_value = 0.0
        var2.max_value = 1.0
        var2.step_size = 0.1
        var2.clamp = lambda x: max(0.0, min(1.0, x))

        optimizer = GlobalOptimizer(system=system, variables=[var1, var2])

        # Simple quadratic merit function: minimize (x-0.2)^2 + (y-0.8)^2
        optimizer._evaluate_design = lambda vals: (vals[0] - 0.2) ** 2 + (vals[1] - 0.8) ** 2

        return optimizer

    def test_genetic_returns_result_object(self, optimizer):
        """GA should return an OptimizationResult with expected attributes"""
        result = optimizer.optimize_genetic(population_size=10, generations=3)
        assert result is not None
        assert hasattr(result, "best_values")
        assert hasattr(result, "final_merit")
        assert isinstance(result.best_values, (list, tuple))
        assert len(result.best_values) == 2

    def test_genetic_respects_variable_bounds(self, optimizer):
        """GA should keep all variables within their specified bounds"""
        result = optimizer.optimize_genetic(population_size=10, generations=5)
        for val, var in zip(result.best_values, optimizer.variables):
            assert var.min_value <= val <= var.max_value

    def test_genetic_zero_generations_edge_case(self, optimizer):
        """GA with zero generations should handle gracefully (runs only refinement)"""
        result = optimizer.optimize_genetic(population_size=5, generations=0)
        assert result is not None
        assert len(result.best_values) == 2

    def test_genetic_callback_invocation(self, optimizer):
        """GA should invoke callback with correct argument types"""
        callback = Mock()
        result = optimizer.optimize_genetic(population_size=5, generations=2, callback=callback)
        
        assert callback.call_count > 0
        # Verify callback receives (iteration, merit, values)
        args = callback.call_args_list[0][0]
        assert isinstance(args[0], int)
        assert isinstance(args[1], (int, float))
        assert isinstance(args[2], list)

    def test_genetic_merit_history_tracking(self, optimizer):
        """GA should track best merit values across generations"""
        result = optimizer.optimize_genetic(population_size=5, generations=3)
        assert hasattr(result, "merit_history")
        assert len(result.merit_history) > 0
        # Merit history should be numeric values
        assert all(isinstance(m, (int, float)) for m in result.merit_history)

    def test_genetic_variable_history_tracking(self, optimizer):
        """GA should track variable values across iterations"""
        result = optimizer.optimize_genetic(population_size=5, generations=2)
        assert hasattr(result, "variable_history")
        assert len(result.variable_history) > 0

    def test_genetic_population_size_variation(self, optimizer):
        """GA should handle different population sizes"""
        result_small = optimizer.optimize_genetic(population_size=3, generations=1)
        
        # Reset for second run
        optimizer.variables[0].current_value = 0.5
        optimizer.variables[1].current_value = 0.5
        
        result_large = optimizer.optimize_genetic(population_size=20, generations=1)
        assert result_small is not None
        assert result_large is not None

    def test_genetic_mutation_rate_parameter(self, optimizer):
        """GA should accept and use mutation_rate parameter"""
        result = optimizer.optimize_genetic(
            population_size=5, generations=1, mutation_rate=0.5
        )
        assert result is not None

    def test_genetic_crossover_rate_parameter(self, optimizer):
        """GA should accept and use crossover_rate parameter"""
        result = optimizer.optimize_genetic(
            population_size=5, generations=1, crossover_rate=0.8
        )
        assert result is not None


class TestOptimizeSimulatedAnnealing:
    """Tests for GlobalOptimizer.optimize_simulated_annealing"""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer for simulated annealing testing"""
        from openlens.global_optimizer import GlobalOptimizer

        system = Mock()

        var1 = Mock()
        var1.name = "x"
        var1.current_value = 0.5
        var1.min_value = 0.0
        var1.max_value = 1.0
        var1.step_size = 0.1
        var1.clamp = lambda x: max(0.0, min(1.0, x))

        var2 = Mock()
        var2.name = "y"
        var2.current_value = 0.5
        var2.min_value = 0.0
        var2.max_value = 1.0
        var2.step_size = 0.1
        var2.clamp = lambda x: max(0.0, min(1.0, x))

        optimizer = GlobalOptimizer(system=system, variables=[var1, var2])
        optimizer._evaluate_design = lambda vals: (vals[0] - 0.2) ** 2 + (vals[1] - 0.8) ** 2

        return optimizer

    def test_sa_returns_result_object(self, optimizer):
        """SA should return an OptimizationResult with expected attributes"""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=10, initial_temperature=10.0
        )
        assert result is not None
        assert hasattr(result, "best_values")
        assert hasattr(result, "final_merit")

    def test_sa_negative_temperature_clamped(self, optimizer):
        """SA should clamp negative initial_temperature to 1e-6"""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=5, initial_temperature=-50.0
        )
        assert result is not None
        assert len(result.best_values) == 2

    def test_sa_zero_temperature_clamped(self, optimizer):
        """SA should clamp zero initial_temperature to 1e-6"""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=5, initial_temperature=0.0
        )
        assert result is not None
        assert len(result.best_values) == 2

    def test_sa_respects_variable_bounds(self, optimizer):
        """SA should keep variables within bounds"""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=20, initial_temperature=10.0
        )
        for val, var in zip(result.best_values, optimizer.variables):
            assert var.min_value <= val <= var.max_value

    def test_sa_zero_iterations_edge_case(self, optimizer):
        """SA with zero iterations should handle gracefully"""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=0, initial_temperature=10.0
        )
        assert result is not None

    def test_sa_callback_invocation(self, optimizer):
        """SA should invoke callback during optimization"""
        callback = Mock()
        result = optimizer.optimize_simulated_annealing(
            max_iterations=5, initial_temperature=10.0, callback=callback
        )
        assert callback.call_count > 0

    def test_sa_cooling_rate_auto_derived(self, optimizer):
        """SA should derive cooling_rate when None is passed"""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=20, initial_temperature=100.0, cooling_rate=None
        )
        assert result is not None

    def test_sa_custom_cooling_rate(self, optimizer):
        """SA should accept explicit cooling_rate value"""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=20, initial_temperature=100.0, cooling_rate=0.95
        )
        assert result is not None

    def test_sa_merit_history_tracking(self, optimizer):
        """SA should track merit history across iterations"""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=10, initial_temperature=10.0
        )
        assert hasattr(result, "merit_history")
        assert len(result.merit_history) > 0

    def test_sa_variable_history_tracking(self, optimizer):
        """SA should track variable values across iterations"""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=10, initial_temperature=10.0
        )
        assert hasattr(result, "variable_history")
        assert len(result.variable_history) > 0

    def test_sa_restart_threshold_api_compatibility(self, optimizer):
        """SA should accept restart_threshold for API compatibility (unused)"""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=5, initial_temperature=10.0, restart_threshold=1e-3
        )
        assert result is not None

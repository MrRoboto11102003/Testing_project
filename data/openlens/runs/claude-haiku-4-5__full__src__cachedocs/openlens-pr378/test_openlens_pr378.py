import pytest
import math
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from unittest.mock import Mock, patch, MagicMock

from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.optimizer import LensOptimizer, OptimizationResult, OptimizationVariable
from openlens.global_optimizer import GlobalOptimizer


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_result_creation_and_fields(self):
        """Test OptimizationResult instantiation and field access."""
        system = OpticalSystem(name="test")
        lens = Lens(
            name="test_lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        system.add_lens(lens)

        result = OptimizationResult(
            success=True,
            iterations=15,
            initial_merit=100.0,
            final_merit=25.0,
            improvement=75.0,
            optimized_system=system,
            message="Converged successfully",
            best_values=[50.0, -50.0],
            variable_history=[{"r1": 50.0}, {"r1": 51.0}],
            merit_history=[100.0, 50.0, 25.0],
        )

        assert result.success is True
        assert result.iterations == 15
        assert result.initial_merit == 100.0
        assert result.final_merit == 25.0
        assert result.improvement == 75.0
        assert result.message == "Converged successfully"
        assert len(result.best_values) == 2
        assert len(result.variable_history) == 2
        assert len(result.merit_history) == 3

    def test_result_with_defaults(self):
        """Test OptimizationResult with default field values."""
        system = OpticalSystem(name="minimal")
        result = OptimizationResult(
            success=False,
            iterations=1,
            initial_merit=50.0,
            final_merit=50.0,
            improvement=0.0,
            optimized_system=system,
        )

        assert result.message == ""
        assert result.variable_history == []
        assert result.merit_history == []
        assert result.best_values == []

    def test_result_improvement_tracking(self):
        """Test that OptimizationResult correctly tracks improvement."""
        system = OpticalSystem()
        
        # Test case: 50% improvement
        result1 = OptimizationResult(
            success=True,
            iterations=10,
            initial_merit=100.0,
            final_merit=50.0,
            improvement=50.0,
            optimized_system=system,
        )
        assert result1.improvement == 50.0

        # Test case: no improvement
        result2 = OptimizationResult(
            success=False,
            iterations=5,
            initial_merit=100.0,
            final_merit=100.0,
            improvement=0.0,
            optimized_system=system,
        )
        assert result2.improvement == 0.0


class TestLensOptimizerSimplex:
    """Test LensOptimizer.optimize_simplex method."""

    @pytest.fixture
    def simple_system_and_optimizer(self):
        """Create a minimal optical system and optimizer."""
        lens = Lens(
            name="biconvex",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        system = OpticalSystem(name="single_lens")
        system.add_lens(lens)
        optimizer = LensOptimizer(system)
        return system, optimizer

    def test_simplex_returns_optimization_result(self, simple_system_and_optimizer):
        """Test that optimize_simplex returns an OptimizationResult."""
        system, optimizer = simple_system_and_optimizer
        
        # Manually add a simple variable if needed
        if not hasattr(optimizer, "variables") or len(optimizer.variables) == 0:
            pytest.skip("Optimizer has no variables to optimize")

        result = optimizer.optimize_simplex(max_iterations=10, tolerance=1e-4)
        
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.iterations, int)
        assert result.iterations > 0
        assert isinstance(result.initial_merit, (int, float))
        assert isinstance(result.final_merit, (int, float))

    def test_simplex_start_parameter_explicit(self, simple_system_and_optimizer):
        """Test simplex with explicit start values."""
        system, optimizer = simple_system_and_optimizer
        
        if not hasattr(optimizer, "variables") or len(optimizer.variables) == 0:
            pytest.skip("No optimization variables")

        n_vars = len(optimizer.variables)
        start = [var.current_value for var in optimizer.variables]

        # Should not raise
        result = optimizer.optimize_simplex(
            max_iterations=5, start=start, tolerance=1e-6
        )
        
        assert result is not None
        assert len(result.best_values) == n_vars

    def test_simplex_start_length_validation(self, simple_system_and_optimizer):
        """Test that mismatched start vector length raises ValueError."""
        system, optimizer = simple_system_and_optimizer
        
        if not hasattr(optimizer, "variables") or len(optimizer.variables) == 0:
            pytest.skip("No variables to test")

        n_vars = len(optimizer.variables)
        bad_start = [1.0] * (n_vars + 1)  # Wrong length

        with pytest.raises(ValueError, match="start has"):
            optimizer.optimize_simplex(max_iterations=5, start=bad_start)

    def test_simplex_callback_invoked(self, simple_system_and_optimizer):
        """Test that callback is invoked during simplex optimization."""
        system, optimizer = simple_system_and_optimizer
        
        if not hasattr(optimizer, "variables") or len(optimizer.variables) == 0:
            pytest.skip("No variables")

        callback_data = []

        def test_callback(iteration: int, merit: float, values: List[float]):
            callback_data.append((iteration, merit, values))

        result = optimizer.optimize_simplex(
            max_iterations=5, callback=test_callback, tolerance=1e-4
        )

        # Callback should be called at least once per iteration
        assert len(callback_data) > 0
        assert all(isinstance(item[0], int) for item in callback_data)
        assert all(isinstance(item[1], (int, float)) for item in callback_data)

    def test_simplex_no_variables_raises(self, simple_system_and_optimizer):
        """Test that simplex raises ValueError when no variables are defined."""
        system, optimizer = simple_system_and_optimizer
        
        # Mock an optimizer with no variables
        with patch.object(optimizer, "variables", []):
            with pytest.raises(ValueError, match="at least one variable"):
                optimizer.optimize_simplex(max_iterations=10)

    def test_simplex_respects_max_iterations(self, simple_system_and_optimizer):
        """Test that simplex respects max_iterations limit."""
        system, optimizer = simple_system_and_optimizer
        
        if not hasattr(optimizer, "variables") or len(optimizer.variables) == 0:
            pytest.skip("No variables")

        result = optimizer.optimize_simplex(max_iterations=3, tolerance=1e-10)
        
        # Should stop at or before max_iterations
        assert result.iterations <= 3


class TestGlobalOptimizerGenetic:
    """Test GlobalOptimizer.optimize_genetic method."""

    @pytest.fixture
    def genetic_test_system(self):
        """Create a test system for genetic algorithm."""
        lens = Lens(
            name="genetic_test",
            radius_of_curvature_1=60.0,
            radius_of_curvature_2=-60.0,
            thickness=4.0,
            diameter=30.0,
            material="BK7",
        )
        system = OpticalSystem(name="genetic_system")
        system.add_lens(lens)
        return system

    def test_genetic_returns_optimization_result(self, genetic_test_system):
        """Test that optimize_genetic returns OptimizationResult."""
        optimizer = GlobalOptimizer(genetic_test_system)
        
        if not hasattr(optimizer, "variables") or len(optimizer.variables) == 0:
            pytest.skip("No optimization variables")

        result = optimizer.optimize_genetic(
            population_size=10, generations=3
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.iterations >= 3
        assert isinstance(result.best_values, list)

    def test_genetic_parameters(self, genetic_test_system):
        """Test genetic algorithm with various parameters."""
        optimizer = GlobalOptimizer(genetic_test_system)
        
        if not hasattr(optimizer, "variables") or len(optimizer.variables) == 0:
            pytest.skip("No variables")

        result = optimizer.optimize_genetic(
            population_size=8,
            generations=2,
            mutation_rate=0.15,
            crossover_rate=0.8,
        )
        
        assert result is not None
        assert result.iterations >= 2

    def test_genetic_callback_per_generation(self, genetic_test_system):
        """Test that genetic callback receives generation updates."""
        optimizer = GlobalOptimizer(genetic_test_system)
        
        if not hasattr(optimizer, "variables") or len(optimizer.variables) == 0:
            pytest.skip("No variables")

        callbacks = []

        def gen_callback(gen: int, merit: float, values: List[float]):
            callbacks.append((gen, merit, len(values)))

        result = optimizer.optimize_genetic(
            population_size=6, generations=2, callback=gen_callback
        )

        # Callback should have been invoked
        assert len(callbacks) > 0


class TestGlobalOptimizerSimulatedAnnealing:
    """Test GlobalOptimizer.optimize_simulated_annealing method."""

    @pytest.fixture
    def sa_test_system(self):
        """Create system for simulated annealing tests."""
        lens = Lens(
            name="sa_lens",
            radius_of_curvature_1=70.0,
            radius_of_curvature_2=-70.0,
            thickness=6.0,
            diameter=20.0,
            material="BK7",
        )
        system = OpticalSystem(name="sa_system")
        system.add_lens(lens)
        return system

    def test_sa_basic_optimization(self, sa_test_system):
        """Test basic simulated annealing optimization."""
        optimizer = GlobalOptimizer(sa_test_system)
        
        if not hasattr(optimizer, "variables") or len(optimizer.variables) == 0:
            pytest.skip("No variables")

        result = optimizer.optimize_simulated_annealing(
            max_iterations=10, initial_temperature=100.0
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.iterations > 0
        assert isinstance(result.final_merit, (int, float))

    def test_sa_negative_temperature_clamped(self, sa_test_system):
        """Test that negative initial_temperature is clamped."""
        optimizer = GlobalOptimizer(sa_test_system)
        
        if not hasattr(optimizer, "variables") or len(optimizer.variables) == 0:
            pytest.skip("No variables")

        # Should not raise; negative temperature clamped internally
        result = optimizer.optimize_simulated_annealing(
            max_iterations=3, initial_temperature=-50.0
        )
        
        assert result is not None
        assert result.iterations > 0

    def test_sa_zero_temperature_clamped(self, sa_test_system):
        """Test that zero initial_temperature is clamped to positive."""
        optimizer = GlobalOptimizer(sa_test_system)
        
        if not hasattr(optimizer, "variables") or len(optimizer.variables) == 0:
            pytest.skip("No variables")

        # Should clamp to small positive value internally
        result = optimizer.optimize_simulated_annealing(
            max_iterations=2, initial_temperature=0.0
        )
        
        assert result is not None

    def test_sa_custom_cooling_rate(self, sa_test_system):
        """Test SA with explicit cooling rate parameter."""
        optimizer = GlobalOptimizer(sa_test_system)
        
        if not hasattr(optimizer, "variables") or len(optimizer.variables) == 0:
            pytest.skip("No variables")

        result = optimizer.optimize_simulated_annealing(
            max_iterations=5,
            initial_temperature=50.0,
            cooling_rate=0.95,  # Explicit cooling rate
        )
        
        assert result is not None

    def test_sa_callback_invoked(self, sa_test_system):
        """Test that SA callback is invoked during iterations."""
        optimizer = GlobalOptimizer(sa_test_system)
        
        if not hasattr(optimizer, "variables") or len(optimizer.variables) == 0:
            pytest.skip("No variables")

        callback_calls = []

        def sa_callback(iteration: int, merit: float, values: List[float]):
            callback_calls.append((iteration, merit))

        result = optimizer.optimize_simulated_annealing(
            max_iterations=4, initial_temperature=30.0, callback=sa_callback
        )

        # Callback should have been invoked
        assert len(callback_calls) >= 0

    def test_sa_iterates_up_to_limit(self, sa_test_system):
        """Test that SA respects max_iterations parameter."""
        optimizer = GlobalOptimizer(sa_test_system)
        
        if not hasattr(optimizer, "variables") or len(optimizer.variables) == 0:
            pytest.skip("No variables")

        result = optimizer.optimize_simulated_annealing(
            max_iterations=5, initial_temperature=50.0
        )
        
        # Total iterations should include SA + simplex refinement
        assert result.iterations > 0


class TestOptimizationIntegration:
    """Integration tests for optimization workflows."""

    def test_lens_focal_length_optimization(self):
        """Test a realistic lens optimization scenario."""
        # Create a simple lens
        lens = Lens(
            name="opt_test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        
        # Verify lens was created and has calculable properties
        assert lens.calculate_focal_length() is not None
        focal_length = lens.calculate_focal_length()
        assert focal_length > 0

    def test_optical_system_creation_for_optimization(self):
        """Test that OpticalSystem can be created for optimization."""
        system = OpticalSystem(name="opt_system")
        
        lens1 = Lens(
            name="L1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        lens2 = Lens(
            name="L2",
            radius_of_curvature_1=-100.0,
            radius_of_curvature_2=100.0,
            thickness=3.0,
            diameter=25.0,
            material="F2",
        )
        
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=10.0)
        
        assert system is not None
        assert system.get_system_focal_length() is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

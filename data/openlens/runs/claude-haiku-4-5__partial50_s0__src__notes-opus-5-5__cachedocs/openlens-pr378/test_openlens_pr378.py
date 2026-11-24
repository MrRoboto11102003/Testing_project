import pytest
from dataclasses import fields
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.optimizer import (
    LensOptimizer,
    OptimizationVariable,
    OptimizationTarget,
    OptimizationResult,
)
from openlens.global_optimizer import GlobalOptimizer


class TestOptimizationResultDataclass:
    """Test OptimizationResult dataclass structure and behavior"""

    def test_construction_with_all_required_fields(self):
        """Construction with all 6 required fields succeeds"""
        sys = OpticalSystem()
        result = OptimizationResult(
            success=True,
            iterations=5,
            initial_merit=100.0,
            final_merit=50.0,
            improvement=50.0,
            optimized_system=sys,
        )
        assert result.success is True
        assert result.iterations == 5
        assert result.initial_merit == 100.0
        assert result.final_merit == 50.0
        assert result.improvement == 50.0
        assert result.optimized_system is sys

    def test_missing_required_field_raises_typeerror(self):
        """Missing a required field raises TypeError"""
        with pytest.raises(TypeError):
            OptimizationResult(
                success=True,
                iterations=1,
                initial_merit=100.0,
                final_merit=50.0,
                improvement=50.0,
                # optimized_system is missing
            )

    def test_default_list_factories_are_independent(self):
        """Each instance gets independent default lists from factory"""
        sys = OpticalSystem()
        r1 = OptimizationResult(
            success=True,
            iterations=1,
            initial_merit=1.0,
            final_merit=0.5,
            improvement=50.0,
            optimized_system=sys,
        )
        r2 = OptimizationResult(
            success=True,
            iterations=1,
            initial_merit=1.0,
            final_merit=0.5,
            improvement=50.0,
            optimized_system=sys,
        )
        r1.variable_history.append({"x": 1.0})
        assert len(r2.variable_history) == 0
        assert len(r1.variable_history) == 1

    def test_message_defaults_to_empty_string(self):
        """message field defaults to empty string"""
        result = OptimizationResult(
            success=True,
            iterations=1,
            initial_merit=1.0,
            final_merit=0.5,
            improvement=50.0,
            optimized_system=None,
        )
        assert result.message == ""

    def test_equality_of_two_identical_results(self):
        """Two instances with identical field values are equal"""
        sys = OpticalSystem()
        r1 = OptimizationResult(
            success=True,
            iterations=1,
            initial_merit=1.0,
            final_merit=0.5,
            improvement=50.0,
            optimized_system=sys,
        )
        r2 = OptimizationResult(
            success=True,
            iterations=1,
            initial_merit=1.0,
            final_merit=0.5,
            improvement=50.0,
            optimized_system=sys,
        )
        assert r1 == r2


class TestOptimizeSimplexEdgeCases:
    """Test LensOptimizer.optimize_simplex edge cases"""

    def test_zero_variables_raises_valueerror(self):
        """Zero variables raises ValueError with specific message"""
        sys = OpticalSystem()
        sys.add_lens(Lens(name="test", radius_of_curvature_1=50.0))
        opt = LensOptimizer(sys, [], [])
        with pytest.raises(ValueError, match="at least one variable"):
            opt.optimize_simplex()

    def test_wrong_start_length_raises_valueerror(self):
        """Wrong start list length raises ValueError"""
        sys = OpticalSystem()
        sys.add_lens(Lens(name="test", radius_of_curvature_1=50.0))
        var = OptimizationVariable(
            "x", 0, "radius_of_curvature_1", 50.0, 20.0, 100.0
        )
        opt = LensOptimizer(sys, [var], [])
        with pytest.raises(ValueError, match="start has.*value"):
            opt.optimize_simplex(start=[1.0, 2.0])


class TestOptimizeSimplexBehavior:
    """Test LensOptimizer.optimize_simplex behavior"""

    def test_returns_optimization_result(self):
        """optimize_simplex returns an OptimizationResult instance"""
        sys = OpticalSystem()
        sys.add_lens(Lens(name="test", radius_of_curvature_1=50.0))
        var = OptimizationVariable(
            "x", 0, "radius_of_curvature_1", 50.0, 20.0, 100.0
        )
        opt = LensOptimizer(sys, [var], [])
        result = opt.optimize_simplex(max_iterations=1)
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.best_values, list)
        assert len(result.best_values) == len([var])

    def test_callback_invoked_during_optimization(self):
        """Callback is called at least once during optimization"""
        sys = OpticalSystem()
        sys.add_lens(Lens(name="test", radius_of_curvature_1=50.0))
        var = OptimizationVariable(
            "x", 0, "radius_of_curvature_1", 50.0, 20.0, 100.0
        )
        opt = LensOptimizer(sys, [var], [])
        calls = []

        def callback(iteration, merit, values):
            calls.append((iteration, merit, list(values)))

        result = opt.optimize_simplex(max_iterations=2, callback=callback)
        assert len(calls) >= 1
        assert all(isinstance(c[0], int) for c in calls)
        assert all(isinstance(c[1], (int, float)) for c in calls)


class TestOptimizeGeneticEdgeCases:
    """Test GlobalOptimizer.optimize_genetic edge cases"""

    def test_population_size_zero_raises_indexerror(self):
        """population_size=0 raises IndexError on assignment"""
        sys = OpticalSystem()
        sys.add_lens(Lens(name="test", radius_of_curvature_1=50.0))
        var = OptimizationVariable(
            "x", 0, "radius_of_curvature_1", 50.0, 20.0, 100.0
        )
        opt = GlobalOptimizer(sys, [var], [], seed=42)
        with pytest.raises(IndexError):
            opt.optimize_genetic(population_size=0, generations=1)

    def test_generations_zero_skips_ga_loop(self):
        """generations=0 produces message indicating 0 generations"""
        sys = OpticalSystem()
        sys.add_lens(Lens(name="test", radius_of_curvature_1=50.0))
        var = OptimizationVariable(
            "x", 0, "radius_of_curvature_1", 50.0, 20.0, 100.0
        )
        opt = GlobalOptimizer(sys, [var], [], seed=42)
        result = opt.optimize_genetic(generations=0, population_size=5)
        assert isinstance(result, OptimizationResult)
        assert "0 generations" in result.message


class TestOptimizeGeneticBehavior:
    """Test GlobalOptimizer.optimize_genetic behavior"""

    def test_returns_optimization_result(self):
        """optimize_genetic returns OptimizationResult with expected fields"""
        sys = OpticalSystem()
        sys.add_lens(Lens(name="test", radius_of_curvature_1=50.0))
        var = OptimizationVariable(
            "x", 0, "radius_of_curvature_1", 50.0, 20.0, 100.0
        )
        opt = GlobalOptimizer(sys, [var], [], seed=42)
        result = opt.optimize_genetic(generations=1, population_size=5)
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.best_values, list)
        assert isinstance(result.iterations, int)
        assert len(result.best_values) == 1

    def test_seed_reproducibility(self):
        """Same seed produces identical best_values"""
        sys1 = OpticalSystem()
        sys1.add_lens(Lens(name="test", radius_of_curvature_1=50.0))
        var1 = OptimizationVariable(
            "x", 0, "radius_of_curvature_1", 50.0, 20.0, 100.0
        )

        sys2 = OpticalSystem()
        sys2.add_lens(Lens(name="test", radius_of_curvature_1=50.0))
        var2 = OptimizationVariable(
            "x", 0, "radius_of_curvature_1", 50.0, 20.0, 100.0
        )

        opt1 = GlobalOptimizer(sys1, [var1], [], seed=12345)
        opt2 = GlobalOptimizer(sys2, [var2], [], seed=12345)

        result1 = opt1.optimize_genetic(generations=1, population_size=3)
        result2 = opt2.optimize_genetic(generations=1, population_size=3)

        assert result1.best_values == result2.best_values


class TestOptimizeSimulatedAnnealingEdgeCases:
    """Test GlobalOptimizer.optimize_simulated_annealing edge cases"""

    def test_zero_max_iterations_produces_correct_message(self):
        """max_iterations=0 produces message with '0 iterations'"""
        sys = OpticalSystem()
        sys.add_lens(Lens(name="test", radius_of_curvature_1=50.0))
        var = OptimizationVariable(
            "x", 0, "radius_of_curvature_1", 50.0, 20.0, 100.0
        )
        opt = GlobalOptimizer(sys, [var], [], seed=42)
        result = opt.optimize_simulated_annealing(max_iterations=0)
        assert isinstance(result, OptimizationResult)
        assert "0 iterations" in result.message


class TestOptimizeSimulatedAnnealingBehavior:
    """Test GlobalOptimizer.optimize_simulated_annealing behavior"""

    def test_returns_optimization_result(self):
        """optimize_simulated_annealing returns OptimizationResult"""
        sys = OpticalSystem()
        sys.add_lens(Lens(name="test", radius_of_curvature_1=50.0))
        var = OptimizationVariable(
            "x", 0, "radius_of_curvature_1", 50.0, 20.0, 100.0
        )
        opt = GlobalOptimizer(sys, [var], [], seed=42)
        result = opt.optimize_simulated_annealing(max_iterations=1)
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.best_values, list)
        assert isinstance(result.iterations, int)
        assert len(result.best_values) == 1

    def test_seed_reproducibility(self):
        """Same seed produces identical results"""
        sys1 = OpticalSystem()
        sys1.add_lens(Lens(name="test", radius_of_curvature_1=50.0))
        var1 = OptimizationVariable(
            "x", 0, "radius_of_curvature_1", 50.0, 20.0, 100.0
        )

        sys2 = OpticalSystem()
        sys2.add_lens(Lens(name="test", radius_of_curvature_1=50.0))
        var2 = OptimizationVariable(
            "x", 0, "radius_of_curvature_1", 50.0, 20.0, 100.0
        )

        opt1 = GlobalOptimizer(sys1, [var1], [], seed=12345)
        opt2 = GlobalOptimizer(sys2, [var2], [], seed=12345)

        result1 = opt1.optimize_simulated_annealing(max_iterations=2)
        result2 = opt2.optimize_simulated_annealing(max_iterations=2)

        assert result1.best_values == result2.best_values
        assert result1.final_merit == result2.final_merit

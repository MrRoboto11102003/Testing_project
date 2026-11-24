import pytest
import types
import logging
from unittest.mock import Mock, patch
import random

# Tests assume openlens is installed and importable
try:
    from openlens.global_optimizer import GlobalOptimizer
    from openlens.optical_system import OpticalSystem
    from openlens.lens import Lens
    from openlens.optimizer import OptimizationVariable, OptimizationTarget
except ImportError:
    pytest.skip("openlens not installed", allow_module_level=True)


@pytest.fixture
def simple_system():
    """Create a minimal optical system for testing."""
    system = OpticalSystem()
    lens = Lens(
        name="TestLens",
        radius_of_curvature_1=50.0,
        radius_of_curvature_2=-50.0,
        thickness=5.0,
        diameter=25.0,
        material="BK7"
    )
    system.add_lens(lens)
    return system


@pytest.fixture
def simple_variable():
    """Create a test optimization variable."""
    return OptimizationVariable(
        name="r1",
        element_index=0,
        attribute="radius_of_curvature_1",
        current_value=50.0,
        min_value=20.0,
        max_value=100.0,
        step_size=5.0
    )


@pytest.fixture
def simple_target():
    """Create a test optimization target."""
    return OptimizationTarget(
        attribute="focal_length",
        target_value=100.0,
        weight=1.0
    )


@pytest.fixture
def stubbed_optimizer(simple_system, simple_variable, simple_target):
    """Create a GlobalOptimizer with stubbed evaluation and refinement."""
    opt = GlobalOptimizer(
        simple_system, [simple_variable], [simple_target], seed=42
    )
    
    # Stub _evaluate_design: minimize distance from r1=50
    opt._evaluate_design = lambda v: (v[0] - 50.0) ** 2 if v else 1e6
    
    # Stub optimize_simplex to simulate refinement
    def fake_simplex(max_iterations=50, start=None, callback=None):
        refined = list(start) if start else [50.0]
        return types.SimpleNamespace(
            best_values=refined,
            final_merit=(refined[0] - 50.0) ** 2 * 0.1,
            iterations=3,
            variable_history=[{"r1": refined[0]}],
            merit_history=[0.1, 0.05, 0.01]
        )
    opt.optimize_simplex = fake_simplex
    
    # Stub _finalize to return a namespace with the passed arguments
    opt._finalize = lambda **kw: types.SimpleNamespace(
        **kw, success=True, message=kw.get("converged_message", "")
    )
    
    return opt


class TestOptimizeGenetic:
    """Tests for GlobalOptimizer.optimize_genetic."""

    def test_genetic_basic_execution(self, stubbed_optimizer):
        """Test that genetic algorithm runs and returns result."""
        result = stubbed_optimizer.optimize_genetic(
            population_size=10, generations=3, mutation_rate=0.1, crossover_rate=0.7
        )
        assert result is not None
        assert hasattr(result, "best_values")
        assert hasattr(result, "final_merit")
        assert hasattr(result, "iterations")

    def test_genetic_zero_generations(self, stubbed_optimizer):
        """Test GA with zero generations skips loop and runs only simplex."""
        result = stubbed_optimizer.optimize_genetic(
            population_size=5, generations=0
        )
        assert result.merit_history == []
        assert "0 generations" in result.converged_message
        assert result.iterations == 0 + 3  # 0 GA + 3 simplex iterations

    def test_genetic_population_size_zero_raises(self, stubbed_optimizer):
        """Test that population_size=0 raises IndexError on initial design placement."""
        with pytest.raises(IndexError):
            stubbed_optimizer.optimize_genetic(population_size=0, generations=1)

    def test_genetic_merit_history_length(self, stubbed_optimizer):
        """Test merit_history has exactly one entry per generation."""
        generations = 5
        result = stubbed_optimizer.optimize_genetic(
            population_size=8, generations=generations
        )
        assert len(result.merit_history) == generations

    def test_genetic_merit_history_non_increasing(self, stubbed_optimizer):
        """Test that merit_history is non-increasing (best-so-far tracking)."""
        result = stubbed_optimizer.optimize_genetic(
            population_size=10, generations=5
        )
        for i in range(1, len(result.merit_history)):
            assert result.merit_history[i] <= result.merit_history[i - 1]

    def test_genetic_variables_not_mutated(self, stubbed_optimizer):
        """Test that optimization variables' current_value is not changed."""
        original = [v.current_value for v in stubbed_optimizer.variables]
        stubbed_optimizer.optimize_genetic(population_size=5, generations=3)
        after = [v.current_value for v in stubbed_optimizer.variables]
        assert original == after

    def test_genetic_callback_invoked(self, stubbed_optimizer):
        """Test that callback is invoked during optimization."""
        calls = []

        def callback(gen, merit, design):
            calls.append((gen, merit, len(design)))

        stubbed_optimizer.optimize_genetic(
            population_size=5, generations=2, callback=callback
        )
        assert len(calls) > 0
        assert calls[0][0] == 0  # First call is generation 0

    def test_genetic_callback_exception_propagates(self, stubbed_optimizer):
        """Test that exceptions raised in callback propagate unchanged."""

        def bad_callback(*args):
            raise RuntimeError("Callback stop signal")

        with pytest.raises(RuntimeError, match="Callback stop signal"):
            stubbed_optimizer.optimize_genetic(
                population_size=5, generations=2, callback=bad_callback
            )

    def test_genetic_deterministic_with_seed(self, simple_system, simple_variable, simple_target):
        """Test that identical seed produces identical results."""
        opt1 = GlobalOptimizer(simple_system, [simple_variable], [simple_target], seed=99)
        opt2 = GlobalOptimizer(simple_system, [simple_variable], [simple_target], seed=99)

        for opt in [opt1, opt2]:
            opt._evaluate_design = lambda v: (v[0] - 50.0) ** 2 if v else 1e6
            opt.optimize_simplex = lambda **kw: types.SimpleNamespace(
                best_values=kw.get("start", [50.0]),
                final_merit=0.01,
                iterations=1,
                variable_history=[],
                merit_history=[0.01]
            )
            opt._finalize = lambda **kw: types.SimpleNamespace(**kw, success=True)

        r1 = opt1.optimize_genetic(population_size=5, generations=2)
        r2 = opt2.optimize_genetic(population_size=5, generations=2)
        assert r1.best_values == r2.best_values


class TestOptimizeSimulatedAnnealing:
    """Tests for GlobalOptimizer.optimize_simulated_annealing."""

    def test_sa_basic_execution(self, stubbed_optimizer):
        """Test that simulated annealing runs and returns result."""
        result = stubbed_optimizer.optimize_simulated_annealing(
            max_iterations=10, initial_temperature=100.0
        )
        assert result is not None
        assert hasattr(result, "best_values")
        assert hasattr(result, "final_merit")
        assert hasattr(result, "initial_merit")

    def test_sa_zero_iterations(self, stubbed_optimizer):
        """Test SA with max_iterations=0 runs only simplex."""
        result = stubbed_optimizer.optimize_simulated_annealing(
            max_iterations=0, initial_temperature=100.0
        )
        assert "0 iterations" in result.converged_message or "0 iterations" in result.out_of_bounds_message

    def test_sa_negative_temperature_clamped_and_logged(self, stubbed_optimizer, caplog):
        """Test that negative initial_temperature is clamped to 1e-6 and logged."""
        with caplog.at_level(logging.WARNING):
            result = stubbed_optimizer.optimize_simulated_annealing(
                max_iterations=3, initial_temperature=-50.0
            )

        assert result is not None
        assert any("not positive" in record.message for record in caplog.records)

    def test_sa_zero_temperature_clamped_and_logged(self, stubbed_optimizer, caplog):
        """Test that zero initial_temperature is clamped to 1e-6 and logged."""
        with caplog.at_level(logging.WARNING):
            result = stubbed_optimizer.optimize_simulated_annealing(
                max_iterations=3, initial_temperature=0.0
            )

        assert result is not None
        assert any("not positive" in record.message for record in caplog.records)

    def test_sa_initial_merit_first_in_history(self, stubbed_optimizer):
        """Test that merit_history[0] equals initial_merit."""
        result = stubbed_optimizer.optimize_simulated_annealing(
            max_iterations=5, initial_temperature=100.0
        )
        assert result.merit_history[0] == result.initial_merit

    def test_sa_callback_invoked_per_iteration(self, stubbed_optimizer):
        """Test that callback is invoked at least once per SA iteration."""
        calls = []

        def callback(iteration, merit, values):
            calls.append(iteration)

        stubbed_optimizer.optimize_simulated_annealing(
            max_iterations=5, initial_temperature=100.0, callback=callback
        )
        # Callback is called in SA loop + simplex refinement
        assert len(calls) > 0

    def test_sa_callback_exception_propagates(self, stubbed_optimizer):
        """Test that exceptions in callback propagate unchanged."""

        def bad_callback(*args):
            raise ValueError("SA user cancel")

        with pytest.raises(ValueError, match="SA user cancel"):
            stubbed_optimizer.optimize_simulated_annealing(
                max_iterations=5, initial_temperature=100.0, callback=bad_callback
            )

    def test_sa_variables_not_mutated(self, stubbed_optimizer):
        """Test that OptimizationVariable.current_value is not changed."""
        original = [v.current_value for v in stubbed_optimizer.variables]
        stubbed_optimizer.optimize_simulated_annealing(
            max_iterations=5, initial_temperature=100.0
        )
        after = [v.current_value for v in stubbed_optimizer.variables]
        assert original == after

    def test_sa_explicit_cooling_rate(self, stubbed_optimizer):
        """Test SA with explicit cooling_rate parameter."""
        result = stubbed_optimizer.optimize_simulated_annealing(
            max_iterations=20,
            initial_temperature=1.0,
            cooling_rate=0.95
        )
        assert result is not None
        assert result.iterations >= 0

    def test_sa_deterministic_with_seed(self, simple_system, simple_variable, simple_target):
        """Test that identical seed produces identical results."""
        opt1 = GlobalOptimizer(simple_system, [simple_variable], [simple_target], seed=55)
        opt2 = GlobalOptimizer(simple_system, [simple_variable], [simple_target], seed=55)

        for opt in [opt1, opt2]:
            opt._evaluate_design = lambda v: (v[0] - 50.0) ** 2 if v else 1e6
            opt.optimize_simplex = lambda **kw: types.SimpleNamespace(
                best_values=kw.get("start", [50.0]),
                final_merit=0.01,
                iterations=1,
                variable_history=[],
                merit_history=[0.01]
            )
            opt._finalize = lambda **kw: types.SimpleNamespace(**kw, success=True)

        r1 = opt1.optimize_simulated_annealing(max_iterations=5, initial_temperature=100.0)
        r2 = opt2.optimize_simulated_annealing(max_iterations=5, initial_temperature=100.0)
        assert r1.best_values == r2.best_values
        assert r1.final_merit == r2.final_merit

    def test_sa_message_contains_iterations(self, stubbed_optimizer):
        """Test that message strings contain iteration information."""
        result = stubbed_optimizer.optimize_simulated_annealing(
            max_iterations=7, initial_temperature=100.0
        )
        msg = result.converged_message or result.out_of_bounds_message
        assert "iterations" in msg.lower()

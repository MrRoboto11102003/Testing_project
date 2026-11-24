import pytest
import types
import logging
from unittest.mock import Mock

from openlens.global_optimizer import GlobalOptimizer
from openlens.optimizer import LensOptimizer, OptimizationResult, OptimizationVariable, OptimizationTarget
from openlens import OpticalSystem, Lens


@pytest.fixture
def simple_system():
    """Create a simple optical system for testing."""
    system = OpticalSystem()
    system.add_lens(Lens())
    return system


@pytest.fixture
def simple_variable():
    """Create a simple optimization variable."""
    return OptimizationVariable(
        name="x",
        index=0,
        parameter_name="radius_of_curvature_1",
        current_value=50.0,
        min_value=20.0,
        max_value=100.0
    )


@pytest.fixture
def simple_target():
    """Create a simple optimization target."""
    return OptimizationTarget(
        target_name="focal_length",
        target_value=100.0
    )


class TestOptimizeGenetic:
    """Tests for GlobalOptimizer.optimize_genetic"""

    def test_genetic_basic_operation(self, simple_system, simple_variable, simple_target):
        """Test basic GA operation returns correct iteration count."""
        opt = GlobalOptimizer(simple_system, [simple_variable], [simple_target], seed=1)
        opt._evaluate_design = Mock(return_value=5.0)
        opt.optimize_simplex = Mock(return_value=types.SimpleNamespace(
            best_values=[50.0], final_merit=0.5, iterations=3,
            variable_history=[{"x": 1.0}], merit_history=[9.0]
        ))
        opt._finalize = Mock(side_effect=lambda **kw: kw)
        
        result = opt.optimize_genetic(population_size=5, generations=2)
        
        assert result["iterations"] == 2 + 3
        assert isinstance(result["merit_history"], list)

    def test_genetic_population_size_zero_raises(self, simple_system, simple_variable, simple_target):
        """Test that population_size=0 raises IndexError."""
        opt = GlobalOptimizer(simple_system, [simple_variable], [simple_target], seed=1)
        
        with pytest.raises(IndexError):
            opt.optimize_genetic(population_size=0, generations=1)

    def test_genetic_callback_exception_propagates(self, simple_system, simple_variable, simple_target):
        """Test that exceptions from callback propagate."""
        opt = GlobalOptimizer(simple_system, [simple_variable], [simple_target], seed=1)
        opt._evaluate_design = Mock(return_value=5.0)
        
        def callback_raises(*args):
            raise RuntimeError("user cancelled")
        
        with pytest.raises(RuntimeError, match="user cancelled"):
            opt.optimize_genetic(population_size=5, generations=1, callback=callback_raises)

    def test_genetic_generations_zero(self, simple_system, simple_variable, simple_target):
        """Test GA with generations=0 skips GA loop."""
        opt = GlobalOptimizer(simple_system, [simple_variable], [simple_target], seed=1)
        opt._evaluate_design = Mock(return_value=5.0)
        opt.optimize_simplex = Mock(return_value=types.SimpleNamespace(
            best_values=[50.0], final_merit=0.5, iterations=0,
            variable_history=[], merit_history=[]
        ))
        opt._finalize = Mock(side_effect=lambda **kw: kw)
        
        result = opt.optimize_genetic(population_size=5, generations=0)
        
        assert result["merit_history"] == []
        assert result["iterations"] == 0

    def test_genetic_seed_reproducible(self, simple_system, simple_variable, simple_target):
        """Test that same seed produces reproducible results."""
        opts = [
            GlobalOptimizer(simple_system, [simple_variable], [simple_target], seed=42),
            GlobalOptimizer(simple_system, [simple_variable], [simple_target], seed=42)
        ]
        
        for opt in opts:
            opt._evaluate_design = Mock(return_value=5.0)
            opt.optimize_simplex = Mock(return_value=types.SimpleNamespace(
                best_values=[50.0], final_merit=0.5, iterations=3,
                variable_history=[], merit_history=[]
            ))
            opt._finalize = Mock(side_effect=lambda **kw: kw)
        
        result1 = opts[0].optimize_genetic(population_size=10, generations=2)
        result2 = opts[1].optimize_genetic(population_size=10, generations=2)
        
        assert result1["iterations"] == result2["iterations"]


class TestOptimizeSimulatedAnnealing:
    """Tests for GlobalOptimizer.optimize_simulated_annealing"""

    def test_sa_basic_operation(self, simple_system, simple_variable, simple_target):
        """Test basic SA operation with stubs."""
        opt = GlobalOptimizer(simple_system, [simple_variable], [simple_target], seed=1)
        opt._evaluate_design = Mock(return_value=5.0)
        opt.optimize_simplex = Mock(return_value=types.SimpleNamespace(
            best_values=[50.0], final_merit=0.5, iterations=3,
            variable_history=[], merit_history=[]
        ))
        opt._finalize = Mock(side_effect=lambda **kw: kw)
        
        result = opt.optimize_simulated_annealing(max_iterations=5, initial_temperature=100.0)
        
        assert "iterations" in result
        assert result["iterations"] > 0

    def test_sa_initial_temperature_negative(self, simple_system, simple_variable, simple_target, caplog):
        """Test that negative initial_temperature is clamped with warning."""
        opt = GlobalOptimizer(simple_system, [simple_variable], [simple_target], seed=1)
        opt._evaluate_design = Mock(return_value=5.0)
        opt.optimize_simplex = Mock(return_value=types.SimpleNamespace(
            best_values=[50.0], final_merit=0.5, iterations=0,
            variable_history=[], merit_history=[]
        ))
        opt._finalize = Mock(side_effect=lambda **kw: kw)
        
        with caplog.at_level(logging.WARNING, logger="openlens.global_optimizer"):
            opt.optimize_simulated_annealing(max_iterations=1, initial_temperature=-5.0)
        
        assert any("initial_temperature" in record.message for record in caplog.records)

    def test_sa_max_iterations_zero(self, simple_system, simple_variable, simple_target):
        """Test SA with max_iterations=0 skips SA loop."""
        opt = GlobalOptimizer(simple_system, [simple_variable], [simple_target], seed=1)
        opt._evaluate_design = Mock(return_value=5.0)
        opt.optimize_simplex = Mock(return_value=types.SimpleNamespace(
            best_values=[50.0], final_merit=0.5, iterations=1,
            variable_history=[], merit_history=[]
        ))
        opt._finalize = Mock(side_effect=lambda **kw: kw)
        
        result = opt.optimize_simulated_annealing(max_iterations=0)
        
        assert result["iterations"] >= 1

    def test_sa_callback_exception_propagates(self, simple_system, simple_variable, simple_target):
        """Test that callback exceptions propagate from SA."""
        opt = GlobalOptimizer(simple_system, [simple_variable], [simple_target], seed=1)
        opt._evaluate_design = Mock(return_value=5.0)
        
        def callback_raises(*args):
            raise ValueError("callback error")
        
        with pytest.raises(ValueError, match="callback error"):
            opt.optimize_simulated_annealing(
                max_iterations=1, initial_temperature=100.0, callback=callback_raises
            )

    def test_sa_seed_reproducible(self, simple_system, simple_variable, simple_target):
        """Test that same seed produces identical SA sequences."""
        opts = [
            GlobalOptimizer(simple_system, [simple_variable], [simple_target], seed=42),
            GlobalOptimizer(simple_system, [simple_variable], [simple_target], seed=42)
        ]
        
        for opt in opts:
            opt._evaluate_design = Mock(return_value=5.0)
            opt.optimize_simplex = Mock(return_value=types.SimpleNamespace(
                best_values=[50.0], final_merit=0.5, iterations=0,
                variable_history=[], merit_history=[]
            ))
            opt._finalize = Mock(side_effect=lambda **kw: kw)
        
        result1 = opts[0].optimize_simulated_annealing(max_iterations=5)
        result2 = opts[1].optimize_simulated_annealing(max_iterations=5)
        
        assert result1["iterations"] == result2["iterations"]


class TestOptimizeSimplex:
    """Tests for LensOptimizer.optimize_simplex"""

    def test_simplex_no_variables_raises(self, simple_system, simple_target):
        """Test that simplex with zero variables raises ValueError."""
        opt = LensOptimizer(simple_system, [], [simple_target])
        
        with pytest.raises(ValueError, match="at least one variable"):
            opt.optimize_simplex()

    def test_simplex_start_wrong_length_raises(self, simple_system, simple_variable, simple_target):
        """Test that simplex with mismatched start length raises ValueError."""
        opt = LensOptimizer(simple_system, [simple_variable], [simple_target])
        
        with pytest.raises(ValueError, match="value"):
            opt.optimize_simplex(start=[1.0, 2.0, 3.0])

    def test_simplex_max_iterations_zero(self, simple_system, simple_variable, simple_target):
        """Test simplex with max_iterations=0."""
        opt = LensOptimizer(simple_system, [simple_variable], [simple_target])
        opt._evaluate_design = Mock(return_value=5.0)
        opt._finalize = Mock(side_effect=lambda **kw: OptimizationResult(
            success=True, iterations=kw["iterations"], initial_merit=kw["initial_merit"],
            final_merit=kw["final_merit"], improvement=0.0,
            optimized_system=kw.get("optimized_system"),
            variable_history=kw.get("variable_history", []),
            merit_history=kw.get("merit_history", [])
        ))
        
        result = opt.optimize_simplex(max_iterations=0)
        
        assert result.iterations == 1


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass"""

    def test_optimization_result_construction(self, simple_system):
        """Test OptimizationResult construction with all required fields."""
        result = OptimizationResult(
            success=True, iterations=10, initial_merit=100.0, final_merit=10.0,
            improvement=90.0, optimized_system=simple_system, message="ok",
            variable_history=[{"x": 1.0}], merit_history=[100.0, 50.0]
        )
        
        assert result.success is True
        assert result.iterations == 10
        assert result.initial_merit == 100.0
        assert result.final_merit == 10.0
        assert result.improvement == 90.0
        assert result.message == "ok"
        assert len(result.variable_history) == 1
        assert len(result.merit_history) == 2

    def test_optimization_result_default_values(self, simple_system):
        """Test OptimizationResult default field values."""
        result = OptimizationResult(
            success=False, iterations=1, initial_merit=100.0,
            final_merit=100.0, improvement=0.0, optimized_system=simple_system
        )
        
        assert result.message == ""
        assert result.variable_history == []
        assert result.merit_history == []
        assert result.best_values == []

    def test_optimization_result_missing_required_raises(self):
        """Test that missing required field raises TypeError."""
        with pytest.raises(TypeError):
            OptimizationResult(
                success=True, iterations=1, initial_merit=100.0,
                final_merit=10.0, improvement=90.0
            )

    def test_optimization_result_equality(self, simple_system):
        """Test OptimizationResult equality comparison."""
        r1 = OptimizationResult(
            success=True, iterations=10, initial_merit=100.0, final_merit=10.0,
            improvement=90.0, optimized_system=simple_system, message="ok"
        )
        r2 = OptimizationResult(
            success=True, iterations=10, initial_merit=100.0, final_merit=10.0,
            improvement=90.0, optimized_system=simple_system, message="ok"
        )
        
        assert r1 == r2

import pytest
import types
import logging
from unittest.mock import Mock

try:
    from openlens.global_optimizer import GlobalOptimizer
    HAS_OPENLENS = True
except ImportError:
    HAS_OPENLENS = False

pytestmark = pytest.mark.skipif(not HAS_OPENLENS, reason="openlens not installed")


def create_mock_variable(name="x", current=50.0, min_val=20.0, max_val=100.0, step_size=5.0):
    """Create a mock OptimizationVariable."""
    var = Mock()
    var.name = name
    var.current_value = current
    var.min_value = min_val
    var.max_value = max_val
    var.step_size = step_size
    var.clamp = lambda x: max(min_val, min(max_val, x))
    return var


def setup_optimizer(num_vars=1, seed=42):
    """Create a GlobalOptimizer with stubbed methods."""
    variables = [create_mock_variable(name=f"var{i}") for i in range(num_vars)]
    system = Mock()
    targets = [Mock()]
    
    opt = GlobalOptimizer(system, variables, targets, seed=seed)
    
    # Stub the unseen methods
    opt._evaluate_design = Mock(return_value=1.0)
    opt.optimize_simplex = Mock(return_value=types.SimpleNamespace(
        best_values=[v.current_value for v in variables],
        final_merit=0.5,
        iterations=3,
        variable_history=[{v.name: v.current_value for v in variables}],
        merit_history=[0.8, 0.6, 0.5]
    ))
    opt._finalize = Mock(side_effect=lambda **kw: types.SimpleNamespace(**kw))
    
    return opt


class TestOptimizeGenetic:
    """Tests for GlobalOptimizer.optimize_genetic"""
    
    def test_genetic_basic_execution(self):
        """Test basic genetic algorithm execution."""
        opt = setup_optimizer()
        opt._evaluate_design.return_value = 1.0
        
        result = opt.optimize_genetic(population_size=10, generations=5)
        
        assert result.iterations == 8
        assert opt.optimize_simplex.called
    
    def test_genetic_zero_generations(self):
        """Test GA with zero generations."""
        opt = setup_optimizer()
        opt._evaluate_design.return_value = 1.0
        
        result = opt.optimize_genetic(population_size=10, generations=0)
        
        assert result.merit_history == []
        assert "0 generations" in result.converged_message
    
    def test_genetic_population_size_zero_raises(self):
        """Test population_size=0 raises IndexError."""
        opt = setup_optimizer()
        with pytest.raises(IndexError):
            opt.optimize_genetic(population_size=0)
    
    def test_genetic_callback_invoked(self):
        """Test callback is invoked per generation."""
        opt = setup_optimizer()
        opt._evaluate_design.return_value = 1.0
        callback = Mock()
        
        result = opt.optimize_genetic(population_size=5, generations=2, callback=callback)
        
        # (1 + population_size) calls per generation = 12 in GA loop
        ga_calls = (1 + 5) * 2
        assert callback.call_count >= ga_calls
    
    def test_genetic_callback_exception_propagates(self):
        """Test exception in callback propagates."""
        opt = setup_optimizer()
        def bad_callback(*args):
            raise RuntimeError("cancelled")
        
        with pytest.raises(RuntimeError, match="cancelled"):
            opt.optimize_genetic(population_size=5, generations=1, callback=bad_callback)
    
    def test_genetic_merit_history_length_and_monotonicity(self):
        """Test merit_history length equals generations and is non-increasing."""
        opt = setup_optimizer()
        opt._evaluate_design.side_effect = [1.0] * 200
        
        result = opt.optimize_genetic(population_size=5, generations=10)
        
        assert len(result.merit_history) == 10
        for i in range(1, len(result.merit_history)):
            assert result.merit_history[i] <= result.merit_history[i-1]
    
    def test_genetic_initial_merit_and_variables_unchanged(self):
        """Test initial_merit is correct and variables are not mutated."""
        var = create_mock_variable(current=50.0)
        system = Mock()
        targets = [Mock()]
        opt = GlobalOptimizer(system, [var], targets, seed=42)
        
        opt._evaluate_design = Mock(return_value=2.5)
        opt.optimize_simplex = Mock(return_value=types.SimpleNamespace(
            best_values=[50.0],
            final_merit=0.5,
            iterations=0,
            variable_history=[],
            merit_history=[]
        ))
        opt._finalize = Mock(side_effect=lambda **kw: types.SimpleNamespace(**kw))
        
        original_current = var.current_value
        result = opt.optimize_genetic(population_size=5, generations=2)
        
        assert result.initial_merit == 2.5
        assert var.current_value == original_current
    
    def test_genetic_message_strings(self):
        """Test converged and out_of_bounds messages contain expected format."""
        opt = setup_optimizer()
        opt._evaluate_design.return_value = 1.0
        
        result = opt.optimize_genetic(population_size=5, generations=7)
        
        assert "GA Completed 7 generations" in result.converged_message
        assert "GA finished 7 generations" in result.out_of_bounds_message


class TestOptimizeSimulatedAnnealing:
    """Tests for GlobalOptimizer.optimize_simulated_annealing"""
    
    def test_sa_basic_execution(self):
        """Test basic simulated annealing execution."""
        opt = setup_optimizer()
        opt._evaluate_design.return_value = 1.0
        
        result = opt.optimize_simulated_annealing(max_iterations=10)
        
        assert result.iterations >= 10
        assert opt.optimize_simplex.called
    
    def test_sa_zero_iterations(self):
        """Test SA with zero iterations."""
        opt = setup_optimizer()
        opt._evaluate_design.return_value = 1.0
        
        result = opt.optimize_simulated_annealing(max_iterations=0)
        
        assert "after 0 iterations" in result.converged_message
    
    def test_sa_negative_temperature_clamped_and_warned(self, caplog):
        """Test negative initial_temperature is clamped and warning logged."""
        opt = setup_optimizer()
        opt._evaluate_design.return_value = 1.0
        
        with caplog.at_level(logging.WARNING):
            result = opt.optimize_simulated_annealing(
                max_iterations=5,
                initial_temperature=-10.0
            )
        
        assert any("initial_temperature" in record.message and "not positive" in record.message
                   for record in caplog.records)
    
    def test_sa_callback_invoked_and_exception_propagates(self):
        """Test callback is invoked and exceptions propagate."""
        opt = setup_optimizer()
        opt._evaluate_design.return_value = 1.0
        callback = Mock()
        
        result = opt.optimize_simulated_annealing(max_iterations=5, callback=callback)
        assert callback.call_count >= 5
        
        # Test exception propagation
        def bad_callback(*args):
            raise ValueError("stop")
        
        with pytest.raises(ValueError, match="stop"):
            opt.optimize_simulated_annealing(max_iterations=5, callback=bad_callback)
    
    def test_sa_initial_merit_and_variables_unchanged(self):
        """Test initial_merit is correct and variables are not mutated."""
        var = create_mock_variable(current=50.0)
        system = Mock()
        targets = [Mock()]
        opt = GlobalOptimizer(system, [var], targets, seed=42)
        
        opt._evaluate_design = Mock(return_value=3.5)
        opt.optimize_simplex = Mock(return_value=types.SimpleNamespace(
            best_values=[50.0],
            final_merit=0.5,
            iterations=0,
            variable_history=[],
            merit_history=[]
        ))
        opt._finalize = Mock(side_effect=lambda **kw: types.SimpleNamespace(**kw))
        
        original_current = var.current_value
        result = opt.optimize_simulated_annealing(max_iterations=5)
        
        assert result.initial_merit == 3.5
        assert result.merit_history[0] == 3.5
        assert var.current_value == original_current
    
    def test_sa_message_strings(self):
        """Test converged and out_of_bounds message format."""
        opt = setup_optimizer()
        opt._evaluate_design.return_value = 1.0
        
        result = opt.optimize_simulated_annealing(max_iterations=5)
        
        assert "SA " in result.converged_message
        assert "iterations" in result.converged_message
        assert "SA finished" in result.out_of_bounds_message
    
    def test_sa_empty_variables_raises(self):
        """Test empty variables with max_iterations >= 1 raises ValueError."""
        system = Mock()
        targets = [Mock()]
        opt = GlobalOptimizer(system, [], targets, seed=42)
        
        opt._evaluate_design = Mock(return_value=1.0)
        opt.optimize_simplex = Mock(return_value=types.SimpleNamespace(
            best_values=[],
            final_merit=0.5,
            iterations=0,
            variable_history=[],
            merit_history=[]
        ))
        opt._finalize = Mock(side_effect=lambda **kw: types.SimpleNamespace(**kw))
        
        with pytest.raises(ValueError):
            opt.optimize_simulated_annealing(max_iterations=1)
    
    def test_sa_reproducibility_with_seed(self):
        """Test same seed produces deterministic results."""
        def create_opt(seed_val):
            var = create_mock_variable()
            system = Mock()
            targets = [Mock()]
            opt = GlobalOptimizer(system, [var], targets, seed=seed_val)
            opt._evaluate_design = Mock(side_effect=lambda v: (v[0] - 50.0) ** 2)
            opt.optimize_simplex = Mock(return_value=types.SimpleNamespace(
                best_values=[50.0],
                final_merit=0.0,
                iterations=0,
                variable_history=[],
                merit_history=[]
            ))
            opt._finalize = Mock(side_effect=lambda **kw: types.SimpleNamespace(**kw))
            return opt
        
        opt1 = create_opt(123)
        opt2 = create_opt(123)
        
        result1 = opt1.optimize_simulated_annealing(max_iterations=5, initial_temperature=10.0)
        result2 = opt2.optimize_simulated_annealing(max_iterations=5, initial_temperature=10.0)
        
        assert result1.best_values == result2.best_values
        assert result1.iterations == result2.iterations

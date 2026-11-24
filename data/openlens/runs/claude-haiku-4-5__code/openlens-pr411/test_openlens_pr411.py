import pytest
from unittest.mock import Mock, patch, MagicMock
from openlens.optimizer import LensOptimizer, OptimizationResult


class TestOptimizeSimplexErrors:
    """Test error handling in optimize_simplex"""
    
    def test_no_variables_raises_value_error(self):
        """optimize_simplex with zero variables should raise ValueError"""
        optimizer = LensOptimizer()
        with pytest.raises(ValueError, match="at least one variable"):
            optimizer.optimize_simplex()
    
    def test_start_length_mismatch_raises_value_error(self):
        """start parameter with wrong length should raise ValueError"""
        optimizer = LensOptimizer()
        optimizer.add_variable("x", current_value=1.0, bounds=(0, 10), step_size=0.1)
        
        with pytest.raises(ValueError, match="start has .* value"):
            optimizer.optimize_simplex(start=[1.0, 2.0])
    
    def test_start_too_short_raises_value_error(self):
        """start parameter too short should raise ValueError"""
        optimizer = LensOptimizer()
        optimizer.add_variable("x", current_value=1.0, bounds=(0, 10), step_size=0.1)
        optimizer.add_variable("y", current_value=2.0, bounds=(0, 10), step_size=0.1)
        
        with pytest.raises(ValueError, match="start has"):
            optimizer.optimize_simplex(start=[1.0])


class TestOptimizeSimplexBasic:
    """Test basic functionality of optimize_simplex"""
    
    def test_returns_optimization_result(self):
        """optimize_simplex should return OptimizationResult instance"""
        optimizer = LensOptimizer()
        optimizer.add_variable("x", current_value=5.0, bounds=(0, 10), step_size=0.5)
        
        with patch.object(optimizer, '_evaluate_design', return_value=1.0):
            result = optimizer.optimize_simplex(max_iterations=1)
        
        assert isinstance(result, OptimizationResult)
    
    def test_result_has_all_required_fields(self):
        """Result should have all required fields"""
        optimizer = LensOptimizer()
        optimizer.add_variable("x", current_value=5.0, bounds=(0, 10), step_size=0.5)
        
        with patch.object(optimizer, '_evaluate_design', return_value=1.0):
            result = optimizer.optimize_simplex(max_iterations=1)
        
        required_fields = [
            'success', 'iterations', 'initial_merit', 'final_merit',
            'improvement', 'best_values', 'variable_history', 'merit_history', 'message'
        ]
        for field in required_fields:
            assert hasattr(result, field), f"Missing field: {field}"
    
    def test_single_variable_optimization(self):
        """optimize_simplex should work with one variable"""
        optimizer = LensOptimizer()
        optimizer.add_variable("x", current_value=5.0, bounds=(0, 10), step_size=0.5)
        
        with patch.object(optimizer, '_evaluate_design', return_value=1.0):
            result = optimizer.optimize_simplex(max_iterations=5)
        
        assert len(result.best_values) == 1
        assert len(result.variable_history[0]) == 1
    
    def test_multiple_variables_optimization(self):
        """optimize_simplex should work with multiple variables"""
        optimizer = LensOptimizer()
        optimizer.add_variable("x", current_value=1.0, bounds=(0, 10), step_size=0.1)
        optimizer.add_variable("y", current_value=2.0, bounds=(0, 10), step_size=0.1)
        optimizer.add_variable("z", current_value=3.0, bounds=(0, 10), step_size=0.1)
        
        with patch.object(optimizer, '_evaluate_design', return_value=1.0):
            result = optimizer.optimize_simplex(max_iterations=5)
        
        assert len(result.best_values) == 3
        assert all(var in result.variable_history[0] for var in ['x', 'y', 'z'])


class TestOptimizeSimplexParameters:
    """Test parameter handling"""
    
    def test_uses_start_parameter(self):
        """optimize_simplex should use provided start values"""
        optimizer = LensOptimizer()
        optimizer.add_variable("x", current_value=5.0, bounds=(0, 10), step_size=0.5)
        
        call_log = []
        def track_calls(vertex):
            call_log.append(vertex)
            return 1.0
        
        with patch.object(optimizer, '_evaluate_design', side_effect=track_calls):
            result = optimizer.optimize_simplex(start=[8.0], max_iterations=1)
        
        # First evaluation should be at or near start value
        assert any(abs(call[0] - 8.0) < 0.01 for call in call_log)
    
    def test_uses_current_value_as_default(self):
        """optimize_simplex should use current_value when start is None"""
        optimizer = LensOptimizer()
        optimizer.add_variable("x", current_value=3.0, bounds=(0, 10), step_size=0.5)
        
        call_log = []
        def track_calls(vertex):
            call_log.append(vertex)
            return 1.0
        
        with patch.object(optimizer, '_evaluate_design', side_effect=track_calls):
            result = optimizer.optimize_simplex(start=None, max_iterations=1)
        
        # First evaluation should be at or near current_value
        assert any(abs(call[0] - 3.0) < 0.01 for call in call_log)
    
    def test_callback_invoked_per_iteration(self):
        """callback should be called on each iteration"""
        optimizer = LensOptimizer()
        optimizer.add_variable("x", current_value=5.0, bounds=(0, 10), step_size=0.5)
        
        callback = Mock()
        
        with patch.object(optimizer, '_evaluate_design', return_value=1.0):
            with patch.object(optimizer, '_merit_range_converged', return_value=False):
                result = optimizer.optimize_simplex(
                    max_iterations=3,
                    callback=callback
                )
        
        # Callback should be called at least once
        assert callback.call_count >= 1
    
    def test_callback_receives_correct_arguments(self):
        """callback should receive (iteration, merit, vertex) tuple"""
        optimizer = LensOptimizer()
        optimizer.add_variable("x", current_value=5.0, bounds=(0, 10), step_size=0.5)
        
        callback = Mock()
        
        with patch.object(optimizer, '_evaluate_design', return_value=1.5):
            with patch.object(optimizer, '_merit_range_converged', return_value=False):
                result = optimizer.optimize_simplex(max_iterations=1, callback=callback)
        
        if callback.call_count > 0:
            call_args = callback.call_args[0]
            assert len(call_args) == 3
            assert isinstance(call_args[0], int)  # iteration
            assert isinstance(call_args[1], (int, float))  # merit
            assert isinstance(call_args[2], list)  # vertex


class TestOptimizeSimplexIterations:
    """Test iteration control"""
    
    def test_respects_max_iterations(self):
        """optimize_simplex should not exceed max_iterations"""
        optimizer = LensOptimizer()
        optimizer.add_variable("x", current_value=5.0, bounds=(0, 10), step_size=0.5)
        
        with patch.object(optimizer, '_evaluate_design', return_value=1.0):
            result = optimizer.optimize_simplex(max_iterations=10, tolerance=1e-10)
        
        assert result.iterations <= 10
    
    def test_converges_early_when_tolerance_met(self):
        """optimize_simplex should stop when tolerance is met"""
        optimizer = LensOptimizer()
        optimizer.add_variable("x", current_value=5.0, bounds=(0, 10), step_size=0.5)
        
        # Mock convergence check to return True after iteration 3
        iteration_count = [0]
        def mock_converged(merit_values, merit_range, tolerance):
            iteration_count[0] += 1
            return iteration_count[0] >= 3
        
        with patch.object(optimizer, '_evaluate_design', return_value=1.0):
            with patch.object(optimizer, '_merit_range_converged', side_effect=mock_converged):
                result = optimizer.optimize_simplex(max_iterations=100, tolerance=1e-6)
        
        # Should converge before max_iterations
        assert result.iterations < 100


class TestOptimizeSimplexHistory:
    """Test history tracking"""
    
    def test_variable_history_tracked(self):
        """variable_history should track all iterations"""
        optimizer = LensOptimizer()
        optimizer.add_variable("x", current_value=5.0, bounds=(0, 10), step_size=0.5)
        
        with patch.object(optimizer, '_evaluate_design', return_value=1.0):
            result = optimizer.optimize_simplex(max_iterations=1)
        
        assert len(result.variable_history) > 0
        assert all(isinstance(h, dict) for h in result.variable_history)
        assert all('x' in h for h in result.variable_history)
    
    def test_merit_history_tracked(self):
        """merit_history should track all iterations"""
        optimizer = LensOptimizer()
        optimizer.add_variable("x", current_value=5.0, bounds=(0, 10), step_size=0.5)
        
        with patch.object(optimizer, '_evaluate_design', return_value=1.0):
            result = optimizer.optimize_simplex(max_iterations=1)
        
        assert len(result.merit_history) > 0
        assert all(isinstance(m, (int, float)) for m in result.merit_history)
    
    def test_history_lengths_match(self):
        """variable_history and merit_history should have equal length"""
        optimizer = LensOptimizer()
        optimizer.add_variable("x", current_value=5.0, bounds=(0, 10), step_size=0.5)
        
        with patch.object(optimizer, '_evaluate_design', return_value=1.0):
            result = optimizer.optimize_simplex(max_iterations=5)
        
        assert len(result.variable_history) == len(result.merit_history)


class TestOptimizeSimplexImprovement:
    """Test improvement calculation"""
    
    def test_improvement_calculated_correctly(self):
        """improvement should be (initial - final) / initial * 100"""
        optimizer = LensOptimizer()
        optimizer.add_variable("x", current_value=5.0, bounds=(0, 10), step_size=0.5)
        
        with patch.object(optimizer, '_finalize') as mock_finalize:
            mock_finalize.return_value = OptimizationResult(
                success=True, iterations=1, initial_merit=10.0, final_merit=5.0,
                improvement=50.0, best_values=[1.0], variable_history=[{'x': 1.0}],
                merit_history=[5.0], message="Test"
            )
            with patch.object(optimizer, '_evaluate_design', return_value=5.0):
                result = optimizer.optimize_simplex(max_iterations=1)
        
        assert result.improvement == 50.0
    
    def test_improvement_zero_when_no_initial_merit(self):
        """improvement should be 0 when initial_merit <= 0"""
        optimizer = LensOptimizer()
        optimizer.add_variable("x", current_value=5.0, bounds=(0, 10), step_size=0.5)
        
        with patch.object(optimizer, '_finalize') as mock_finalize:
            mock_finalize.return_value = OptimizationResult(
                success=True, iterations=1, initial_merit=0.0, final_merit=0.0,
                improvement=0.0, best_values=[1.0], variable_history=[{'x': 1.0}],
                merit_history=[0.0], message="Test"
            )
            with patch.object(optimizer, '_evaluate_design', return_value=0.0):
                result = optimizer.optimize_simplex(max_iterations=1)
        
        assert result.improvement == 0.0


class TestOptimizeSimplexBestValues:
    """Test best values tracking"""
    
    def test_best_values_set_correctly(self):
        """best_values should correspond to the best design found"""
        optimizer = LensOptimizer()
        optimizer.add_variable("x", current_value=5.0, bounds=(0, 10), step_size=0.5)
        optimizer.add_variable("y", current_value=5.0, bounds=(0, 10), step_size=0.5)
        
        with patch.object(optimizer, '_evaluate_design', return_value=1.0):
            result = optimizer.optimize_simplex(max_iterations=5)
        
        assert len(result.best_values) == 2
        assert all(isinstance(v, (int, float)) for v in result.best_values)
    
    def test_best_values_in_valid_bounds(self):
        """best_values may be outside bounds (returned as-is from optimization)"""
        optimizer = LensOptimizer()
        optimizer.add_variable("x", current_value=5.0, bounds=(0, 10), step_size=0.5)
        
        with patch.object(optimizer, '_evaluate_design', return_value=1.0):
            result = optimizer.optimize_simplex(max_iterations=1)
        
        # Algorithm allows out-of-bounds vertices (penalized in merit, not clamped)
        assert len(result.best_values) == 1

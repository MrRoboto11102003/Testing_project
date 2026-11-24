import pytest
from unittest.mock import Mock
from openlens.optimizer import LensOptimizer, OptimizationVariable
from openlens.optical_system import OpticalSystem


class TestLensOptimizerSimplexOptimize:
    """Test suite for LensOptimizer.optimize_simplex method."""
    
    @pytest.fixture
    def empty_system(self):
        """Empty optical system (no elements, minimal penalty)."""
        return OpticalSystem()
    
    @pytest.fixture
    def single_var_optimizer(self, empty_system):
        """Optimizer with one variable."""
        var = OptimizationVariable(
            name="x",
            parameter="dummy",
            current_value=5.0,
            min_bound=0.0,
            max_bound=10.0,
            step_size=1.0
        )
        return LensOptimizer(empty_system, [var], [])
    
    # ===== Input Validation =====
    
    def test_no_variables_raises_valueerror(self, empty_system):
        """Optimizer with no variables should raise ValueError."""
        opt = LensOptimizer(empty_system, [], [])
        with pytest.raises(ValueError, match="needs at least one variable"):
            opt.optimize_simplex()
    
    def test_wrong_start_length_raises_valueerror(self, single_var_optimizer):
        """Wrong start list length should raise ValueError with correct message."""
        with pytest.raises(ValueError, match="start has 2 value.*but there are 1"):
            single_var_optimizer.optimize_simplex(start=[1.0, 2.0])
    
    def test_empty_start_with_variables_raises_valueerror(self, single_var_optimizer):
        """Empty start list with variables should raise ValueError."""
        with pytest.raises(ValueError, match="start has 0 value"):
            single_var_optimizer.optimize_simplex(start=[])
    
    # ===== Convergence at Iteration Zero =====
    
    def test_converge_at_iteration_zero(self, single_var_optimizer):
        """When initial simplex has zero merit range, should converge immediately."""
        single_var_optimizer._evaluate_design = Mock(return_value=0.0)
        
        result = single_var_optimizer.optimize_simplex()
        
        assert result.success is True
        assert result.iterations == 1
        assert result.final_merit == 0.0
        assert result.best_values == [5.0]
        assert len(result.variable_history) == 1
        assert len(result.merit_history) == 1
        assert "Converged after 1 iterations" in result.message
    
    # ===== Max Iterations Zero =====
    
    def test_max_iterations_zero_evaluates_simplex_only(self, single_var_optimizer):
        """max_iterations=0 means loop never runs, only simplex initialization."""
        single_var_optimizer._evaluate_design = Mock(return_value=0.0)
        
        result = single_var_optimizer.optimize_simplex(max_iterations=0)
        
        # n_vars=1, so n+1=2 evaluations (initial simplex)
        assert result.iterations == 1
        assert single_var_optimizer._evaluate_design.call_count == 2
    
    def test_max_iterations_zero_callback_not_called(self, single_var_optimizer):
        """Callback should never be called when max_iterations=0."""
        single_var_optimizer._evaluate_design = Mock(return_value=0.0)
        callback = Mock()
        
        single_var_optimizer.optimize_simplex(max_iterations=0, callback=callback)
        
        assert callback.call_count == 0
    
    # ===== Callback Behavior =====
    
    def test_callback_called_with_correct_arguments(self, single_var_optimizer):
        """Callback should be called with (iteration_index, best_merit, best_values)."""
        single_var_optimizer._evaluate_design = Mock(return_value=1.5)
        callback = Mock()
        
        single_var_optimizer.optimize_simplex(max_iterations=2, callback=callback)
        
        assert callback.call_count >= 1
        # First call at iteration 0
        first_call = callback.call_args_list[0][0]
        iteration_idx, merit, values = first_call
        assert iteration_idx == 0
        assert merit == 1.5
        assert isinstance(values, list)
    
    # ===== History Tracking =====
    
    def test_variable_and_merit_history_same_length(self, single_var_optimizer):
        """variable_history and merit_history must always have equal length."""
        single_var_optimizer._evaluate_design = Mock(return_value=1.5)
        
        result = single_var_optimizer.optimize_simplex(max_iterations=5)
        
        assert len(result.variable_history) == len(result.merit_history)
    
    def test_merit_history_non_increasing(self, single_var_optimizer):
        """merit_history should be non-increasing (represents best-so-far)."""
        merits = [0.5, 1.0, 0.3, 0.8, 0.2, 0.1, 0.05]
        single_var_optimizer._evaluate_design = Mock(side_effect=merits)
        
        result = single_var_optimizer.optimize_simplex(max_iterations=5)
        
        for i in range(1, len(result.merit_history)):
            assert result.merit_history[i] <= result.merit_history[i-1]
    
    def test_variable_history_contains_all_variable_names(self, empty_system):
        """Each dict in variable_history should have correct variable names."""
        var1 = OptimizationVariable("alpha", "dummy", 1.0, 0.0, 10.0, step_size=0.5)
        var2 = OptimizationVariable("beta", "dummy", 2.0, 0.0, 10.0, step_size=0.5)
        opt = LensOptimizer(empty_system, [var1, var2], [])
        opt._evaluate_design = Mock(return_value=1.0)
        
        result = opt.optimize_simplex(max_iterations=1)
        
        for hist_dict in result.variable_history:
            assert set(hist_dict.keys()) == {"alpha", "beta"}
    
    # ===== Out-of-Bounds Behavior =====
    
    def test_out_of_bounds_variables_incur_penalty(self, empty_system):
        """Out-of-bounds variables should incur bound penalty."""
        var = OptimizationVariable("x", "dummy", 5.0, 0.0, 1.0, step_size=1.0)
        opt = LensOptimizer(empty_system, [var], [])
        
        result = opt.optimize_simplex()
        
        # Value 5.0 is far outside bounds [0, 1], penalty should be large
        assert result.initial_merit > 1e6
        assert result.success is False
    
    def test_abort_when_majority_vertices_stuck_out_of_bounds(self, empty_system):
        """Should abort if majority of vertices stuck out of bounds with low merit spread."""
        var = OptimizationVariable("x", "dummy", 5.0, 0.0, 1.0, step_size=1.0)
        opt = LensOptimizer(empty_system, [var], [])
        # Return same merit for all vertices, so merit_range < tolerance
        opt._evaluate_design = Mock(return_value=0.0)
        
        result = opt.optimize_simplex()
        
        # Both vertices (5.0, 6.0) out of bounds [0, 1], zero merit range
        assert result.success is False
        assert "stuck outside variable bounds" in result.message
        assert result.iterations == 1
    
    # ===== Result Structure =====
    
    def test_result_has_all_required_fields(self, single_var_optimizer):
        """OptimizationResult should have all documented fields."""
        single_var_optimizer._evaluate_design = Mock(return_value=1.0)
        
        result = single_var_optimizer.optimize_simplex()
        
        required_fields = [
            'success', 'iterations', 'initial_merit', 'final_merit',
            'improvement', 'best_values', 'optimized_system',
            'variable_history', 'merit_history', 'message'
        ]
        for field in required_fields:
            assert hasattr(result, field), f"Missing field: {field}"
    
    def test_improvement_calculated_correctly(self, empty_system):
        """improvement = (initial - final) / initial * 100."""
        var = OptimizationVariable("x", "dummy", 0.0, -100.0, 100.0, step_size=1.0)
        opt = LensOptimizer(empty_system, [var], [])
        # Merits: initial=10.0, vertex2=11.0, then converge to ~2.0
        opt._evaluate_design = Mock(side_effect=[10.0, 11.0, 2.0, 1.0])
        
        result = opt.optimize_simplex(max_iterations=5)
        
        expected = (result.initial_merit - result.final_merit) / result.initial_merit * 100
        assert abs(result.improvement - expected) < 0.1
    
    def test_improvement_is_zero_when_initial_merit_zero(self, single_var_optimizer):
        """improvement should be int 0 when initial_merit is 0 (avoid division by zero)."""
        single_var_optimizer._evaluate_design = Mock(return_value=0.0)
        
        result = single_var_optimizer.optimize_simplex()
        
        assert result.improvement == 0
    
    # ===== Search Functionality =====
    
    def test_simple_quadratic_function_optimization(self, empty_system):
        """Test that simplex can optimize a simple (x-3)^2 function."""
        var = OptimizationVariable("x", "dummy", 0.0, -100.0, 100.0, step_size=1.0)
        opt = LensOptimizer(empty_system, [var], [])
        opt._evaluate_design = lambda vals: (vals[0] - 3.0) ** 2
        
        result = opt.optimize_simplex(max_iterations=200, tolerance=1e-6)
        
        assert result.success is True
        # Optimum should be near x=3
        assert abs(result.best_values[0] - 3.0) < 0.01
        assert result.final_merit < result.initial_merit
        assert result.final_merit < 0.1
    
    # ===== Side Effects =====
    
    def test_original_system_not_mutated(self, single_var_optimizer):
        """Original system should not be modified by optimization."""
        original_system = single_var_optimizer.system
        single_var_optimizer._evaluate_design = Mock(return_value=0.0)
        
        single_var_optimizer.optimize_simplex()
        
        assert single_var_optimizer.system is original_system
    
    def test_variable_current_values_not_mutated(self, single_var_optimizer):
        """Variable current_values should not be changed by optimize_simplex."""
        original_value = single_var_optimizer.variables[0].current_value
        single_var_optimizer._evaluate_design = Mock(return_value=0.0)
        
        single_var_optimizer.optimize_simplex(start=[7.5])
        
        assert single_var_optimizer.variables[0].current_value == original_value
    
    def test_optimized_system_is_separate_copy(self, single_var_optimizer):
        """result.optimized_system should be a different object from original."""
        single_var_optimizer._evaluate_design = Mock(return_value=0.0)
        
        result = single_var_optimizer.optimize_simplex()
        
        assert result.optimized_system is not single_var_optimizer.system

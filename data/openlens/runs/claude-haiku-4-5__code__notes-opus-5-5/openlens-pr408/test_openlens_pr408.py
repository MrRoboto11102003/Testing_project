import pytest
from openlens import OpticalSystem, Lens
from openlens.optimizer import LensOptimizer, OptimizationVariable, OptimizationTarget


class SimpleMerit:
    """Test merit function based on lens thickness"""
    def __init__(self, func):
        self.func = func
    
    def evaluate(self, system):
        thickness = system.elements[0].lens.thickness
        return self.func(thickness)


class TestOptimizeGradientDescent:
    
    @pytest.fixture
    def system_and_var(self):
        """Create system and variable for testing"""
        system = OpticalSystem()
        system.add_lens(Lens(thickness=5.0))
        variable = OptimizationVariable("thickness", 0, "thickness", 5.0, 1.0, 20.0)
        return system, variable
    
    def test_smooth_quadratic_converges(self, system_and_var):
        """Test convergence on smooth quadratic (x-3)^2"""
        system, variable = system_and_var
        merit_fn = SimpleMerit(lambda x: (x - 3.0) ** 2)
        
        optimizer = LensOptimizer(system, [variable], [OptimizationTarget("system_length", 0.0)])
        optimizer.merit_function = merit_fn
        
        result = optimizer.optimize_gradient_descent(max_iterations=100, learning_rate=0.1, tolerance=1e-6)
        
        assert result.success is True
        assert result.iterations > 0
        assert result.final_merit < 1e-4
        assert result.initial_merit == pytest.approx((5.0 - 3.0) ** 2)
        assert result.improvement > 90
        assert len(result.merit_history) == len(result.variable_history)
    
    def test_constant_merit_zero_gradient(self, system_and_var):
        """Test zero gradient (constant merit function)"""
        system, variable = system_and_var
        merit_fn = SimpleMerit(lambda x: 2.5)
        
        optimizer = LensOptimizer(system, [variable], [OptimizationTarget("system_length", 0.0)])
        optimizer.merit_function = merit_fn
        
        result = optimizer.optimize_gradient_descent(max_iterations=100, learning_rate=0.1, tolerance=1e-6)
        
        assert result.iterations == 1
        assert len(result.merit_history) == 2
        assert result.merit_history[0] == result.merit_history[1] == 2.5
        assert result.improvement == 0
        assert "Completed" in result.message
    
    def test_stall_at_non_differentiable_point(self, system_and_var):
        """Test stalling when starting at minimum of |x-3|"""
        system, variable = system_and_var
        variable.current_value = 3.0
        merit_fn = SimpleMerit(lambda x: abs(x - 3.0))
        
        optimizer = LensOptimizer(system, [variable], [OptimizationTarget("system_length", 0.0)])
        optimizer.merit_function = merit_fn
        
        result = optimizer.optimize_gradient_descent(max_iterations=100, learning_rate=0.1, tolerance=1e-6)
        
        assert result.iterations == 1
        assert len(result.merit_history) == 1
        assert len(result.variable_history) == 1
        assert result.final_merit == 0.0
        assert result.improvement == 0
    
    def test_max_iterations_zero_no_loop(self, system_and_var):
        """Test that max_iterations=0 runs no loop iterations"""
        system, variable = system_and_var
        merit_fn = SimpleMerit(lambda x: (x - 3.0) ** 2)
        
        optimizer = LensOptimizer(system, [variable], [OptimizationTarget("system_length", 0.0)])
        optimizer.merit_function = merit_fn
        
        result = optimizer.optimize_gradient_descent(max_iterations=0)
        
        assert result.iterations == 1
        assert len(result.merit_history) == 1
        assert len(result.variable_history) == 1
        assert result.final_merit == result.initial_merit
        assert result.improvement == 0
        assert result.message == "Completed 1 iterations"
    
    def test_negative_max_iterations(self, system_and_var):
        """Test that negative max_iterations results in no loop iterations"""
        system, variable = system_and_var
        merit_fn = SimpleMerit(lambda x: (x - 3.0) ** 2)
        
        optimizer = LensOptimizer(system, [variable], [OptimizationTarget("system_length", 0.0)])
        optimizer.merit_function = merit_fn
        
        result = optimizer.optimize_gradient_descent(max_iterations=-1)
        
        assert result.iterations == 1
        assert len(result.merit_history) == 1
    
    def test_out_of_bounds_initial_value(self):
        """Test starting outside variable bounds"""
        system = OpticalSystem()
        system.add_lens(Lens(thickness=25.0))
        variable = OptimizationVariable("thickness", 0, "thickness", 25.0, 1.0, 20.0)
        
        merit_fn = SimpleMerit(lambda x: (x - 3.0) ** 2)
        optimizer = LensOptimizer(system, [variable], [OptimizationTarget("system_length", 0.0)])
        optimizer.merit_function = merit_fn
        
        result = optimizer.optimize_gradient_descent(max_iterations=0)
        
        assert result.success is False
        assert "outside variable bounds" in result.message
        assert "Finished 1 iterations" in result.message
    
    def test_zero_variables_allowed(self):
        """Test that zero variables is allowed"""
        system = OpticalSystem()
        system.add_lens(Lens(thickness=5.0))
        
        merit_fn = SimpleMerit(lambda x: 1.0)
        optimizer = LensOptimizer(system, [], [OptimizationTarget("system_length", 0.0)])
        optimizer.merit_function = merit_fn
        
        result = optimizer.optimize_gradient_descent(max_iterations=100)
        
        assert result.iterations == 1
        assert result.success is True
        assert len(result.variable_history) == 1
        assert result.variable_history[0] == {}
        assert result.message == "Completed 1 iterations"
    
    def test_tolerance_stops_early(self, system_and_var):
        """Test that loose tolerance causes early convergence"""
        system, variable = system_and_var
        merit_fn = SimpleMerit(lambda x: (x - 3.0) ** 2)
        
        optimizer = LensOptimizer(system, [variable], [OptimizationTarget("system_length", 0.0)])
        optimizer.merit_function = merit_fn
        
        result = optimizer.optimize_gradient_descent(max_iterations=1000, learning_rate=0.1, tolerance=1.0)
        
        assert result.iterations <= 20
        assert result.success is True
    
    def test_history_lengths_and_monotonicity(self, system_and_var):
        """Test that histories are consistent and merit is non-increasing"""
        system, variable = system_and_var
        merit_fn = SimpleMerit(lambda x: (x - 3.0) ** 2)
        
        optimizer = LensOptimizer(system, [variable], [OptimizationTarget("system_length", 0.0)])
        optimizer.merit_function = merit_fn
        
        result = optimizer.optimize_gradient_descent(max_iterations=100, tolerance=1e-6)
        
        assert len(result.variable_history) == len(result.merit_history)
        assert result.variable_history[0]["thickness"] == 5.0
        # Merit history should be non-increasing
        for i in range(1, len(result.merit_history)):
            assert result.merit_history[i] <= result.merit_history[i-1]
    
    def test_improvement_percentage_calculation(self, system_and_var):
        """Test improvement percentage is correctly computed"""
        system, variable = system_and_var
        merit_fn = SimpleMerit(lambda x: (x - 3.0) ** 2)
        
        optimizer = LensOptimizer(system, [variable], [OptimizationTarget("system_length", 0.0)])
        optimizer.merit_function = merit_fn
        
        result = optimizer.optimize_gradient_descent(max_iterations=100, tolerance=1e-6)
        
        expected = (result.initial_merit - result.final_merit) / result.initial_merit * 100
        assert result.improvement == pytest.approx(expected)
        assert result.improvement >= 0
    
    def test_improvement_zero_when_initial_merit_zero(self):
        """Test improvement is 0 when initial merit is 0"""
        system = OpticalSystem()
        system.add_lens(Lens(thickness=3.0))
        variable = OptimizationVariable("thickness", 0, "thickness", 3.0, 1.0, 20.0)
        
        merit_fn = SimpleMerit(lambda x: (x - 3.0) ** 2)
        optimizer = LensOptimizer(system, [variable], [OptimizationTarget("system_length", 0.0)])
        optimizer.merit_function = merit_fn
        
        result = optimizer.optimize_gradient_descent(max_iterations=100)
        
        assert result.initial_merit == 0.0
        assert result.improvement == 0
    
    def test_system_not_mutated(self, system_and_var):
        """Test that original system and variables are unchanged"""
        system, variable = system_and_var
        original_thickness = system.elements[0].lens.thickness
        original_current_value = variable.current_value
        merit_fn = SimpleMerit(lambda x: (x - 3.0) ** 2)
        
        optimizer = LensOptimizer(system, [variable], [OptimizationTarget("system_length", 0.0)])
        optimizer.merit_function = merit_fn
        
        result = optimizer.optimize_gradient_descent(max_iterations=100, tolerance=1e-6)
        
        assert system.elements[0].lens.thickness == original_thickness
        assert variable.current_value == original_current_value
        assert result.optimized_system is not system
    
    def test_best_values_always_empty(self, system_and_var):
        """Test that best_values is always an empty list"""
        system, variable = system_and_var
        merit_fn = SimpleMerit(lambda x: (x - 3.0) ** 2)
        
        optimizer = LensOptimizer(system, [variable], [OptimizationTarget("system_length", 0.0)])
        optimizer.merit_function = merit_fn
        
        result = optimizer.optimize_gradient_descent(max_iterations=100)
        
        assert result.best_values == []
    
    def test_merit_function_error_propagates(self, system_and_var):
        """Test that errors from merit_function propagate"""
        system, variable = system_and_var
        
        class FailingMerit:
            def evaluate(self, system):
                raise RuntimeError("Test merit evaluation error")
        
        optimizer = LensOptimizer(system, [variable], [OptimizationTarget("system_length", 0.0)])
        optimizer.merit_function = FailingMerit()
        
        with pytest.raises(RuntimeError, match="Test merit evaluation error"):
            optimizer.optimize_gradient_descent(max_iterations=10)

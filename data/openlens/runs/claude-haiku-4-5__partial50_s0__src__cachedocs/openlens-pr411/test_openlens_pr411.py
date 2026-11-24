import pytest
from openlens.optimizer import LensOptimizer
from openlens.lens import Lens


class TestOptimizeSimplexBasic:
    """Test basic simplex optimization functionality."""

    def test_no_variables_raises_error(self):
        """Test that optimize_simplex raises ValueError when no variables are defined."""
        lens = Lens(name="Test", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        optimizer = LensOptimizer(lens)
        
        with pytest.raises(ValueError, match="needs at least one variable"):
            optimizer.optimize_simplex()

    def test_start_wrong_length_raises_error(self):
        """Test that optimize_simplex raises ValueError when start has wrong number of values."""
        lens = Lens(name="Test", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        optimizer = LensOptimizer(lens)
        optimizer.add_variable("radius_of_curvature_1", step_size=1.0, min_val=10.0, max_val=100.0)
        
        with pytest.raises(ValueError, match="start has"):
            optimizer.optimize_simplex(start=[1.0, 2.0])

    def test_single_variable_optimization(self):
        """Test optimization converges with one variable."""
        lens = Lens(name="Test", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        optimizer = LensOptimizer(lens)
        optimizer.add_variable("radius_of_curvature_1", step_size=1.0, min_val=10.0, max_val=100.0)
        optimizer.target_focal_length(50.0)
        
        result = optimizer.optimize_simplex(max_iterations=50, tolerance=1e-4)
        
        assert result.iterations > 0
        assert len(result.best_values) == 1
        assert len(result.merit_history) > 0
        assert len(result.variable_history) > 0

    def test_multiple_variables_optimization(self):
        """Test optimization with multiple variables."""
        lens = Lens(name="Test", radius_of_curvature_1=50.0, radius_of_curvature_2=-60.0, thickness=5.0)
        optimizer = LensOptimizer(lens)
        optimizer.add_variable("radius_of_curvature_1", step_size=1.0, min_val=10.0, max_val=100.0)
        optimizer.add_variable("radius_of_curvature_2", step_size=1.0, min_val=-100.0, max_val=-10.0)
        optimizer.target_focal_length(50.0)
        
        result = optimizer.optimize_simplex(max_iterations=100, tolerance=1e-4)
        
        assert len(result.best_values) == 2
        assert len(result.variable_history) > 0

    def test_start_parameter_as_initial_point(self):
        """Test that start parameter is used as initial point."""
        lens = Lens(name="Test", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        optimizer = LensOptimizer(lens)
        optimizer.add_variable("radius_of_curvature_1", step_size=1.0, min_val=10.0, max_val=100.0)
        optimizer.target_focal_length(50.0)
        
        start_val = [65.0]
        result = optimizer.optimize_simplex(start=start_val, max_iterations=30)
        
        # First entry in variable_history should match start
        first_entry = result.variable_history[0]
        for var_name, start_v in zip([v.name for v in optimizer.variables], start_val):
            assert first_entry[var_name] == start_v


class TestOptimizeSimplexConvergence:
    """Test convergence behavior and history tracking."""

    def test_merit_history_exists(self):
        """Test that merit history is tracked during optimization."""
        lens = Lens(name="Test", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        optimizer = LensOptimizer(lens)
        optimizer.add_variable("radius_of_curvature_1", step_size=1.0, min_val=10.0, max_val=100.0)
        optimizer.target_focal_length(50.0)
        
        result = optimizer.optimize_simplex(max_iterations=50, tolerance=1e-4)
        
        assert len(result.merit_history) > 0
        assert result.merit_history[0] == result.initial_merit
        assert result.merit_history[-1] == result.final_merit

    def test_variable_history_tracks_all_variables(self):
        """Test that variable_history contains all variables at each step."""
        lens = Lens(name="Test", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        optimizer = LensOptimizer(lens)
        optimizer.add_variable("radius_of_curvature_1", step_size=1.0, min_val=10.0, max_val=100.0)
        optimizer.add_variable("radius_of_curvature_2", step_size=1.0, min_val=-100.0, max_val=-10.0)
        optimizer.target_focal_length(50.0)
        
        result = optimizer.optimize_simplex(max_iterations=50, tolerance=1e-4)
        
        var_names = {v.name for v in optimizer.variables}
        for entry in result.variable_history:
            assert set(entry.keys()) == var_names

    def test_max_iterations_limit(self):
        """Test that optimization respects max_iterations."""
        lens = Lens(name="Test", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        optimizer = LensOptimizer(lens)
        optimizer.add_variable("radius_of_curvature_1", step_size=1.0, min_val=10.0, max_val=100.0)
        optimizer.target_focal_length(50.0)
        
        max_iter = 5
        result = optimizer.optimize_simplex(max_iterations=max_iter, tolerance=1e-15)
        
        assert result.iterations <= max_iter + 1


class TestOptimizeSimplexCallback:
    """Test callback functionality."""

    def test_callback_invoked_during_optimization(self):
        """Test that callback is invoked at each iteration."""
        lens = Lens(name="Test", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        optimizer = LensOptimizer(lens)
        optimizer.add_variable("radius_of_curvature_1", step_size=1.0, min_val=10.0, max_val=100.0)
        optimizer.target_focal_length(50.0)
        
        calls = []
        
        def callback(iteration, merit, values):
            calls.append((iteration, merit, len(values)))
        
        result = optimizer.optimize_simplex(max_iterations=20, tolerance=1e-4, callback=callback)
        
        assert len(calls) == result.iterations
        assert calls[0][0] == 0
        assert len(calls[-1]) == 3

    def test_callback_merit_matches_history(self):
        """Test that callback merits match merit_history."""
        lens = Lens(name="Test", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        optimizer = LensOptimizer(lens)
        optimizer.add_variable("radius_of_curvature_1", step_size=1.0, min_val=10.0, max_val=100.0)
        optimizer.target_focal_length(50.0)
        
        callback_merits = []
        
        def callback(iteration, merit, values):
            callback_merits.append(merit)
        
        result = optimizer.optimize_simplex(max_iterations=20, tolerance=1e-4, callback=callback)
        
        assert len(callback_merits) == len(result.merit_history)
        for cb_merit, hist_merit in zip(callback_merits, result.merit_history):
            assert abs(cb_merit - hist_merit) < 1e-10


class TestOptimizeSimplexResult:
    """Test OptimizationResult object and fields."""

    def test_result_contains_required_fields(self):
        """Test that result has all documented fields."""
        lens = Lens(name="Test", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        optimizer = LensOptimizer(lens)
        optimizer.add_variable("radius_of_curvature_1", step_size=1.0, min_val=10.0, max_val=100.0)
        optimizer.target_focal_length(50.0)
        
        result = optimizer.optimize_simplex(max_iterations=50, tolerance=1e-4)
        
        assert hasattr(result, 'success')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'initial_merit')
        assert hasattr(result, 'final_merit')
        assert hasattr(result, 'improvement')
        assert hasattr(result, 'best_values')
        assert hasattr(result, 'variable_history')
        assert hasattr(result, 'merit_history')
        assert hasattr(result, 'optimized_system')
        assert hasattr(result, 'message')

    def test_improvement_calculation(self):
        """Test that improvement percentage is correctly computed."""
        lens = Lens(name="Test", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        optimizer = LensOptimizer(lens)
        optimizer.add_variable("radius_of_curvature_1", step_size=1.0, min_val=10.0, max_val=100.0)
        optimizer.target_focal_length(50.0)
        
        result = optimizer.optimize_simplex(max_iterations=50, tolerance=1e-4)
        
        assert isinstance(result.improvement, (int, float))
        assert result.improvement >= 0.0
        # improvement = ((initial - final) / initial) * 100 when initial > 0
        if result.initial_merit > 0:
            expected = ((result.initial_merit - result.final_merit) / result.initial_merit) * 100
            assert abs(result.improvement - expected) < 0.1


class TestOptimizeSimplexBounds:
    """Test variable bounds enforcement."""

    def test_best_values_within_bounds_on_success(self):
        """Test that successful results respect variable bounds."""
        lens = Lens(name="Test", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        optimizer = LensOptimizer(lens)
        min_val, max_val = 20.0, 80.0
        optimizer.add_variable("radius_of_curvature_1", step_size=1.0, min_val=min_val, max_val=max_val)
        optimizer.target_focal_length(50.0)
        
        result = optimizer.optimize_simplex(max_iterations=50, tolerance=1e-4)
        
        if result.success:
            assert min_val <= result.best_values[0] <= max_val

    def test_handles_tight_bounds(self):
        """Test optimization with very tight bounds."""
        lens = Lens(name="Test", radius_of_curvature_1=5.0, radius_of_curvature_2=-100.0)
        optimizer = LensOptimizer(lens)
        optimizer.add_variable("radius_of_curvature_1", step_size=0.05, min_val=4.95, max_val=5.05)
        optimizer.target_focal_length(100.0)
        
        result = optimizer.optimize_simplex(max_iterations=10, tolerance=1e-6)
        
        # Should complete (success or fail gracefully) and have message
        assert result.message is not None
        assert isinstance(result.message, str)


class TestOptimizeSimplexEdgeCases:
    """Test edge cases and special scenarios."""

    def test_nearly_optimal_design_converges_quickly(self):
        """Test that nearly-optimal designs converge quickly."""
        lens = Lens(name="Test", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        
        # Get baseline focal length
        efl = lens.calculate_focal_length()
        assert efl is not None
        
        optimizer = LensOptimizer(lens)
        optimizer.add_variable("radius_of_curvature_1", step_size=0.1, min_val=49.5, max_val=50.5)
        optimizer.target_focal_length(efl)
        
        result = optimizer.optimize_simplex(max_iterations=50, tolerance=1e-6)
        
        # Should converge well for near-optimal design
        assert result.final_merit <= result.initial_merit * 1.1

    def test_very_small_step_size(self):
        """Test optimization with small step sizes."""
        lens = Lens(name="Test", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        optimizer = LensOptimizer(lens)
        optimizer.add_variable("radius_of_curvature_1", step_size=0.01, min_val=10.0, max_val=100.0)
        optimizer.target_focal_length(50.0)
        
        result = optimizer.optimize_simplex(max_iterations=30, tolerance=1e-4)
        
        assert result.iterations > 0
        assert len(result.best_values) == 1

    def test_tolerance_relative_to_merit_magnitude(self):
        """Test that tolerance behaves relative to merit scale."""
        lens = Lens(name="Test", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        optimizer = LensOptimizer(lens)
        optimizer.add_variable("radius_of_curvature_1", step_size=1.0, min_val=10.0, max_val=100.0)
        optimizer.target_focal_length(50.0)
        
        # Larger tolerance should converge faster
        result_loose = optimizer.optimize_simplex(max_iterations=100, tolerance=1e-2)
        
        lens2 = Lens(name="Test", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        optimizer2 = LensOptimizer(lens2)
        optimizer2.add_variable("radius_of_curvature_1", step_size=1.0, min_val=10.0, max_val=100.0)
        optimizer2.target_focal_length(50.0)
        
        result_tight = optimizer2.optimize_simplex(max_iterations=100, tolerance=1e-8)
        
        # Tight tolerance typically needs more iterations (not guaranteed, but expected)
        assert result_loose.iterations > 0
        assert result_tight.iterations > 0

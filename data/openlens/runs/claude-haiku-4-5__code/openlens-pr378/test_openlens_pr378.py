import pytest
from unittest.mock import Mock, patch
from openlens.optimizer import LensOptimizer, OptimizationResult, OptimizationVariable
from openlens.global_optimizer import GlobalOptimizer
from openlens.optical_system import OpticalSystem


class TestOptimizationResultBasics:
    """Tests for OptimizationResult dataclass"""

    def test_optimization_result_creation_required_fields(self):
        """OptimizationResult can be created with required fields"""
        mock_system = Mock(spec=OpticalSystem)
        result = OptimizationResult(
            success=True,
            iterations=10,
            initial_merit=100.0,
            final_merit=50.0,
            improvement=50.0,
            optimized_system=mock_system,
        )
        assert result.success is True
        assert result.iterations == 10
        assert result.initial_merit == 100.0
        assert result.final_merit == 50.0
        assert result.improvement == 50.0

    def test_optimization_result_default_optional_fields(self):
        """OptimizationResult optional fields have correct defaults"""
        mock_system = Mock(spec=OpticalSystem)
        result = OptimizationResult(
            success=False,
            iterations=5,
            initial_merit=100.0,
            final_merit=100.0,
            improvement=0.0,
            optimized_system=mock_system,
        )
        assert result.variable_history == []
        assert result.merit_history == []
        assert result.message == ""
        assert result.best_values == []

    def test_optimization_result_all_fields_assignable(self):
        """OptimizationResult accepts all documented fields"""
        mock_system = Mock(spec=OpticalSystem)
        history = [{"radius": 1.0, "thickness": 2.0}]
        merit_hist = [100.0, 50.0]
        best_vals = [1.5, 2.5]
        
        result = OptimizationResult(
            success=True,
            iterations=2,
            initial_merit=100.0,
            final_merit=50.0,
            improvement=50.0,
            optimized_system=mock_system,
            variable_history=history,
            merit_history=merit_hist,
            message="Converged successfully",
            best_values=best_vals,
        )
        assert result.variable_history == history
        assert result.merit_history == merit_hist
        assert result.message == "Converged successfully"
        assert result.best_values == best_vals


class TestSimplexEdgeCases:
    """Tests for optimize_simplex documented edge cases"""

    def test_simplex_no_variables_raises_error(self):
        """optimize_simplex raises ValueError when n_vars==0"""
        with patch.object(LensOptimizer, 'optimize_simplex',
                         side_effect=ValueError("optimize_simplex needs at least one variable")):
            with pytest.raises(ValueError, match="at least one variable"):
                optimizer = Mock(spec=LensOptimizer)
                LensOptimizer.optimize_simplex(optimizer)

    def test_simplex_start_wrong_length_raises_error(self):
        """optimize_simplex raises ValueError if start length != n_vars"""
        with patch.object(LensOptimizer, 'optimize_simplex',
                         side_effect=ValueError("start has 3 value(s) but there are 2 variable(s)")):
            with pytest.raises(ValueError, match="start has.*value"):
                optimizer = Mock(spec=LensOptimizer)
                LensOptimizer.optimize_simplex(optimizer, start=[1.0, 2.0, 3.0])

    def test_simplex_returns_optimization_result(self):
        """optimize_simplex returns OptimizationResult instance"""
        with patch.object(LensOptimizer, 'optimize_simplex') as mock_method:
            expected = OptimizationResult(
                success=True, iterations=5, initial_merit=100.0, final_merit=90.0,
                improvement=10.0, optimized_system=Mock(spec=OpticalSystem),
                best_values=[0.5],
            )
            mock_method.return_value = expected
            result = LensOptimizer.optimize_simplex(Mock(spec=LensOptimizer))
            assert isinstance(result, OptimizationResult)
            assert result.iterations == 5

    def test_simplex_with_explicit_start_parameter(self):
        """optimize_simplex accepts and uses start parameter"""
        with patch.object(LensOptimizer, 'optimize_simplex') as mock_method:
            start_vals = [1.5, 2.5]
            mock_method.return_value = OptimizationResult(
                success=True, iterations=1, initial_merit=100.0, final_merit=95.0,
                improvement=5.0, optimized_system=Mock(spec=OpticalSystem),
                best_values=start_vals,
            )
            result = LensOptimizer.optimize_simplex(Mock(spec=LensOptimizer), start=start_vals)
            assert result.best_values == start_vals

    def test_simplex_with_callback(self):
        """optimize_simplex invokes callback during iterations"""
        callback = Mock()
        with patch.object(LensOptimizer, 'optimize_simplex') as mock_method:
            mock_method.return_value = OptimizationResult(
                success=True, iterations=2, initial_merit=100.0, final_merit=90.0,
                improvement=10.0, optimized_system=Mock(spec=OpticalSystem),
            )
            LensOptimizer.optimize_simplex(Mock(spec=LensOptimizer), callback=callback)
            mock_method.assert_called_once()

    def test_simplex_max_iterations_parameter(self):
        """optimize_simplex accepts max_iterations parameter"""
        with patch.object(LensOptimizer, 'optimize_simplex') as mock_method:
            mock_method.return_value = OptimizationResult(
                success=True, iterations=100, initial_merit=100.0, final_merit=50.0,
                improvement=50.0, optimized_system=Mock(spec=OpticalSystem),
            )
            LensOptimizer.optimize_simplex(Mock(spec=LensOptimizer), max_iterations=100)
            mock_method.assert_called_once()

    def test_simplex_tolerance_parameter(self):
        """optimize_simplex accepts tolerance parameter"""
        with patch.object(LensOptimizer, 'optimize_simplex') as mock_method:
            mock_method.return_value = OptimizationResult(
                success=True, iterations=10, initial_merit=100.0, final_merit=99.9,
                improvement=0.1, optimized_system=Mock(spec=OpticalSystem),
            )
            LensOptimizer.optimize_simplex(Mock(spec=LensOptimizer), tolerance=1e-6)
            mock_method.assert_called_once()


class TestGeneticAlgorithmBehavior:
    """Tests for optimize_genetic documented behavior"""

    def test_genetic_returns_optimization_result(self):
        """optimize_genetic returns OptimizationResult"""
        with patch.object(GlobalOptimizer, 'optimize_genetic') as mock_method:
            expected = OptimizationResult(
                success=True, iterations=50, initial_merit=100.0, final_merit=40.0,
                improvement=60.0, optimized_system=Mock(spec=OpticalSystem),
                best_values=[0.5],
            )
            mock_method.return_value = expected
            result = GlobalOptimizer.optimize_genetic(Mock(spec=GlobalOptimizer))
            assert isinstance(result, OptimizationResult)
            assert result.final_merit == 40.0

    def test_genetic_population_size_parameter(self):
        """optimize_genetic accepts population_size parameter"""
        with patch.object(GlobalOptimizer, 'optimize_genetic') as mock_method:
            mock_method.return_value = OptimizationResult(
                success=True, iterations=50, initial_merit=100.0, final_merit=50.0,
                improvement=50.0, optimized_system=Mock(spec=OpticalSystem),
            )
            GlobalOptimizer.optimize_genetic(Mock(spec=GlobalOptimizer), population_size=100)
            mock_method.assert_called_once()

    def test_genetic_generations_parameter(self):
        """optimize_genetic accepts generations parameter"""
        with patch.object(GlobalOptimizer, 'optimize_genetic') as mock_method:
            mock_method.return_value = OptimizationResult(
                success=True, iterations=0, initial_merit=100.0, final_merit=100.0,
                improvement=0.0, optimized_system=Mock(spec=OpticalSystem),
                merit_history=[],
            )
            GlobalOptimizer.optimize_genetic(Mock(spec=GlobalOptimizer), generations=0)
            mock_method.assert_called_once()

    def test_genetic_mutation_rate_parameter(self):
        """optimize_genetic accepts mutation_rate parameter"""
        with patch.object(GlobalOptimizer, 'optimize_genetic') as mock_method:
            mock_method.return_value = OptimizationResult(
                success=True, iterations=1, initial_merit=100.0, final_merit=100.0,
                improvement=0.0, optimized_system=Mock(spec=OpticalSystem),
            )
            GlobalOptimizer.optimize_genetic(Mock(spec=GlobalOptimizer), mutation_rate=0.2)
            mock_method.assert_called_once()

    def test_genetic_crossover_rate_parameter(self):
        """optimize_genetic accepts crossover_rate parameter"""
        with patch.object(GlobalOptimizer, 'optimize_genetic') as mock_method:
            mock_method.return_value = OptimizationResult(
                success=True, iterations=1, initial_merit=100.0, final_merit=100.0,
                improvement=0.0, optimized_system=Mock(spec=OpticalSystem),
            )
            GlobalOptimizer.optimize_genetic(Mock(spec=GlobalOptimizer), crossover_rate=0.8)
            mock_method.assert_called_once()

    def test_genetic_callback_parameter(self):
        """optimize_genetic accepts callback parameter"""
        callback = Mock()
        with patch.object(GlobalOptimizer, 'optimize_genetic') as mock_method:
            mock_method.return_value = OptimizationResult(
                success=True, iterations=1, initial_merit=100.0, final_merit=100.0,
                improvement=0.0, optimized_system=Mock(spec=OpticalSystem),
            )
            GlobalOptimizer.optimize_genetic(Mock(spec=GlobalOptimizer), callback=callback)
            mock_method.assert_called_once()


class TestSimulatedAnnealingBehavior:
    """Tests for optimize_simulated_annealing documented behavior"""

    def test_simulated_annealing_returns_optimization_result(self):
        """optimize_simulated_annealing returns OptimizationResult"""
        with patch.object(GlobalOptimizer, 'optimize_simulated_annealing') as mock_method:
            expected = OptimizationResult(
                success=True, iterations=100, initial_merit=100.0, final_merit=40.0,
                improvement=60.0, optimized_system=Mock(spec=OpticalSystem),
                best_values=[0.5],
            )
            mock_method.return_value = expected
            result = GlobalOptimizer.optimize_simulated_annealing(Mock(spec=GlobalOptimizer))
            assert isinstance(result, OptimizationResult)
            assert result.iterations == 100

    def test_simulated_annealing_max_iterations_parameter(self):
        """optimize_simulated_annealing accepts max_iterations parameter"""
        with patch.object(GlobalOptimizer, 'optimize_simulated_annealing') as mock_method:
            mock_method.return_value = OptimizationResult(
                success=True, iterations=1000, initial_merit=100.0, final_merit=50.0,
                improvement=50.0, optimized_system=Mock(spec=OpticalSystem),
            )
            GlobalOptimizer.optimize_simulated_annealing(
                Mock(spec=GlobalOptimizer), max_iterations=1000
            )
            mock_method.assert_called_once()

    def test_simulated_annealing_initial_temperature_parameter(self):
        """optimize_simulated_annealing accepts initial_temperature parameter"""
        with patch.object(GlobalOptimizer, 'optimize_simulated_annealing') as mock_method:
            mock_method.return_value = OptimizationResult(
                success=True, iterations=1, initial_merit=100.0, final_merit=100.0,
                improvement=0.0, optimized_system=Mock(spec=OpticalSystem),
            )
            GlobalOptimizer.optimize_simulated_annealing(
                Mock(spec=GlobalOptimizer), initial_temperature=50.0
            )
            mock_method.assert_called_once()

    def test_simulated_annealing_cooling_rate_parameter(self):
        """optimize_simulated_annealing accepts cooling_rate parameter"""
        with patch.object(GlobalOptimizer, 'optimize_simulated_annealing') as mock_method:
            mock_method.return_value = OptimizationResult(
                success=True, iterations=1, initial_merit=100.0, final_merit=100.0,
                improvement=0.0, optimized_system=Mock(spec=OpticalSystem),
            )
            GlobalOptimizer.optimize_simulated_annealing(
                Mock(spec=GlobalOptimizer), cooling_rate=0.95
            )
            mock_method.assert_called_once()

    def test_simulated_annealing_negative_temperature_clamped(self):
        """optimize_simulated_annealing clamps negative initial_temperature"""
        with patch.object(GlobalOptimizer, 'optimize_simulated_annealing') as mock_method:
            mock_method.return_value = OptimizationResult(
                success=True, iterations=1, initial_merit=100.0, final_merit=100.0,
                improvement=0.0, optimized_system=Mock(spec=OpticalSystem),
            )
            result = GlobalOptimizer.optimize_simulated_annealing(
                Mock(spec=GlobalOptimizer), initial_temperature=-10.0
            )
            assert result.success is True

    def test_simulated_annealing_zero_temperature_clamped(self):
        """optimize_simulated_annealing clamps zero initial_temperature"""
        with patch.object(GlobalOptimizer, 'optimize_simulated_annealing') as mock_method:
            mock_method.return_value = OptimizationResult(
                success=True, iterations=1, initial_merit=100.0, final_merit=100.0,
                improvement=0.0, optimized_system=Mock(spec=OpticalSystem),
            )
            result = GlobalOptimizer.optimize_simulated_annealing(
                Mock(spec=GlobalOptimizer), initial_temperature=0.0
            )
            assert result.success is True

    def test_simulated_annealing_zero_max_iterations_handled(self):
        """optimize_simulated_annealing handles zero max_iterations"""
        with patch.object(GlobalOptimizer, 'optimize_simulated_annealing') as mock_method:
            mock_method.return_value = OptimizationResult(
                success=True, iterations=0, initial_merit=100.0, final_merit=100.0,
                improvement=0.0, optimized_system=Mock(spec=OpticalSystem),
            )
            result = GlobalOptimizer.optimize_simulated_annealing(
                Mock(spec=GlobalOptimizer), max_iterations=0
            )
            assert result.iterations == 0

    def test_simulated_annealing_cooling_rate_none_derived(self):
        """optimize_simulated_annealing derives cooling_rate when None"""
        with patch.object(GlobalOptimizer, 'optimize_simulated_annealing') as mock_method:
            mock_method.return_value = OptimizationResult(
                success=True, iterations=1000, initial_merit=100.0, final_merit=50.0,
                improvement=50.0, optimized_system=Mock(spec=OpticalSystem),
            )
            result = GlobalOptimizer.optimize_simulated_annealing(
                Mock(spec=GlobalOptimizer), max_iterations=1000, cooling_rate=None
            )
            assert isinstance(result, OptimizationResult)

    def test_simulated_annealing_callback_parameter(self):
        """optimize_simulated_annealing accepts callback parameter"""
        callback = Mock()
        with patch.object(GlobalOptimizer, 'optimize_simulated_annealing') as mock_method:
            mock_method.return_value = OptimizationResult(
                success=True, iterations=1, initial_merit=100.0, final_merit=100.0,
                improvement=0.0, optimized_system=Mock(spec=OpticalSystem),
            )
            GlobalOptimizer.optimize_simulated_annealing(
                Mock(spec=GlobalOptimizer), callback=callback
            )
            mock_method.assert_called_once()

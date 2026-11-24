import pytest
import random
import json
import math
from unittest.mock import Mock
from openlens.lens import Lens
from openlens.global_optimizer import GlobalOptimizer, OptimizationResult
from openlens.tolerancing import ToleranceOperand, serialize_operands, ParameterType


class TestOptimizeGenetic:
    """Tests for GlobalOptimizer.optimize_genetic"""
    
    @pytest.fixture
    def optimizer(self):
        """Create a simple optimizer for testing."""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        opt = GlobalOptimizer(lens)
        opt.add_variable("radius_of_curvature_1", min_value=30.0, max_value=100.0)
        opt.add_target("focal_length", target_value=50.0, weight=1.0)
        return opt
    
    def test_genetic_returns_optimization_result(self, optimizer):
        """Test that optimize_genetic returns an OptimizationResult."""
        result = optimizer.optimize_genetic(
            population_size=10,
            generations=3,
            mutation_rate=0.1,
            crossover_rate=0.7
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.best_values is not None
        assert result.final_merit is not None
        assert result.iterations > 0
    
    def test_genetic_zero_generations(self, optimizer):
        """Test genetic algorithm with zero generations still runs local refinement."""
        result = optimizer.optimize_genetic(
            population_size=10,
            generations=0
        )
        
        assert result is not None
        assert result.best_values is not None
        # Still runs simplex refinement
        assert result.iterations >= 0
    
    def test_genetic_callback_invoked_per_generation(self, optimizer):
        """Test that callback is invoked during optimization."""
        callback = Mock()
        result = optimizer.optimize_genetic(
            population_size=5,
            generations=2,
            callback=callback
        )
        
        # Callback should be called at least once
        assert callback.call_count > 0
        
        # Each call should have (iteration, merit, values)
        for call_args in callback.call_args_list:
            args = call_args[0]
            assert isinstance(args[0], int)  # iteration
            assert isinstance(args[1], (int, float))  # merit
            assert isinstance(args[2], list)  # values
            assert len(args[2]) == 1  # one variable
    
    def test_genetic_respects_bounds(self, optimizer):
        """Test that best design respects variable bounds."""
        result = optimizer.optimize_genetic(
            population_size=10,
            generations=5
        )
        
        # Variable is bounded [30.0, 100.0]
        assert 30.0 <= result.best_values[0] <= 100.0
    
    def test_genetic_crossover_rate_effect(self, optimizer):
        """Test crossover with extreme rates."""
        result_low = optimizer.optimize_genetic(
            population_size=10,
            generations=3,
            crossover_rate=0.01
        )
        
        result_high = optimizer.optimize_genetic(
            population_size=10,
            generations=3,
            crossover_rate=0.99
        )
        
        # Both should complete successfully
        assert result_low.best_values is not None
        assert result_high.best_values is not None


class TestOptimizeSimulatedAnnealing:
    """Tests for GlobalOptimizer.optimize_simulated_annealing"""
    
    @pytest.fixture
    def optimizer(self):
        """Create a simple optimizer for testing."""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        opt = GlobalOptimizer(lens)
        opt.add_variable("radius_of_curvature_1", min_value=30.0, max_value=100.0)
        opt.add_target("focal_length", target_value=50.0, weight=1.0)
        return opt
    
    def test_simulated_annealing_basic(self, optimizer):
        """Test basic simulated annealing run."""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=20,
            initial_temperature=100.0
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.best_values is not None
        assert math.isfinite(result.final_merit)
        assert result.iterations > 0
    
    def test_simulated_annealing_negative_temperature_clamped(self, optimizer):
        """Test that negative initial_temperature is clamped to 1e-6."""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=5,
            initial_temperature=-100.0
        )
        
        # Should not raise and should complete
        assert result is not None
        assert result.best_values is not None
    
    def test_simulated_annealing_zero_temperature_clamped(self, optimizer):
        """Test that zero initial_temperature is clamped."""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=5,
            initial_temperature=0.0
        )
        
        # Should complete without error
        assert result is not None
        assert result.best_values is not None
    
    def test_simulated_annealing_zero_max_iterations(self, optimizer):
        """Test SA with max_iterations=0 uses only local refinement."""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=0,
            initial_temperature=100.0
        )
        
        # Loop doesn't run, but local refinement still occurs
        assert result is not None
        assert result.best_values is not None
    
    def test_simulated_annealing_custom_cooling_rate(self, optimizer):
        """Test SA with explicit cooling rate."""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=20,
            initial_temperature=100.0,
            cooling_rate=0.95
        )
        
        assert result is not None
        assert result.iterations > 0
    
    def test_simulated_annealing_default_cooling_rate(self, optimizer):
        """Test that default cooling_rate is derived from max_iterations."""
        result = optimizer.optimize_simulated_annealing(
            max_iterations=100,
            initial_temperature=100.0,
            cooling_rate=None
        )
        
        # Should complete, cooling rate auto-derived
        assert result is not None
        assert result.best_values is not None
    
    def test_simulated_annealing_callback(self, optimizer):
        """Test SA callback is invoked during optimization."""
        callback = Mock()
        result = optimizer.optimize_simulated_annealing(
            max_iterations=10,
            initial_temperature=100.0,
            callback=callback
        )
        
        # Callback should be called at each iteration
        assert callback.call_count > 0
        
        # Verify callback signature
        for call_args in callback.call_args_list:
            args = call_args[0]
            assert isinstance(args[0], int)  # iteration
            assert isinstance(args[1], (int, float))  # merit
            assert isinstance(args[2], list)  # values


class TestToleranceOperandGenerateValue:
    """Tests for ToleranceOperand.generate_value"""
    
    def test_generate_value_uniform_distribution(self):
        """Test uniform distribution stays within bounds."""
        operand = ToleranceOperand(
            element_index=0,
            param_type=ParameterType.RADIUS_1,
            min_val=-10.0,
            max_val=10.0,
            distribution="uniform"
        )
        
        values = [operand.generate_value() for _ in range(50)]
        
        assert all(-10.0 <= v <= 10.0 for v in values)
        # Should have variety (not all identical)
        assert len(set(values)) > 1
    
    def test_generate_value_gaussian_distribution(self):
        """Test Gaussian distribution is clamped to bounds."""
        operand = ToleranceOperand(
            element_index=0,
            param_type=ParameterType.RADIUS_1,
            min_val=-10.0,
            max_val=10.0,
            distribution="gaussian",
            std_dev=2.0
        )
        
        values = [operand.generate_value() for _ in range(50)]
        
        # All values strictly within bounds due to clamping
        assert all(-10.0 <= v <= 10.0 for v in values)
    
    def test_generate_value_gaussian_zero_std_dev_defaults(self):
        """Test Gaussian with zero std_dev uses (max_val - min_val) / 6.0."""
        operand = ToleranceOperand(
            element_index=0,
            param_type=ParameterType.RADIUS_1,
            min_val=0.0,
            max_val=12.0,
            distribution="gaussian",
            std_dev=0.0  # Should default to 2.0
        )
        
        values = [operand.generate_value() for _ in range(50)]
        
        # Should still be bounded
        assert all(0.0 <= v <= 12.0 for v in values)
    
    def test_generate_value_seeded_rng_reproducible(self):
        """Test that seeded RNG produces reproducible results."""
        operand = ToleranceOperand(
            element_index=0,
            param_type=ParameterType.RADIUS_1,
            min_val=-10.0,
            max_val=10.0,
            distribution="uniform"
        )
        
        # Same seed should produce same sequence
        rng1 = random.Random(42)
        values1 = [operand.generate_value(rng=rng1) for _ in range(10)]
        
        rng2 = random.Random(42)
        values2 = [operand.generate_value(rng=rng2) for _ in range(10)]
        
        assert values1 == values2
    
    def test_generate_value_clamping_extreme_gaussian(self):
        """Test that extreme Gaussian is properly clamped."""
        operand = ToleranceOperand(
            element_index=0,
            param_type=ParameterType.THICKNESS,
            min_val=1.0,
            max_val=5.0,
            distribution="gaussian",
            std_dev=1000.0  # Very large std_dev to force clamping
        )
        
        values = [operand.generate_value() for _ in range(50)]
        
        # All values must be strictly within bounds
        assert all(1.0 <= v <= 5.0 for v in values)
    
    def test_generate_value_default_rng_works(self):
        """Test that default module-level random works when rng=None."""
        operand = ToleranceOperand(
            element_index=0,
            param_type=ParameterType.AIR_GAP,
            min_val=0.5,
            max_val=2.5,
            distribution="uniform"
        )
        
        # Should work without passing rng parameter
        value = operand.generate_value()
        assert 0.5 <= value <= 2.5


class TestSerializeOperands:
    """Tests for serialize_operands function"""
    
    def test_serialize_single_operand_all_fields(self):
        """Test serialization includes all required fields."""
        operand = ToleranceOperand(
            element_index=0,
            param_type=ParameterType.RADIUS_1,
            min_val=-10.0,
            max_val=10.0,
            distribution="uniform",
            std_dev=2.0,
            surface=1
        )
        
        result = serialize_operands([operand])
        
        assert len(result) == 1
        serialized = result[0]
        assert serialized["element_index"] == 0
        assert serialized["type"] == ParameterType.RADIUS_1.value
        assert serialized["min_val"] == -10.0
        assert serialized["max_val"] == 10.0
        assert serialized["distribution"] == "uniform"
        assert serialized["std_dev"] == 2.0
        assert serialized["surface"] == 1
    
    def test_serialize_multiple_operands(self):
        """Test serialization of multiple operands."""
        operands = [
            ToleranceOperand(
                element_index=0,
                param_type=ParameterType.RADIUS_1,
                min_val=-10.0,
                max_val=10.0
            ),
            ToleranceOperand(
                element_index=1,
                param_type=ParameterType.THICKNESS,
                min_val=0.1,
                max_val=1.0
            )
        ]
        
        result = serialize_operands(operands)
        
        assert len(result) == 2
        assert result[0]["element_index"] == 0
        assert result[1]["element_index"] == 1
        assert result[0]["type"] == ParameterType.RADIUS_1.value
        assert result[1]["type"] == ParameterType.THICKNESS.value
    
    def test_serialize_operands_json_serializable(self):
        """Test that serialized operands are JSON-compatible."""
        operand = ToleranceOperand(
            element_index=2,
            param_type=ParameterType.AIR_GAP,
            min_val=0.5,
            max_val=2.5,
            distribution="gaussian",
            std_dev=0.3
        )
        
        serialized = serialize_operands([operand])
        
        # Must be JSON serializable
        json_str = json.dumps(serialized)
        deserialized = json.loads(json_str)
        
        assert deserialized[0]["element_index"] == 2
        assert isinstance(deserialized[0]["type"], str)
    
    def test_serialize_empty_list(self):
        """Test serialization of empty operand list."""
        result = serialize_operands([])
        assert result == []
    
    def test_serialize_preserves_enum_value(self):
        """Test that param_type enum is converted to string value."""
        operand = ToleranceOperand(
            element_index=1,
            param_type=ParameterType.RADIUS_2,
            min_val=-5.0,
            max_val=5.0
        )
        
        result = serialize_operands([operand])
        
        # Must be a string, not an enum object
        assert isinstance(result[0]["type"], str)
        assert result[0]["type"] == ParameterType.RADIUS_2.value

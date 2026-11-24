import pytest
import random
import json
import math
from unittest.mock import Mock, patch, MagicMock

# Import actual library
from openlens.global_optimizer import GlobalOptimizer, OptimizationVariable, OptimizationResult
from openlens.tolerancing import ToleranceOperand, serialize_operands, deserialize_operands, ParamType


def create_simple_optimizer():
    """Helper: create an optimizer for a simple 2D quadratic objective."""
    variables = [
        OptimizationVariable(
            name="x",
            min_value=-10.0,
            max_value=10.0,
            current_value=0.0,
            step_size=1.0
        ),
        OptimizationVariable(
            name="y",
            min_value=-10.0,
            max_value=10.0,
            current_value=0.0,
            step_size=1.0
        ),
    ]
    # Simple quadratic: (x - 3)^2 + (y + 2)^2, minimum at (3, -2)
    return GlobalOptimizer(
        variables=variables,
        objective_fn=lambda **kwargs: (kwargs.get('x', 0) - 3)**2 + (kwargs.get('y', 0) + 2)**2
    )


class TestGeneticOptimization:
    """Tests for GlobalOptimizer.optimize_genetic"""

    def test_optimize_genetic_basic(self):
        """Test genetic optimization with default-like parameters."""
        optimizer = create_simple_optimizer()
        result = optimizer.optimize_genetic(population_size=20, generations=5)
        
        assert isinstance(result, OptimizationResult)
        assert len(result.best_values) == 2
        assert math.isfinite(result.final_merit)

    def test_optimize_genetic_zero_generations(self):
        """Test genetic optimization with zero generations."""
        optimizer = create_simple_optimizer()
        result = optimizer.optimize_genetic(generations=0)
        
        # Should still return valid result, possibly just initial/refined design
        assert result is not None
        assert len(result.best_values) == 2

    def test_optimize_genetic_callback_called(self):
        """Test that callback is invoked during genetic optimization."""
        optimizer = create_simple_optimizer()
        calls = []
        
        def track_callback(gen, merit, design):
            calls.append((gen, merit, design))
        
        optimizer.optimize_genetic(
            population_size=10,
            generations=3,
            callback=track_callback
        )
        
        # Callback should be called at least once
        assert len(calls) > 0
        # Verify callback signature
        for gen, merit, design in calls:
            assert isinstance(gen, int)
            assert isinstance(merit, (int, float))
            assert isinstance(design, list)
            assert len(design) == 2

    def test_optimize_genetic_mutation_rate_extremes(self):
        """Test genetic optimization with extreme mutation rates."""
        optimizer = create_simple_optimizer()
        
        # Very low mutation
        result_low = optimizer.optimize_genetic(
            population_size=10,
            generations=3,
            mutation_rate=0.0
        )
        
        # Very high mutation
        result_high = optimizer.optimize_genetic(
            population_size=10,
            generations=3,
            mutation_rate=1.0
        )
        
        assert math.isfinite(result_low.final_merit)
        assert math.isfinite(result_high.final_merit)

    def test_optimize_genetic_crossover_rate_extremes(self):
        """Test genetic optimization with extreme crossover rates."""
        optimizer = create_simple_optimizer()
        
        result_no_cross = optimizer.optimize_genetic(
            population_size=10,
            generations=3,
            crossover_rate=0.0
        )
        result_full_cross = optimizer.optimize_genetic(
            population_size=10,
            generations=3,
            crossover_rate=1.0
        )
        
        assert math.isfinite(result_no_cross.final_merit)
        assert math.isfinite(result_full_cross.final_merit)


class TestSimulatedAnnealingOptimization:
    """Tests for GlobalOptimizer.optimize_simulated_annealing"""

    def test_optimize_simulated_annealing_basic(self):
        """Test simulated annealing with default-like parameters."""
        optimizer = create_simple_optimizer()
        result = optimizer.optimize_simulated_annealing(max_iterations=50)
        
        assert isinstance(result, OptimizationResult)
        assert len(result.best_values) == 2
        assert math.isfinite(result.final_merit)

    def test_optimize_simulated_annealing_zero_iterations(self):
        """Test simulated annealing with zero iterations."""
        optimizer = create_simple_optimizer()
        result = optimizer.optimize_simulated_annealing(max_iterations=0)
        
        # Should complete without UnboundLocalError per documentation
        assert result is not None

    def test_optimize_simulated_annealing_negative_temperature_clamped(self):
        """Test that negative initial_temperature is clamped internally."""
        optimizer = create_simple_optimizer()
        # Should not raise ZeroDivisionError per documentation
        result = optimizer.optimize_simulated_annealing(
            max_iterations=10,
            initial_temperature=-100.0
        )
        
        assert result is not None
        assert math.isfinite(result.final_merit)

    def test_optimize_simulated_annealing_custom_cooling_rate(self):
        """Test simulated annealing with explicit cooling rate."""
        optimizer = create_simple_optimizer()
        result = optimizer.optimize_simulated_annealing(
            max_iterations=30,
            initial_temperature=100.0,
            cooling_rate=0.95
        )
        
        assert len(result.best_values) == 2
        assert math.isfinite(result.final_merit)

    def test_optimize_simulated_annealing_callback_called(self):
        """Test that callback is invoked during simulated annealing."""
        optimizer = create_simple_optimizer()
        calls = []
        
        def track_callback(iteration, merit, design):
            calls.append((iteration, merit, design))
        
        optimizer.optimize_simulated_annealing(
            max_iterations=20,
            callback=track_callback
        )
        
        assert len(calls) > 0
        for iteration, merit, design in calls:
            assert isinstance(iteration, int)
            assert isinstance(merit, (int, float))
            assert isinstance(design, list)

    def test_optimize_simulated_annealing_automatic_cooling_rate(self):
        """Test that None cooling_rate is automatically derived."""
        optimizer = create_simple_optimizer()
        # cooling_rate=None (default) should derive from max_iterations
        result = optimizer.optimize_simulated_annealing(
            max_iterations=100,
            initial_temperature=100.0,
            cooling_rate=None
        )
        
        assert result is not None
        assert math.isfinite(result.final_merit)


class TestToleranceOperandGenerateValue:
    """Tests for ToleranceOperand.generate_value"""

    def test_generate_value_gaussian_with_std_dev(self):
        """Test Gaussian distribution with explicit std_dev."""
        operand = ToleranceOperand(
            element_index=0,
            param_type=ParamType.THICKNESS,
            min_val=-2.0,
            max_val=2.0,
            distribution="gaussian",
            std_dev=0.5
        )
        
        samples = [operand.generate_value() for _ in range(100)]
        
        # All values must be clamped to bounds
        assert all(-2.0 <= s <= 2.0 for s in samples)

    def test_generate_value_gaussian_default_std_dev(self):
        """Test Gaussian without std_dev uses (max - min) / 6 as sigma."""
        operand = ToleranceOperand(
            element_index=0,
            param_type=ParamType.RADIUS,
            min_val=-3.0,
            max_val=3.0,
            distribution="gaussian",
            std_dev=0  # Should default to (3 - (-3)) / 6 = 1.0
        )
        
        samples = [operand.generate_value() for _ in range(50)]
        assert all(-3.0 <= s <= 3.0 for s in samples)

    def test_generate_value_uniform_distribution(self):
        """Test uniform distribution generation."""
        operand = ToleranceOperand(
            element_index=0,
            param_type=ParamType.THICKNESS,
            min_val=-1.0,
            max_val=1.0,
            distribution="uniform"
        )
        
        samples = [operand.generate_value() for _ in range(100)]
        assert all(-1.0 <= s <= 1.0 for s in samples)

    def test_generate_value_reproducibility_with_seed(self):
        """Test that generate_value is reproducible with seeded RNG."""
        operand = ToleranceOperand(
            element_index=0,
            param_type=ParamType.THICKNESS,
            min_val=-1.0,
            max_val=1.0,
            distribution="gaussian",
            std_dev=0.3
        )
        
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        
        val1 = operand.generate_value(rng1)
        val2 = operand.generate_value(rng2)
        
        assert val1 == val2

    def test_generate_value_clamping_enforced(self):
        """Test that extreme std_dev forces clamping to bounds."""
        operand = ToleranceOperand(
            element_index=0,
            param_type=ParamType.THICKNESS,
            min_val=-0.05,
            max_val=0.05,
            distribution="gaussian",
            std_dev=100.0  # Very large std_dev
        )
        
        for _ in range(50):
            val = operand.generate_value()
            assert -0.05 <= val <= 0.05

    def test_generate_value_default_rng_works(self):
        """Test that generate_value(rng=None) uses module-level random."""
        operand = ToleranceOperand(
            element_index=0,
            param_type=ParamType.THICKNESS,
            min_val=-1.0,
            max_val=1.0,
            distribution="uniform"
        )
        
        # Should not raise when rng is None
        val = operand.generate_value(rng=None)
        assert -1.0 <= val <= 1.0


class TestSerializeOperands:
    """Tests for serialize_operands function"""

    def test_serialize_single_operand(self):
        """Test serialization of a single operand."""
        operand = ToleranceOperand(
            element_index=1,
            param_type=ParamType.THICKNESS,
            min_val=-0.5,
            max_val=0.5,
            distribution="gaussian",
            std_dev=0.1
        )
        
        result = serialize_operands([operand])
        
        assert len(result) == 1
        assert result[0]['element_index'] == 1
        assert result[0]['type'] == ParamType.THICKNESS.value
        assert result[0]['min_val'] == -0.5
        assert result[0]['max_val'] == 0.5
        assert result[0]['distribution'] == "gaussian"
        assert result[0]['std_dev'] == 0.1

    def test_serialize_multiple_operands(self):
        """Test serialization of multiple operands."""
        operands = [
            ToleranceOperand(
                element_index=0,
                param_type=ParamType.THICKNESS,
                min_val=-0.5,
                max_val=0.5
            ),
            ToleranceOperand(
                element_index=1,
                param_type=ParamType.RADIUS,
                min_val=-1.0,
                max_val=1.0,
                distribution="uniform"
            ),
        ]
        
        result = serialize_operands(operands)
        
        assert len(result) == 2
        assert result[0]['element_index'] == 0
        assert result[1]['element_index'] == 1

    def test_serialize_operands_json_serializable(self):
        """Test that serialized operands can be JSON dumped/loaded."""
        operands = [
            ToleranceOperand(
                element_index=0,
                param_type=ParamType.THICKNESS,
                min_val=-0.5,
                max_val=0.5,
                distribution="gaussian",
                std_dev=0.1
            ),
        ]
        
        serialized = serialize_operands(operands)
        
        # Should round-trip through JSON
        json_str = json.dumps(serialized)
        parsed = json.loads(json_str)
        
        assert parsed[0]['element_index'] == 0
        assert parsed[0]['min_val'] == -0.5

    def test_serialize_includes_all_fields(self):
        """Test that serialization includes all expected fields."""
        operand = ToleranceOperand(
            element_index=2,
            param_type=ParamType.DECENTRATION_X,
            min_val=-0.2,
            max_val=0.2,
            distribution="uniform"
        )
        
        result = serialize_operands([operand])
        serialized = result[0]
        
        # All fields documented in serialize_operands
        expected_fields = {'element_index', 'type', 'min_val', 'max_val', 
                          'distribution', 'std_dev', 'surface'}
        assert all(key in serialized for key in expected_fields)

    def test_serialize_empty_list(self):
        """Test serialization of empty operand list."""
        result = serialize_operands([])
        assert result == []

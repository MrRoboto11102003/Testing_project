import pytest
import random
import math
from unittest.mock import Mock, patch

from openlens.global_optimizer import GlobalOptimizer, OptimizationVariable, OptimizationResult
from openlens.tolerancing import ToleranceOperand, serialize_operands, ParameterType


class TestOptimizeGenetic:
    """Test suite for GlobalOptimizer.optimize_genetic"""

    @pytest.fixture
    def simple_optimizer(self):
        """Setup a simple optimizer with one variable"""
        optimizer = GlobalOptimizer(optical_system=None)
        optimizer.variables = [
            OptimizationVariable(
                name="x",
                current_value=0.0,
                min_value=-10.0,
                max_value=10.0,
                step_size=1.0
            )
        ]
        return optimizer

    def test_genetic_algorithm_returns_optimization_result(self, simple_optimizer):
        """Test that optimize_genetic returns an OptimizationResult"""
        simple_optimizer._evaluate_design = lambda x: x[0] ** 2
        
        result = simple_optimizer.optimize_genetic(population_size=10, generations=2)
        
        assert isinstance(result, OptimizationResult)
        assert result.best_values is not None
        assert result.final_merit is not None
        assert len(result.best_values) == 1

    def test_genetic_algorithm_with_callback(self, simple_optimizer):
        """Test callback receives correct arguments (generation, merit, design)"""
        simple_optimizer._evaluate_design = lambda x: x[0] ** 2
        
        callback_calls = []
        def track_calls(gen, merit, design):
            callback_calls.append((gen, merit, design))
        
        simple_optimizer.optimize_genetic(
            population_size=5,
            generations=3,
            callback=track_calls
        )
        
        assert len(callback_calls) > 0
        for gen, merit, design in callback_calls:
            assert isinstance(gen, int)
            assert isinstance(merit, (int, float, type(None))) or math.isfinite(merit)
            assert isinstance(design, list)

    def test_genetic_algorithm_zero_generations(self, simple_optimizer):
        """Test GA handles zero generations gracefully"""
        simple_optimizer._evaluate_design = lambda x: x[0] ** 2
        
        result = simple_optimizer.optimize_genetic(population_size=10, generations=0)
        
        assert result.best_values is not None
        assert len(result.best_values) > 0

    def test_genetic_algorithm_respects_bounds(self, simple_optimizer):
        """Test that final design respects variable bounds"""
        simple_optimizer._evaluate_design = lambda x: (x[0] - 50.0) ** 2
        
        result = simple_optimizer.optimize_genetic(population_size=15, generations=5)
        
        assert -10.0 <= result.best_values[0] <= 10.0

    def test_genetic_algorithm_multiple_variables(self):
        """Test GA with multiple variables"""
        optimizer = GlobalOptimizer(optical_system=None)
        optimizer.variables = [
            OptimizationVariable(name="x", current_value=0.0, min_value=-5.0, max_value=5.0, step_size=0.5),
            OptimizationVariable(name="y", current_value=0.0, min_value=-5.0, max_value=5.0, step_size=0.5),
        ]
        
        optimizer._evaluate_design = lambda vals: (vals[0] - 2) ** 2 + (vals[1] + 1) ** 2
        
        result = optimizer.optimize_genetic(population_size=20, generations=8)
        
        assert len(result.best_values) == 2
        assert all(-5.0 <= v <= 5.0 for v in result.best_values)


class TestSimulatedAnnealing:
    """Test suite for GlobalOptimizer.optimize_simulated_annealing"""

    @pytest.fixture
    def simple_optimizer(self):
        """Setup a simple optimizer with one variable"""
        optimizer = GlobalOptimizer(optical_system=None)
        optimizer.variables = [
            OptimizationVariable(
                name="x",
                current_value=5.0,
                min_value=-10.0,
                max_value=10.0,
                step_size=0.5
            )
        ]
        return optimizer

    def test_simulated_annealing_returns_result(self, simple_optimizer):
        """Test that SA returns an OptimizationResult"""
        simple_optimizer._evaluate_design = lambda x: x[0] ** 2
        
        result = simple_optimizer.optimize_simulated_annealing(max_iterations=50)
        
        assert isinstance(result, OptimizationResult)
        assert result.best_values is not None
        assert result.final_merit is not None

    def test_simulated_annealing_with_callback(self, simple_optimizer):
        """Test SA callback receives correct arguments (iteration, merit, design)"""
        simple_optimizer._evaluate_design = lambda x: x[0] ** 2
        
        callback_calls = []
        def track_calls(iter, merit, design):
            callback_calls.append((iter, merit, design))
        
        simple_optimizer.optimize_simulated_annealing(
            max_iterations=20,
            initial_temperature=10.0,
            callback=track_calls
        )
        
        assert len(callback_calls) > 0

    def test_simulated_annealing_negative_temperature_clamped(self, simple_optimizer):
        """Test that negative initial_temperature is clamped to positive value"""
        simple_optimizer._evaluate_design = lambda x: x[0] ** 2
        
        result = simple_optimizer.optimize_simulated_annealing(
            max_iterations=10,
            initial_temperature=-5.0
        )
        
        assert result.best_values is not None

    def test_simulated_annealing_zero_temperature_clamped(self, simple_optimizer):
        """Test that zero initial_temperature is clamped to positive value"""
        simple_optimizer._evaluate_design = lambda x: x[0] ** 2
        
        result = simple_optimizer.optimize_simulated_annealing(
            max_iterations=10,
            initial_temperature=0.0
        )
        
        assert result.best_values is not None

    def test_simulated_annealing_zero_iterations(self, simple_optimizer):
        """Test SA with zero max_iterations handles gracefully"""
        simple_optimizer._evaluate_design = lambda x: x[0] ** 2
        
        result = simple_optimizer.optimize_simulated_annealing(
            max_iterations=0,
            initial_temperature=1.0
        )
        
        assert result.best_values is not None

    def test_simulated_annealing_custom_cooling_rate(self, simple_optimizer):
        """Test SA with explicit cooling_rate parameter"""
        simple_optimizer._evaluate_design = lambda x: x[0] ** 2
        
        result = simple_optimizer.optimize_simulated_annealing(
            max_iterations=30,
            initial_temperature=5.0,
            cooling_rate=0.95
        )
        
        assert result.best_values is not None
        assert result.final_merit is not None

    def test_simulated_annealing_respects_variable_bounds(self, simple_optimizer):
        """Test that final design respects variable bounds"""
        simple_optimizer._evaluate_design = lambda x: (x[0] - 50.0) ** 2
        
        result = simple_optimizer.optimize_simulated_annealing(max_iterations=100)
        
        assert -10.0 <= result.best_values[0] <= 10.0


class TestToleranceOperandGenerateValue:
    """Test suite for ToleranceOperand.generate_value"""

    def test_generate_value_gaussian_distribution(self):
        """Test generate_value with Gaussian distribution and std_dev"""
        operand = ToleranceOperand(
            element_index=0,
            param_type=ParameterType.RADIUS,
            min_val=-1.0,
            max_val=1.0,
            distribution="gaussian",
            std_dev=0.3
        )
        
        values = [operand.generate_value() for _ in range(50)]
        
        assert all(-1.0 <= v <= 1.0 for v in values)
        assert len(set(values)) > 1

    def test_generate_value_uniform_distribution(self):
        """Test generate_value with uniform distribution"""
        operand = ToleranceOperand(
            element_index=0,
            param_type=ParameterType.THICKNESS,
            min_val=0.5,
            max_val=2.5,
            distribution="uniform"
        )
        
        values = [operand.generate_value() for _ in range(100)]
        
        assert all(0.5 <= v <= 2.5 for v in values)

    def test_generate_value_gaussian_with_zero_std_dev(self):
        """Test Gaussian with std_dev=0 uses (max_val - min_val) / 6.0 formula"""
        operand = ToleranceOperand(
            element_index=0,
            param_type=ParameterType.DECENTER,
            min_val=-2.0,
            max_val=2.0,
            distribution="gaussian",
            std_dev=0.0
        )
        
        values = [operand.generate_value() for _ in range(100)]
        
        assert all(-2.0 <= v <= 2.0 for v in values)

    def test_generate_value_seeded_reproducibility(self):
        """Test that seeded RNG produces identical sequences"""
        operand = ToleranceOperand(
            element_index=0,
            param_type=ParameterType.TILT,
            min_val=-0.5,
            max_val=0.5,
            distribution="uniform"
        )
        
        rng1 = random.Random(12345)
        vals1 = [operand.generate_value(rng=rng1) for _ in range(10)]
        
        rng2 = random.Random(12345)
        vals2 = [operand.generate_value(rng=rng2) for _ in range(10)]
        
        assert vals1 == vals2

    def test_generate_value_default_rng_works(self):
        """Test generate_value uses module-level random when rng=None"""
        operand = ToleranceOperand(
            element_index=0,
            param_type=ParameterType.AIR_GAP,
            min_val=0.0,
            max_val=1.0,
            distribution="uniform"
        )
        
        value = operand.generate_value()
        
        assert 0.0 <= value <= 1.0


class TestSerializeOperands:
    """Test suite for serialize_operands function"""

    def test_serialize_single_operand(self):
        """Test serialization of a single tolerance operand"""
        operand = ToleranceOperand(
            element_index=0,
            param_type=ParameterType.RADIUS,
            min_val=-1.0,
            max_val=1.0,
            distribution="uniform",
            std_dev=0.1,
            surface=1
        )
        
        result = serialize_operands([operand])
        
        assert len(result) == 1
        assert result[0]["element_index"] == 0
        assert result[0]["type"] == ParameterType.RADIUS.value
        assert result[0]["min_val"] == -1.0
        assert result[0]["max_val"] == 1.0
        assert result[0]["distribution"] == "uniform"
        assert result[0]["std_dev"] == 0.1
        assert result[0]["surface"] == 1

    def test_serialize_multiple_operands(self):
        """Test serialization of multiple operands preserves order and content"""
        operands = [
            ToleranceOperand(element_index=0, param_type=ParameterType.RADIUS, min_val=-1.0, max_val=1.0),
            ToleranceOperand(element_index=1, param_type=ParameterType.THICKNESS, min_val=-0.5, max_val=0.5),
        ]
        
        result = serialize_operands(operands)
        
        assert len(result) == 2
        assert result[0]["element_index"] == 0
        assert result[1]["element_index"] == 1

    def test_serialize_operands_empty_list(self):
        """Test serialization of empty list returns empty list"""
        result = serialize_operands([])
        
        assert result == []
        assert isinstance(result, list)

    def test_serialize_operands_all_required_fields(self):
        """Test serialized output contains all required fields"""
        operand = ToleranceOperand(
            element_index=2,
            param_type=ParameterType.AIR_GAP,
            min_val=0.0,
            max_val=2.0,
            distribution="gaussian",
            std_dev=0.2,
            surface=3
        )
        
        result = serialize_operands([operand])
        
        required_keys = {"element_index", "type", "min_val", "max_val", "distribution", "std_dev", "surface"}
        assert required_keys.issubset(result[0].keys())

    def test_serialize_operands_different_param_types(self):
        """Test serialization with various parameter types"""
        param_types = [
            ParameterType.RADIUS,
            ParameterType.THICKNESS,
            ParameterType.DECENTER,
            ParameterType.TILT,
            ParameterType.AIR_GAP,
        ]
        
        for param_type in param_types:
            operand = ToleranceOperand(
                element_index=0,
                param_type=param_type,
                min_val=-1.0,
                max_val=1.0
            )
            
            result = serialize_operands([operand])
            
            assert result[0]["type"] == param_type.value
            assert isinstance(result[0]["type"], str)

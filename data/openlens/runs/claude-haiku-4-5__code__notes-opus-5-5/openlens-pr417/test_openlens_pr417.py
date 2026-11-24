import pytest
import random
import types
import json
from openlens.global_optimizer import GlobalOptimizer
from openlens.tolerancing import (
    ToleranceOperand,
    serialize_operands,
    deserialize_operands,
    ToleranceType,
)
from openlens.optimizer import OptimizationVariable, OptimizationTarget
from openlens.optical_system import OpticalSystem
from openlens.lens import Lens


class TestOptimizeGenetic:
    """Tests for GlobalOptimizer.optimize_genetic"""

    def setup_method(self):
        """Create a minimal optimizer with stubbed methods."""
        system = OpticalSystem()
        system.add_lens(Lens())
        
        var = OptimizationVariable(
            "x", 0, "radius_of_curvature_1",
            current_value=50.0, min_value=20.0, max_value=100.0
        )
        target = OptimizationTarget("focal_length", 100.0)
        
        self.optimizer = GlobalOptimizer(system, [var], [target], seed=42)
        self.var = var
        
        # Stub unseen methods
        self.optimizer._evaluate_design = lambda v: (v[0] - 50.0) ** 2
        
        def fake_simplex(**kw):
            return types.SimpleNamespace(
                best_values=list(kw["start"]),
                final_merit=0.5,
                iterations=3,
                variable_history=[{"x": 1.0}],
                merit_history=[0.5]
            )
        
        self.optimizer.optimize_simplex = fake_simplex
        self.optimizer._finalize = lambda **kw: kw

    def test_optimize_genetic_basic(self):
        """Test basic GA execution."""
        result = self.optimizer.optimize_genetic(
            population_size=10, generations=5
        )
        assert result["iterations"] == 8
        assert len(result["merit_history"]) == 5
        assert "GA Completed 5 generations" in result["converged_message"]

    def test_optimize_genetic_merit_history_non_increasing(self):
        """Test merit_history is non-increasing."""
        result = self.optimizer.optimize_genetic(
            population_size=20, generations=10
        )
        history = result["merit_history"]
        for i in range(1, len(history)):
            assert history[i] >= history[i-1]

    def test_optimize_genetic_zero_generations(self):
        """Test GA with zero generations."""
        result = self.optimizer.optimize_genetic(
            population_size=10, generations=0
        )
        assert result["iterations"] == 3
        assert len(result["merit_history"]) == 0
        assert "GA Completed 0 generations" in result["converged_message"]

    def test_optimize_genetic_callback_invoked(self):
        """Test callback is invoked during GA."""
        callback_calls = []
        def callback(gen, merit, design):
            callback_calls.append((gen, merit, list(design)))
        
        self.optimizer.optimize_genetic(
            population_size=5, generations=2, callback=callback
        )
        assert len(callback_calls) > 0
        assert callback_calls[0][0] == 0

    def test_optimize_genetic_population_size_zero_raises(self):
        """Test population_size=0 raises IndexError."""
        with pytest.raises(IndexError):
            self.optimizer.optimize_genetic(population_size=0)

    def test_optimize_genetic_variables_not_mutated(self):
        """Test original variables are not mutated."""
        original_value = self.var.current_value
        self.optimizer.optimize_genetic(population_size=10, generations=5)
        assert self.var.current_value == original_value


class TestOptimizeSimulatedAnnealing:
    """Tests for GlobalOptimizer.optimize_simulated_annealing"""

    def setup_method(self):
        """Create a minimal optimizer with stubbed methods."""
        system = OpticalSystem()
        system.add_lens(Lens())
        
        var = OptimizationVariable(
            "r1", 0, "radius_of_curvature_1",
            current_value=50.0, min_value=20.0, max_value=200.0, step_size=5.0
        )
        target = OptimizationTarget("focal_length", 100.0)
        
        self.optimizer = GlobalOptimizer(system, [var], [target], seed=42)
        self.var = var
        
        # Stub unseen methods
        self.optimizer._evaluate_design = lambda v: (v[0] - 100.0) ** 2
        
        def fake_simplex(**kw):
            return types.SimpleNamespace(
                best_values=list(kw["start"]),
                final_merit=1.0,
                iterations=2,
                variable_history=[],
                merit_history=[]
            )
        
        self.optimizer.optimize_simplex = fake_simplex
        self.optimizer._finalize = lambda **kw: kw

    def test_optimize_sa_basic(self):
        """Test basic SA execution."""
        result = self.optimizer.optimize_simulated_annealing(
            max_iterations=5, initial_temperature=100.0
        )
        assert result["iterations"] > 0
        assert len(result["merit_history"]) > 0
        assert result["merit_history"][0] == result["initial_merit"]

    def test_optimize_sa_zero_iterations(self):
        """Test SA with zero iterations."""
        result = self.optimizer.optimize_simulated_annealing(
            max_iterations=0, initial_temperature=100.0
        )
        assert "after 0 iterations" in result["converged_message"]

    def test_optimize_sa_callback_invoked(self):
        """Test callback is invoked."""
        callback_calls = []
        def callback(iteration, merit, values):
            callback_calls.append((iteration, merit, list(values)))
        
        self.optimizer.optimize_simulated_annealing(
            max_iterations=3, initial_temperature=100.0, callback=callback
        )
        assert len(callback_calls) > 0
        assert callback_calls[0][0] == 0

    def test_optimize_sa_negative_temperature_clamped(self, caplog):
        """Test negative temperature is clamped and logged."""
        import logging
        caplog.set_level(logging.WARNING)
        
        self.optimizer.optimize_simulated_annealing(
            max_iterations=2, initial_temperature=-5.0
        )
        assert any("not positive" in record.message for record in caplog.records)

    def test_optimize_sa_initial_merit_in_history(self):
        """Test merit_history[0] equals initial_merit."""
        result = self.optimizer.optimize_simulated_annealing(
            max_iterations=5, initial_temperature=100.0
        )
        assert result["merit_history"][0] == result["initial_merit"]

    def test_optimize_sa_variables_not_mutated(self):
        """Test original variables are not mutated."""
        original_value = self.var.current_value
        self.optimizer.optimize_simulated_annealing(
            max_iterations=5, initial_temperature=100.0
        )
        assert self.var.current_value == original_value

    def test_optimize_sa_explicit_cooling_rate(self):
        """Test SA with explicit cooling rate."""
        result = self.optimizer.optimize_simulated_annealing(
            max_iterations=20, initial_temperature=1.0, cooling_rate=0.5
        )
        assert result["iterations"] > 0


class TestToleranceOperandGenerateValue:
    """Tests for ToleranceOperand.generate_value"""

    def test_generate_value_uniform_default(self):
        """Test uniform distribution (default)."""
        op = ToleranceOperand(0, ToleranceType.THICKNESS, -0.1, 0.1)
        rng = random.Random(42)
        val = op.generate_value(rng)
        
        assert -0.1 <= val <= 0.1
        rng = random.Random(42)
        expected = rng.uniform(-0.1, 0.1)
        assert val == expected

    def test_generate_value_gaussian_clamped(self):
        """Test gaussian distribution is clamped."""
        op = ToleranceOperand(
            0, ToleranceType.RADIUS_1, -0.1, 0.1,
            distribution="gaussian", std_dev=0.05
        )
        rng = random.Random(42)
        for _ in range(50):
            val = op.generate_value(rng)
            assert -0.1 <= val <= 0.1

    def test_generate_value_gaussian_sigma_from_range(self):
        """Test gaussian with std_dev=0 uses range/6."""
        op = ToleranceOperand(
            0, ToleranceType.DECENTER_X, -0.3, 0.3,
            distribution="gaussian", std_dev=0.0
        )
        
        class FakeRng:
            def __init__(self):
                self.calls = []
            def gauss(self, mu, sigma):
                self.calls.append(("gauss", mu, sigma))
                return 0.0
        
        fake = FakeRng()
        op.generate_value(fake)
        
        assert fake.calls[0][0] == "gauss"
        assert fake.calls[0][1] == 0
        assert fake.calls[0][2] == pytest.approx(0.1)

    def test_generate_value_gaussian_case_sensitive(self):
        """Test distribution name is case-sensitive."""
        op = ToleranceOperand(
            0, ToleranceType.REFRACTIVE_INDEX, -0.1, 0.1,
            distribution="Gaussian"
        )
        rng = random.Random(42)
        val = op.generate_value(rng)
        
        rng = random.Random(42)
        expected = rng.uniform(-0.1, 0.1)
        assert val == expected

    def test_generate_value_no_rng_uses_stdlib(self):
        """Test None rng uses stdlib random."""
        op = ToleranceOperand(0, ToleranceType.THICKNESS, -0.5, 0.5)
        
        random.seed(123)
        val1 = op.generate_value(rng=None)
        
        random.seed(123)
        val2 = random.uniform(-0.5, 0.5)
        
        assert val1 == val2

    def test_generate_value_uniform_min_equals_max(self):
        """Test uniform when min equals max."""
        op = ToleranceOperand(0, ToleranceType.THICKNESS, 0.5, 0.5)
        rng = random.Random(42)
        val = op.generate_value(rng)
        assert val == 0.5

    def test_generate_value_gaussian_min_equals_max(self):
        """Test gaussian when min equals max."""
        op = ToleranceOperand(
            0, ToleranceType.THICKNESS, 0.5, 0.5,
            distribution="gaussian", std_dev=0.0
        )
        rng = random.Random(42)
        val = op.generate_value(rng)
        assert val == 0.5


class TestSerializeOperands:
    """Tests for serialize_operands"""

    def test_serialize_operands_empty_list(self):
        """Test serialization of empty list."""
        result = serialize_operands([])
        assert result == []

    def test_serialize_operands_single_operand(self):
        """Test serialization of single operand."""
        op = ToleranceOperand(0, ToleranceType.RADIUS_1, -0.1, 0.1)
        result = serialize_operands([op])
        
        assert len(result) == 1
        assert result[0]["element_index"] == 0
        assert result[0]["type"] == "Radius 1"
        assert result[0]["min_val"] == -0.1
        assert result[0]["max_val"] == 0.1
        assert result[0]["distribution"] == "uniform"
        assert result[0]["std_dev"] == 0.0
        assert result[0]["surface"] == 1

    def test_serialize_operands_key_order(self):
        """Test dict keys are in correct order."""
        op = ToleranceOperand(0, ToleranceType.THICKNESS, -0.2, 0.2)
        result = serialize_operands([op])
        
        expected_keys = [
            "element_index", "type", "min_val", "max_val",
            "distribution", "std_dev", "surface"
        ]
        assert list(result[0].keys()) == expected_keys

    def test_serialize_operands_gaussian_operand(self):
        """Test serialization of gaussian operand."""
        op = ToleranceOperand(
            2, ToleranceType.IRREGULARITY, -0.25, 0.25,
            distribution="gaussian", std_dev=0.05, surface=2
        )
        result = serialize_operands([op])
        
        assert result[0]["element_index"] == 2
        assert result[0]["type"] == "Surface Irregularity"
        assert result[0]["distribution"] == "gaussian"
        assert result[0]["std_dev"] == 0.05
        assert result[0]["surface"] == 2

    def test_serialize_operands_multiple_operands(self):
        """Test serialization of multiple operands."""
        ops = [
            ToleranceOperand(0, ToleranceType.RADIUS_1, -0.1, 0.1),
            ToleranceOperand(1, ToleranceType.DECENTER_X, -0.05, 0.05),
            ToleranceOperand(0, ToleranceType.TILT_Y, -0.01, 0.01),
        ]
        result = serialize_operands(ops)
        
        assert len(result) == 3
        assert result[0]["type"] == "Radius 1"
        assert result[1]["type"] == "Decenter X"
        assert result[2]["type"] == "Tilt Y"

    def test_serialize_operands_roundtrip(self):
        """Test roundtrip through deserialize_operands."""
        ops = [
            ToleranceOperand(0, ToleranceType.RADIUS_1, -0.1, 0.1),
            ToleranceOperand(1, ToleranceType.THICKNESS, -0.05, 0.05,
                           distribution="gaussian", std_dev=0.01, surface=2),
        ]
        
        serialized = serialize_operands(ops)
        deserialized = deserialize_operands(serialized)
        
        assert len(deserialized) == len(ops)
        for orig, deser in zip(ops, deserialized):
            assert orig.element_index == deser.element_index
            assert orig.param_type == deser.param_type
            assert orig.min_val == deser.min_val
            assert orig.max_val == deser.max_val

    def test_serialize_operands_json_serializable(self):
        """Test output is JSON serializable."""
        ops = [
            ToleranceOperand(0, ToleranceType.RADIUS_1, -0.1, 0.1),
            ToleranceOperand(1, ToleranceType.THICKNESS, -0.05, 0.05),
        ]
        
        result = serialize_operands(ops)
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    def test_serialize_operands_enum_values(self):
        """Test enum values are converted correctly."""
        type_pairs = [
            (ToleranceType.RADIUS_1, "Radius 1"),
            (ToleranceType.DECENTER_X, "Decenter X"),
            (ToleranceType.TILT_Y, "Tilt Y"),
            (ToleranceType.IRREGULARITY, "Surface Irregularity"),
        ]
        
        for enum_val, expected_str in type_pairs:
            op = ToleranceOperand(0, enum_val, -0.1, 0.1)
            result = serialize_operands([op])
            assert result[0]["type"] == expected_str

    def test_serialize_operands_does_not_mutate(self):
        """Test input operands are not mutated."""
        op = ToleranceOperand(0, ToleranceType.THICKNESS, -0.1, 0.1)
        original_element = op.element_index
        original_min = op.min_val
        
        serialize_operands([op])
        
        assert op.element_index == original_element
        assert op.min_val == original_min

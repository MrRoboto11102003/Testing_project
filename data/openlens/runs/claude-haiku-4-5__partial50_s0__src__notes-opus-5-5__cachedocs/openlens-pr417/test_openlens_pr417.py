import pytest
import random
import types
from unittest.mock import MagicMock, patch
from openlens.tolerancing import (
    ToleranceOperand,
    serialize_operands,
    deserialize_operands,
    ToleranceType,
)


class TestSerializeOperands:
    """Tests for serialize_operands function."""

    def test_serialize_single_operand_basic(self):
        """Test serializing a single basic tolerance operand."""
        operand = ToleranceOperand(0, ToleranceType.RADIUS_1, -0.1, 0.1)
        result = serialize_operands([operand])

        assert len(result) == 1
        assert result[0]["element_index"] == 0
        assert result[0]["type"] == "Radius 1"
        assert result[0]["min_val"] == -0.1
        assert result[0]["max_val"] == 0.1
        assert result[0]["distribution"] == "uniform"
        assert result[0]["std_dev"] == 0.0
        assert result[0]["surface"] == 1

    def test_serialize_gaussian_operand(self):
        """Test serializing operand with gaussian distribution."""
        operand = ToleranceOperand(
            2,
            ToleranceType.THICKNESS,
            -0.05,
            0.05,
            "gaussian",
            0.02,
            surface=2,
        )
        result = serialize_operands([operand])

        assert result[0]["element_index"] == 2
        assert result[0]["type"] == "Thickness"
        assert result[0]["distribution"] == "gaussian"
        assert result[0]["std_dev"] == 0.02
        assert result[0]["surface"] == 2

    def test_serialize_multiple_operands(self):
        """Test serializing multiple operands in order."""
        operands = [
            ToleranceOperand(0, ToleranceType.RADIUS_1, -0.1, 0.1),
            ToleranceOperand(1, ToleranceType.RADIUS_2, -0.2, 0.2),
        ]
        result = serialize_operands(operands)

        assert len(result) == 2
        assert result[0]["element_index"] == 0
        assert result[0]["type"] == "Radius 1"
        assert result[1]["element_index"] == 1
        assert result[1]["type"] == "Radius 2"

    def test_serialize_empty_list(self):
        """Test serializing empty list."""
        result = serialize_operands([])
        assert result == []

    def test_serialize_roundtrip(self):
        """Test that serialize -> deserialize produces equal operands."""
        operands = [
            ToleranceOperand(0, ToleranceType.THICKNESS, -0.05, 0.05),
            ToleranceOperand(
                1, ToleranceType.REFRACTIVE_INDEX, -0.01, 0.01, "gaussian", 0.005
            ),
        ]
        serialized = serialize_operands(operands)
        deserialized = deserialize_operands(serialized)

        assert deserialized == operands

    def test_serialize_enum_values_correct(self):
        """Test that enum values are converted to strings correctly."""
        test_cases = [
            (ToleranceType.RADIUS_1, "Radius 1"),
            (ToleranceType.THICKNESS, "Thickness"),
            (ToleranceType.DECENTER_X, "Decenter X"),
            (ToleranceType.AIR_GAP, "Air Gap"),
        ]
        for enum_val, expected_str in test_cases:
            operand = ToleranceOperand(0, enum_val, -1, 1)
            result = serialize_operands([operand])
            assert result[0]["type"] == expected_str


class TestToleranceOperandGenerateValue:
    """Tests for ToleranceOperand.generate_value method."""

    def test_generate_value_uniform_basic(self):
        """Test uniform distribution generation."""
        operand = ToleranceOperand(0, ToleranceType.THICKNESS, -0.1, 0.1, "uniform")
        rng = random.Random(42)
        value = operand.generate_value(rng)

        assert -0.1 <= value <= 0.1

    def test_generate_value_uniform_reproducible(self):
        """Test that uniform generation is reproducible with same seed."""
        operand = ToleranceOperand(0, ToleranceType.THICKNESS, -0.5, 0.5, "uniform")
        value1 = operand.generate_value(random.Random(123))
        value2 = operand.generate_value(random.Random(123))

        assert value1 == value2

    def test_generate_value_gaussian_auto_sigma(self):
        """Test gaussian distribution with auto-computed sigma."""
        operand = ToleranceOperand(
            0, ToleranceType.THICKNESS, -0.6, 0.6, "gaussian", std_dev=0.0
        )
        # With std_dev=0, sigma should be (0.6 - (-0.6))/6 = 0.2
        rng = random.Random(42)
        value = operand.generate_value(rng)

        # Value should be clamped to bounds
        assert -0.6 <= value <= 0.6

    def test_generate_value_gaussian_explicit_sigma(self):
        """Test gaussian distribution with explicit sigma."""
        operand = ToleranceOperand(
            0, ToleranceType.REFRACTIVE_INDEX, -0.1, 0.1, "gaussian", std_dev=0.05
        )
        rng = random.Random(42)
        value = operand.generate_value(rng)

        # Gaussian draw clamped to bounds
        assert -0.1 <= value <= 0.1

    def test_generate_value_gaussian_clamping(self):
        """Test that gaussian values are clamped to min/max bounds."""
        operand = ToleranceOperand(
            0, ToleranceType.THICKNESS, -0.1, 0.1, "gaussian", std_dev=100.0
        )
        # Large sigma should force many clamped values
        values = [operand.generate_value(random.Random(i)) for i in range(100)]

        # All values must be within bounds
        assert all(-0.1 <= v <= 0.1 for v in values)
        # With large sigma, some should hit the boundaries
        assert any(v == -0.1 or v == 0.1 for v in values)

    def test_generate_value_gaussian_zero_range(self):
        """Test gaussian when min_val == max_val."""
        operand = ToleranceOperand(
            0, ToleranceType.THICKNESS, 0.5, 0.5, "gaussian", std_dev=0.0
        )
        value = operand.generate_value(random.Random(42))

        # Should clamp to the only valid value
        assert value == 0.5

    def test_generate_value_no_rng_uses_module_random(self):
        """Test that rng=None uses module-level random."""
        operand = ToleranceOperand(0, ToleranceType.THICKNESS, -0.1, 0.1, "uniform")
        
        with patch("openlens.tolerancing.random") as mock_random:
            mock_random.uniform.return_value = 0.05
            value = operand.generate_value(None)

            assert value == 0.05
            mock_random.uniform.assert_called_once_with(-0.1, 0.1)

    def test_generate_value_gaussian_with_no_rng(self):
        """Test gaussian distribution with rng=None uses module random."""
        operand = ToleranceOperand(
            0, ToleranceType.THICKNESS, -0.1, 0.1, "gaussian", std_dev=0.05
        )
        
        with patch("openlens.tolerancing.random") as mock_random:
            mock_random.gauss.return_value = 0.05
            value = operand.generate_value(None)

            assert -0.1 <= value <= 0.1
            mock_random.gauss.assert_called_once()

    def test_generate_value_invalid_distribution_uses_uniform(self):
        """Test that invalid distribution string falls back to uniform."""
        operand = ToleranceOperand(
            0, ToleranceType.THICKNESS, -0.1, 0.1, "invalid_distribution"
        )
        rng = random.Random(42)
        value = operand.generate_value(rng)

        # Should use uniform path
        assert -0.1 <= value <= 0.1


class TestGlobalOptimizerGenetic:
    """Tests for GlobalOptimizer.optimize_genetic method."""

    def _create_optimizer(self, seed=42):
        """Helper to create a minimal GlobalOptimizer with mocked dependencies."""
        from openlens.global_optimizer import GlobalOptimizer
        from openlens.optical_system import OpticalSystem
        from openlens.lens import Lens
        from openlens.optimizer import OptimizationVariable, OptimizationTarget

        system = OpticalSystem()
        system.add_lens(Lens(name="test", radius_of_curvature_1=50.0))

        var = OptimizationVariable(
            "r1", 0, "radius_of_curvature_1", 50.0, 20.0, 100.0
        )
        target = OptimizationTarget("focal_length", 100.0)

        opt = GlobalOptimizer(system, [var], [target], seed=seed)
        return opt, var

    def test_genetic_zero_generations(self):
        """Test genetic algorithm with zero generations."""
        opt, var = self._create_optimizer()

        # Stub methods
        opt._evaluate_design = lambda v: (v[0] - 50.0) ** 2
        opt.optimize_simplex = lambda **kw: types.SimpleNamespace(
            best_values=list(kw["start"]),
            final_merit=0.1,
            iterations=5,
            variable_history=[],
            merit_history=[],
        )
        opt._finalize = lambda **kw: types.SimpleNamespace(**kw)

        result = opt.optimize_genetic(generations=0)

        # With 0 generations, merit_history should be empty
        assert result.merit_history == []
        assert result.iterations == 0 + 5

    def test_genetic_population_size_zero_raises(self):
        """Test that population_size=0 raises IndexError."""
        opt, var = self._create_optimizer()

        opt._evaluate_design = lambda v: 0.0
        opt.optimize_simplex = lambda **kw: types.SimpleNamespace(
            best_values=[], final_merit=0, iterations=0,
            variable_history=[], merit_history=[]
        )
        opt._finalize = lambda **kw: types.SimpleNamespace(**kw)

        with pytest.raises(IndexError):
            opt.optimize_genetic(population_size=0)

    def test_genetic_merit_history_length(self):
        """Test that merit_history has one entry per generation."""
        opt, var = self._create_optimizer()

        opt._evaluate_design = lambda v: (v[0] - 50.0) ** 2
        opt.optimize_simplex = lambda **kw: types.SimpleNamespace(
            best_values=list(kw["start"]),
            final_merit=0.1,
            iterations=3,
            variable_history=[],
            merit_history=[],
        )
        opt._finalize = lambda **kw: types.SimpleNamespace(**kw)

        result = opt.optimize_genetic(population_size=10, generations=5)

        assert len(result.merit_history) == 5

    def test_genetic_merit_history_non_increasing(self):
        """Test that merit history is non-increasing (best-so-far)."""
        opt, var = self._create_optimizer()

        opt._evaluate_design = lambda v: (v[0] - 50.0) ** 2
        opt.optimize_simplex = lambda **kw: types.SimpleNamespace(
            best_values=list(kw["start"]),
            final_merit=0.1,
            iterations=3,
            variable_history=[],
            merit_history=[],
        )
        opt._finalize = lambda **kw: types.SimpleNamespace(**kw)

        result = opt.optimize_genetic(population_size=10, generations=5)

        # Merit history should be non-increasing
        for i in range(1, len(result.merit_history)):
            assert result.merit_history[i] <= result.merit_history[i - 1]


class TestGlobalOptimizerSimulatedAnnealing:
    """Tests for GlobalOptimizer.optimize_simulated_annealing method."""

    def _create_optimizer(self, seed=42):
        """Helper to create a minimal GlobalOptimizer with mocked dependencies."""
        from openlens.global_optimizer import GlobalOptimizer
        from openlens.optical_system import OpticalSystem
        from openlens.lens import Lens
        from openlens.optimizer import OptimizationVariable, OptimizationTarget

        system = OpticalSystem()
        system.add_lens(Lens(name="test", radius_of_curvature_1=50.0))

        var = OptimizationVariable(
            "r1", 0, "radius_of_curvature_1", 50.0, 20.0, 100.0
        )
        target = OptimizationTarget("focal_length", 100.0)

        opt = GlobalOptimizer(system, [var], [target], seed=seed)
        return opt, var

    def test_simulated_annealing_zero_iterations(self):
        """Test simulated annealing with zero iterations."""
        opt, var = self._create_optimizer()

        opt._evaluate_design = lambda v: 0.0
        opt.optimize_simplex = lambda **kw: types.SimpleNamespace(
            best_values=[], final_merit=0, iterations=5,
            variable_history=[], merit_history=[]
        )
        opt._finalize = lambda **kw: types.SimpleNamespace(**kw)

        result = opt.optimize_simulated_annealing(max_iterations=0)

        # With 0 iterations, SA should be skipped
        assert result.iterations == 0 + 5

    def test_simulated_annealing_negative_temperature_clamped(self):
        """Test that negative initial_temperature is clamped with warning."""
        opt, var = self._create_optimizer()

        opt._evaluate_design = lambda v: (v[0] - 50.0) ** 2
        opt.optimize_simplex = lambda **kw: types.SimpleNamespace(
            best_values=list(kw["start"]),
            final_merit=0.1,
            iterations=3,
            variable_history=[],
            merit_history=[],
        )
        opt._finalize = lambda **kw: types.SimpleNamespace(**kw)

        with patch("openlens.global_optimizer.logger") as mock_logger:
            result = opt.optimize_simulated_annealing(
                max_iterations=1, initial_temperature=-5.0
            )
            # Should log a warning about clamping
            mock_logger.warning.assert_called_once()

    def test_simulated_annealing_callback_called(self):
        """Test that callback is called during optimization."""
        opt, var = self._create_optimizer()

        opt._evaluate_design = lambda v: (v[0] - 50.0) ** 2
        opt.optimize_simplex = lambda **kw: types.SimpleNamespace(
            best_values=list(kw["start"]),
            final_merit=0.1,
            iterations=0,
            variable_history=[],
            merit_history=[],
        )
        opt._finalize = lambda **kw: types.SimpleNamespace(**kw)

        callback_calls = []

        def mock_callback(iteration, merit, values):
            callback_calls.append((iteration, merit, list(values)))

        result = opt.optimize_simulated_annealing(
            max_iterations=3, callback=mock_callback
        )

        # Callback should be called at least once per iteration
        assert len(callback_calls) >= 3

    def test_simulated_annealing_reproducibility(self):
        """Test that same seed produces same result."""
        opt1, var1 = self._create_optimizer(seed=42)
        opt2, var2 = self._create_optimizer(seed=42)

        for opt in [opt1, opt2]:
            opt._evaluate_design = lambda v: (v[0] - 50.0) ** 2
            opt.optimize_simplex = lambda **kw: types.SimpleNamespace(
                best_values=list(kw["start"]),
                final_merit=0.1,
                iterations=3,
                variable_history=[],
                merit_history=[],
            )
            opt._finalize = lambda **kw: types.SimpleNamespace(**kw)

        result1 = opt1.optimize_simulated_annealing(max_iterations=5)
        result2 = opt2.optimize_simulated_annealing(max_iterations=5)

        # Same seed should produce same merit histories
        assert result1.merit_history == result2.merit_history

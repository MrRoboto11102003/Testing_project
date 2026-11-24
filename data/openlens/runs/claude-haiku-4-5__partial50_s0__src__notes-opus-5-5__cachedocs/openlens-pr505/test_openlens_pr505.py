import pytest
import math
import statistics
from unittest.mock import patch, MagicMock

from openlens.tolerancing import (
    InverseSensitivityAnalyzer,
    MonteCarloAnalyzer,
    ToleranceOperand,
    ToleranceType,
)
from openlens.optical_system import OpticalSystem
from openlens.lens import Lens


@pytest.fixture
def simple_system():
    """Create a simple optical system with one lens."""
    sys = OpticalSystem()
    lens = Lens(
        name="test",
        radius_of_curvature_1=50.0,
        radius_of_curvature_2=-50.0,
        thickness=5.0,
        diameter=25.0,
        material="BK7",
    )
    sys.add_lens(lens)
    return sys


@pytest.fixture
def simple_tolerances():
    """Create simple tolerance operands."""
    return [
        ToleranceOperand(
            element_index=0,
            param_type=ToleranceType.RADIUS_1,
            min_val=-0.1,
            max_val=0.1,
        ),
        ToleranceOperand(
            element_index=0,
            param_type=ToleranceType.THICKNESS,
            min_val=-0.05,
            max_val=0.05,
        ),
    ]


def fake_spot_factory(values_seq):
    """Factory for mocking SpotDiagram with a sequence of return values.
    
    Each value can be a float (rms_radius), None (vignetted), or Exception.
    """
    it = iter(values_seq)

    class FakeSpotDiagram:
        def __init__(self, system):
            pass

        def trace_spot(self, **kw):
            v = next(it)
            if isinstance(v, Exception):
                raise v
            return {"rms_radius": v}

    return FakeSpotDiagram


class TestInverseSensitivityAnalyzerCalculateSensitivities:
    """Tests for InverseSensitivityAnalyzer.calculate_sensitivities"""

    def test_unsupported_criterion_raises_valueerror_first(
        self, simple_system, simple_tolerances, monkeypatch
    ):
        """Test that unsupported criterion raises ValueError before any tracing."""
        # Create a factory that would fail if instantiated
        def bad_factory(values):
            raise AssertionError("Should not be called")

        monkeypatch.setattr("openlens.tolerancing.SpotDiagram", bad_factory)

        analyzer = InverseSensitivityAnalyzer(simple_system, simple_tolerances)

        with pytest.raises(ValueError, match="Unsupported criterion 'mtf'"):
            analyzer.calculate_sensitivities(criterion="mtf")

    def test_returns_list_with_correct_structure(
        self, simple_system, simple_tolerances, monkeypatch
    ):
        """Test that result is a list of dicts with all required keys."""
        # nominal=1.0, then results for 2 operands
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_factory([1.0, 1.1, 0.95]),
        )

        analyzer = InverseSensitivityAnalyzer(simple_system, simple_tolerances)
        result = analyzer.calculate_sensitivities()

        assert isinstance(result, list)
        assert len(result) == 2

        expected_keys = {
            "operand_index",
            "type",
            "element",
            "test_value",
            "result_value",
            "change",
            "sensitivity",
        }
        for entry in result:
            assert isinstance(entry, dict)
            assert set(entry.keys()) == expected_keys

    def test_operand_index_type_element_test_value(
        self, simple_system, simple_tolerances, monkeypatch
    ):
        """Test that operand metadata is recorded correctly."""
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_factory([1.0, 1.1, 0.95]),
        )

        analyzer = InverseSensitivityAnalyzer(simple_system, simple_tolerances)
        result = analyzer.calculate_sensitivities()

        # First operand
        assert result[0]["operand_index"] == 0
        assert result[0]["type"] == "RADIUS_1"
        assert result[0]["element"] == 0
        assert result[0]["test_value"] == 0.1  # max_val

        # Second operand
        assert result[1]["operand_index"] == 1
        assert result[1]["type"] == "THICKNESS"
        assert result[1]["element"] == 0
        assert result[1]["test_value"] == 0.05  # max_val

    def test_sensitivity_formula(self, simple_system, simple_tolerances, monkeypatch):
        """Test that sensitivity = change / test_value."""
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_factory([1.0, 1.5, 0.98]),
        )

        analyzer = InverseSensitivityAnalyzer(simple_system, simple_tolerances)
        result = analyzer.calculate_sensitivities()

        # First: change = 1.5 - 1.0 = 0.5, sensitivity = 0.5 / 0.1 = 5.0
        assert result[0]["change"] == pytest.approx(0.5)
        assert result[0]["sensitivity"] == pytest.approx(5.0)

        # Second: change = 0.98 - 1.0 = -0.02, sensitivity = -0.02 / 0.05 = -0.4
        assert result[1]["change"] == pytest.approx(-0.02)
        assert result[1]["sensitivity"] == pytest.approx(-0.4)

    def test_nominal_trace_vignetted_produces_nan(
        self, simple_system, simple_tolerances, monkeypatch
    ):
        """Test that vignetted nominal (None) makes all changes/sensitivities NaN."""
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_factory([None, 1.5, 0.95]),
        )

        analyzer = InverseSensitivityAnalyzer(simple_system, simple_tolerances)
        result = analyzer.calculate_sensitivities()

        for entry in result:
            assert math.isnan(entry["change"])
            assert math.isnan(entry["sensitivity"])

    def test_perturbed_trace_vignetted_produces_nan(
        self, simple_system, simple_tolerances, monkeypatch
    ):
        """Test that vignetted perturbed trace makes change/sensitivity NaN."""
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_factory([1.0, None, 1.2]),
        )

        analyzer = InverseSensitivityAnalyzer(simple_system, simple_tolerances)
        result = analyzer.calculate_sensitivities()

        # First operand: perturbed is None → inf, change and sensitivity are NaN
        assert result[0]["result_value"] == float("inf")
        assert math.isnan(result[0]["change"])
        assert math.isnan(result[0]["sensitivity"])

        # Second operand: normal
        assert result[1]["result_value"] == 1.2
        assert result[1]["change"] == pytest.approx(0.2)

    def test_exception_during_operand_trace_becomes_inf_and_nan(
        self, simple_system, simple_tolerances, monkeypatch
    ):
        """Test that exceptions during operand trace become inf/NaN."""
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_factory([1.0, RuntimeError("trace failed"), 1.2]),
        )

        analyzer = InverseSensitivityAnalyzer(simple_system, simple_tolerances)
        result = analyzer.calculate_sensitivities()

        # Exception becomes inf for result_value, NaN for change/sensitivity
        assert result[0]["result_value"] == float("inf")
        assert math.isnan(result[0]["change"])
        assert math.isnan(result[0]["sensitivity"])

    def test_max_val_zero_gives_int_zero_sensitivity(
        self, simple_system, monkeypatch
    ):
        """Test that sensitivity is int 0 when max_val is 0, not NaN."""
        tolerances = [
            ToleranceOperand(0, ToleranceType.RADIUS_1, min_val=0.0, max_val=0.0),
        ]

        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_factory([1.0, 1.0]),
        )

        analyzer = InverseSensitivityAnalyzer(simple_system, tolerances)
        result = analyzer.calculate_sensitivities()

        # max_val=0: sensitivity should be int 0, not NaN
        assert result[0]["sensitivity"] == 0
        assert isinstance(result[0]["sensitivity"], int)
        assert result[0]["change"] == pytest.approx(0.0)

    def test_empty_tolerances_returns_empty_list(
        self, simple_system, monkeypatch
    ):
        """Test that empty tolerances returns empty list (but nominal still traced)."""
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_factory([1.0]),
        )

        analyzer = InverseSensitivityAnalyzer(simple_system, [])
        result = analyzer.calculate_sensitivities()

        assert result == []


class TestMonteCarloAnalyzerRun:
    """Tests for MonteCarloAnalyzer.run"""

    def test_num_trials_zero_raises_valueerror(
        self, simple_system, simple_tolerances
    ):
        """Test that num_trials=0 raises ValueError."""
        analyzer = MonteCarloAnalyzer(simple_system, simple_tolerances)

        with pytest.raises(ValueError, match="num_trials must be positive, got 0"):
            analyzer.run(num_trials=0)

    def test_num_trials_negative_raises_valueerror(
        self, simple_system, simple_tolerances
    ):
        """Test that negative num_trials raises ValueError."""
        analyzer = MonteCarloAnalyzer(simple_system, simple_tolerances)

        with pytest.raises(ValueError, match="num_trials must be positive, got -5"):
            analyzer.run(num_trials=-5)

    def test_num_trials_check_before_criterion_check(
        self, simple_system, simple_tolerances
    ):
        """Test that num_trials is validated before criterion."""
        analyzer = MonteCarloAnalyzer(simple_system, simple_tolerances)

        # Both invalid, but num_trials error expected first
        with pytest.raises(ValueError, match="num_trials must be positive"):
            analyzer.run(num_trials=0, criterion="mtf")

    def test_unsupported_criterion_raises_valueerror(
        self, simple_system, simple_tolerances, monkeypatch
    ):
        """Test that unsupported criterion raises ValueError."""
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_factory([1.0]),
        )

        analyzer = MonteCarloAnalyzer(simple_system, simple_tolerances)

        with pytest.raises(ValueError, match="Unsupported criterion 'mtf'"):
            analyzer.run(num_trials=1, criterion="mtf")

    def test_returns_dict_with_all_required_keys(
        self, simple_system, simple_tolerances, monkeypatch
    ):
        """Test that run returns a dict with exactly 12 required keys."""
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_factory([1.0, 0.95, 1.1]),
        )

        analyzer = MonteCarloAnalyzer(simple_system, simple_tolerances)
        result = analyzer.run(num_trials=2)

        expected_keys = {
            "nominal",
            "mean",
            "std_dev",
            "min",
            "max",
            "yield",
            "trials",
            "criterion",
            "limit",
            "90th_percentile",
            "vignetted_trials",
            "skipped_operands",
        }
        assert set(result.keys()) == expected_keys

    def test_basic_statistics_calculation(
        self, simple_system, simple_tolerances, monkeypatch
    ):
        """Test that statistics are computed correctly from trials."""
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_factory([1.0, 0.9, 1.0, 1.1]),
        )

        analyzer = MonteCarloAnalyzer(simple_system, simple_tolerances)
        result = analyzer.run(num_trials=3)

        assert result["nominal"] == 1.0
        assert result["trials"] == 3
        assert result["min"] == pytest.approx(0.9)
        assert result["max"] == pytest.approx(1.1)
        assert result["mean"] == pytest.approx(1.0)
        expected_std = statistics.stdev([0.9, 1.0, 1.1])
        assert result["std_dev"] == pytest.approx(expected_std)

    def test_pass_fail_yield_calculation(
        self, simple_system, simple_tolerances, monkeypatch
    ):
        """Test that yield is calculated as percentage of passing trials."""
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_factory([1.0, 0.04, 0.05, 0.06]),
        )

        analyzer = MonteCarloAnalyzer(simple_system, simple_tolerances)
        result = analyzer.run(num_trials=3, criterion_limit=0.05)

        # Pass: 0.04, 0.05; Fail: 0.06 → 2/3 = 66.67%
        assert result["yield"] == pytest.approx(66.66666666666666)
        assert analyzer.results[0]["passed"] is True
        assert analyzer.results[1]["passed"] is True
        assert analyzer.results[2]["passed"] is False

    def test_criterion_limit_infinity_all_pass(
        self, simple_system, simple_tolerances, monkeypatch
    ):
        """Test that criterion_limit=inf gives 100% yield."""
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_factory([1.0, 100.0, 1000.0]),
        )

        analyzer = MonteCarloAnalyzer(simple_system, simple_tolerances)
        result = analyzer.run(num_trials=2, criterion_limit=float("inf"))

        assert result["yield"] == 100.0

    def test_vignetted_trials_excluded_from_finite_stats(
        self, simple_system, simple_tolerances, monkeypatch
    ):
        """Test that vignetted trials (None→inf) are excluded from stats."""
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_factory([1.0, None, 1.1]),
        )

        analyzer = MonteCarloAnalyzer(simple_system, simple_tolerances)
        result = analyzer.run(num_trials=2)

        assert result["vignetted_trials"] == 1
        # Only 1.1 is finite
        assert result["mean"] == pytest.approx(1.1)
        # max includes inf
        assert result["max"] == float("inf")

    def test_single_trial_std_dev_is_zero(
        self, simple_system, simple_tolerances, monkeypatch
    ):
        """Test that std_dev is int 0 for single trial."""
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_factory([1.0, 1.1]),
        )

        analyzer = MonteCarloAnalyzer(simple_system, simple_tolerances)
        result = analyzer.run(num_trials=1)

        assert result["std_dev"] == 0
        assert isinstance(result["std_dev"], int)

    def test_empty_tolerances_deterministic_nominal(
        self, simple_system, monkeypatch
    ):
        """Test that empty tolerances means all trials equal nominal."""
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_factory([1.0, 1.0, 1.0, 1.0]),
        )

        analyzer = MonteCarloAnalyzer(simple_system, [])
        result = analyzer.run(num_trials=3, criterion_limit=1.0)

        assert result["nominal"] == result["mean"]
        assert result["min"] == result["max"] == 1.0
        assert result["std_dev"] == 0
        assert result["yield"] == 100.0

    def test_results_list_per_trial_structure(
        self, simple_system, simple_tolerances, monkeypatch
    ):
        """Test that self.results contains correct per-trial records."""
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_factory([1.0, 0.95, 1.1]),
        )

        analyzer = MonteCarloAnalyzer(simple_system, simple_tolerances)
        analyzer.run(num_trials=2)

        assert len(analyzer.results) == 2
        for i, record in enumerate(analyzer.results):
            assert record["trial"] == i
            assert isinstance(record["perturbations"], dict)
            assert isinstance(record["compensators"], dict)
            assert isinstance(record["value"], (int, float))
            assert isinstance(record["passed"], bool)

    def test_criterion_and_limit_echoed_in_stats(
        self, simple_system, simple_tolerances, monkeypatch
    ):
        """Test that criterion and limit are echoed in result dict."""
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_factory([1.0, 0.95]),
        )

        analyzer = MonteCarloAnalyzer(simple_system, simple_tolerances)
        result = analyzer.run(
            num_trials=1, criterion="rms_spot_radius", criterion_limit=0.1
        )

        assert result["criterion"] == "rms_spot_radius"
        assert result["limit"] == 0.1
        assert result["trials"] == 1

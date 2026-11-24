import pytest
import math
import statistics
from unittest.mock import MagicMock
import random

from openlens.tolerancing import (
    InverseSensitivityAnalyzer,
    MonteCarloAnalyzer,
    ToleranceOperand,
    ToleranceType,
)
from openlens.optical_system import OpticalSystem
from openlens.lens import Lens


class FakeSpotDiagram:
    """Mock SpotDiagram that returns scripted rms_radius values."""
    
    def __init__(self, values_iter):
        self.values_iter = values_iter
    
    def trace_spot(self, **kwargs):
        v = next(self.values_iter)
        if isinstance(v, Exception):
            raise v
        return {"rms_radius": v}


def fake_spot_diagram_factory(values):
    """Factory for FakeSpotDiagram with scripted values."""
    values_iter = iter(values)
    
    class Factory:
        def __call__(self, system):
            return FakeSpotDiagram(values_iter)
    
    return Factory()


# ============= Tests for InverseSensitivityAnalyzer.calculate_sensitivities =============

class TestInverseSensitivityAnalyzerCalculateSensitivities:
    
    def test_criterion_unsupported_raises_before_tracing(self, monkeypatch):
        """ValueError for unsupported criterion should be raised before any SpotDiagram calls."""
        system = OpticalSystem()
        system.add_lens(Lens())
        tol = ToleranceOperand(0, ToleranceType.THICKNESS, -0.1, 0.1)
        analyzer = InverseSensitivityAnalyzer(system, [tol])
        
        fake_factory = MagicMock(side_effect=RuntimeError("Should not be called"))
        monkeypatch.setattr("openlens.tolerancing.SpotDiagram", fake_factory)
        
        with pytest.raises(ValueError, match=r"Unsupported criterion 'mtf'"):
            analyzer.calculate_sensitivities(criterion="mtf")
        
        assert not fake_factory.called
    
    def test_supported_criterion_accepted(self, monkeypatch):
        """Criterion 'rms_spot_radius' should be accepted."""
        system = OpticalSystem()
        system.add_lens(Lens())
        tol = ToleranceOperand(0, ToleranceType.THICKNESS, -0.1, 0.1)
        analyzer = InverseSensitivityAnalyzer(system, [tol])
        
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_diagram_factory([1.0, 1.1])
        )
        
        result = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
        assert len(result) == 1
        assert result[0]["sensitivity"] == pytest.approx(1.0)
    
    def test_empty_tolerances_returns_empty_list(self, monkeypatch):
        """Empty tolerances should return empty list but still trace nominal."""
        system = OpticalSystem()
        system.add_lens(Lens())
        analyzer = InverseSensitivityAnalyzer(system, [])
        
        call_count = [0]
        
        class CountingFactory:
            def __call__(self, sys):
                call_count[0] += 1
                return FakeSpotDiagram(iter([1.0]))
        
        monkeypatch.setattr("openlens.tolerancing.SpotDiagram", CountingFactory())
        
        result = analyzer.calculate_sensitivities()
        assert result == []
        assert call_count[0] == 1
    
    def test_return_dict_has_all_required_keys(self, monkeypatch):
        """Result should contain all 7 required keys."""
        system = OpticalSystem()
        system.add_lens(Lens())
        tol = ToleranceOperand(0, ToleranceType.THICKNESS, -0.1, 0.1)
        analyzer = InverseSensitivityAnalyzer(system, [tol])
        
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_diagram_factory([1.0, 1.05])
        )
        
        result = analyzer.calculate_sensitivities()
        required_keys = {
            "operand_index", "type", "element", "test_value",
            "result_value", "change", "sensitivity"
        }
        assert set(result[0].keys()) == required_keys
    
    def test_sensitivity_equals_change_divided_by_test_value(self, monkeypatch):
        """Sensitivity should be change / test_value when both are finite."""
        system = OpticalSystem()
        system.add_lens(Lens())
        tol = ToleranceOperand(0, ToleranceType.THICKNESS, -0.05, 0.1)
        analyzer = InverseSensitivityAnalyzer(system, [tol])
        
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_diagram_factory([1.0, 1.5])
        )
        
        result = analyzer.calculate_sensitivities()
        entry = result[0]
        
        assert entry["test_value"] == 0.1
        assert entry["change"] == pytest.approx(0.5)
        assert entry["sensitivity"] == pytest.approx(5.0)
    
    def test_zero_max_val_gives_integer_zero_sensitivity(self, monkeypatch):
        """When max_val is 0, sensitivity should be integer 0."""
        system = OpticalSystem()
        system.add_lens(Lens())
        tol = ToleranceOperand(0, ToleranceType.THICKNESS, 0.0, 0.0)
        analyzer = InverseSensitivityAnalyzer(system, [tol])
        
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_diagram_factory([1.0, 1.0])
        )
        
        result = analyzer.calculate_sensitivities()
        
        assert result[0]["test_value"] == 0.0
        assert result[0]["sensitivity"] == 0
        assert isinstance(result[0]["sensitivity"], int)
    
    def test_vignetted_nominal_gives_nan_change_and_sensitivity(self, monkeypatch):
        """When nominal is None (vignetted), change and sensitivity should be NaN."""
        system = OpticalSystem()
        system.add_lens(Lens())
        tol = ToleranceOperand(0, ToleranceType.THICKNESS, -0.1, 0.1)
        analyzer = InverseSensitivityAnalyzer(system, [tol])
        
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_diagram_factory([None, 1.5])
        )
        
        result = analyzer.calculate_sensitivities()
        entry = result[0]
        
        assert entry["result_value"] == 1.5
        assert math.isnan(entry["change"])
        assert math.isnan(entry["sensitivity"])
    
    def test_vignetted_perturbed_gives_inf_result_and_nan_stats(self, monkeypatch):
        """When perturbed trace returns None, result_value is inf, change and sensitivity NaN."""
        system = OpticalSystem()
        system.add_lens(Lens())
        tol = ToleranceOperand(0, ToleranceType.THICKNESS, -0.1, 0.1)
        analyzer = InverseSensitivityAnalyzer(system, [tol])
        
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_diagram_factory([1.0, None])
        )
        
        result = analyzer.calculate_sensitivities()
        entry = result[0]
        
        assert math.isinf(entry["result_value"])
        assert math.isnan(entry["change"])
        assert math.isnan(entry["sensitivity"])
    
    def test_exception_during_trace_caught_and_logged(self, monkeypatch):
        """Exception during perturbed trace should be caught, mapped to inf, change/sensitivity to NaN."""
        system = OpticalSystem()
        system.add_lens(Lens())
        tol = ToleranceOperand(0, ToleranceType.THICKNESS, -0.1, 0.1)
        analyzer = InverseSensitivityAnalyzer(system, [tol])
        
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_diagram_factory([1.0, RuntimeError("Trace failed")])
        )
        
        result = analyzer.calculate_sensitivities()
        entry = result[0]
        
        assert math.isinf(entry["result_value"])
        assert math.isnan(entry["change"])
        assert math.isnan(entry["sensitivity"])
    
    def test_nominal_trace_exception_propagates(self, monkeypatch):
        """Exception during nominal trace should propagate immediately."""
        system = OpticalSystem()
        system.add_lens(Lens())
        analyzer = InverseSensitivityAnalyzer(system, [])
        
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_diagram_factory([RuntimeError("Nominal failed")])
        )
        
        with pytest.raises(RuntimeError, match="Nominal failed"):
            analyzer.calculate_sensitivities()
    
    def test_type_field_is_param_type_name(self, monkeypatch):
        """Type field should be enum name (e.g. 'RADIUS_1'), not value."""
        system = OpticalSystem()
        system.add_lens(Lens())
        tol = ToleranceOperand(0, ToleranceType.RADIUS_1, -0.1, 0.1)
        analyzer = InverseSensitivityAnalyzer(system, [tol])
        
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_diagram_factory([1.0, 1.1])
        )
        
        result = analyzer.calculate_sensitivities()
        
        assert result[0]["type"] == ToleranceType.RADIUS_1.name
        assert result[0]["type"] == "RADIUS_1"
    
    def test_operand_index_matches_position(self, monkeypatch):
        """operand_index should match position in tolerances list."""
        system = OpticalSystem()
        system.add_lens(Lens())
        tols = [
            ToleranceOperand(0, ToleranceType.THICKNESS, -0.1, 0.1),
            ToleranceOperand(0, ToleranceType.RADIUS_1, -0.05, 0.05),
            ToleranceOperand(0, ToleranceType.RADIUS_2, -0.02, 0.02),
        ]
        analyzer = InverseSensitivityAnalyzer(system, tols)
        
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_diagram_factory([1.0, 1.1, 1.05, 1.02])
        )
        
        result = analyzer.calculate_sensitivities()
        
        assert len(result) == 3
        assert result[0]["operand_index"] == 0
        assert result[1]["operand_index"] == 1
        assert result[2]["operand_index"] == 2


# ============= Tests for MonteCarloAnalyzer.run =============

class TestMonteCarloAnalyzerRun:
    
    def test_num_trials_zero_raises_value_error(self):
        """num_trials <= 0 should raise ValueError."""
        system = OpticalSystem()
        system.add_lens(Lens())
        analyzer = MonteCarloAnalyzer(system, [])
        
        with pytest.raises(ValueError, match=r"num_trials must be positive, got 0"):
            analyzer.run(num_trials=0)
    
    def test_num_trials_negative_raises_value_error(self):
        """Negative num_trials should raise ValueError."""
        system = OpticalSystem()
        system.add_lens(Lens())
        analyzer = MonteCarloAnalyzer(system, [])
        
        with pytest.raises(ValueError, match=r"num_trials must be positive, got -5"):
            analyzer.run(num_trials=-5)
    
    def test_num_trials_validation_before_criterion(self):
        """num_trials validation should happen before criterion validation."""
        system = OpticalSystem()
        system.add_lens(Lens())
        analyzer = MonteCarloAnalyzer(system, [])
        
        with pytest.raises(ValueError, match=r"num_trials must be positive"):
            analyzer.run(num_trials=0, criterion="unsupported")
    
    def test_criterion_unsupported_raises_value_error(self, monkeypatch):
        """Unsupported criterion should raise ValueError."""
        system = OpticalSystem()
        system.add_lens(Lens())
        analyzer = MonteCarloAnalyzer(system, [])
        
        with pytest.raises(ValueError, match=r"Unsupported criterion 'mtf'"):
            analyzer.run(num_trials=1, criterion="mtf")
    
    def test_return_dict_has_all_required_keys(self, monkeypatch):
        """Result should have exactly 12 required keys."""
        system = OpticalSystem()
        system.add_lens(Lens())
        analyzer = MonteCarloAnalyzer(system, [])
        
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_diagram_factory([1.0, 1.0])
        )
        
        result = analyzer.run(num_trials=1)
        
        required_keys = {
            "nominal", "mean", "std_dev", "min", "max", "yield",
            "trials", "criterion", "limit", "90th_percentile",
            "vignetted_trials", "skipped_operands"
        }
        assert set(result.keys()) == required_keys
    
    def test_empty_tolerances_gives_deterministic_results(self, monkeypatch):
        """With no tolerances, all trials match nominal value."""
        system = OpticalSystem()
        system.add_lens(Lens())
        analyzer = MonteCarloAnalyzer(system, [])
        
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_diagram_factory([1.0, 1.0])
        )
        
        result = analyzer.run(num_trials=1, criterion_limit=0.5)
        
        assert result["nominal"] == 1.0
        assert result["mean"] == 1.0
        assert result["min"] == 1.0
        assert result["max"] == 1.0
        assert result["std_dev"] == 0
        assert result["yield"] == 100.0
    
    def test_criterion_limit_all_pass(self, monkeypatch):
        """Very high criterion_limit should give 100% yield."""
        system = OpticalSystem()
        system.add_lens(Lens())
        analyzer = MonteCarloAnalyzer(system, [])
        
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_diagram_factory([1.0, 1.0])
        )
        
        result = analyzer.run(num_trials=1, criterion_limit=float("inf"))
        
        assert result["yield"] == 100.0
    
    def test_criterion_limit_all_fail(self, monkeypatch):
        """Very low criterion_limit should give 0% yield."""
        system = OpticalSystem()
        system.add_lens(Lens())
        analyzer = MonteCarloAnalyzer(system, [])
        
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_diagram_factory([1.0, 1.0])
        )
        
        result = analyzer.run(num_trials=1, criterion_limit=-1.0)
        
        assert result["yield"] == 0.0
        assert result["trials"] == 1
    
    def test_criterion_and_limit_echoed_in_result(self, monkeypatch):
        """criterion and limit values should be echoed in result."""
        system = OpticalSystem()
        system.add_lens(Lens())
        analyzer = MonteCarloAnalyzer(system, [])
        
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_diagram_factory([1.0, 1.0])
        )
        
        result = analyzer.run(num_trials=1, criterion="rms_spot_radius", criterion_limit=0.075)
        
        assert result["criterion"] == "rms_spot_radius"
        assert result["limit"] == 0.075
        assert result["trials"] == 1
    
    def test_std_dev_is_integer_zero_for_single_trial(self, monkeypatch):
        """std_dev should be integer 0 for single trial (len(finite) <= 1)."""
        system = OpticalSystem()
        system.add_lens(Lens())
        analyzer = MonteCarloAnalyzer(system, [])
        
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_diagram_factory([1.0, 1.0])
        )
        
        result = analyzer.run(num_trials=1)
        
        assert result["std_dev"] == 0
        assert isinstance(result["std_dev"], int)
    
    def test_vignetted_trials_excluded_from_mean_stdev_min_percentile(self, monkeypatch):
        """Vignetted trials (inf) should be excluded from mean, stdev, min, percentile but kept in max."""
        system = OpticalSystem()
        system.add_lens(Lens())
        analyzer = MonteCarloAnalyzer(system, [])
        
        # Nominal 1.0, trials: 1.0, None (vignetted->inf), 1.2
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_diagram_factory([1.0, 1.0, None, 1.2])
        )
        
        result = analyzer.run(num_trials=3)
        
        finite_vals = [1.0, 1.2]
        assert result["mean"] == pytest.approx(statistics.mean(finite_vals))
        assert result["min"] == pytest.approx(1.0)
        assert result["vignetted_trials"] == 1
        assert math.isinf(result["max"])
    
    def test_results_attribute_reset_each_run(self, monkeypatch):
        """self.results should be reset at start of each run."""
        system = OpticalSystem()
        system.add_lens(Lens())
        analyzer = MonteCarloAnalyzer(system, [])
        
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_diagram_factory([1.0, 1.0, 1.0])
        )
        
        analyzer.run(num_trials=2)
        first_len = len(analyzer.results)
        
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_diagram_factory([1.0, 1.0])
        )
        analyzer.run(num_trials=1)
        second_len = len(analyzer.results)
        
        assert first_len == 2
        assert second_len == 1
    
    def test_pass_failed_per_trial_based_on_criterion_limit(self, monkeypatch):
        """Each trial should have passed bool: passed = (value <= limit)."""
        system = OpticalSystem()
        system.add_lens(Lens())
        analyzer = MonteCarloAnalyzer(system, [])
        
        monkeypatch.setattr(
            "openlens.tolerancing.SpotDiagram",
            fake_spot_diagram_factory([1.0, 0.04, 0.06])
        )
        
        result = analyzer.run(num_trials=2, criterion_limit=0.05)
        
        assert len(analyzer.results) == 2
        assert analyzer.results[0]["passed"] is True
        assert analyzer.results[1]["passed"] is False
        assert result["yield"] == 50.0

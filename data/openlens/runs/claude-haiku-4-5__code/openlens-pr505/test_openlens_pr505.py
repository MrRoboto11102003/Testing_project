import pytest
import math
import statistics
from unittest.mock import Mock, patch
from typing import Dict, Any, List


try:
    from openlens.tolerancing import InverseSensitivityAnalyzer, MonteCarloAnalyzer
    HAS_OPENLENS = True
except ImportError:
    HAS_OPENLENS = False
    pytestmark = pytest.mark.skip(reason="openlens not installed")


class MockParamType:
    def __init__(self, name: str):
        self.name = name


class MockTolerance:
    def __init__(self, param_type_name: str = "RADIUS", element_index: int = 0, max_val: float = 0.1):
        self.param_type = MockParamType(param_type_name)
        self.element_index = element_index
        self.max_val = max_val


@pytest.fixture
def inverse_analyzer():
    """Create an InverseSensitivityAnalyzer with mocked dependencies"""
    tolerances = [
        MockTolerance("RADIUS", 0, 0.1),
        MockTolerance("THICKNESS", 1, 0.05),
    ]
    
    analyzer = Mock(spec=InverseSensitivityAnalyzer)
    analyzer.system = Mock()
    analyzer.tolerances = tolerances
    analyzer.SUPPORTED_CRITERIA = "rms_spot_radius"
    analyzer._get_system_state = Mock(return_value={})
    analyzer._set_system_state = Mock()
    analyzer._apply_single_tolerance = Mock()
    
    analyzer.calculate_sensitivities = (
        InverseSensitivityAnalyzer.calculate_sensitivities.__get__(analyzer)
    )
    return analyzer


@pytest.fixture
def monte_carlo_analyzer():
    """Create a MonteCarloAnalyzer with mocked dependencies"""
    tolerances = [
        MockTolerance("RADIUS", 0, 0.1),
        MockTolerance("THICKNESS", 1, 0.05),
    ]
    
    analyzer = Mock(spec=MonteCarloAnalyzer)
    analyzer.nominal_system = Mock()
    analyzer.tolerances = tolerances
    analyzer.SUPPORTED_CRITERIA = "rms_spot_radius"
    analyzer.results = []
    analyzer._skipped_operands = []
    analyzer._get_system_state = Mock(return_value={})
    analyzer._set_system_state = Mock()
    analyzer._apply_tolerances = Mock(return_value={})
    analyzer._optimize_compensators = Mock(return_value={})
    analyzer._spot_rms = Mock(return_value=0.05)
    
    analyzer.run = MonteCarloAnalyzer.run.__get__(analyzer)
    return analyzer


class TestInverseSensitivityAnalyzerCalculateSensitivities:
    """Tests for InverseSensitivityAnalyzer.calculate_sensitivities"""
    
    def test_unsupported_criterion_raises_valueerror(self, inverse_analyzer):
        """Should raise ValueError for unsupported criterion"""
        with pytest.raises(ValueError) as exc_info:
            inverse_analyzer.calculate_sensitivities(criterion="invalid_metric")
        assert "Unsupported criterion" in str(exc_info.value)
    
    def test_supported_criterion_returns_list(self, inverse_analyzer):
        """Should accept supported criterion and return list of results"""
        with patch('openlens.tolerancing.SpotDiagram') as mock_spot:
            mock_spot.return_value.trace_spot.return_value = {"rms_radius": 0.05}
            result = inverse_analyzer.calculate_sensitivities(criterion="rms_spot_radius")
            assert isinstance(result, list)
            assert len(result) == len(inverse_analyzer.tolerances)
    
    def test_result_contains_required_keys(self, inverse_analyzer):
        """Each result dict should contain all required keys"""
        with patch('openlens.tolerancing.SpotDiagram') as mock_spot:
            mock_spot.return_value.trace_spot.return_value = {"rms_radius": 0.05}
            result = inverse_analyzer.calculate_sensitivities()
            required_keys = {
                "operand_index", "type", "element", "test_value",
                "result_value", "change", "sensitivity"
            }
            for item in result:
                assert required_keys.issubset(item.keys())
    
    def test_vignetted_nominal_handled(self, inverse_analyzer):
        """Vignetted nominal (None) should be treated as inf without crashing"""
        with patch('openlens.tolerancing.SpotDiagram') as mock_spot:
            mock_spot.return_value.trace_spot.return_value = {"rms_radius": None}
            result = inverse_analyzer.calculate_sensitivities()
            assert isinstance(result, list)
    
    def test_vignetted_test_case_gives_nan(self, inverse_analyzer):
        """Vignetted test case (None) should result in nan sensitivity"""
        with patch('openlens.tolerancing.SpotDiagram') as mock_spot:
            call_count = [0]
            def side_effect(*args, **kwargs):
                call_count[0] += 1
                return {"rms_radius": 0.05 if call_count[0] == 1 else None}
            
            mock_spot.return_value.trace_spot.side_effect = side_effect
            result = inverse_analyzer.calculate_sensitivities()
            
            assert math.isinf(result[0]["result_value"])
            assert math.isnan(result[0]["sensitivity"])
    
    def test_sensitivity_arithmetic(self, inverse_analyzer):
        """Sensitivity should equal (result - nominal) / delta"""
        with patch('openlens.tolerancing.SpotDiagram') as mock_spot:
            nominal, result_val, delta = 0.050, 0.080, 0.100
            call_count = [0]
            def side_effect(*args, **kwargs):
                call_count[0] += 1
                return {"rms_radius": nominal if call_count[0] == 1 else result_val}
            
            mock_spot.return_value.trace_spot.side_effect = side_effect
            result = inverse_analyzer.calculate_sensitivities()
            
            expected_sensitivity = (result_val - nominal) / delta
            assert abs(result[0]["sensitivity"] - expected_sensitivity) < 1e-9
    
    def test_zero_delta_sensitivity(self, inverse_analyzer):
        """Zero delta should give zero sensitivity"""
        inverse_analyzer.tolerances[0].max_val = 0.0
        with patch('openlens.tolerancing.SpotDiagram') as mock_spot:
            mock_spot.return_value.trace_spot.return_value = {"rms_radius": 0.05}
            result = inverse_analyzer.calculate_sensitivities()
            assert result[0]["sensitivity"] == 0
    
    def test_exception_gives_inf_result(self, inverse_analyzer):
        """Exception during trace should result in inf value"""
        with patch('openlens.tolerancing.SpotDiagram') as mock_spot:
            call_count = [0]
            def side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return {"rms_radius": 0.05}
                raise RuntimeError("Trace failed")
            
            mock_spot.return_value.trace_spot.side_effect = side_effect
            result = inverse_analyzer.calculate_sensitivities()
            
            assert math.isinf(result[0]["result_value"])
    
    def test_state_restoration(self, inverse_analyzer):
        """System state should be restored after each tolerance"""
        with patch('openlens.tolerancing.SpotDiagram') as mock_spot:
            mock_spot.return_value.trace_spot.return_value = {"rms_radius": 0.05}
            inverse_analyzer.calculate_sensitivities()
            
            assert inverse_analyzer._set_system_state.call_count == len(inverse_analyzer.tolerances)


class TestMonteCarloAnalyzerRun:
    """Tests for MonteCarloAnalyzer.run"""
    
    def test_zero_trials_raises_valueerror(self, monte_carlo_analyzer):
        """num_trials <= 0 should raise ValueError"""
        with pytest.raises(ValueError) as exc_info:
            monte_carlo_analyzer.run(num_trials=0)
        assert "num_trials must be positive" in str(exc_info.value)
    
    def test_unsupported_criterion_raises_valueerror(self, monte_carlo_analyzer):
        """Unsupported criterion should raise ValueError"""
        with pytest.raises(ValueError) as exc_info:
            monte_carlo_analyzer.run(criterion="bad_metric")
        assert "Unsupported criterion" in str(exc_info.value)
    
    def test_valid_run_returns_stats_dict(self, monte_carlo_analyzer):
        """Valid run should return dict with all required statistical keys"""
        result = monte_carlo_analyzer.run(num_trials=5)
        required_keys = {
            "nominal", "mean", "std_dev", "min", "max", "yield",
            "trials", "criterion", "limit", "90th_percentile",
            "vignetted_trials", "skipped_operands"
        }
        assert required_keys.issubset(result.keys())
    
    def test_yield_percentage_calculation(self, monte_carlo_analyzer):
        """Yield should be (pass_count / num_trials) * 100"""
        monte_carlo_analyzer._spot_rms.side_effect = [0.03, 0.04, 0.06, 0.07]
        result = monte_carlo_analyzer.run(num_trials=4, criterion_limit=0.05)
        
        assert result["yield"] == 50.0
    
    def test_all_pass_yield_100(self, monte_carlo_analyzer):
        """All passing trials should give 100% yield"""
        monte_carlo_analyzer._spot_rms.return_value = 0.03
        result = monte_carlo_analyzer.run(num_trials=10, criterion_limit=0.05)
        assert result["yield"] == 100.0
    
    def test_all_fail_yield_0(self, monte_carlo_analyzer):
        """All failing trials should give 0% yield"""
        monte_carlo_analyzer._spot_rms.return_value = 0.10
        result = monte_carlo_analyzer.run(num_trials=10, criterion_limit=0.05)
        assert result["yield"] == 0.0
    
    def test_vignetted_trials_counted(self, monte_carlo_analyzer):
        """Vignetted trials (inf) should be counted separately"""
        monte_carlo_analyzer._spot_rms.side_effect = [0.03, 0.04, float("inf"), 0.06]
        result = monte_carlo_analyzer.run(num_trials=4, criterion_limit=0.05)
        
        assert result["vignetted_trials"] == 1
        assert len(monte_carlo_analyzer.results) == 4
    
    def test_exception_continues_trials(self, monte_carlo_analyzer):
        """Exception in one trial should not abort the entire run"""
        monte_carlo_analyzer._apply_tolerances.side_effect = [
            {}, RuntimeError("Failed"), {}, {}
        ]
        monte_carlo_analyzer._spot_rms.side_effect = [0.03, 0.04, 0.05, 0.06]
        
        result = monte_carlo_analyzer.run(num_trials=4)
        
        assert result["trials"] == 4
    
    def test_state_restored_on_exception(self, monte_carlo_analyzer):
        """System state should be restored even after exception"""
        monte_carlo_analyzer._apply_tolerances.side_effect = RuntimeError("Failed")
        monte_carlo_analyzer.run(num_trials=3)
        
        assert monte_carlo_analyzer._set_system_state.call_count >= 3
    
    def test_skipped_operands_recorded(self, monte_carlo_analyzer):
        """Skipped operands should be included in results"""
        monte_carlo_analyzer._skipped_operands = ["OP_1", "OP_2"]
        result = monte_carlo_analyzer.run(num_trials=2)
        
        assert result["skipped_operands"] == ["OP_1", "OP_2"]
    
    def test_criterion_and_limit_recorded(self, monte_carlo_analyzer):
        """Criterion and limit should be recorded in results"""
        result = monte_carlo_analyzer.run(
            num_trials=3,
            criterion="rms_spot_radius",
            criterion_limit=0.075
        )
        
        assert result["criterion"] == "rms_spot_radius"
        assert result["limit"] == 0.075
    
    def test_nominal_analyzed_first(self, monte_carlo_analyzer):
        """Nominal system should be analyzed first"""
        nominal_val = 0.042
        monte_carlo_analyzer._spot_rms.side_effect = [nominal_val, 0.05, 0.06]
        result = monte_carlo_analyzer.run(num_trials=2)
        
        assert result["nominal"] == nominal_val


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

import pytest
import math
import statistics
from unittest.mock import Mock, patch


class TestInverseSensitivityAnalyzer:
    """Tests for InverseSensitivityAnalyzer.calculate_sensitivities"""
    
    def test_unsupported_criterion_raises_valueerror(self):
        """Should raise ValueError for unsupported criterion"""
        from openlens.tolerancing import InverseSensitivityAnalyzer
        
        analyzer = InverseSensitivityAnalyzer(Mock(), [])
        
        with pytest.raises(ValueError, match="Unsupported criterion"):
            analyzer.calculate_sensitivities(criterion="invalid_metric")
    
    def test_valid_criterion_rms_spot_radius_accepted(self):
        """Should accept 'rms_spot_radius' criterion"""
        from openlens.tolerancing import InverseSensitivityAnalyzer
        
        analyzer = InverseSensitivityAnalyzer(Mock(), [])
        
        # Should not raise, returns empty list for no tolerances
        result = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
        assert isinstance(result, list)
    
    def test_empty_tolerances_returns_empty_list(self):
        """Should return empty list when no tolerances defined"""
        from openlens.tolerancing import InverseSensitivityAnalyzer
        
        analyzer = InverseSensitivityAnalyzer(Mock(), [])
        result = analyzer.calculate_sensitivities()
        
        assert result == []
    
    def test_result_dict_contains_required_keys(self):
        """Each result dict should contain required keys"""
        from openlens.tolerancing import InverseSensitivityAnalyzer
        
        tol = Mock()
        tol.param_type = Mock()
        tol.param_type.name = "RADIUS"
        tol.element_index = 0
        tol.max_val = 0.1
        
        analyzer = InverseSensitivityAnalyzer(Mock(), [tol])
        
        with patch.object(analyzer, '_get_system_state', return_value={}):
            with patch.object(analyzer, '_set_system_state'):
                with patch.object(analyzer, '_apply_single_tolerance'):
                    with patch('openlens.analysis.spot_diagram.SpotDiagram') as MockSpot:
                        mock_spot = Mock()
                        mock_spot.trace_spot.return_value = {"rms_radius": 0.05}
                        MockSpot.return_value = mock_spot
                        
                        result = analyzer.calculate_sensitivities()
                        
                        assert len(result) == 1
                        result_dict = result[0]
                        required_keys = {"operand_index", "type", "element", "test_value",
                                       "result_value", "change", "sensitivity"}
                        assert set(result_dict.keys()) == required_keys
    
    def test_vignetted_nominal_uses_inf(self):
        """When nominal spot is None, should convert to inf"""
        from openlens.tolerancing import InverseSensitivityAnalyzer
        
        tol = Mock()
        tol.param_type = Mock(name="RADIUS")
        tol.element_index = 0
        tol.max_val = 0.1
        
        analyzer = InverseSensitivityAnalyzer(Mock(), [tol])
        
        with patch.object(analyzer, '_get_system_state', return_value={}):
            with patch.object(analyzer, '_set_system_state'):
                with patch.object(analyzer, '_apply_single_tolerance'):
                    with patch('openlens.analysis.spot_diagram.SpotDiagram') as MockSpot:
                        mock_spot = Mock()
                        # Nominal vignetted (None), perturbed finite
                        mock_spot.trace_spot.side_effect = [
                            {"rms_radius": None},
                            {"rms_radius": 0.1}
                        ]
                        MockSpot.return_value = mock_spot
                        
                        result = analyzer.calculate_sensitivities()
                        
                        # change = 0.1 - inf = -inf
                        assert math.isinf(result[0]["change"])
    
    def test_exception_during_analysis_produces_inf_and_continues(self):
        """Exception during tolerance application should produce inf, not crash"""
        from openlens.tolerancing import InverseSensitivityAnalyzer
        
        tol = Mock()
        tol.param_type = Mock(name="RADIUS")
        tol.element_index = 0
        tol.max_val = 0.1
        
        analyzer = InverseSensitivityAnalyzer(Mock(), [tol])
        
        with patch.object(analyzer, '_get_system_state', return_value={}):
            with patch.object(analyzer, '_set_system_state'):
                with patch.object(analyzer, '_apply_single_tolerance', 
                                 side_effect=RuntimeError("Bad apply")):
                    with patch('openlens.analysis.spot_diagram.SpotDiagram') as MockSpot:
                        mock_spot = Mock()
                        mock_spot.trace_spot.return_value = {"rms_radius": 0.05}
                        MockSpot.return_value = mock_spot
                        
                        result = analyzer.calculate_sensitivities()
                        
                        # Failed analysis should give inf
                        assert math.isinf(result[0]["result_value"])
    
    def test_nonfinite_values_produce_nan_sensitivity(self):
        """Non-finite change and sensitivity should both be NaN"""
        from openlens.tolerancing import InverseSensitivityAnalyzer
        
        tol = Mock()
        tol.param_type = Mock(name="RADIUS")
        tol.element_index = 0
        tol.max_val = 0.1
        
        analyzer = InverseSensitivityAnalyzer(Mock(), [tol])
        
        with patch.object(analyzer, '_get_system_state', return_value={}):
            with patch.object(analyzer, '_set_system_state'):
                with patch.object(analyzer, '_apply_single_tolerance'):
                    with patch('openlens.analysis.spot_diagram.SpotDiagram') as MockSpot:
                        mock_spot = Mock()
                        # Nominal inf, perturbed inf -> change = NaN
                        mock_spot.trace_spot.side_effect = [
                            {"rms_radius": None},
                            {"rms_radius": None}
                        ]
                        MockSpot.return_value = mock_spot
                        
                        result = analyzer.calculate_sensitivities()
                        
                        assert math.isnan(result[0]["change"])
                        assert math.isnan(result[0]["sensitivity"])


class TestMonteCarloAnalyzer:
    """Tests for MonteCarloAnalyzer.run"""
    
    def test_num_trials_zero_raises_valueerror(self):
        """Should raise ValueError when num_trials is 0"""
        from openlens.tolerancing import MonteCarloAnalyzer
        
        analyzer = MonteCarloAnalyzer(Mock(), [])
        
        with pytest.raises(ValueError, match="num_trials must be positive"):
            analyzer.run(num_trials=0)
    
    def test_num_trials_negative_raises_valueerror(self):
        """Should raise ValueError when num_trials is negative"""
        from openlens.tolerancing import MonteCarloAnalyzer
        
        analyzer = MonteCarloAnalyzer(Mock(), [])
        
        with pytest.raises(ValueError, match="num_trials must be positive"):
            analyzer.run(num_trials=-5)
    
    def test_unsupported_criterion_raises_valueerror(self):
        """Should raise ValueError for unsupported criterion"""
        from openlens.tolerancing import MonteCarloAnalyzer
        
        analyzer = MonteCarloAnalyzer(Mock(), [])
        
        with pytest.raises(ValueError, match="Unsupported criterion"):
            analyzer.run(criterion="mtf_cutoff")
    
    def test_yield_calculation_basic(self):
        """Yield should be calculated as (pass_count / num_trials) * 100"""
        from openlens.tolerancing import MonteCarloAnalyzer
        
        analyzer = MonteCarloAnalyzer(Mock(), [])
        
        with patch.object(analyzer, '_spot_rms') as mock_spot:
            # Nominal + 10 trials, all pass
            mock_spot.side_effect = [0.03] + [0.03] * 10
            
            with patch.object(analyzer, '_get_system_state', return_value={}):
                with patch.object(analyzer, '_set_system_state'):
                    with patch.object(analyzer, '_apply_tolerances', return_value={}):
                        with patch.object(analyzer, '_optimize_compensators', return_value={}):
                            result = analyzer.run(num_trials=10, criterion_limit=0.05)
                            
                            assert result["yield"] == 100.0
    
    def test_return_dict_has_all_statistical_keys(self):
        """Should return dict with all expected keys"""
        from openlens.tolerancing import MonteCarloAnalyzer
        
        analyzer = MonteCarloAnalyzer(Mock(), [])
        
        expected_keys = {
            "nominal", "mean", "std_dev", "min", "max", "yield",
            "trials", "criterion", "limit", "90th_percentile",
            "vignetted_trials", "skipped_operands"
        }
        
        with patch.object(analyzer, '_spot_rms', return_value=0.03):
            with patch.object(analyzer, '_get_system_state', return_value={}):
                with patch.object(analyzer, '_set_system_state'):
                    with patch.object(analyzer, '_apply_tolerances', return_value={}):
                        with patch.object(analyzer, '_optimize_compensators', return_value={}):
                            result = analyzer.run(num_trials=1)
                            
                            assert set(result.keys()) == expected_keys
    
    def test_vignetted_trials_converted_to_inf(self):
        """Vignetted trials (rms_radius None, returned as inf) counted separately"""
        from openlens.tolerancing import MonteCarloAnalyzer
        
        analyzer = MonteCarloAnalyzer(Mock(), [])
        
        with patch.object(analyzer, '_spot_rms') as mock_spot:
            # Nominal + 3 trials: one inf (vignetted), two finite
            mock_spot.side_effect = [0.03, float('inf'), 0.04, 0.05]
            
            with patch.object(analyzer, '_get_system_state', return_value={}):
                with patch.object(analyzer, '_set_system_state'):
                    with patch.object(analyzer, '_apply_tolerances', return_value={}):
                        with patch.object(analyzer, '_optimize_compensators', return_value={}):
                            result = analyzer.run(num_trials=3, criterion_limit=0.1)
                            
                            assert result["vignetted_trials"] == 1
    
    def test_statistics_exclude_nonfinite_values(self):
        """Mean and stdev should use only finite values, max should include inf"""
        from openlens.tolerancing import MonteCarloAnalyzer
        
        analyzer = MonteCarloAnalyzer(Mock(), [])
        
        with patch.object(analyzer, '_spot_rms') as mock_spot:
            # Nominal 0.03, trials: 0.04, 0.05, inf
            mock_spot.side_effect = [0.03, 0.04, 0.05, float('inf')]
            
            with patch.object(analyzer, '_get_system_state', return_value={}):
                with patch.object(analyzer, '_set_system_state'):
                    with patch.object(analyzer, '_apply_tolerances', return_value={}):
                        with patch.object(analyzer, '_optimize_compensators', return_value={}):
                            result = analyzer.run(num_trials=3)
                            
                            # mean = (0.04 + 0.05) / 2 = 0.045
                            assert abs(result["mean"] - 0.045) < 1e-9
                            # max should still be inf
                            assert math.isinf(result["max"])
    
    def test_pass_fail_evaluation_against_limit(self):
        """Trials should be evaluated as passed when value <= limit"""
        from openlens.tolerancing import MonteCarloAnalyzer
        
        analyzer = MonteCarloAnalyzer(Mock(), [])
        
        with patch.object(analyzer, '_spot_rms') as mock_spot:
            # Nominal 0.03, trials: 0.04 (pass), 0.07 (fail), 0.05 (pass)
            mock_spot.side_effect = [0.03, 0.04, 0.07, 0.05]
            
            with patch.object(analyzer, '_get_system_state', return_value={}):
                with patch.object(analyzer, '_set_system_state'):
                    with patch.object(analyzer, '_apply_tolerances', return_value={}):
                        with patch.object(analyzer, '_optimize_compensators', return_value={}):
                            result = analyzer.run(num_trials=3, criterion_limit=0.06)
                            
                            # 2 out of 3 pass
                            assert result["yield"] == pytest.approx(66.667, rel=0.01)
    
    def test_state_restored_after_each_trial(self):
        """System state should be restored after each trial"""
        from openlens.tolerancing import MonteCarloAnalyzer
        
        analyzer = MonteCarloAnalyzer(Mock(), [])
        
        with patch.object(analyzer, '_spot_rms', return_value=0.03):
            with patch.object(analyzer, '_get_system_state', return_value={"state": "saved"}) as mock_get:
                with patch.object(analyzer, '_set_system_state') as mock_set:
                    with patch.object(analyzer, '_apply_tolerances', return_value={}):
                        with patch.object(analyzer, '_optimize_compensators', return_value={}):
                            analyzer.run(num_trials=3)
                            
                            # get called once for nominal, set called once per trial
                            assert mock_set.call_count == 3
    
    def test_compensators_applied_to_each_trial(self):
        """Compensators should be optimized for each trial"""
        from openlens.tolerancing import MonteCarloAnalyzer
        
        analyzer = MonteCarloAnalyzer(Mock(), [])
        
        with patch.object(analyzer, '_spot_rms', return_value=0.03):
            with patch.object(analyzer, '_get_system_state', return_value={}):
                with patch.object(analyzer, '_set_system_state'):
                    with patch.object(analyzer, '_apply_tolerances', return_value={}):
                        with patch.object(analyzer, '_optimize_compensators',
                                        return_value={"focus_shift_mm": 0.1}) as mock_comp:
                            analyzer.run(num_trials=2)
                            
                            # Compensator called once per trial
                            assert mock_comp.call_count == 2
    
    def test_criterion_and_limit_stored_in_result(self):
        """Criterion and limit should be stored in result for reproducibility"""
        from openlens.tolerancing import MonteCarloAnalyzer
        
        analyzer = MonteCarloAnalyzer(Mock(), [])
        
        with patch.object(analyzer, '_spot_rms', return_value=0.03):
            with patch.object(analyzer, '_get_system_state', return_value={}):
                with patch.object(analyzer, '_set_system_state'):
                    with patch.object(analyzer, '_apply_tolerances', return_value={}):
                        with patch.object(analyzer, '_optimize_compensators', return_value={}):
                            result = analyzer.run(num_trials=1, criterion_limit=0.075)
                            
                            assert result["criterion"] == "rms_spot_radius"
                            assert result["limit"] == 0.075
                            assert result["trials"] == 1

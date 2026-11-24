import pytest
import math
from unittest.mock import Mock, patch

from openlens.tolerancing import InverseSensitivityAnalyzer, MonteCarloAnalyzer


class TestInverseSensitivityAnalyzerCalculateSensitivities:
    """Tests for InverseSensitivityAnalyzer.calculate_sensitivities"""

    @pytest.fixture
    def analyzer_setup(self):
        """Setup analyzer with mocked system and tolerances"""
        system = Mock()
        tolerances = [
            Mock(param_type=Mock(name="RADIUS"), element_index=0, max_val=0.1),
            Mock(param_type=Mock(name="THICKNESS"), element_index=0, max_val=0.05),
        ]
        analyzer = InverseSensitivityAnalyzer(system, tolerances)
        return analyzer, system, tolerances

    def test_valid_criterion_accepted(self, analyzer_setup):
        """Test calculate_sensitivities accepts valid criterion 'rms_spot_radius'"""
        analyzer, _, tolerances = analyzer_setup

        with patch.object(analyzer, "_get_system_state", return_value={}):
            with patch.object(analyzer, "_set_system_state"):
                with patch.object(analyzer, "_apply_single_tolerance"):
                    with patch("openlens.tolerancing.SpotDiagram") as mock_spot_cls:
                        mock_spot = Mock()
                        mock_spot.trace_spot.return_value = {"rms_radius": 0.02}
                        mock_spot_cls.return_value = mock_spot

                        result = analyzer.calculate_sensitivities(criterion="rms_spot_radius")

                        assert isinstance(result, list)
                        assert len(result) == len(tolerances)

    def test_invalid_criterion_raises_valueerror(self, analyzer_setup):
        """Test unsupported criterion raises ValueError with criterion name"""
        analyzer, _, _ = analyzer_setup

        with pytest.raises(ValueError) as exc_info:
            analyzer.calculate_sensitivities(criterion="invalid_metric")

        assert "Unsupported criterion" in str(exc_info.value)
        assert "invalid_metric" in str(exc_info.value)

    def test_return_structure_has_required_keys(self, analyzer_setup):
        """Test return value is list of dicts with expected keys"""
        analyzer, _, _ = analyzer_setup

        with patch.object(analyzer, "_get_system_state", return_value={}):
            with patch.object(analyzer, "_set_system_state"):
                with patch.object(analyzer, "_apply_single_tolerance"):
                    with patch("openlens.tolerancing.SpotDiagram") as mock_spot_cls:
                        mock_spot = Mock()
                        mock_spot.trace_spot.return_value = {"rms_radius": 0.03}
                        mock_spot_cls.return_value = mock_spot

                        result = analyzer.calculate_sensitivities()

                        required_keys = {
                            "operand_index",
                            "type",
                            "element",
                            "test_value",
                            "result_value",
                            "change",
                            "sensitivity",
                        }
                        for item in result:
                            assert isinstance(item, dict)
                            assert required_keys.issubset(item.keys())

    def test_vignetted_system_none_as_infinity(self, analyzer_setup):
        """Test None rms_radius (vignetted system) handled as inf"""
        analyzer, _, _ = analyzer_setup

        with patch.object(analyzer, "_get_system_state", return_value={}):
            with patch.object(analyzer, "_set_system_state"):
                with patch.object(analyzer, "_apply_single_tolerance"):
                    with patch("openlens.tolerancing.SpotDiagram") as mock_spot_cls:
                        mock_spot = Mock()
                        # All calls return None (vignetted)
                        mock_spot.trace_spot.return_value = {"rms_radius": None}
                        mock_spot_cls.return_value = mock_spot

                        result = analyzer.calculate_sensitivities()

                        # Should complete without raising
                        assert isinstance(result, list)
                        # With both nominal and perturbed vignetted, change/sensitivity should be NaN
                        for item in result:
                            assert math.isnan(item["change"]) or math.isfinite(item["change"])

    def test_exception_caught_state_restored(self, analyzer_setup):
        """Test exceptions during perturbation caught and state restored"""
        analyzer, _, tolerances = analyzer_setup

        with patch.object(analyzer, "_get_system_state", return_value={}):
            with patch.object(analyzer, "_set_system_state") as mock_restore:
                with patch.object(analyzer, "_apply_single_tolerance"):
                    with patch("openlens.tolerancing.SpotDiagram") as mock_spot_cls:
                        mock_spot = Mock()
                        mock_spot.trace_spot.side_effect = [
                            {"rms_radius": 0.02},  # nominal
                            RuntimeError("Trace failed"),
                            RuntimeError("Trace failed"),
                        ]
                        mock_spot_cls.return_value = mock_spot

                        result = analyzer.calculate_sensitivities()

                        # Should complete despite exception
                        assert isinstance(result, list)
                        # State restored for each tolerance
                        assert mock_restore.call_count >= len(tolerances)


class TestMonteCarloAnalyzerRun:
    """Tests for MonteCarloAnalyzer.run"""

    @pytest.fixture
    def analyzer_setup(self):
        """Setup analyzer with mocked system and tolerances"""
        system = Mock()
        tolerances = [
            Mock(
                param_type=Mock(name="RADIUS"),
                element_index=0,
                max_val=0.1,
                min_val=-0.1,
            ),
        ]
        analyzer = MonteCarloAnalyzer(system, tolerances)
        return analyzer, system, tolerances

    def test_valid_run_returns_stats_dict(self, analyzer_setup):
        """Test valid run returns dict with statistics"""
        analyzer, _, _ = analyzer_setup

        with patch.object(analyzer, "_spot_rms", return_value=0.02):
            with patch.object(analyzer, "_apply_tolerances", return_value={}):
                with patch.object(analyzer, "_optimize_compensators", return_value={}):
                    with patch.object(analyzer, "_get_system_state", return_value={}):
                        with patch.object(analyzer, "_set_system_state"):
                            result = analyzer.run(num_trials=5)

                            required_keys = {
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
                            assert all(k in result for k in required_keys)

    def test_num_trials_zero_raises_valueerror(self, analyzer_setup):
        """Test num_trials=0 raises ValueError"""
        analyzer, _, _ = analyzer_setup

        with pytest.raises(ValueError) as exc_info:
            analyzer.run(num_trials=0)

        assert "num_trials must be positive" in str(exc_info.value)

    def test_num_trials_negative_raises_valueerror(self, analyzer_setup):
        """Test negative num_trials raises ValueError"""
        analyzer, _, _ = analyzer_setup

        with pytest.raises(ValueError) as exc_info:
            analyzer.run(num_trials=-5)

        assert "num_trials must be positive" in str(exc_info.value)

    def test_invalid_criterion_raises_valueerror(self, analyzer_setup):
        """Test unsupported criterion raises ValueError"""
        analyzer, _, _ = analyzer_setup

        with pytest.raises(ValueError) as exc_info:
            analyzer.run(num_trials=5, criterion="unsupported_metric")

        assert "Unsupported criterion" in str(exc_info.value)
        assert "unsupported_metric" in str(exc_info.value)

    def test_yield_calculation_percentage(self, analyzer_setup):
        """Test yield calculated as percentage of passing trials"""
        analyzer, _, _ = analyzer_setup

        # 3 pass (<=0.05), 2 fail (>0.05) = 60% yield
        spot_values = [0.03, 0.02, 0.06, 0.07, 0.04]
        analyzer._spot_rms = Mock(side_effect=[0.02] + spot_values)
        analyzer._apply_tolerances = Mock(return_value={})
        analyzer._optimize_compensators = Mock(return_value={})

        with patch.object(analyzer, "_get_system_state", return_value={}):
            with patch.object(analyzer, "_set_system_state"):
                result = analyzer.run(num_trials=5, criterion_limit=0.05)

                assert result["yield"] == 60.0

    def test_statistics_finite_and_sensible(self, analyzer_setup):
        """Test statistics are calculated and reasonable"""
        analyzer, _, _ = analyzer_setup

        spot_values = [0.01, 0.02, 0.03, 0.02, 0.01]
        analyzer._spot_rms = Mock(side_effect=[0.02] + spot_values)
        analyzer._apply_tolerances = Mock(return_value={})
        analyzer._optimize_compensators = Mock(return_value={})

        with patch.object(analyzer, "_get_system_state", return_value={}):
            with patch.object(analyzer, "_set_system_state"):
                result = analyzer.run(num_trials=5)

                assert math.isfinite(result["mean"])
                assert math.isfinite(result["std_dev"])
                assert result["min"] <= result["mean"] <= result["max"]
                assert result["trials"] == 5

    def test_vignetted_trials_tracked_as_infinity(self, analyzer_setup):
        """Test vignetted trials (inf values) tracked separately"""
        analyzer, _, _ = analyzer_setup

        spot_values = [0.02, 0.03, float("inf"), 0.02, float("inf")]
        analyzer._spot_rms = Mock(side_effect=[0.02] + spot_values)
        analyzer._apply_tolerances = Mock(return_value={})
        analyzer._optimize_compensators = Mock(return_value={})

        with patch.object(analyzer, "_get_system_state", return_value={}):
            with patch.object(analyzer, "_set_system_state"):
                result = analyzer.run(num_trials=5)

                assert result["vignetted_trials"] == 2
                assert math.isinf(result["max"])
                assert math.isfinite(result["mean"])

    def test_exception_in_trial_continues(self, analyzer_setup):
        """Test exception in trial doesn't stop run"""
        analyzer, _, _ = analyzer_setup

        call_count = [0]

        def spot_rms_with_errors(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] in [2, 4]:
                raise RuntimeError("Trace failed")
            return 0.02

        analyzer._spot_rms = Mock(side_effect=spot_rms_with_errors)
        analyzer._apply_tolerances = Mock(return_value={})
        analyzer._optimize_compensators = Mock(return_value={})

        with patch.object(analyzer, "_get_system_state", return_value={}):
            with patch.object(analyzer, "_set_system_state"):
                result = analyzer.run(num_trials=5)

                assert isinstance(result, dict)
                assert result["trials"] == 5

    def test_state_restoration_on_error(self, analyzer_setup):
        """Test system state restored even when trial raises"""
        analyzer, _, _ = analyzer_setup

        analyzer._spot_rms = Mock(return_value=0.02)
        analyzer._apply_tolerances = Mock(side_effect=RuntimeError("Apply failed"))
        analyzer._optimize_compensators = Mock(return_value={})

        set_state_mock = Mock()

        with patch.object(analyzer, "_get_system_state", return_value={}):
            with patch.object(analyzer, "_set_system_state", side_effect=set_state_mock):
                result = analyzer.run(num_trials=3)

                assert set_state_mock.call_count >= 3

    def test_compensators_applied_in_trial(self, analyzer_setup):
        """Test that compensators are retrieved and used in spot calculation"""
        analyzer, _, _ = analyzer_setup

        analyzer._spot_rms = Mock(return_value=0.02)
        analyzer._apply_tolerances = Mock(return_value={})

        focus_shift = {"focus_shift_mm": 0.5}
        analyzer._optimize_compensators = Mock(return_value=focus_shift)

        with patch.object(analyzer, "_get_system_state", return_value={}):
            with patch.object(analyzer, "_set_system_state"):
                result = analyzer.run(num_trials=2)

                # Verify _optimize_compensators was called
                assert analyzer._optimize_compensators.call_count >= 2
                # Verify _spot_rms called with focus_shift
                call_kwargs = [call.kwargs for call in analyzer._spot_rms.call_args_list]
                assert any("focus_shift_mm" in kw for kw in call_kwargs)

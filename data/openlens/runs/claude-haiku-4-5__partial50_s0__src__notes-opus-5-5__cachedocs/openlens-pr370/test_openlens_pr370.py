import pytest
from unittest.mock import Mock, patch
from openlens.performance_metrics import PerformanceMetrics
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem, create_doublet


class TestPerformanceMetricsCalculateBackFocalLength:
    """Test suite for PerformanceMetrics.calculate_back_focal_length()"""

    # ===== Constructor Validation =====

    def test_constructor_rejects_none(self):
        """Constructor raises ValueError when given None"""
        with pytest.raises(ValueError, match="Must provide Lens or OpticalSystem"):
            PerformanceMetrics(None)

    def test_constructor_rejects_invalid_types(self):
        """Constructor raises ValueError for int, dict, string, list, object"""
        for invalid_input in [42, {}, "string", [], object()]:
            with pytest.raises(ValueError, match="Must provide Lens or OpticalSystem"):
                PerformanceMetrics(invalid_input)

    # ===== Lens Delegation =====

    def test_lens_delegates_exactly(self):
        """Returns exact value from Lens.calculate_back_focal_length()"""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=10.0,
            diameter=25.0,
            material="BK7"
        )
        pm = PerformanceMetrics(lens)
        expected = lens.calculate_back_focal_length()
        assert pm.calculate_back_focal_length() == pytest.approx(expected)

    def test_lens_passes_through_none(self):
        """None from Lens.calculate_back_focal_length() is passed through unchanged"""
        lens = Mock(spec=Lens)
        lens.calculate_back_focal_length.return_value = None
        pm = PerformanceMetrics(lens)
        assert pm.calculate_back_focal_length() is None

    def test_lens_thick_lens_vs_thin_approximation(self):
        """Uses thick-lens BFL, not thin-lens f - t/2 approximation"""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=15.0,
            diameter=25.0,
            material="BK7"
        )
        pm = PerformanceMetrics(lens)
        result = pm.calculate_back_focal_length()

        if result is not None:
            # Verify not using thin-lens approximation
            f = lens.calculate_focal_length()
            thin_approx = f - lens.thickness / 2
            assert result != pytest.approx(thin_approx, rel=1e-2)

    # ===== System Delegation =====

    def test_system_delegates_exactly(self):
        """Returns exact value from OpticalSystem.calculate_back_focal_length()"""
        system = create_doublet(100.0, 50.0)
        pm = PerformanceMetrics(system)
        expected = system.calculate_back_focal_length()
        assert pm.calculate_back_focal_length() == pytest.approx(expected)

    def test_system_called_with_no_arguments(self):
        """OpticalSystem.calculate_back_focal_length() is called with no args"""
        system = Mock(spec=OpticalSystem)
        system.calculate_back_focal_length.return_value = 42.5
        pm = PerformanceMetrics(system)

        result = pm.calculate_back_focal_length()
        assert result == 42.5
        system.calculate_back_focal_length.assert_called_once_with()

    def test_system_bfl_not_efl_regression(self):
        """System BFL differs from EFL (not the EFL bug regression)"""
        system = create_doublet(100.0, 50.0)
        pm = PerformanceMetrics(system)
        bfl = pm.calculate_back_focal_length()
        efl = system.get_system_focal_length()

        if bfl is not None and efl is not None:
            # Principal planes offset from last surface; should differ noticeably
            assert bfl != pytest.approx(efl, rel=1e-3)

    # ===== Priority: Lens Wins Over System =====

    def test_lens_takes_priority_over_system(self):
        """When both are set, Lens is used and System is ignored"""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        system = Mock(spec=OpticalSystem)
        system.calculate_back_focal_length.return_value = 999.0

        pm = PerformanceMetrics(lens)
        pm.system = system  # Manually set system to verify it's ignored

        expected = lens.calculate_back_focal_length()
        assert pm.calculate_back_focal_length() == pytest.approx(expected)
        system.calculate_back_focal_length.assert_not_called()

    # ===== Fallback Case =====

    def test_returns_none_when_both_cleared(self):
        """Returns None when both lens and system are manually set to None"""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        pm = PerformanceMetrics(lens)
        pm.lens = None
        pm.system = None

        assert pm.calculate_back_focal_length() is None

    # ===== Monkeypatch / Spy Tests =====

    def test_monkeypatch_lens_method(self):
        """Monkeypatched Lens method is called and result returned"""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        pm = PerformanceMetrics(lens)

        with patch.object(lens, 'calculate_back_focal_length', return_value=123.456):
            assert pm.calculate_back_focal_length() == 123.456

    def test_monkeypatch_system_method(self):
        """Monkeypatched OpticalSystem method is called and result returned"""
        system = create_doublet(100.0, 50.0)
        pm = PerformanceMetrics(system)

        with patch.object(system, 'calculate_back_focal_length', return_value=77.777):
            assert pm.calculate_back_focal_length() == 77.777

    def test_system_method_called_with_exact_signature(self):
        """System method is called with exactly zero positional and keyword args"""
        system = create_doublet(100.0, 50.0)
        pm = PerformanceMetrics(system)

        calls = []
        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return 42.0

        with patch.object(system, 'calculate_back_focal_length', side_effect=spy):
            pm.calculate_back_focal_length()
            assert calls == [((), {})]

    # ===== Integration Tests =====

    def test_integration_thick_biconvex_lens(self):
        """Integration: thick biconvex lens returns valid numeric BFL"""
        lens = Lens(
            name="Thick Biconvex",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=10.0,
            diameter=25.0,
            material="BK7"
        )
        pm = PerformanceMetrics(lens)
        bfl = pm.calculate_back_focal_length()

        assert isinstance(bfl, (int, float)) or bfl is None
        assert bfl is not None

    def test_integration_optical_doublet_system(self):
        """Integration: doublet system returns valid numeric BFL"""
        system = create_doublet(100.0, 50.0)
        pm = PerformanceMetrics(system)
        bfl = pm.calculate_back_focal_length()

        assert isinstance(bfl, (int, float)) or bfl is None
        assert bfl is not None

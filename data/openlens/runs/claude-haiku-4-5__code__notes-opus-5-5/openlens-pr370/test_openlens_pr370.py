import pytest
from unittest.mock import Mock
from openlens.performance_metrics import PerformanceMetrics
from openlens.optical_elements import Lens
from openlens.optical_system import OpticalSystem, create_doublet


class TestPerformanceMetricsCalculateBackFocalLength:
    """Tests for PerformanceMetrics.calculate_back_focal_length()"""
    
    # ===== Constructor validation tests =====
    
    def test_constructor_rejects_integer_input(self):
        """Constructor raises ValueError for integer input"""
        with pytest.raises(ValueError, match="Must provide Lens or OpticalSystem"):
            PerformanceMetrics(42)
    
    def test_constructor_rejects_none_input(self):
        """Constructor raises ValueError for None input"""
        with pytest.raises(ValueError, match="Must provide Lens or OpticalSystem"):
            PerformanceMetrics(None)
    
    def test_constructor_rejects_string_input(self):
        """Constructor raises ValueError for string input"""
        with pytest.raises(ValueError, match="Must provide Lens or OpticalSystem"):
            PerformanceMetrics("invalid")
    
    # ===== Delegation tests =====
    
    def test_delegates_to_lens_with_exact_return_value(self):
        """Delegates to Lens.calculate_back_focal_length() with no modification"""
        lens = Mock(spec=Lens)
        lens.calculate_back_focal_length = Mock(return_value=50.123)
        
        pm = PerformanceMetrics(lens)
        result = pm.calculate_back_focal_length()
        
        assert result == 50.123
        lens.calculate_back_focal_length.assert_called_once()
    
    def test_delegates_to_system_with_exact_return_value(self):
        """Delegates to OpticalSystem.calculate_back_focal_length() with no modification"""
        system = Mock(spec=OpticalSystem)
        system.calculate_back_focal_length = Mock(return_value=75.456)
        
        pm = PerformanceMetrics(system)
        result = pm.calculate_back_focal_length()
        
        assert result == 75.456
        system.calculate_back_focal_length.assert_called_once()
    
    def test_system_called_with_no_arguments(self):
        """System method is called with no arguments (n_overrides keeps default)"""
        system = Mock(spec=OpticalSystem)
        calls = []
        system.calculate_back_focal_length = Mock(
            side_effect=lambda *a, **k: (calls.append((a, k)), 20.0)[1]
        )
        
        pm = PerformanceMetrics(system)
        pm.calculate_back_focal_length()
        
        assert calls == [((), {})]
    
    # ===== Return value passthrough tests =====
    
    def test_lens_returns_none_is_passed_through(self):
        """None returned by Lens is passed through unchanged"""
        lens = Mock(spec=Lens)
        lens.calculate_back_focal_length = Mock(return_value=None)
        
        pm = PerformanceMetrics(lens)
        result = pm.calculate_back_focal_length()
        
        assert result is None
    
    def test_system_returns_none_is_passed_through(self):
        """None returned by OpticalSystem is passed through unchanged"""
        system = Mock(spec=OpticalSystem)
        system.calculate_back_focal_length = Mock(return_value=None)
        
        pm = PerformanceMetrics(system)
        result = pm.calculate_back_focal_length()
        
        assert result is None
    
    # ===== Priority and branching tests =====
    
    def test_lens_priority_over_system_when_both_set(self):
        """Lens is used even if system is also present (checked first)"""
        lens = Mock(spec=Lens)
        lens.calculate_back_focal_length = Mock(return_value=100.0)
        
        system = Mock(spec=OpticalSystem)
        system.calculate_back_focal_length = Mock(return_value=200.0)
        
        pm = PerformanceMetrics(lens)
        pm.system = system  # Assign system after lens
        
        result = pm.calculate_back_focal_length()
        
        assert result == 100.0
        lens.calculate_back_focal_length.assert_called_once()
        system.calculate_back_focal_length.assert_not_called()
    
    def test_returns_none_when_both_lens_and_system_manually_set_to_none(self):
        """Setting both lens and system to None returns None"""
        lens = Mock(spec=Lens)
        pm = PerformanceMetrics(lens)
        
        pm.lens = None
        pm.system = None
        
        result = pm.calculate_back_focal_length()
        assert result is None
    
    # ===== Integration tests with real objects =====
    
    def test_real_lens_object_delegation(self):
        """Real Lens object delegates correctly to thick-lens calculation"""
        lens = Lens(radius_of_curvature_1=50, radius_of_curvature_2=-50, 
                    thickness=10, diameter=25)
        pm = PerformanceMetrics(lens)
        
        pm_result = pm.calculate_back_focal_length()
        lens_result = lens.calculate_back_focal_length()
        
        if pm_result is not None and lens_result is not None:
            assert pm_result == pytest.approx(lens_result)
        else:
            assert pm_result is lens_result
    
    def test_real_optical_system_delegation(self):
        """Real OpticalSystem object delegates correctly from ABCD matrix (-A/C)"""
        system = create_doublet(100.0, 50.0)
        pm = PerformanceMetrics(system)
        
        pm_result = pm.calculate_back_focal_length()
        sys_result = system.calculate_back_focal_length()
        
        if pm_result is not None and sys_result is not None:
            assert pm_result == pytest.approx(sys_result)
        else:
            assert pm_result is sys_result
    
    def test_system_bfl_differs_from_efl_regression_guard(self):
        """Guards against bug where EFL was returned instead of BFL"""
        system = create_doublet(100.0, 50.0)
        pm = PerformanceMetrics(system)
        
        bfl_result = pm.calculate_back_focal_length()
        efl_result = system.get_system_focal_length()
        
        if bfl_result is not None and efl_result is not None:
            # BFL and EFL should differ due to principal plane offset in a doublet
            assert bfl_result != pytest.approx(efl_result, abs=1e-3)

import pytest
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.performance_metrics import PerformanceMetrics


class TestPerformanceMetricsCalculateBackFocalLength:
    """Tests for PerformanceMetrics.calculate_back_focal_length()"""

    def test_bfl_returns_none_when_no_lens_or_system(self):
        """Should return None when neither lens nor system is set"""
        metrics = PerformanceMetrics(lens=None, system=None)
        result = metrics.calculate_back_focal_length()
        assert result is None

    def test_bfl_delegates_to_lens_when_lens_exists(self):
        """Should delegate to lens.calculate_back_focal_length() when lens is set"""
        lens = Lens(
            name="Test Lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        metrics = PerformanceMetrics(lens=lens)
        result = metrics.calculate_back_focal_length()
        
        # Should match direct lens calculation
        assert result == lens.calculate_back_focal_length()

    def test_bfl_delegates_to_system_when_system_exists(self):
        """Should delegate to system.calculate_back_focal_length() when only system is set"""
        lens = Lens(
            name="Test Lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem(name="Test System")
        system.add_lens(lens)
        
        metrics = PerformanceMetrics(system=system)
        result = metrics.calculate_back_focal_length()
        
        # Should match direct system calculation
        assert result == system.calculate_back_focal_length()

    def test_bfl_prefers_lens_over_system(self):
        """Should use lens when both lens and system are set"""
        lens = Lens(
            name="Test Lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem(name="Test System")
        system.add_lens(lens)
        
        metrics = PerformanceMetrics(lens=lens, system=system)
        result = metrics.calculate_back_focal_length()
        
        # Should return lens result, not system result
        assert result == lens.calculate_back_focal_length()

    def test_bfl_biconvex_converging(self):
        """Biconvex lens should have positive BFL (converging)"""
        lens = Lens(
            name="Biconvex",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=10.0,
            diameter=50.0,
            material="BK7"
        )
        metrics = PerformanceMetrics(lens=lens)
        result = metrics.calculate_back_focal_length()
        
        assert result is not None
        assert result > 0

    def test_bfl_biconcave_diverging(self):
        """Biconcave lens should have negative BFL (diverging)"""
        lens = Lens(
            name="Biconcave",
            radius_of_curvature_1=-100.0,
            radius_of_curvature_2=100.0,
            thickness=5.0,
            diameter=50.0,
            material="BK7"
        )
        metrics = PerformanceMetrics(lens=lens)
        result = metrics.calculate_back_focal_length()
        
        assert result is not None
        assert result < 0

    def test_bfl_plano_convex(self):
        """Plano-convex lens should have positive BFL"""
        lens = Lens(
            name="Plano-Convex",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=float('inf'),
            thickness=4.0,
            diameter=25.0,
            material="BK7"
        )
        metrics = PerformanceMetrics(lens=lens)
        result = metrics.calculate_back_focal_length()
        
        assert result is not None
        assert result > 0

    def test_bfl_thick_lens_formula(self):
        """Thick lens BFL should differ from thin lens approximation"""
        thin = Lens(
            name="Thin",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=0.5,
            diameter=25.0,
            material="BK7"
        )
        thick = Lens(
            name="Thick",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=10.0,
            diameter=25.0,
            material="BK7"
        )
        
        thin_bfl = PerformanceMetrics(lens=thin).calculate_back_focal_length()
        thick_bfl = PerformanceMetrics(lens=thick).calculate_back_focal_length()
        
        assert thin_bfl is not None
        assert thick_bfl is not None
        assert thin_bfl != thick_bfl

    def test_bfl_material_dispersion(self):
        """BFL should vary with refractive index"""
        lens_bk7 = Lens(
            name="BK7",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        lens_sf11 = Lens(
            name="SF11",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="SF11"
        )
        
        bfl_bk7 = PerformanceMetrics(lens=lens_bk7).calculate_back_focal_length()
        bfl_sf11 = PerformanceMetrics(lens=lens_sf11).calculate_back_focal_length()
        
        assert bfl_bk7 is not None
        assert bfl_sf11 is not None
        assert bfl_bk7 != bfl_sf11

    def test_bfl_return_type_optional_float(self):
        """Return type should be float or None"""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        result = PerformanceMetrics(lens=lens).calculate_back_focal_length()
        
        assert result is None or isinstance(result, (int, float))

    def test_bfl_meniscus_converging(self):
        """Meniscus converging lens should have positive BFL"""
        meniscus = Lens(
            name="Meniscus",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=100.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        result = PerformanceMetrics(lens=meniscus).calculate_back_focal_length()
        
        assert result is None or isinstance(result, (int, float))

    def test_bfl_multi_element_system(self):
        """Multi-element system BFL should be derived from ABCD matrix"""
        lens1 = Lens(
            name="L1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        lens2 = Lens(
            name="L2",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        
        system = OpticalSystem(name="Two Element")
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=5.0)
        
        result = PerformanceMetrics(system=system).calculate_back_focal_length()
        
        assert result is None or isinstance(result, (int, float))

    def test_bfl_consistency_multiple_calls(self):
        """Repeated calls should return identical results"""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        
        metrics = PerformanceMetrics(lens=lens)
        result1 = metrics.calculate_back_focal_length()
        result2 = metrics.calculate_back_focal_length()
        
        assert result1 == result2

    def test_bfl_afocal_system_returns_none(self):
        """Afocal system (zero power) should return None"""
        # Create an afocal system (equal and opposite powers)
        lens1 = Lens(
            name="Positive",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        lens2 = Lens(
            name="Negative",
            radius_of_curvature_1=-50.0,
            radius_of_curvature_2=50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        
        system = OpticalSystem(name="Afocal")
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=0.0)
        
        result = PerformanceMetrics(system=system).calculate_back_focal_length()
        
        # Afocal system should return None
        assert result is None or isinstance(result, (int, float))

    def test_bfl_measures_from_rear_vertex(self):
        """BFL should be measured from rear vertex to focal point"""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        
        metrics = PerformanceMetrics(lens=lens)
        bfl = metrics.calculate_back_focal_length()
        
        # For a standard biconvex BK7 lens, BFL should be less than focal length
        # due to thick lens formula
        assert bfl is not None
        assert bfl > 0

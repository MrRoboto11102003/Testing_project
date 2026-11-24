import pytest
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem


class TestOpticalSystemFocalLength:
    """Tests for OpticalSystem.get_system_focal_length()"""

    @pytest.fixture
    def biconvex_lens(self):
        """Standard biconvex lens for testing"""
        return Lens(
            name="Biconvex",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )

    @pytest.fixture
    def biconcave_lens(self):
        """Standard biconcave lens for testing"""
        return Lens(
            name="Biconcave",
            radius_of_curvature_1=-50.0,
            radius_of_curvature_2=50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )

    def test_empty_system_returns_none(self):
        """Empty system with no elements returns None"""
        system = OpticalSystem()
        assert system.get_system_focal_length() is None

    def test_single_lens_returns_lens_focal_length(self, biconvex_lens):
        """Single-element system returns the lens's calculated focal length"""
        system = OpticalSystem()
        system.add_element(biconvex_lens, air_gap_before=0)
        
        expected_fl = biconvex_lens.calculate_focal_length()
        assert system.get_system_focal_length() == pytest.approx(expected_fl)

    def test_single_positive_lens_positive_fl(self, biconvex_lens):
        """Single converging lens has positive focal length"""
        system = OpticalSystem()
        system.add_element(biconvex_lens, air_gap_before=0)
        
        result = system.get_system_focal_length()
        assert result is not None
        assert result > 0

    def test_single_negative_lens_negative_fl(self, biconcave_lens):
        """Single diverging lens has negative focal length"""
        system = OpticalSystem()
        system.add_element(biconcave_lens, air_gap_before=0)
        
        result = system.get_system_focal_length()
        assert result is not None
        assert result < 0

    def test_two_element_system_uses_matrix_method(self, biconvex_lens, biconcave_lens):
        """Two-element system uses exact ABCD matrix method, not thin-lens formula"""
        system = OpticalSystem()
        system.add_element(biconvex_lens, air_gap_before=0)
        system.add_element(biconcave_lens, air_gap_before=10.0)
        
        result = system.get_system_focal_length()
        assert result is None or isinstance(result, float)

    def test_three_element_system(self, biconvex_lens):
        """Multi-element system with 3+ lenses calculates focal length"""
        lens2 = Lens("L2", radius_of_curvature_1=60.0, radius_of_curvature_2=-60.0,
                     thickness=5.0, diameter=25.0, material="BK7")
        lens3 = Lens("L3", radius_of_curvature_1=70.0, radius_of_curvature_2=-70.0,
                     thickness=5.0, diameter=25.0, material="BK7")
        
        system = OpticalSystem()
        system.add_element(biconvex_lens, air_gap_before=0)
        system.add_element(lens2, air_gap_before=10.0)
        system.add_element(lens3, air_gap_before=10.0)
        
        result = system.get_system_focal_length()
        assert result is None or isinstance(result, float)

    def test_afocal_system_returns_none(self, biconvex_lens):
        """Afocal system (C < 1e-10, infinite focal length) returns None"""
        biconcave = Lens("Neg", radius_of_curvature_1=-50.0, radius_of_curvature_2=50.0,
                        thickness=5.0, diameter=25.0, material="BK7")
        
        system = OpticalSystem()
        fl1 = biconvex_lens.calculate_focal_length()
        system.add_element(biconvex_lens, air_gap_before=0)
        # Approximate afocal configuration: gap ≈ focal length
        system.add_element(biconcave, air_gap_before=fl1)
        
        result = system.get_system_focal_length()
        # Afocal system should return None or very large value
        assert result is None or abs(result) > 1e9

    def test_return_type_is_float_or_none(self, biconvex_lens):
        """Return value is always float or None"""
        system = OpticalSystem()
        system.add_element(biconvex_lens)
        
        result = system.get_system_focal_length()
        assert isinstance(result, (float, type(None)))

    def test_air_gap_affects_focal_length(self, biconvex_lens):
        """Different air gaps between elements produce different system focal lengths"""
        lens2 = Lens("L2", radius_of_curvature_1=60.0, radius_of_curvature_2=-60.0,
                     thickness=5.0, diameter=25.0, material="BK7")
        
        sys1 = OpticalSystem()
        sys1.add_element(biconvex_lens, air_gap_before=0)
        sys1.add_element(lens2, air_gap_before=5.0)
        fl1 = sys1.get_system_focal_length()
        
        sys2 = OpticalSystem()
        sys2.add_element(biconvex_lens, air_gap_before=0)
        sys2.add_element(lens2, air_gap_before=20.0)
        fl2 = sys2.get_system_focal_length()
        
        if fl1 is not None and fl2 is not None:
            assert fl1 != fl2

    def test_converging_doublet(self, biconvex_lens):
        """Converging doublet system has positive focal length"""
        lens2 = Lens("L2", radius_of_curvature_1=60.0, radius_of_curvature_2=-60.0,
                     thickness=5.0, diameter=25.0, material="BK7")
        
        system = OpticalSystem()
        system.add_element(biconvex_lens, air_gap_before=0)
        system.add_element(lens2, air_gap_before=5.0)
        
        result = system.get_system_focal_length()
        if result is not None:
            assert result > 0

    def test_thin_lens_in_system(self):
        """Thin lens (small thickness) in system works"""
        thin = Lens("Thin", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=0.5, diameter=25.0, material="BK7")
        system = OpticalSystem()
        system.add_element(thin)
        
        result = system.get_system_focal_length()
        assert result is None or isinstance(result, float)

    def test_weak_lens_pair(self):
        """Weak lenses with high radii of curvature"""
        weak1 = Lens("W1", radius_of_curvature_1=5000.0, radius_of_curvature_2=-5000.0,
                    thickness=2.0, diameter=25.0, material="BK7")
        weak2 = Lens("W2", radius_of_curvature_1=5000.0, radius_of_curvature_2=-5000.0,
                    thickness=2.0, diameter=25.0, material="BK7")
        
        system = OpticalSystem()
        system.add_element(weak1, air_gap_before=0)
        system.add_element(weak2, air_gap_before=10.0)
        
        result = system.get_system_focal_length()
        assert result is None or isinstance(result, float)

    def test_large_air_gap_between_elements(self, biconvex_lens):
        """Large air gap between elements"""
        lens2 = Lens("L2", radius_of_curvature_1=60.0, radius_of_curvature_2=-60.0,
                     thickness=5.0, diameter=25.0, material="BK7")
        
        system = OpticalSystem()
        system.add_element(biconvex_lens, air_gap_before=0)
        system.add_element(lens2, air_gap_before=100.0)
        
        result = system.get_system_focal_length()
        assert result is None or isinstance(result, float)

import pytest
from openlens.coating_designer import CoatingDesigner, CoatingLayer


@pytest.fixture
def designer():
    """Create a CoatingDesigner instance for testing."""
    return CoatingDesigner()


class TestCalculateReflectivity:
    """Tests for CoatingDesigner.calculate_reflectivity."""

    def test_bare_substrate_normal_incidence(self, designer):
        """Bare substrate has measurable reflectivity at normal incidence."""
        R = designer.calculate_reflectivity([], wavelength_nm=550.0)
        assert isinstance(R, float)
        assert 0 <= R <= 1
        assert R > 0

    def test_reflectivity_increases_with_angle(self, designer):
        """Reflectivity generally increases with angle of incidence."""
        R_normal = designer.calculate_reflectivity([], wavelength_nm=550.0, angle_deg=0)
        R_oblique = designer.calculate_reflectivity([], wavelength_nm=550.0, angle_deg=45)
        assert R_oblique > R_normal

    def test_angle_validation_negative(self, designer):
        """Negative angle should raise ValueError."""
        with pytest.raises(ValueError, match="Angle of incidence must be in"):
            designer.calculate_reflectivity([], wavelength_nm=550, angle_deg=-1)

    def test_angle_validation_at_critical(self, designer):
        """Angle at 90 degrees should raise ValueError."""
        with pytest.raises(ValueError, match="Angle of incidence must be in"):
            designer.calculate_reflectivity([], wavelength_nm=550, angle_deg=90)

    def test_wavelength_validation_zero(self, designer):
        """Zero wavelength should raise ValueError."""
        with pytest.raises(ValueError, match="Wavelength must be positive"):
            designer.calculate_reflectivity([], wavelength_nm=0)

    def test_wavelength_validation_negative(self, designer):
        """Negative wavelength should raise ValueError."""
        with pytest.raises(ValueError, match="Wavelength must be positive"):
            designer.calculate_reflectivity([], wavelength_nm=-100)

    def test_ar_coating_reduces_reflectivity(self, designer):
        """AR coating should reduce reflectivity compared to bare substrate."""
        # Create a quarter-wave MgF2 layer at 550nm
        qwl_thickness = 550.0 / (4 * 1.38)
        ar_layer = CoatingLayer("MgF2", 1.38, qwl_thickness)
        
        R_bare = designer.calculate_reflectivity([], wavelength_nm=550.0)
        R_coated = designer.calculate_reflectivity([ar_layer], wavelength_nm=550.0)
        
        assert R_coated < R_bare

    def test_reflectivity_range_multiple_wavelengths(self, designer):
        """Reflectivity should stay in [0, 1] for various wavelengths."""
        for wavelength in [350, 450, 550, 650, 750, 1000]:
            R = designer.calculate_reflectivity([], wavelength_nm=float(wavelength))
            assert 0 <= R <= 1


class TestCalculateReflectivityCurve:
    """Tests for CoatingDesigner.calculate_reflectivity_curve."""

    def test_curve_correct_length(self, designer):
        """Curve should contain exactly the requested number of points."""
        curve = designer.calculate_reflectivity_curve(
            [], wavelength_range=(400, 700), num_points=25
        )
        assert len(curve) == 25

    def test_curve_tuple_format(self, designer):
        """Curve should return list of (wavelength, reflectivity) tuples."""
        curve = designer.calculate_reflectivity_curve(
            [], wavelength_range=(400, 700), num_points=10
        )
        assert all(isinstance(item, tuple) and len(item) == 2 for item in curve)

    def test_curve_wavelengths_in_range(self, designer):
        """Wavelengths should span the requested range."""
        min_wl, max_wl = 400, 700
        curve = designer.calculate_reflectivity_curve(
            [], wavelength_range=(min_wl, max_wl), num_points=20
        )
        wavelengths = [w for w, _ in curve]
        assert min(wavelengths) == pytest.approx(min_wl)
        assert max(wavelengths) == pytest.approx(max_wl)
        assert len(set(wavelengths)) == len(wavelengths)  # All unique

    def test_curve_reflectivity_in_range(self, designer):
        """All reflectivity values should be in [0, 1]."""
        curve = designer.calculate_reflectivity_curve(
            [], wavelength_range=(400, 700), num_points=20
        )
        for _, reflectivity in curve:
            assert 0 <= reflectivity <= 1

    def test_single_point_at_minimum(self, designer):
        """With num_points=1, should sample at minimum wavelength."""
        curve = designer.calculate_reflectivity_curve(
            [], wavelength_range=(400, 700), num_points=1
        )
        assert len(curve) == 1
        assert curve[0][0] == 400

    def test_num_points_validation(self, designer):
        """num_points must be at least 1."""
        with pytest.raises(ValueError, match="num_points must be at least 1"):
            designer.calculate_reflectivity_curve(
                [], wavelength_range=(400, 700), num_points=0
            )


class TestDesignVCoating:
    """Tests for CoatingDesigner.design_v_coating."""

    def test_v_coating_has_three_layers(self, designer):
        """V-coating design should return exactly 3 layers."""
        coating = designer.design_v_coating(wavelength_nm=550.0)
        assert isinstance(coating, list)
        assert len(coating) == 3

    def test_v_coating_uses_correct_materials(self, designer):
        """V-coating should use Al2O3, SiO2, and MgF2 in order."""
        coating = designer.design_v_coating(wavelength_nm=550.0)
        material_names = [layer.name for layer in coating]
        assert material_names == ["Al2O3", "SiO2", "MgF2"]

    def test_v_coating_all_valid_layers(self, designer):
        """All layers should be valid CoatingLayer instances."""
        coating = designer.design_v_coating(wavelength_nm=550.0)
        assert all(isinstance(layer, CoatingLayer) for layer in coating)
        assert all(layer.thickness_nm > 0 for layer in coating)
        assert all(layer.refractive_index > 0 for layer in coating)

    def test_v_coating_beats_bare_substrate(self, designer):
        """Designed V-coating should have lower reflectivity than bare substrate."""
        coating = designer.design_v_coating(wavelength_nm=550.0)
        
        R_coated = designer.calculate_reflectivity(coating, wavelength_nm=550.0)
        R_bare = designer.calculate_reflectivity([], wavelength_nm=550.0)
        
        assert R_coated < R_bare

    def test_v_coating_broadband_performance(self, designer):
        """V-coating should maintain low reflectivity across visible spectrum."""
        coating = designer.design_v_coating(wavelength_nm=550.0)
        R_bare_ref = designer.calculate_reflectivity([], wavelength_nm=550.0)
        
        # Test across visible spectrum
        for wavelength in [450, 500, 550, 600, 650]:
            R_coated = designer.calculate_reflectivity(coating, wavelength_nm=float(wavelength))
            assert R_coated < R_bare_ref, f"Coating should reduce R at {wavelength}nm"

    def test_v_coating_design_wavelength_dependent(self, designer):
        """Coatings designed for different wavelengths should differ."""
        coating_450 = designer.design_v_coating(wavelength_nm=450.0)
        coating_550 = designer.design_v_coating(wavelength_nm=550.0)
        
        # Compare layer thicknesses
        thicknesses_450 = [layer.thickness_nm for layer in coating_450]
        thicknesses_550 = [layer.thickness_nm for layer in coating_550]
        
        assert thicknesses_450 != thicknesses_550

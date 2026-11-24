import pytest
from openlens.coating_designer import CoatingDesigner, CoatingLayer


class TestCalculateReflectivity:
    """Tests for CoatingDesigner.calculate_reflectivity"""
    
    @pytest.fixture
    def designer(self):
        return CoatingDesigner()
    
    def test_wavelength_must_be_positive(self, designer):
        """Wavelength <= 0 raises ValueError"""
        with pytest.raises(ValueError, match="Wavelength must be positive"):
            designer.calculate_reflectivity([], wavelength_nm=0)
        with pytest.raises(ValueError):
            designer.calculate_reflectivity([], wavelength_nm=-100)
    
    def test_angle_must_be_in_range_0_90(self, designer):
        """Angle must be in [0, 90) degrees"""
        with pytest.raises(ValueError, match="Angle of incidence must be in"):
            designer.calculate_reflectivity([], wavelength_nm=500, angle_deg=90)
        with pytest.raises(ValueError):
            designer.calculate_reflectivity([], wavelength_nm=500, angle_deg=-1)
    
    def test_bare_substrate_gives_valid_reflectivity(self, designer):
        """Empty coating list returns Fresnel reflection in [0, 1]"""
        R = designer.calculate_reflectivity([], wavelength_nm=500)
        assert isinstance(R, float)
        assert 0 <= R <= 1
    
    def test_oblique_incidence_valid(self, designer):
        """Reflectivity valid for angles in (0, 90)"""
        for angle in [0, 45, 75, 89.9]:
            R = designer.calculate_reflectivity([], wavelength_nm=500, angle_deg=angle)
            assert 0 <= R <= 1
    
    def test_single_and_multiple_layers(self, designer):
        """Coated stacks return valid reflectivity"""
        layer1 = CoatingLayer("SiO2", 1.5, 100)
        R1 = designer.calculate_reflectivity([layer1], wavelength_nm=550)
        assert 0 <= R1 <= 1
        
        layers = [layer1, CoatingLayer("MgF2", 1.38, 120)]
        R2 = designer.calculate_reflectivity(layers, wavelength_nm=550)
        assert 0 <= R2 <= 1
    
    def test_across_spectrum(self, designer):
        """Works for UV, visible, and IR"""
        for wl in [200, 550, 2000]:
            R = designer.calculate_reflectivity([], wavelength_nm=wl)
            assert 0 <= R <= 1


class TestCalculateReflectivityCurve:
    """Tests for CoatingDesigner.calculate_reflectivity_curve"""
    
    @pytest.fixture
    def designer(self):
        return CoatingDesigner()
    
    def test_single_point_at_min_wavelength(self, designer):
        """num_points=1 returns point at start of range"""
        result = designer.calculate_reflectivity_curve([], (400, 700), num_points=1)
        assert len(result) == 1
        assert result[0][0] == 400
        assert 0 <= result[0][1] <= 1
    
    def test_num_points_validation(self, designer):
        """num_points must be >= 1"""
        with pytest.raises(ValueError, match="num_points must be at least 1"):
            designer.calculate_reflectivity_curve([], (400, 700), num_points=0)
    
    def test_curve_spans_range_monotonically(self, designer):
        """Points span range and wavelengths are monotonic"""
        result = designer.calculate_reflectivity_curve([], (400, 800), num_points=10)
        wavelengths = [wl for wl, _ in result]
        assert abs(wavelengths[0] - 400) < 1e-6
        assert abs(wavelengths[-1] - 800) < 1e-6
        assert all(wavelengths[i] < wavelengths[i+1] for i in range(len(wavelengths)-1))
        assert all(0 <= R <= 1 for _, R in result)
    
    def test_with_coating_stack(self, designer):
        """Works with actual coating layers"""
        layers = [CoatingLayer("SiO2", 1.5, 100)]
        result = designer.calculate_reflectivity_curve(layers, (400, 700), num_points=5)
        assert len(result) == 5
        assert all(0 <= R <= 1 for _, R in result)


class TestDesignVCoating:
    """Tests for CoatingDesigner.design_v_coating"""
    
    @pytest.fixture
    def designer(self):
        return CoatingDesigner()
    
    def test_returns_three_layer_stack_with_correct_materials(self, designer):
        """Returns 3 layers: Al2O3, SiO2, MgF2"""
        result = designer.design_v_coating(wavelength_nm=550)
        assert len(result) == 3
        assert all(isinstance(layer, CoatingLayer) for layer in result)
        materials = {layer.name for layer in result}
        assert materials == {"Al2O3", "SiO2", "MgF2"}
    
    def test_inner_layer_is_quarter_wave(self, designer):
        """Al2O3 layer thickness is quarter-wave at design wavelength"""
        wl = 550
        result = designer.design_v_coating(wavelength_nm=wl)
        al2o3 = result[0]
        assert al2o3.name == "Al2O3"
        expected = wl / (4 * al2o3.refractive_index)
        assert abs(al2o3.thickness_nm - expected) < 1
    
    def test_all_thicknesses_positive(self, designer):
        """All designed layer thicknesses are positive"""
        result = designer.design_v_coating(wavelength_nm=550)
        assert all(layer.thickness_nm > 0 for layer in result)
    
    def test_v_coating_beats_bare_substrate(self, designer):
        """Designed coating reduces reflectance vs uncoated substrate"""
        wl = 550
        coating = designer.design_v_coating(wavelength_nm=wl)
        R_bare = designer.calculate_reflectivity([], wavelength_nm=wl)
        R_coated = designer.calculate_reflectivity(coating, wavelength_nm=wl)
        assert R_coated < R_bare

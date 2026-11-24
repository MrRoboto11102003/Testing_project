import pytest
import math
import re
from openlens.coating_designer import CoatingDesigner, CoatingLayer


class TestCalculateReflectivity:
    """Test CoatingDesigner.calculate_reflectivity method."""

    def test_bare_substrate_1_5_normal_incidence(self):
        """Test bare Fresnel reflection with n=1.5 at normal incidence."""
        designer = CoatingDesigner(substrate_index=1.5)
        R = designer.calculate_reflectivity([], 550.0, 0.0)
        expected = (0.5 / 2.5) ** 2
        assert R == pytest.approx(expected, rel=1e-9)

    def test_bare_bk7_default_normal_incidence(self):
        """Test bare BK7 (default n=1.5168) at normal incidence."""
        designer = CoatingDesigner()
        R = designer.calculate_reflectivity([], 550.0, 0.0)
        expected = ((0.5168 / 2.5168) ** 2)
        assert R == pytest.approx(expected, rel=1e-9)

    def test_bare_wavelength_independent(self):
        """Bare Fresnel reflectance does not depend on wavelength."""
        designer = CoatingDesigner()
        R1 = designer.calculate_reflectivity([], 400.0, 0.0)
        R2 = designer.calculate_reflectivity([], 550.0, 0.0)
        R3 = designer.calculate_reflectivity([], 700.0, 0.0)
        assert R1 == pytest.approx(R2, abs=1e-15)
        assert R2 == pytest.approx(R3, abs=1e-15)

    def test_quarter_wave_mgf2_layer(self):
        """Test quarter-wave MgF2 coating on BK7."""
        designer = CoatingDesigner(substrate_index=1.5168)
        wl = 550.0
        n_mgf2 = 1.38
        thickness = wl / (4 * n_mgf2)
        layers = [CoatingLayer("MgF2", n_mgf2, thickness)]
        R = designer.calculate_reflectivity(layers, wl, 0.0)
        expected = ((1.5168 - n_mgf2**2) / (1.5168 + n_mgf2**2)) ** 2
        assert R == pytest.approx(expected, abs=1e-12)

    def test_zero_thickness_layer_equals_bare(self):
        """Zero-thickness layer acts as if absent."""
        designer = CoatingDesigner()
        layers = [CoatingLayer("MgF2", 1.38, 0.0)]
        R_bare = designer.calculate_reflectivity([], 550.0, 0.0)
        R_layer = designer.calculate_reflectivity(layers, 550.0, 0.0)
        assert R_layer == pytest.approx(R_bare, abs=1e-15)

    def test_half_wave_layer_equals_bare(self):
        """Half-wave layer acts as if absent at normal incidence."""
        designer = CoatingDesigner()
        wl = 550.0
        n_mgf2 = 1.38
        thickness = wl / (2 * n_mgf2)
        layers = [CoatingLayer("MgF2", n_mgf2, thickness)]
        R_bare = designer.calculate_reflectivity([], wl, 0.0)
        R_layer = designer.calculate_reflectivity(layers, wl, 0.0)
        assert R_layer == pytest.approx(R_bare, abs=1e-10)

    def test_wavelength_zero_raises(self):
        """Wavelength of 0 raises ValueError with specific message."""
        designer = CoatingDesigner()
        with pytest.raises(ValueError, match=re.escape("Wavelength must be positive, got 0")):
            designer.calculate_reflectivity([], 0, 0.0)

    def test_negative_wavelength_raises(self):
        """Negative wavelength raises ValueError."""
        designer = CoatingDesigner()
        with pytest.raises(ValueError, match=re.escape("Wavelength must be positive, got -5.0")):
            designer.calculate_reflectivity([], -5.0, 0.0)

    def test_negative_angle_raises(self):
        """Negative angle raises ValueError."""
        designer = CoatingDesigner()
        with pytest.raises(ValueError, match=re.escape("Angle of incidence must be in [0, 90), got -1")):
            designer.calculate_reflectivity([], 550.0, -1)

    def test_angle_at_90_raises(self):
        """Angle at 90 degrees raises ValueError."""
        designer = CoatingDesigner()
        with pytest.raises(ValueError, match=re.escape("Angle of incidence must be in [0, 90), got 90")):
            designer.calculate_reflectivity([], 550.0, 90)

    def test_wavelength_validation_before_angle(self):
        """Wavelength validation runs before angle validation."""
        designer = CoatingDesigner()
        with pytest.raises(ValueError, match="Wavelength must be positive"):
            designer.calculate_reflectivity([], -1, 95)

    def test_return_type_float(self):
        """Result is always a Python float, even with int inputs."""
        designer = CoatingDesigner()
        R = designer.calculate_reflectivity([], 550, 0)
        assert isinstance(R, float)

    def test_return_clamped_0_1(self):
        """Result is always in [0, 1]."""
        designer = CoatingDesigner()
        R = designer.calculate_reflectivity([], 550.0, 0.0)
        assert 0.0 <= R <= 1.0


class TestCalculateReflectivityCurve:
    """Test CoatingDesigner.calculate_reflectivity_curve method."""

    def test_default_num_points_100(self):
        """Default curve should have 100 points."""
        designer = CoatingDesigner()
        curve = designer.calculate_reflectivity_curve([], (400, 700))
        assert len(curve) == 100

    def test_custom_num_points(self):
        """Curve length should equal num_points parameter."""
        designer = CoatingDesigner()
        for n in [1, 10, 50, 200]:
            curve = designer.calculate_reflectivity_curve([], (400, 700), n)
            assert len(curve) == n

    def test_return_tuples(self):
        """Should return list of (wavelength, reflectivity) tuples."""
        designer = CoatingDesigner()
        curve = designer.calculate_reflectivity_curve([], (400, 700), 5)
        assert isinstance(curve, list)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in curve)

    def test_single_point_at_min(self):
        """Single point should be at min_wl, ignoring max_wl."""
        designer = CoatingDesigner()
        curve = designer.calculate_reflectivity_curve([], (500, 900), 1)
        assert len(curve) == 1
        assert curve[0][0] == 500

    def test_wavelength_spacing_linear(self):
        """Wavelengths should be linearly spaced from min to max."""
        designer = CoatingDesigner()
        curve = designer.calculate_reflectivity_curve([], (400, 700), 4)
        wavelengths = [w for w, _ in curve]
        assert wavelengths == pytest.approx([400.0, 500.0, 600.0, 700.0], rel=1e-9)

    def test_reflectivity_matches_single_call(self):
        """Each reflectivity value should equal calculate_reflectivity at that wavelength."""
        designer = CoatingDesigner()
        curve = designer.calculate_reflectivity_curve([], (400, 700), 10)
        for wl, R in curve:
            R_direct = designer.calculate_reflectivity([], wl)
            assert R == pytest.approx(R_direct, abs=1e-15)

    def test_uncoated_constant_reflectivity(self):
        """Uncoated curve should have constant R across wavelengths."""
        designer = CoatingDesigner()
        curve = designer.calculate_reflectivity_curve([], (400, 700), 50)
        reflectivities = [R for _, R in curve]
        assert all(abs(R - reflectivities[0]) < 1e-14 for R in reflectivities)

    def test_num_points_zero_raises(self):
        """num_points < 1 raises ValueError."""
        designer = CoatingDesigner()
        with pytest.raises(ValueError, match="num_points must be at least 1"):
            designer.calculate_reflectivity_curve([], (400, 700), 0)

    def test_num_points_negative_raises(self):
        """Negative num_points raises ValueError."""
        designer = CoatingDesigner()
        with pytest.raises(ValueError, match="num_points must be at least 1"):
            designer.calculate_reflectivity_curve([], (400, 700), -3)

    def test_wavelength_range_wrong_length_raises(self):
        """wavelength_range must unpack to exactly 2 values."""
        designer = CoatingDesigner()
        with pytest.raises(ValueError):
            designer.calculate_reflectivity_curve([], (400,), 10)
        with pytest.raises(ValueError):
            designer.calculate_reflectivity_curve([], (400, 500, 600), 10)

    def test_reversed_wavelength_range(self):
        """Reversed range (max, min) should give descending wavelengths."""
        designer = CoatingDesigner()
        curve = designer.calculate_reflectivity_curve([], (700, 400), 4)
        wavelengths = [w for w, _ in curve]
        assert wavelengths == pytest.approx([700.0, 600.0, 500.0, 400.0], rel=1e-9)

    def test_coated_varies_with_wavelength(self):
        """Quarter-wave coated surface should show R variation across visible band."""
        designer = CoatingDesigner()
        wl_design = 550.0
        n_mgf2 = 1.38
        thickness = wl_design / (4 * n_mgf2)
        layers = [CoatingLayer("MgF2", n_mgf2, thickness)]
        curve = designer.calculate_reflectivity_curve(layers, (400, 700), 50)
        reflectivities = [R for _, R in curve]
        R_min = min(reflectivities)
        R_max = max(reflectivities)
        assert (R_max - R_min) > 1e-5


class TestDesignVCoating:
    """Test CoatingDesigner.design_v_coating method."""

    def test_returns_three_layers(self):
        """design_v_coating should return exactly 3 CoatingLayer objects."""
        designer = CoatingDesigner()
        result = designer.design_v_coating(550.0)
        assert len(result) == 3
        assert all(isinstance(layer, CoatingLayer) for layer in result)

    def test_materials_and_indices(self):
        """Should return Al2O3, SiO2, MgF2 with correct refractive indices."""
        designer = CoatingDesigner()
        result = designer.design_v_coating(550.0)
        assert result[0].material == "Al2O3"
        assert result[0].refractive_index == 1.63
        assert result[1].material == "SiO2"
        assert result[1].refractive_index == 1.46
        assert result[2].material == "MgF2"
        assert result[2].refractive_index == 1.38

    def test_inner_layer_quarter_wave(self):
        """Inner Al2O3 layer should always be quarter-wave."""
        designer = CoatingDesigner()
        for wl in [400.0, 550.0, 700.0]:
            result = designer.design_v_coating(wl)
            expected = wl / (4 * 1.63)
            assert result[0].thickness_nm == pytest.approx(expected, rel=1e-9)

    def test_inner_independent_of_substrate(self):
        """Inner thickness should not depend on substrate index."""
        wl = 550.0
        result1 = CoatingDesigner(substrate_index=1.5).design_v_coating(wl)
        result2 = CoatingDesigner(substrate_index=1.6).design_v_coating(wl)
        assert result1[0].thickness_nm == pytest.approx(result2[0].thickness_nm, abs=1e-12)

    def test_deterministic(self):
        """Repeated calls with same parameters should give identical results."""
        designer = CoatingDesigner()
        result1 = designer.design_v_coating(550.0)
        result2 = designer.design_v_coating(550.0)
        for layer1, layer2 in zip(result1, result2):
            assert layer1.material == layer2.material
            assert layer1.refractive_index == layer2.refractive_index
            assert layer1.thickness_nm == pytest.approx(layer2.thickness_nm, abs=1e-14)

    def test_beats_uncoated(self):
        """Result should beat or equal uncoated reflectance over visible band."""
        designer = CoatingDesigner()
        wl = 550.0
        result = designer.design_v_coating(wl)
        
        probes = (400, 450, 500, 550, 600, 650, 700)
        coated_cost = sum(designer.calculate_reflectivity(result, p) for p in probes)
        uncoated_cost = sum(designer.calculate_reflectivity([], p) for p in probes)
        
        assert coated_cost <= uncoated_cost + 1e-9

    def test_non_negative_thicknesses(self):
        """All returned thicknesses must be non-negative."""
        designer = CoatingDesigner()
        result = designer.design_v_coating(550.0)
        for layer in result:
            assert layer.thickness_nm >= 0.0

    def test_zero_wavelength_gives_zero_thicknesses(self):
        """Design at wavelength 0 should give zero thicknesses."""
        designer = CoatingDesigner()
        result = designer.design_v_coating(0.0)
        for layer in result:
            assert layer.thickness_nm == pytest.approx(0.0, abs=1e-12)

import pytest
import math
import re
from openlens.coating_designer import CoatingDesigner, CoatingLayer


class TestCalculateReflectivity:
    """Tests for CoatingDesigner.calculate_reflectivity"""

    def test_bare_substrate_reference_value(self):
        """Test reflectivity of bare BK7 substrate at normal incidence."""
        designer = CoatingDesigner(substrate_index=1.5168)
        R = designer.calculate_reflectivity([], 550.0, 0)
        # R = ((n-1)/(n+1))^2 for bare substrate
        expected = ((1.5168 - 1.0) / (1.5168 + 1.0)) ** 2
        assert R == pytest.approx(expected, rel=1e-9)

    def test_wavelength_must_be_positive(self):
        """Test that non-positive wavelengths raise ValueError."""
        designer = CoatingDesigner()
        with pytest.raises(ValueError, match="Wavelength must be positive, got 0"):
            designer.calculate_reflectivity([], 0, 0)
        with pytest.raises(ValueError, match="Wavelength must be positive, got -5.0"):
            designer.calculate_reflectivity([], -5.0, 0)

    def test_angle_must_be_in_valid_range(self):
        """Test that angle must be in [0, 90)."""
        designer = CoatingDesigner()
        # Escape regex special characters in the error message
        angle_pattern = re.escape("Angle of incidence must be in [0, 90), got 90")
        with pytest.raises(ValueError, match=angle_pattern):
            designer.calculate_reflectivity([], 550, 90)
        
        angle_pattern = re.escape("Angle of incidence must be in [0, 90), got -1")
        with pytest.raises(ValueError, match=angle_pattern):
            designer.calculate_reflectivity([], 550, -1)

    def test_wavelength_check_before_angle_check(self):
        """Test that wavelength validation occurs before angle validation."""
        designer = CoatingDesigner()
        # Both invalid: wavelength should be checked first
        with pytest.raises(ValueError, match="Wavelength must be positive"):
            designer.calculate_reflectivity([], -1, 95)

    def test_zero_thickness_layer_equals_bare(self):
        """Test that a zero-thickness layer is transparent (identity matrix)."""
        designer = CoatingDesigner()
        R_bare = designer.calculate_reflectivity([], 550.0, 0)
        layer = CoatingLayer("MgF2", 1.38, 0.0)
        R_zero = designer.calculate_reflectivity([layer], 550.0, 0)
        assert R_zero == pytest.approx(R_bare, rel=1e-9)

    def test_half_wave_layer_equals_bare(self):
        """Test that half-wave layer (λ/2n) equals bare substrate."""
        designer = CoatingDesigner()
        R_bare = designer.calculate_reflectivity([], 550.0, 0)
        # Half-wave thickness
        thickness = 550.0 / (2 * 1.38)
        layer = CoatingLayer("MgF2", 1.38, thickness)
        R_half = designer.calculate_reflectivity([layer], 550.0, 0)
        assert R_half == pytest.approx(R_bare, rel=1e-9)

    def test_quarter_wave_layer_reduces_reflectivity(self):
        """Test that quarter-wave AR layer reduces reflectivity."""
        designer = CoatingDesigner(substrate_index=1.5168)
        R_bare = designer.calculate_reflectivity([], 550.0, 0)
        # Quarter-wave MgF2 layer
        thickness = 550.0 / (4 * 1.38)
        layer = CoatingLayer("MgF2", 1.38, thickness)
        R_ar = designer.calculate_reflectivity([layer], 550.0, 0)
        assert R_ar < R_bare

    def test_integer_and_float_inputs(self):
        """Test that integer inputs are accepted."""
        designer = CoatingDesigner()
        R_int = designer.calculate_reflectivity([], 550, 0)
        R_float = designer.calculate_reflectivity([], 550.0, 0.0)
        assert isinstance(R_int, float)
        assert R_int == pytest.approx(R_float, rel=1e-9)


class TestCalculateReflectivityCurve:
    """Tests for CoatingDesigner.calculate_reflectivity_curve"""

    def test_default_num_points_is_100(self):
        """Test that default num_points parameter is 100."""
        designer = CoatingDesigner()
        curve = designer.calculate_reflectivity_curve([], (400, 700))
        assert len(curve) == 100

    def test_single_point_at_min_wavelength(self):
        """Test that num_points=1 returns single point at min_wl."""
        designer = CoatingDesigner()
        curve = designer.calculate_reflectivity_curve([], (500, 900), 1)
        assert len(curve) == 1
        assert curve[0][0] == 500  # min_wl is used

    def test_num_points_validation(self):
        """Test that num_points < 1 raises ValueError."""
        designer = CoatingDesigner()
        with pytest.raises(ValueError, match="num_points must be at least 1"):
            designer.calculate_reflectivity_curve([], (400, 700), 0)
        with pytest.raises(ValueError, match="num_points must be at least 1"):
            designer.calculate_reflectivity_curve([], (400, 700), -3)

    def test_curve_matches_individual_calls(self):
        """Test that each point in curve equals calculate_reflectivity result."""
        designer = CoatingDesigner()
        layer = CoatingLayer("MgF2", 1.38, 99.6)
        curve = designer.calculate_reflectivity_curve([layer], (400, 700), 5)
        
        for wl, R in curve:
            R_direct = designer.calculate_reflectivity([layer], wl)
            # Must be bit-for-bit equal (same code path)
            assert R == R_direct

    def test_curve_returns_tuples(self):
        """Test that curve returns list of (wavelength, reflectivity) tuples."""
        designer = CoatingDesigner()
        curve = designer.calculate_reflectivity_curve([], (400, 700), 3)
        assert len(curve) == 3
        assert all(isinstance(point, tuple) and len(point) == 2 for point in curve)

    def test_uncoated_reflectivity_constant_across_wavelengths(self):
        """Test that uncoated substrate R is constant (independent of λ)."""
        designer = CoatingDesigner()
        curve = designer.calculate_reflectivity_curve([], (400, 700), 10)
        R_values = [R for _, R in curve]
        # All should equal the first value
        assert all(R == pytest.approx(R_values[0], rel=1e-9) for R in R_values)

    def test_reversed_wavelength_range(self):
        """Test that reversed (max, min) wavelength range works."""
        designer = CoatingDesigner()
        curve = designer.calculate_reflectivity_curve([], (700, 400), 4)
        wavelengths = [w for w, _ in curve]
        assert wavelengths == pytest.approx([700.0, 600.0, 500.0, 400.0], rel=1e-9)


class TestDesignVCoating:
    """Tests for CoatingDesigner.design_v_coating"""

    def test_design_returns_three_layers(self):
        """Test that design_v_coating always returns exactly 3 CoatingLayer objects."""
        designer = CoatingDesigner()
        result = designer.design_v_coating(550.0)
        assert len(result) == 3
        assert all(isinstance(layer, CoatingLayer) for layer in result)

    def test_design_materials_and_indices(self):
        """Test that layers are Al2O3/SiO2/MgF2 with correct indices."""
        designer = CoatingDesigner()
        result = designer.design_v_coating(550.0)
        
        materials = [layer.material for layer in result]
        indices = [layer.refractive_index for layer in result]
        
        assert materials == ["Al2O3", "SiO2", "MgF2"]
        assert indices == pytest.approx([1.63, 1.46, 1.38], rel=1e-9)

    def test_inner_layer_is_quarter_wave(self):
        """Test that inner Al2O3 layer is always quarter-wave at design wavelength."""
        designer = CoatingDesigner()
        for wl in [400.0, 550.0, 700.0]:
            result = designer.design_v_coating(wl)
            expected = wl / (4 * 1.63)
            assert result[0].thickness_nm == pytest.approx(expected, rel=1e-9)

    def test_design_deterministic(self):
        """Test that design_v_coating produces deterministic output."""
        designer = CoatingDesigner()
        result1 = designer.design_v_coating(550.0)
        result2 = designer.design_v_coating(550.0)
        # CoatingLayer is a dataclass, so == compares all fields
        assert result1 == result2

    def test_design_beats_bare_or_stays_at_start(self):
        """Test that result either beats bare substrate or uses quarter-wave start."""
        designer = CoatingDesigner()
        wl = 550.0
        result = designer.design_v_coating(wl)
        
        # Compute cost at probe wavelengths
        probes = (400, 450, 500, 550, 600, 650, 700)
        result_cost = sum(designer.calculate_reflectivity(result, p) for p in probes)
        bare_cost = sum(designer.calculate_reflectivity([], p) for p in probes)
        
        # Either beats bare by more than tolerance (1e-9), or equals quarter-wave start
        tolerance = 1e-9
        if result_cost >= bare_cost - tolerance:
            # Must be at the initial quarter-wave point
            qm = wl / (4 * 1.46)
            qo = wl / (4 * 1.38)
            assert result[1].thickness_nm == pytest.approx(qm, rel=1e-9)
            assert result[2].thickness_nm == pytest.approx(qo, rel=1e-9)

    def test_outer_thicknesses_non_negative(self):
        """Test that optimized thicknesses are always non-negative."""
        designer = CoatingDesigner()
        result = designer.design_v_coating(550.0)
        assert result[1].thickness_nm >= 0
        assert result[2].thickness_nm >= 0

    def test_design_independent_of_substrate_for_inner_layer(self):
        """Test that inner layer depends only on wavelength, not substrate."""
        wl = 550.0
        designer1 = CoatingDesigner(substrate_index=1.5)
        designer2 = CoatingDesigner(substrate_index=1.6)
        
        result1 = designer1.design_v_coating(wl)
        result2 = designer2.design_v_coating(wl)
        
        # Inner thickness should be identical
        assert result1[0].thickness_nm == pytest.approx(result2[0].thickness_nm, rel=1e-9)

    def test_design_zero_wavelength(self):
        """Test that zero wavelength doesn't raise an error."""
        designer = CoatingDesigner()
        result = designer.design_v_coating(0.0)
        assert len(result) == 3
        # All thicknesses should be 0 or very close to 0
        assert all(layer.thickness_nm >= 0 for layer in result)

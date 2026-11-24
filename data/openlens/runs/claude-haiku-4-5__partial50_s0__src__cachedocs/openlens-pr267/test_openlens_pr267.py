import pytest
import math
from openlens.coating_designer import CoatingDesigner, CoatingLayer


class TestCalculateReflectivity:
    """Test CoatingDesigner.calculate_reflectivity method."""

    def setup_method(self):
        """Create a CoatingDesigner instance for each test."""
        self.designer = CoatingDesigner()

    def test_wavelength_negative_raises_error(self):
        """Negative wavelength raises ValueError."""
        with pytest.raises(ValueError, match="Wavelength must be positive"):
            self.designer.calculate_reflectivity([], -550.0)

    def test_wavelength_zero_raises_error(self):
        """Zero wavelength raises ValueError."""
        with pytest.raises(ValueError, match="Wavelength must be positive"):
            self.designer.calculate_reflectivity([], 0.0)

    def test_angle_negative_raises_error(self):
        """Negative angle raises ValueError."""
        with pytest.raises(ValueError, match="Angle of incidence must be in"):
            self.designer.calculate_reflectivity([], 550.0, angle_deg=-1.0)

    def test_angle_90_or_greater_raises_error(self):
        """Angle >= 90 degrees raises ValueError."""
        with pytest.raises(ValueError, match="Angle of incidence must be in"):
            self.designer.calculate_reflectivity([], 550.0, angle_deg=90.0)
        with pytest.raises(ValueError, match="Angle of incidence must be in"):
            self.designer.calculate_reflectivity([], 550.0, angle_deg=100.0)

    def test_empty_layers_returns_valid_reflectivity(self):
        """Bare substrate (no coating) returns valid reflectivity in [0,1]."""
        R = self.designer.calculate_reflectivity([], 550.0, angle_deg=0.0)
        assert isinstance(R, float)
        assert 0.0 <= R <= 1.0

    def test_reflectivity_in_valid_range_various_angles(self):
        """Reflectivity is always in [0,1] for various angles."""
        for wl in [400.0, 550.0, 800.0]:
            for angle in [0.0, 30.0, 60.0, 89.9]:
                R = self.designer.calculate_reflectivity([], wl, angle_deg=angle)
                assert 0.0 <= R <= 1.0, f"R={R} out of range for wl={wl}, angle={angle}"

    def test_normal_incidence(self):
        """Normal incidence (angle=0) works correctly."""
        R = self.designer.calculate_reflectivity([], 550.0, angle_deg=0.0)
        assert 0.0 <= R <= 1.0

    def test_grazing_angle_near_90(self):
        """Grazing angle (near 90 degrees) works correctly."""
        R = self.designer.calculate_reflectivity([], 550.0, angle_deg=89.99)
        assert 0.0 <= R <= 1.0


class TestCalculateReflectivityCurve:
    """Test CoatingDesigner.calculate_reflectivity_curve method."""

    def setup_method(self):
        """Create a CoatingDesigner instance for each test."""
        self.designer = CoatingDesigner()

    def test_num_points_zero_raises_error(self):
        """num_points=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_points must be at least 1"):
            self.designer.calculate_reflectivity_curve([], (400.0, 700.0), num_points=0)

    def test_num_points_negative_raises_error(self):
        """Negative num_points raises ValueError."""
        with pytest.raises(ValueError, match="num_points must be at least 1"):
            self.designer.calculate_reflectivity_curve([], (400.0, 700.0), num_points=-5)

    def test_single_point_at_range_start(self):
        """Single point curve samples at start of wavelength range."""
        curve = self.designer.calculate_reflectivity_curve([], (400.0, 700.0), num_points=1)
        assert len(curve) == 1
        wl, R = curve[0]
        assert wl == 400.0
        assert 0.0 <= R <= 1.0

    def test_multiple_points_correct_count(self):
        """Curve returns correct number of points."""
        for num_pts in [2, 5, 10, 100]:
            curve = self.designer.calculate_reflectivity_curve([], (400.0, 700.0), num_points=num_pts)
            assert len(curve) == num_pts

    def test_wavelength_range_endpoints(self):
        """Curve endpoints match wavelength range boundaries."""
        curve = self.designer.calculate_reflectivity_curve([], (400.0, 700.0), num_points=5)
        wavelengths = [wl for wl, _ in curve]
        assert wavelengths[0] == 400.0
        assert wavelengths[-1] == 700.0
        assert all(400.0 <= wl <= 700.0 for wl in wavelengths)

    def test_wavelengths_linearly_spaced(self):
        """Wavelengths are linearly spaced within range."""
        curve = self.designer.calculate_reflectivity_curve([], (400.0, 700.0), num_points=6)
        wavelengths = [wl for wl, _ in curve]
        # Verify equal spacing
        spacings = [wavelengths[i+1] - wavelengths[i] for i in range(len(wavelengths)-1)]
        assert all(abs(spacings[0] - s) < 1e-6 for s in spacings)

    def test_reflectivity_values_in_range(self):
        """All returned reflectivity values are in [0,1]."""
        curve = self.designer.calculate_reflectivity_curve([], (400.0, 700.0), num_points=10)
        for wl, R in curve:
            assert 0.0 <= R <= 1.0
            assert isinstance(R, float)


class TestDesignVCoating:
    """Test CoatingDesigner.design_v_coating method."""

    def setup_method(self):
        """Create a CoatingDesigner instance for each test."""
        self.designer = CoatingDesigner()

    def test_returns_coating_layers(self):
        """design_v_coating returns a list of CoatingLayer objects."""
        layers = self.designer.design_v_coating(550.0)
        assert isinstance(layers, list)
        assert len(layers) > 0
        assert all(isinstance(layer, CoatingLayer) for layer in layers)

    def test_layer_properties_valid(self):
        """All returned layers have positive refractive index and thickness."""
        layers = self.designer.design_v_coating(550.0)
        for layer in layers:
            assert hasattr(layer, 'refractive_index')
            assert hasattr(layer, 'thickness_nm')
            assert layer.refractive_index > 0
            assert layer.thickness_nm > 0

    def test_v_coating_reduces_reflectivity(self):
        """V-coating reduces reflectivity compared to uncoated substrate."""
        wavelength = 550.0
        R_bare = self.designer.calculate_reflectivity([], wavelength, angle_deg=0.0)
        layers = self.designer.design_v_coating(wavelength)
        R_coated = self.designer.calculate_reflectivity(layers, wavelength, angle_deg=0.0)
        assert R_coated < R_bare, f"Coated R={R_coated} not less than bare R={R_bare}"

    def test_v_coating_across_visible_spectrum(self):
        """V-coating design produces valid layers across visible wavelengths."""
        for wl in [450.0, 550.0, 650.0]:
            layers = self.designer.design_v_coating(wl)
            assert len(layers) > 0
            for layer in layers:
                assert layer.refractive_index > 0
                assert layer.thickness_nm > 0

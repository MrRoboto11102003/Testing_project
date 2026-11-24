import pytest
import numpy as np
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.analysis.psf_mtf import ImageQualityAnalyzer


class TestCalculateMTF:
    """Tests for ImageQualityAnalyzer.calculate_mtf"""
    
    @pytest.fixture
    def analyzer(self):
        """Fixture: ImageQualityAnalyzer with a single lens."""
        lens = Lens(
            name="test_lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        system = OpticalSystem(name="test_system")
        system.add_lens(lens)
        return ImageQualityAnalyzer(system)
    
    def test_calculate_mtf_returns_correct_structure(self, analyzer):
        """MTF returns dict with freq, mtf_tan, mtf_sag as numpy arrays."""
        result = analyzer.calculate_mtf()
        assert isinstance(result, dict)
        assert all(k in result for k in ["freq", "mtf_tan", "mtf_sag"])
        assert all(isinstance(result[k], np.ndarray) for k in result)
        assert len(result["freq"]) == len(result["mtf_tan"]) == len(result["mtf_sag"])
    
    def test_calculate_mtf_respects_max_freq_parameter(self, analyzer):
        """Frequencies should not exceed max_freq."""
        result = analyzer.calculate_mtf(max_freq=50.0)
        if len(result["freq"]) > 0:
            assert np.all(result["freq"] <= 50.0)
    
    def test_calculate_mtf_handles_field_angle_wavelength_focus(self, analyzer):
        """Should handle field_angle_deg, wavelength_nm, and focus_shift_mm."""
        result = analyzer.calculate_mtf(
            field_angle_deg=5.0,
            wavelength_nm=650.0,
            focus_shift_mm=0.1,
            max_freq=100.0
        )
        assert isinstance(result, dict)
        assert all(k in result for k in ["freq", "mtf_tan", "mtf_sag"])
    
    def test_calculate_mtf_geometric_and_diffraction_modes(self, analyzer):
        """Should compute MTF for both geometric and diffraction modes."""
        result_geom = analyzer.calculate_mtf(use_diffraction=False)
        result_diff = analyzer.calculate_mtf(use_diffraction=True)
        assert all(isinstance(r, dict) for r in [result_geom, result_diff])
    
    def test_calculate_mtf_empty_system_returns_empty_arrays(self):
        """Empty system should return empty MTF arrays."""
        empty_system = OpticalSystem(name="empty")
        analyzer = ImageQualityAnalyzer(empty_system)
        result = analyzer.calculate_mtf()
        assert len(result["freq"]) == 0
        assert len(result["mtf_tan"]) == 0
        assert len(result["mtf_sag"]) == 0


class TestSimulateImage:
    """Tests for ImageQualityAnalyzer.simulate_image"""
    
    @pytest.fixture
    def analyzer(self):
        """Fixture: ImageQualityAnalyzer with a single lens."""
        lens = Lens(
            name="test_lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        system = OpticalSystem(name="test_system")
        system.add_lens(lens)
        return ImageQualityAnalyzer(system)
    
    def test_simulate_image_output_shape_dtype_range(self, analyzer):
        """Output shape matches input, dtype is float64, range [0, 1]."""
        for shape in [(32, 32), (64, 48, 3)]:
            image = np.ones(shape)
            result = analyzer.simulate_image(image)
            assert result.shape == image.shape
            assert result.dtype == np.float64
            assert np.all((result >= 0.0) & (result <= 1.0))
    
    def test_simulate_image_accepts_uint8_and_float_inputs(self, analyzer):
        """Should accept and normalize both uint8 and float inputs."""
        image_uint8 = np.full((32, 32), 128, dtype=np.uint8)
        image_float32 = np.full((32, 32), 0.5, dtype=np.float32)
        
        result_uint = analyzer.simulate_image(image_uint8)
        result_float = analyzer.simulate_image(image_float32)
        
        assert result_uint.dtype == np.float64
        assert result_float.dtype == np.float64
        assert np.all((result_uint >= 0) & (result_uint <= 1))
        assert np.all((result_float >= 0) & (result_float <= 1))
    
    def test_simulate_image_grayscale_rgb_channels(self, analyzer):
        """Should handle both 2-D grayscale and 3-D RGB images."""
        gray = np.ones((32, 32))
        rgb = np.ones((32, 32, 3))
        
        result_gray = analyzer.simulate_image(gray)
        result_rgb = analyzer.simulate_image(rgb)
        
        assert result_gray.shape == (32, 32)
        assert result_rgb.shape == (32, 32, 3)
        assert result_gray.ndim == 2  # Grayscale stays 2-D
    
    def test_simulate_image_channel_count_mismatch_raises_valueerror(self, analyzer):
        """Mismatched channel count should raise ValueError."""
        image_2ch = np.ones((32, 32, 2))  # 2 channels but 3 wavelengths
        with pytest.raises(ValueError, match="channels"):
            analyzer.simulate_image(image_2ch)
    
    def test_simulate_image_invalid_dimensions_raises_valueerror(self, analyzer):
        """1-D and 4-D arrays should raise ValueError."""
        with pytest.raises(ValueError, match="2-D|3-D"):
            analyzer.simulate_image(np.ones(64))
        
        with pytest.raises(ValueError, match="2-D|3-D"):
            analyzer.simulate_image(np.ones((2, 32, 32, 3)))
    
    def test_simulate_image_field_angle_focus_shift_pixel_size_params(self, analyzer):
        """Should accept field_angle_deg, focus_shift_mm, and pixel_size_mm."""
        image = np.ones((32, 32))
        result = analyzer.simulate_image(
            image,
            field_angle_deg=10.0,
            focus_shift_mm=0.1,
            pixel_size_mm=0.005
        )
        assert result.dtype == np.float64
        assert result.shape == image.shape
        assert np.all((result >= 0) & (result <= 1))

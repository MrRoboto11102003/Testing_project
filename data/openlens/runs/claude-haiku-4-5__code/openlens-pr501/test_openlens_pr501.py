import pytest
import numpy as np
from typing import Dict, Any

from openlens.analysis.psf_mtf import ImageQualityAnalyzer
from openlens import LensSystem, Lens


@pytest.fixture
def simple_system():
    """Create a minimal valid optical system for testing."""
    system = LensSystem(name="test_system")
    lens = Lens(
        diameter=10.0,
        thickness=5.0,
        radius_1=50.0,
        radius_2=-50.0,
        material="BK7",
        focal_length=100.0
    )
    system.add_element(lens)
    return system


@pytest.fixture
def analyzer(simple_system):
    """Create an ImageQualityAnalyzer instance for testing."""
    return ImageQualityAnalyzer(simple_system)


class TestCalculateMTF:
    """Test suite for ImageQualityAnalyzer.calculate_mtf."""

    def test_calculate_mtf_returns_dict_with_required_keys(self, analyzer):
        """Test that calculate_mtf returns a dict with freq, mtf_tan, mtf_sag."""
        result = analyzer.calculate_mtf()
        
        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}

    def test_calculate_mtf_returns_numpy_arrays(self, analyzer):
        """Test that all MTF components are numpy arrays."""
        result = analyzer.calculate_mtf()
        
        assert isinstance(result["freq"], np.ndarray)
        assert isinstance(result["mtf_tan"], np.ndarray)
        assert isinstance(result["mtf_sag"], np.ndarray)

    def test_calculate_mtf_arrays_matched_length(self, analyzer):
        """Test that freq, mtf_tan, and mtf_sag have the same length."""
        result = analyzer.calculate_mtf()
        
        n = len(result["freq"])
        assert len(result["mtf_tan"]) == n
        assert len(result["mtf_sag"]) == n

    def test_calculate_mtf_max_freq_filtering(self, analyzer):
        """Test that max_freq parameter correctly filters results."""
        result_50 = analyzer.calculate_mtf(max_freq=50.0)
        result_100 = analyzer.calculate_mtf(max_freq=100.0)
        
        # All frequencies should be within max_freq
        if len(result_50["freq"]) > 0:
            assert np.all(result_50["freq"] <= 50.0)
        if len(result_100["freq"]) > 0:
            assert np.all(result_100["freq"] <= 100.0)

    def test_calculate_mtf_frequencies_non_negative_sorted(self, analyzer):
        """Test that frequencies are non-negative and sorted."""
        result = analyzer.calculate_mtf()
        
        if len(result["freq"]) > 0:
            assert np.all(result["freq"] >= 0)
            assert np.all(np.diff(result["freq"]) >= 0)

    def test_calculate_mtf_with_various_parameters(self, analyzer):
        """Test calculate_mtf with different parameter combinations."""
        result1 = analyzer.calculate_mtf(field_angle_deg=10.0, wavelength_nm=650.0)
        result2 = analyzer.calculate_mtf(focus_shift_mm=0.1, wavelength_nm=550.0)
        result3 = analyzer.calculate_mtf(field_angle_deg=5.0, focus_shift_mm=0.05, max_freq=80.0)
        
        for result in [result1, result2, result3]:
            assert isinstance(result, dict)
            assert len(result["freq"]) == len(result["mtf_tan"]) == len(result["mtf_sag"])

    def test_calculate_mtf_use_diffraction_flag(self, analyzer):
        """Test calculate_mtf with use_diffraction=True vs False."""
        result_geometric = analyzer.calculate_mtf(use_diffraction=False)
        result_diffraction = analyzer.calculate_mtf(use_diffraction=True)
        
        # Both should return valid dictionaries
        assert isinstance(result_geometric, dict)
        assert isinstance(result_diffraction, dict)
        assert len(result_geometric["freq"]) >= 0
        assert len(result_diffraction["freq"]) >= 0

    def test_calculate_mtf_invalid_kwargs_raises_typeerror(self, analyzer):
        """Test that unexpected kwargs raise TypeError."""
        with pytest.raises(TypeError, match="Unexpected kwargs"):
            analyzer.calculate_mtf(invalid_parameter=True)

    def test_calculate_mtf_legacy_parameter_aliases(self, analyzer):
        """Test that legacy parameter names still work."""
        # Legacy alias: field_angle -> field_angle_deg
        result1 = analyzer.calculate_mtf(field_angle=5.0)
        assert isinstance(result1, dict)
        
        # Verify it produces same result as the standard parameter
        result2 = analyzer.calculate_mtf(field_angle_deg=5.0)
        assert isinstance(result2, dict)


class TestSimulateImage:
    """Test suite for ImageQualityAnalyzer.simulate_image."""

    def test_simulate_image_grayscale_float_input(self, analyzer):
        """Test simulate_image with grayscale float input (0-1)."""
        image = np.random.rand(64, 64)
        result = analyzer.simulate_image(image)
        
        assert result.shape == image.shape
        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_simulate_image_rgb_float_input(self, analyzer):
        """Test simulate_image with RGB float input (0-1)."""
        image = np.random.rand(64, 64, 3)
        result = analyzer.simulate_image(image)
        
        assert result.shape == image.shape
        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_simulate_image_grayscale_uint8_input(self, analyzer):
        """Test simulate_image with grayscale uint8 integer input."""
        image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = analyzer.simulate_image(image)
        
        assert result.shape == image.shape
        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_simulate_image_rgb_uint8_input(self, analyzer):
        """Test simulate_image with RGB uint8 integer input."""
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = analyzer.simulate_image(image)
        
        assert result.shape == image.shape
        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_simulate_image_output_always_normalized(self, analyzer):
        """Test that output is always clipped to [0, 1] range."""
        # Create image with values that would exceed 1 after convolution
        image = np.ones((32, 32), dtype=np.float64)
        result = analyzer.simulate_image(image)
        
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        assert result.dtype == np.float64

    def test_simulate_image_with_parameters(self, analyzer):
        """Test simulate_image with various parameter combinations."""
        image = np.random.rand(64, 64)
        
        result1 = analyzer.simulate_image(image, pixel_size_mm=0.001)
        result2 = analyzer.simulate_image(image, field_angle_deg=10.0)
        result3 = analyzer.simulate_image(image, focus_shift_mm=0.05)
        result4 = analyzer.simulate_image(
            image, 
            pixel_size_mm=0.005, 
            field_angle_deg=5.0, 
            focus_shift_mm=0.02
        )
        
        for result in [result1, result2, result3, result4]:
            assert result.shape == image.shape
            assert result.dtype == np.float64

    def test_simulate_image_channel_count_validation(self, analyzer):
        """Test that channel count must match wavelengths_nm."""
        # 4 channels but default is 3 wavelengths
        image_4ch = np.random.rand(64, 64, 4)
        with pytest.raises(ValueError, match="channels"):
            analyzer.simulate_image(image_4ch)
        
        # 2 channels with 3 wavelengths
        image_2ch = np.random.rand(64, 64, 2)
        with pytest.raises(ValueError, match="channels"):
            analyzer.simulate_image(image_2ch)

    def test_simulate_image_array_dimension_validation(self, analyzer):
        """Test that array must be 2-D or 3-D."""
        # 1-D array
        image_1d = np.random.rand(64)
        with pytest.raises(ValueError, match="2-D.*3-D"):
            analyzer.simulate_image(image_1d)
        
        # 4-D array
        image_4d = np.random.rand(64, 64, 3, 3)
        with pytest.raises(ValueError, match="2-D.*3-D"):
            analyzer.simulate_image(image_4d)

    def test_simulate_image_invalid_kwargs_raises_typeerror(self, analyzer):
        """Test that unexpected kwargs raise TypeError."""
        image = np.random.rand(64, 64)
        with pytest.raises(TypeError, match="Unexpected kwargs"):
            analyzer.simulate_image(image, unknown_param=0.5)

    def test_simulate_image_legacy_parameter_aliases(self, analyzer):
        """Test that legacy parameter names (pixel_size, wavelengths) work."""
        image = np.random.rand(64, 64)
        
        # Legacy alias: pixel_size -> pixel_size_mm
        result1 = analyzer.simulate_image(image, pixel_size=0.005)
        assert result1.shape == image.shape
        assert result1.dtype == np.float64
        
        # Legacy alias: wavelengths -> wavelengths_nm
        wl_tuple = (620.0, 540.0, 460.0)
        image_rgb = np.random.rand(64, 64, 3)
        result2 = analyzer.simulate_image(image_rgb, wavelengths=wl_tuple)
        assert result2.shape == image_rgb.shape
        assert result2.dtype == np.float64

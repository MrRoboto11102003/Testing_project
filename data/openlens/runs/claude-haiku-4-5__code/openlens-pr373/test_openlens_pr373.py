import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from openlens.analysis.psf_mtf import ImageQualityAnalyzer


class TestImageQualityAnalyzerCalculateMTF:
    """Comprehensive tests for ImageQualityAnalyzer.calculate_mtf method."""
    
    @pytest.fixture
    def mock_system_with_elements(self):
        """Create a mock optical system with elements."""
        system = Mock()
        system.name = "TestSystem"
        element = Mock()
        element.lens = Mock()
        element.lens.diameter = 10.0
        system.elements = [element]
        system.get_system_focal_length.return_value = 100.0
        return system
    
    @pytest.fixture
    def mock_system_empty(self):
        """Create a mock optical system without elements."""
        system = Mock()
        system.elements = []
        return system
    
    @pytest.fixture
    def analyzer(self, mock_system_with_elements):
        """Create an analyzer with mocked required components."""
        analyzer = ImageQualityAnalyzer()
        analyzer.system = mock_system_with_elements
        analyzer.wavefront_sensor = Mock()
        analyzer.spot_analyzer = Mock()
        analyzer.calculate_psf = Mock(return_value={
            "image": np.ones((128, 128)),
            "step_size": 0.002
        })
        return analyzer
    
    @pytest.fixture
    def analyzer_empty(self, mock_system_empty):
        """Create an analyzer with an empty system."""
        analyzer = ImageQualityAnalyzer()
        analyzer.system = mock_system_empty
        return analyzer
    
    def test_calculate_mtf_returns_dict(self, analyzer):
        """Test that calculate_mtf returns a dictionary with required structure."""
        result = analyzer.calculate_mtf()
        assert isinstance(result, dict)
        assert "freq" in result
        assert "mtf_tan" in result
        assert "mtf_sag" in result
    
    def test_calculate_mtf_returns_numpy_arrays(self, analyzer):
        """Test that all return values are numpy arrays."""
        result = analyzer.calculate_mtf()
        assert isinstance(result["freq"], np.ndarray)
        assert isinstance(result["mtf_tan"], np.ndarray)
        assert isinstance(result["mtf_sag"], np.ndarray)
    
    def test_calculate_mtf_array_lengths_match(self, analyzer):
        """Test that freq, mtf_tan, and mtf_sag have matching lengths."""
        result = analyzer.calculate_mtf()
        assert len(result["freq"]) == len(result["mtf_tan"])
        assert len(result["freq"]) == len(result["mtf_sag"])
    
    def test_calculate_mtf_frequencies_non_negative(self, analyzer):
        """Test that all frequencies are non-negative."""
        result = analyzer.calculate_mtf()
        if len(result["freq"]) > 0:
            assert np.all(result["freq"] >= 0)
    
    def test_calculate_mtf_frequencies_respect_max_freq(self, analyzer):
        """Test that all frequencies are below max_freq limit."""
        max_freq_values = [20.0, 50.0, 100.0]
        for max_freq in max_freq_values:
            result = analyzer.calculate_mtf(max_freq=max_freq)
            if len(result["freq"]) > 0:
                assert np.all(result["freq"] <= max_freq)
    
    def test_calculate_mtf_mtf_values_bounded(self, analyzer):
        """Test that MTF values are in valid range [0, 1]."""
        result = analyzer.calculate_mtf()
        assert np.all(result["mtf_tan"] >= -1e-10)
        assert np.all(result["mtf_tan"] <= 1.0 + 1e-10)
        assert np.all(result["mtf_sag"] >= -1e-10)
        assert np.all(result["mtf_sag"] <= 1.0 + 1e-10)
    
    def test_calculate_mtf_empty_system_returns_empty(self, analyzer_empty):
        """Test that empty system returns empty arrays for all outputs."""
        result = analyzer_empty.calculate_mtf()
        assert len(result["freq"]) == 0
        assert len(result["mtf_tan"]) == 0
        assert len(result["mtf_sag"]) == 0
    
    def test_calculate_mtf_accepts_field_angle(self, analyzer):
        """Test that calculate_mtf accepts various field angles."""
        for field_angle in [0.0, 5.0, 10.0, 20.0]:
            result = analyzer.calculate_mtf(field_angle_deg=field_angle)
            assert isinstance(result, dict)
            assert len(result["freq"]) == len(result["mtf_tan"])
    
    def test_calculate_mtf_accepts_wavelength(self, analyzer):
        """Test that calculate_mtf accepts various wavelengths."""
        for wavelength in [405.0, 532.0, 650.0]:
            result = analyzer.calculate_mtf(wavelength_nm=wavelength)
            assert isinstance(result, dict)
    
    def test_calculate_mtf_accepts_focus_shift(self, analyzer):
        """Test that calculate_mtf accepts defocus parameter."""
        for focus_shift in [-0.1, 0.0, 0.1]:
            result = analyzer.calculate_mtf(focus_shift_mm=focus_shift)
            assert isinstance(result, dict)
            assert len(result["freq"]) == len(result["mtf_tan"])
    
    def test_calculate_mtf_legacy_field_angle_kwarg(self, analyzer):
        """Test that legacy 'field_angle' kwarg works as alias."""
        result = analyzer.calculate_mtf(field_angle=5.0)
        assert isinstance(result, dict)
    
    def test_calculate_mtf_unexpected_kwargs_raises_typeerror(self, analyzer):
        """Test that unexpected kwargs raise TypeError."""
        with pytest.raises(TypeError, match="Unexpected kwargs"):
            analyzer.calculate_mtf(invalid_parameter=123)
    
    def test_calculate_mtf_max_freq_affects_output_range(self, analyzer):
        """Test that larger max_freq can produce more frequency samples."""
        result_low = analyzer.calculate_mtf(max_freq=20.0)
        result_high = analyzer.calculate_mtf(max_freq=100.0)
        # Higher max_freq should not produce fewer frequency points
        assert len(result_high["freq"]) >= len(result_low["freq"])

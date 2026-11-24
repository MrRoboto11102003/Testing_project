import pytest
import numpy as np
import re
from unittest.mock import Mock

from openlens.optical_system import OpticalSystem
from openlens.lens import Lens
from openlens.analysis.psf_mtf import ImageQualityAnalyzer


class TestCalculateMTF:
    """Tests for ImageQualityAnalyzer.calculate_mtf"""
    
    @pytest.fixture
    def empty_system(self):
        return OpticalSystem()
    
    @pytest.fixture
    def simple_system(self):
        sys = OpticalSystem()
        sys.add_lens(Lens())
        return sys
    
    @pytest.fixture
    def analyzer_empty(self, empty_system):
        return ImageQualityAnalyzer(empty_system)
    
    @pytest.fixture
    def analyzer_simple(self, simple_system):
        return ImageQualityAnalyzer(simple_system)
    
    def test_empty_system_returns_empty_arrays(self, analyzer_empty, caplog):
        """Empty system returns three empty float64 arrays"""
        result = analyzer_empty.calculate_mtf()
        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}
        assert result["freq"].size == 0
        assert result["mtf_tan"].size == 0
        assert result["mtf_sag"].size == 0
        assert "MTF: system has no elements" in caplog.text
    
    def test_empty_system_diffraction_branch(self, analyzer_empty):
        """Empty system with use_diffraction=True returns empty arrays"""
        result = analyzer_empty.calculate_mtf(use_diffraction=True)
        assert result["freq"].size == 0
        assert result["mtf_tan"].size == 0
        assert result["mtf_sag"].size == 0
    
    def test_geometric_mtf_dict_structure(self, analyzer_simple):
        """Geometric MTF returns dict with correct keys"""
        result = analyzer_simple.calculate_mtf()
        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}
    
    def test_geometric_array_lengths_equal(self, analyzer_simple):
        """All returned arrays have equal length"""
        result = analyzer_simple.calculate_mtf()
        assert len(result["freq"]) == len(result["mtf_tan"]) == len(result["mtf_sag"])
    
    def test_geometric_frequency_starts_at_zero(self, analyzer_simple):
        """Geometric MTF includes frequency 0"""
        result = analyzer_simple.calculate_mtf()
        if len(result["freq"]) > 0:
            assert result["freq"][0] == 0.0
    
    def test_geometric_normalized_at_dc(self, analyzer_simple):
        """MTF is normalized to 1.0 at DC (frequency 0)"""
        result = analyzer_simple.calculate_mtf()
        if len(result["freq"]) > 0:
            assert np.isclose(result["mtf_tan"][0], 1.0, atol=1e-9)
            assert np.isclose(result["mtf_sag"][0], 1.0, atol=1e-9)
    
    def test_geometric_mtf_range(self, analyzer_simple):
        """MTF values are in [0, 1]"""
        result = analyzer_simple.calculate_mtf()
        assert np.all(result["mtf_tan"] >= -1e-9)
        assert np.all(result["mtf_tan"] <= 1.0 + 1e-9)
        assert np.all(result["mtf_sag"] >= -1e-9)
        assert np.all(result["mtf_sag"] <= 1.0 + 1e-9)
    
    def test_geometric_frequency_monotonic(self, analyzer_simple):
        """Frequencies are non-decreasing"""
        result = analyzer_simple.calculate_mtf()
        if len(result["freq"]) > 1:
            assert np.all(np.diff(result["freq"]) >= -1e-15)
    
    def test_max_freq_cutoff_respected(self, analyzer_simple):
        """Frequencies respect max_freq parameter"""
        result = analyzer_simple.calculate_mtf(max_freq=20.0)
        if len(result["freq"]) > 0:
            assert np.all(result["freq"] <= 20.0 + 1e-9)
    
    def test_max_freq_zero(self, analyzer_simple):
        """max_freq=0 returns only DC component"""
        result = analyzer_simple.calculate_mtf(max_freq=0.0)
        if len(result["freq"]) > 0:
            assert len(result["freq"]) == 1
            assert np.isclose(result["freq"][0], 0.0)
    
    def test_max_freq_negative(self, analyzer_simple):
        """Negative max_freq returns empty arrays"""
        result = analyzer_simple.calculate_mtf(max_freq=-1.0)
        assert len(result["freq"]) == 0
        assert len(result["mtf_tan"]) == 0
        assert len(result["mtf_sag"]) == 0
    
    def test_field_angle_alias(self, analyzer_simple):
        """Legacy field_angle parameter accepted"""
        result = analyzer_simple.calculate_mtf(field_angle=5.0)
        assert "freq" in result
    
    def test_unexpected_kwargs_error(self, analyzer_simple):
        """Unexpected kwargs raise TypeError"""
        with pytest.raises(TypeError, match=re.escape("Unexpected kwargs: ['invalid_key']")):
            analyzer_simple.calculate_mtf(invalid_key=1)
    
    def test_unexpected_kwargs_on_empty_system(self, analyzer_empty):
        """Unexpected kwargs raise TypeError even on empty system"""
        with pytest.raises(TypeError, match=re.escape("Unexpected kwargs: ['foo']")):
            analyzer_empty.calculate_mtf(foo=1)
    
    def test_geometric_uniform_spacing(self, analyzer_simple):
        """Geometric MTF frequencies have uniform spacing"""
        result = analyzer_simple.calculate_mtf(max_freq=50.0)
        if len(result["freq"]) > 2:
            diffs = np.diff(result["freq"])
            assert np.allclose(diffs, diffs[0], rtol=1e-10)
    
    def test_diffraction_dict_structure(self, analyzer_simple):
        """Diffraction MTF returns valid dict with equal length arrays"""
        result = analyzer_simple.calculate_mtf(use_diffraction=True)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}
        assert len(result["freq"]) == len(result["mtf_tan"]) == len(result["mtf_sag"])


class TestSimulateImage:
    """Tests for ImageQualityAnalyzer.simulate_image"""
    
    @pytest.fixture
    def analyzer(self):
        sys = OpticalSystem()
        sys.add_lens(Lens())
        return ImageQualityAnalyzer(sys)
    
    @pytest.fixture
    def empty_analyzer(self):
        return ImageQualityAnalyzer(OpticalSystem())
    
    def test_grayscale_2d_shape_preserved(self, analyzer):
        """2-D grayscale input produces 2-D output of same shape"""
        img = np.ones((8, 8), dtype=np.float64) * 0.5
        result = analyzer.simulate_image(img)
        assert result.shape == (8, 8)
        assert result.ndim == 2
    
    def test_rgb_3d_shape_preserved(self, analyzer):
        """3-D RGB input produces 3-D output of same shape"""
        img = np.ones((8, 8, 3), dtype=np.float64) * 0.5
        result = analyzer.simulate_image(img)
        assert result.shape == (8, 8, 3)
        assert result.ndim == 3
    
    def test_output_always_float64(self, analyzer):
        """Output is always float64 regardless of input dtype"""
        img_uint8 = np.array([[0, 128, 255]], dtype=np.uint8)
        result = analyzer.simulate_image(img_uint8)
        assert result.dtype == np.float64
    
    def test_output_clipped_to_valid_range(self, analyzer):
        """Output is clipped to [0, 1]"""
        img = np.ones((8, 8), dtype=np.float64) * 2.0
        result = analyzer.simulate_image(img)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
    
    def test_uint8_integer_normalization(self, analyzer):
        """uint8 input scaled by dtype max value"""
        def mock_psf(**kw):
            k = np.zeros((3, 3), dtype=np.float64)
            k[1, 1] = 1.0
            return {"image": k}
        
        analyzer.calculate_psf = mock_psf
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        result = analyzer.simulate_image(img)
        assert np.isclose(result[0, 1], 128.0/255.0, atol=0.01)
    
    def test_1d_array_raises_valueerror(self, analyzer):
        """1-D array raises ValueError"""
        img = np.array([1, 2, 3], dtype=np.float64)
        with pytest.raises(ValueError, match="must be 2-D \\(H, W\\) or 3-D"):
            analyzer.simulate_image(img)
    
    def test_4d_array_raises_valueerror(self, analyzer):
        """4-D array raises ValueError"""
        img = np.ones((2, 2, 2, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="must be 2-D \\(H, W\\) or 3-D"):
            analyzer.simulate_image(img)
    
    def test_scalar_array_raises_valueerror(self, analyzer):
        """Scalar array raises ValueError"""
        img = np.array(1.0)
        with pytest.raises(ValueError, match="must be 2-D \\(H, W\\) or 3-D"):
            analyzer.simulate_image(img)
    
    def test_channel_mismatch_raises_valueerror(self, analyzer):
        """Channel count mismatch raises ValueError"""
        img = np.ones((8, 8, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="has 2 channels but 3 wavelengths"):
            analyzer.simulate_image(img)
    
    def test_plain_list_raises_attributeerror(self, analyzer):
        """Plain list without .shape raises AttributeError"""
        with pytest.raises(AttributeError):
            analyzer.simulate_image([[0.5, 0.5]])
    
    def test_unexpected_kwargs_raise_typeerror(self, analyzer):
        """Unexpected kwargs raise TypeError"""
        img = np.ones((8, 8), dtype=np.float64)
        with pytest.raises(TypeError, match=re.escape("Unexpected kwargs: ['foo']")):
            analyzer.simulate_image(img, foo=1)
    
    def test_pixel_size_alias_accepted(self, analyzer):
        """Legacy pixel_size alias is accepted"""
        def mock_psf(**kw):
            k = np.zeros((3, 3), dtype=np.float64)
            k[1, 1] = 1.0
            return {"image": k}
        
        analyzer.calculate_psf = mock_psf
        img = np.ones((8, 8), dtype=np.float64)
        result = analyzer.simulate_image(img, pixel_size=0.01)
        assert result.shape == (8, 8)
    
    def test_wavelengths_alias_accepted(self, analyzer):
        """Legacy wavelengths alias is accepted"""
        def mock_psf(**kw):
            k = np.zeros((3, 3), dtype=np.float64)
            k[1, 1] = 1.0
            return {"image": k}
        
        analyzer.calculate_psf = mock_psf
        img = np.ones((8, 8, 2), dtype=np.float64)
        result = analyzer.simulate_image(img, wavelengths=(500.0, 600.0))
        assert result.shape == (8, 8, 2)
    
    def test_input_not_mutated(self, analyzer):
        """Input array is not mutated by simulate_image"""
        def mock_psf(**kw):
            k = np.zeros((3, 3), dtype=np.float64)
            k[1, 1] = 1.0
            return {"image": k}
        
        analyzer.calculate_psf = mock_psf
        img = np.ones((8, 8), dtype=np.float64) * 0.5
        img_orig = img.copy()
        result = analyzer.simulate_image(img)
        assert np.array_equal(img, img_orig)
        assert result is not img
    
    def test_uses_diffraction_psf(self, analyzer):
        """simulate_image calls calculate_psf with use_diffraction=True"""
        calls = []
        def mock_psf(**kw):
            calls.append(kw)
            k = np.zeros((3, 3), dtype=np.float64)
            k[1, 1] = 1.0
            return {"image": k}
        
        analyzer.calculate_psf = mock_psf
        img = np.ones((8, 8), dtype=np.float64)
        analyzer.simulate_image(img)
        
        assert all(c.get("use_diffraction") is True for c in calls)
    
    def test_grayscale_per_wavelength_convolution(self, analyzer):
        """Grayscale image processes each wavelength"""
        calls = []
        wavelengths = (500.0, 600.0, 700.0)
        def mock_psf(**kw):
            calls.append(kw["wavelength_nm"])
            k = np.zeros((3, 3), dtype=np.float64)
            k[1, 1] = 1.0
            return {"image": k}
        
        analyzer.calculate_psf = mock_psf
        img = np.ones((8, 8), dtype=np.float64)
        analyzer.simulate_image(img, wavelengths_nm=wavelengths)
        
        assert len(calls) == 3
        assert calls == list(wavelengths)
    
    def test_output_finite_values(self, analyzer):
        """Output contains only finite values"""
        img = np.ones((8, 8), dtype=np.float64) * 0.5
        result = analyzer.simulate_image(img)
        assert np.all(np.isfinite(result))

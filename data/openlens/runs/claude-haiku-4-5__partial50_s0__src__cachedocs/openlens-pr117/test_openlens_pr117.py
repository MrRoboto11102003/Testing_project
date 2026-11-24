import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from openlens.vector3 import Vector3


class TestVector3Normalize:
    """Tests for Vector3.normalize() method."""

    def test_normalize_basic_vector(self):
        """Test normalizing a standard non-zero vector."""
        v = Vector3(3, 4, 0)
        normalized = v.normalize()
        assert abs(normalized.magnitude() - 1.0) < 1e-10
        assert abs(normalized.x - 0.6) < 1e-10
        assert abs(normalized.y - 0.8) < 1e-10

    def test_normalize_unit_vector(self):
        """Test that a unit vector normalizes to itself."""
        v = Vector3(1, 0, 0)
        normalized = v.normalize()
        assert abs(normalized.magnitude() - 1.0) < 1e-10
        assert abs(normalized.x - 1.0) < 1e-10

    def test_normalize_negative_vector(self):
        """Test normalizing a vector with negative components."""
        v = Vector3(-3, -4, 0)
        normalized = v.normalize()
        assert abs(normalized.magnitude() - 1.0) < 1e-10
        assert abs(normalized.x + 0.6) < 1e-10
        assert abs(normalized.y + 0.8) < 1e-10

    def test_normalize_zero_vector(self):
        """Test that zero vector normalizes to zero vector."""
        v = Vector3(0, 0, 0)
        normalized = v.normalize()
        assert normalized.x == 0 and normalized.y == 0 and normalized.z == 0

    def test_normalize_near_zero_vector(self):
        """Test that very small vector (magnitude < 1e-15) normalizes to zero."""
        v = Vector3(1e-16, 1e-16, 1e-16)
        normalized = v.normalize()
        assert normalized.x == 0 and normalized.y == 0 and normalized.z == 0

    def test_normalize_3d_vector(self):
        """Test normalizing a full 3D vector."""
        v = Vector3(1, 2, 2)  # magnitude = 3
        normalized = v.normalize()
        assert abs(normalized.magnitude() - 1.0) < 1e-10
        assert abs(normalized.x - 1/3) < 1e-10
        assert abs(normalized.y - 2/3) < 1e-10
        assert abs(normalized.z - 2/3) < 1e-10


class TestBeamSynthesisPropagator:
    """Tests for BeamSynthesisPropagator.propagate_to_image() method."""

    def test_propagate_to_image_empty_system(self):
        """Test that empty system returns empty arrays."""
        from openlens.analysis.beam_synthesis import BeamSynthesisPropagator
        
        mock_system = Mock()
        mock_system.elements = []
        propagator = BeamSynthesisPropagator(mock_system)
        
        dy, dz, intensity = propagator.propagate_to_image()
        
        assert len(dy) == 0
        assert len(dz) == 0
        assert len(intensity) == 0

    def test_propagate_to_image_return_types(self):
        """Test that return values are numpy arrays."""
        from openlens.analysis.beam_synthesis import BeamSynthesisPropagator
        
        # Create a minimal mock system with one element
        mock_lens = Mock()
        mock_lens.diameter = 10.0
        mock_element = Mock()
        mock_element.lens = mock_lens
        mock_element.position = 0.0
        
        mock_system = Mock()
        mock_system.elements = [mock_element]
        
        propagator = BeamSynthesisPropagator(mock_system)
        
        # This should run without crashing
        with patch.object(propagator, '_element_tracers', return_value=[]):
            with patch.object(propagator, '_default_image_plane', return_value=100.0):
                dy, dz, intensity = propagator.propagate_to_image(
                    grid_size=4,  # Small for testing
                    detector_pixels=4
                )
        
        assert isinstance(dy, np.ndarray)
        assert isinstance(dz, np.ndarray)
        assert isinstance(intensity, np.ndarray)

    def test_propagate_to_image_detector_grid_shape(self):
        """Test that detector grid has correct shape."""
        from openlens.analysis.beam_synthesis import BeamSynthesisPropagator
        
        mock_lens = Mock()
        mock_lens.diameter = 10.0
        mock_element = Mock()
        mock_element.lens = mock_lens
        mock_element.position = 0.0
        
        mock_system = Mock()
        mock_system.elements = [mock_element]
        
        propagator = BeamSynthesisPropagator(mock_system)
        
        pixels = 32
        with patch.object(propagator, '_element_tracers', return_value=[]):
            with patch.object(propagator, '_default_image_plane', return_value=100.0):
                dy, dz, intensity = propagator.propagate_to_image(
                    grid_size=4,
                    detector_pixels=pixels
                )
        
        # For empty beamlets, intensity should be zeros but grids should match
        assert dy.shape == (pixels, pixels)
        assert dz.shape == (pixels, pixels)
        assert intensity.shape == (pixels, pixels)


class TestImageQualityAnalyzerMTF:
    """Tests for ImageQualityAnalyzer.calculate_mtf() method."""

    def test_calculate_mtf_empty_system(self):
        """Test that empty system returns empty MTF."""
        from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        
        mock_system = Mock()
        mock_system.elements = []
        mock_system.name = "empty"
        
        analyzer = ImageQualityAnalyzer(mock_system)
        result = analyzer.calculate_mtf()
        
        assert isinstance(result, dict)
        assert "freq" in result
        assert "mtf_tan" in result
        assert "mtf_sag" in result
        assert len(result["freq"]) == 0
        assert len(result["mtf_tan"]) == 0
        assert len(result["mtf_sag"]) == 0

    def test_calculate_mtf_return_structure(self):
        """Test that MTF result has required keys."""
        from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        
        mock_system = Mock()
        mock_element = Mock()
        mock_lens = Mock()
        mock_lens.diameter = 10.0
        mock_element.lens = mock_lens
        mock_system.elements = [mock_element]
        mock_system.name = "test"
        mock_system.get_system_focal_length = Mock(return_value=50.0)
        
        analyzer = ImageQualityAnalyzer(mock_system)
        
        with patch.object(analyzer, 'spot_analyzer'):
            with patch.object(analyzer.spot_analyzer, 'trace_spot', 
                            return_value={'rms_radius': 0.01}):
                with patch.object(analyzer, 'calculate_psf',
                                return_value={'image': np.ones((128, 128)) / 128**2,
                                             'step_size': 0.002}):
                    result = analyzer.calculate_mtf(use_diffraction=False)
        
        assert isinstance(result, dict)
        assert all(k in result for k in ["freq", "mtf_tan", "mtf_sag"])
        assert all(isinstance(result[k], np.ndarray) for k in ["freq", "mtf_tan", "mtf_sag"])

    def test_calculate_mtf_max_freq_filtering(self):
        """Test that MTF respects max_freq parameter."""
        from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        
        mock_system = Mock()
        mock_element = Mock()
        mock_lens = Mock()
        mock_lens.diameter = 10.0
        mock_element.lens = mock_lens
        mock_system.elements = [mock_element]
        mock_system.name = "test"
        mock_system.get_system_focal_length = Mock(return_value=50.0)
        
        analyzer = ImageQualityAnalyzer(mock_system)
        
        with patch.object(analyzer, 'spot_analyzer'):
            with patch.object(analyzer.spot_analyzer, 'trace_spot',
                            return_value={'rms_radius': 0.01}):
                with patch.object(analyzer, 'calculate_psf',
                                return_value={'image': np.ones((128, 128)) / 128**2,
                                             'step_size': 0.002}):
                    result = analyzer.calculate_mtf(max_freq=50.0, use_diffraction=False)
        
        if len(result["freq"]) > 0:
            assert np.max(result["freq"]) <= 50.0


class TestImageSimulation:
    """Tests for ImageQualityAnalyzer.simulate_image() method."""

    @patch('openlens.analysis.psf_mtf.ImageQualityAnalyzer.calculate_psf')
    def test_simulate_image_grayscale(self, mock_psf):
        """Test simulating a grayscale image."""
        from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        
        mock_system = Mock()
        analyzer = ImageQualityAnalyzer(mock_system)
        
        # Mock PSF calculation
        mock_psf.return_value = {
            'image': np.ones((5, 5)) / 25,
            'step_size': 0.001
        }
        
        input_img = np.ones((10, 10), dtype=np.float32) * 0.5
        result = analyzer.simulate_image(input_img, pixel_size_mm=0.005)
        
        assert result.dtype == np.float64
        assert result.shape == input_img.shape
        assert 0 <= result.min()
        assert result.max() <= 1

    @patch('openlens.analysis.psf_mtf.ImageQualityAnalyzer.calculate_psf')
    def test_simulate_image_rgb(self, mock_psf):
        """Test simulating an RGB image."""
        from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        
        mock_system = Mock()
        analyzer = ImageQualityAnalyzer(mock_system)
        
        mock_psf.return_value = {
            'image': np.ones((5, 5)) / 25,
            'step_size': 0.001
        }
        
        input_img = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        result = analyzer.simulate_image(input_img, pixel_size_mm=0.005)
        
        assert result.dtype == np.float64
        assert result.shape == input_img.shape
        assert result.ndim == 3
        assert result.shape[2] == 3

    @patch('openlens.analysis.psf_mtf.ImageQualityAnalyzer.calculate_psf')
    def test_simulate_image_integer_input(self, mock_psf):
        """Test simulating with integer (uint8) input."""
        from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        
        mock_system = Mock()
        analyzer = ImageQualityAnalyzer(mock_system)
        
        mock_psf.return_value = {
            'image': np.ones((5, 5)) / 25,
            'step_size': 0.001
        }
        
        input_img = np.ones((10, 10), dtype=np.uint8) * 128
        result = analyzer.simulate_image(input_img, pixel_size_mm=0.005)
        
        assert result.dtype == np.float64
        assert 0 <= result.min()
        assert result.max() <= 1

    def test_simulate_image_invalid_2d_dimension(self):
        """Test that invalid 1D input raises ValueError."""
        from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        
        mock_system = Mock()
        analyzer = ImageQualityAnalyzer(mock_system)
        
        input_img = np.array([0.1, 0.2, 0.3])  # 1D array
        
        with pytest.raises(ValueError, match="must be 2-D or 3-D"):
            analyzer.simulate_image(input_img)

    def test_simulate_image_channel_mismatch(self):
        """Test that channel count mismatch raises ValueError."""
        from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        
        mock_system = Mock()
        analyzer = ImageQualityAnalyzer(mock_system)
        
        input_img = np.ones((10, 10, 4), dtype=np.float32)  # 4 channels
        wavelengths = (650.0, 550.0, 450.0)  # 3 wavelengths
        
        with pytest.raises(ValueError, match="channels but.*wavelengths"):
            analyzer.simulate_image(input_img, wavelengths_nm=wavelengths)

    @patch('openlens.analysis.psf_mtf.ImageQualityAnalyzer.calculate_psf')
    def test_simulate_image_output_normalized(self, mock_psf):
        """Test that output is normalized to [0, 1]."""
        from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        
        mock_system = Mock()
        analyzer = ImageQualityAnalyzer(mock_system)
        
        mock_psf.return_value = {
            'image': np.ones((5, 5)) / 25,
            'step_size': 0.001
        }
        
        # Input with values outside [0, 1]
        input_img = np.ones((10, 10), dtype=np.float32) * 2.0
        result = analyzer.simulate_image(input_img, pixel_size_mm=0.005)
        
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    @patch('openlens.analysis.psf_mtf.ImageQualityAnalyzer.calculate_psf')
    def test_simulate_image_grayscale_to_grayscale(self, mock_psf):
        """Test that grayscale input produces grayscale output."""
        from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        
        mock_system = Mock()
        analyzer = ImageQualityAnalyzer(mock_system)
        
        mock_psf.return_value = {
            'image': np.ones((5, 5)) / 25,
            'step_size': 0.001
        }
        
        input_img = np.ones((10, 10), dtype=np.float32) * 0.5
        result = analyzer.simulate_image(input_img, pixel_size_mm=0.005)
        
        assert result.ndim == 2
        assert result.shape == (10, 10)

    @patch('openlens.analysis.psf_mtf.ImageQualityAnalyzer.calculate_psf')
    def test_simulate_image_rgb_channels_independent(self, mock_psf):
        """Test that RGB channels are processed independently."""
        from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        
        mock_system = Mock()
        analyzer = ImageQualityAnalyzer(mock_system)
        
        # Return different PSF for each call (simulating different wavelengths)
        def psf_side_effect(*args, **kwargs):
            return {
                'image': np.ones((5, 5)) / 25,
                'step_size': 0.001
            }
        
        mock_psf.side_effect = psf_side_effect
        
        input_img = np.zeros((10, 10, 3), dtype=np.float32)
        input_img[:, :, 0] = 0.5  # Red channel
        input_img[:, :, 1] = 0.3  # Green channel
        input_img[:, :, 2] = 0.7  # Blue channel
        
        result = analyzer.simulate_image(input_img, pixel_size_mm=0.005)
        
        # Verify all channels are processed
        assert result.shape == (10, 10, 3)
        assert mock_psf.call_count >= 3

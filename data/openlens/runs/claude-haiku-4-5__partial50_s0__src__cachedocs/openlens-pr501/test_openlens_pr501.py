import pytest
import numpy as np
from unittest.mock import Mock
from openlens.analysis.psf_mtf import ImageQualityAnalyzer


class TestCalculateMTF:
    """Test ImageQualityAnalyzer.calculate_mtf"""

    @pytest.fixture
    def analyzer(self):
        """Create an ImageQualityAnalyzer with mocked dependencies"""
        system = Mock()
        system.name = "Test System"
        system.elements = [Mock()]
        system.get_system_focal_length = Mock(return_value=100.0)

        analyzer = ImageQualityAnalyzer.__new__(ImageQualityAnalyzer)
        analyzer.system = system
        analyzer.spot_analyzer = Mock()
        analyzer.wavefront_sensor = Mock()
        return analyzer

    def test_empty_system_returns_empty_mtf(self):
        """Empty system should return empty arrays for MTF"""
        system = Mock()
        system.name = "Empty System"
        system.elements = []

        analyzer = ImageQualityAnalyzer.__new__(ImageQualityAnalyzer)
        analyzer.system = system

        result = analyzer.calculate_mtf()

        assert isinstance(result, dict)
        assert len(result["freq"]) == 0
        assert len(result["mtf_tan"]) == 0
        assert len(result["mtf_sag"]) == 0

    def test_calculate_mtf_returns_valid_dictionary_structure(self, analyzer):
        """MTF calculation should return dict with required keys"""
        analyzer.spot_analyzer.trace_spot = Mock(
            return_value={"rms_radius": 0.01}
        )
        analyzer.calculate_psf = Mock(
            return_value={
                "image": np.ones((128, 128)),
                "step_size": 0.001,
            }
        )

        result = analyzer.calculate_mtf()

        assert isinstance(result, dict)
        assert "freq" in result
        assert "mtf_tan" in result
        assert "mtf_sag" in result
        assert isinstance(result["freq"], np.ndarray)
        assert isinstance(result["mtf_tan"], np.ndarray)
        assert isinstance(result["mtf_sag"], np.ndarray)

    def test_mtf_respects_max_freq_limit(self, analyzer):
        """MTF should not exceed specified max_freq"""
        analyzer.spot_analyzer.trace_spot = Mock(
            return_value={"rms_radius": 0.01}
        )
        analyzer.calculate_psf = Mock(
            return_value={
                "image": np.ones((128, 128)),
                "step_size": 0.001,
            }
        )

        result = analyzer.calculate_mtf(max_freq=50.0)

        if len(result["freq"]) > 0:
            assert np.all(result["freq"] <= 50.0)

    def test_mtf_frequencies_non_negative(self, analyzer):
        """MTF frequencies should all be non-negative"""
        analyzer.spot_analyzer.trace_spot = Mock(
            return_value={"rms_radius": 0.01}
        )
        analyzer.calculate_psf = Mock(
            return_value={
                "image": np.ones((128, 128)),
                "step_size": 0.001,
            }
        )

        result = analyzer.calculate_mtf()

        if len(result["freq"]) > 0:
            assert np.all(result["freq"] >= 0)

    def test_mtf_values_in_valid_range(self, analyzer):
        """MTF values should be between 0 and 1"""
        analyzer.spot_analyzer.trace_spot = Mock(
            return_value={"rms_radius": 0.01}
        )
        analyzer.calculate_psf = Mock(
            return_value={
                "image": np.ones((128, 128)),
                "step_size": 0.001,
            }
        )

        result = analyzer.calculate_mtf()

        if len(result["mtf_tan"]) > 0:
            assert np.all(result["mtf_tan"] >= 0)
            assert np.all(result["mtf_tan"] <= 1.0)
        if len(result["mtf_sag"]) > 0:
            assert np.all(result["mtf_sag"] >= 0)
            assert np.all(result["mtf_sag"] <= 1.0)

    def test_mtf_with_field_angle_parameter(self, analyzer):
        """MTF should accept field_angle_deg parameter"""
        analyzer.spot_analyzer.trace_spot = Mock(
            return_value={"rms_radius": 0.01}
        )
        analyzer.calculate_psf = Mock(
            return_value={
                "image": np.ones((128, 128)),
                "step_size": 0.001,
            }
        )

        result = analyzer.calculate_mtf(field_angle_deg=5.0)

        assert isinstance(result, dict)
        analyzer.spot_analyzer.trace_spot.assert_called()

    def test_mtf_with_wavelength_parameter(self, analyzer):
        """MTF should accept different wavelengths"""
        analyzer.spot_analyzer.trace_spot = Mock(
            return_value={"rms_radius": 0.01}
        )
        analyzer.calculate_psf = Mock(
            return_value={
                "image": np.ones((128, 128)),
                "step_size": 0.001,
            }
        )

        result = analyzer.calculate_mtf(wavelength_nm=486.1)

        assert isinstance(result, dict)

    def test_mtf_legacy_field_angle_alias(self, analyzer):
        """MTF should accept legacy field_angle parameter"""
        analyzer.spot_analyzer.trace_spot = Mock(
            return_value={"rms_radius": 0.01}
        )
        analyzer.calculate_psf = Mock(
            return_value={
                "image": np.ones((128, 128)),
                "step_size": 0.001,
            }
        )

        result = analyzer.calculate_mtf(field_angle=3.0)

        assert isinstance(result, dict)

    def test_mtf_invalid_kwargs_raises_typeerror(self, analyzer):
        """MTF should raise TypeError for unexpected kwargs"""
        with pytest.raises(TypeError, match="Unexpected kwargs"):
            analyzer.calculate_mtf(invalid_param=True)


class TestSimulateImage:
    """Test ImageQualityAnalyzer.simulate_image"""

    @pytest.fixture
    def analyzer(self):
        """Create an ImageQualityAnalyzer with mocked PSF calculation"""
        system = Mock()
        analyzer = ImageQualityAnalyzer.__new__(ImageQualityAnalyzer)
        analyzer.system = system
        analyzer.calculate_psf = Mock(
            return_value={"image": np.ones((32, 32))}
        )
        return analyzer

    def test_simulate_image_2d_grayscale_float_input(self, analyzer):
        """Simulate image should handle 2D grayscale float input"""
        image = np.random.rand(64, 64).astype(np.float32)

        result = analyzer.simulate_image(image)

        assert result.shape == image.shape
        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_simulate_image_3d_rgb_float_input(self, analyzer):
        """Simulate image should handle 3D RGB float input"""
        image = np.random.rand(64, 64, 3).astype(np.float32)

        result = analyzer.simulate_image(image)

        assert result.shape == image.shape
        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_simulate_image_integer_input_scaled_to_float(self, analyzer):
        """Simulate image should scale integer input to [0, 1] range"""
        image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

        result = analyzer.simulate_image(image)

        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_simulate_image_output_always_normalized(self, analyzer):
        """Simulate image output should always be clipped to [0, 1]"""
        image = np.ones((32, 32), dtype=np.float32) * 2.0

        result = analyzer.simulate_image(image)

        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_simulate_image_shape_preserved_various_dimensions(self, analyzer):
        """Simulate image should preserve input shape"""
        test_shapes = [(32, 32), (100, 100), (50, 60, 3)]

        for shape in test_shapes:
            image = np.random.rand(*shape).astype(np.float32)
            if len(shape) == 3:
                result = analyzer.simulate_image(
                    image, wavelengths_nm=(650, 550, 450)
                )
            else:
                result = analyzer.simulate_image(image)

            assert result.shape == image.shape

    def test_simulate_image_legacy_pixel_size_alias(self, analyzer):
        """Simulate image should accept legacy pixel_size parameter"""
        image = np.random.rand(32, 32).astype(np.float32)

        result = analyzer.simulate_image(image, pixel_size=0.005)

        assert result.shape == image.shape

    def test_simulate_image_legacy_wavelengths_alias(self, analyzer):
        """Simulate image should accept legacy wavelengths parameter"""
        image = np.random.rand(32, 32, 3).astype(np.float32)

        result = analyzer.simulate_image(
            image, wavelengths=(650, 550, 450)
        )

        assert result.shape == image.shape

    def test_simulate_image_invalid_kwargs_raises_typeerror(self, analyzer):
        """Simulate image should raise TypeError for unexpected kwargs"""
        image = np.random.rand(32, 32).astype(np.float32)

        with pytest.raises(TypeError, match="Unexpected kwargs"):
            analyzer.simulate_image(image, invalid_param=True)

    def test_simulate_image_wrong_dimensions_raises_valueerror(self, analyzer):
        """Simulate image should raise ValueError for wrong dimensions"""
        image = np.random.rand(32, 32, 32, 3).astype(np.float32)

        with pytest.raises(ValueError, match="must be 2-D"):
            analyzer.simulate_image(image)

    def test_simulate_image_channel_mismatch_raises_valueerror(self, analyzer):
        """Simulate image should raise ValueError for channel/wavelength mismatch"""
        image = np.random.rand(32, 32, 3).astype(np.float32)

        with pytest.raises(ValueError, match="channels but"):
            analyzer.simulate_image(
                image, wavelengths_nm=(650, 550)  # Only 2, not 3
            )

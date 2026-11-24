import pytest
from unittest.mock import Mock, MagicMock, patch, call
import numpy as np
import tempfile
import os


class TestPlotAiryDisk:
    """Tests for DiffractionCalculator.plot_airy_disk"""

    @pytest.fixture
    def calculator_with_mocked_methods(self):
        """Create calculator with mocked helper methods"""
        from openlens.diffraction import DiffractionCalculator
        calc = DiffractionCalculator()
        calc.airy_disk_radius = MagicMock(return_value=2.5)
        calc.airy_pattern = MagicMock(return_value=(
            np.linspace(0, 10, 100),
            np.sin(np.linspace(0, 10, 100))
        ))
        calc.wavelength = 550e-9
        return calc

    @patch('openlens.diffraction.plt')
    @patch('openlens.diffraction.MATPLOTLIB_AVAILABLE', True)
    def test_plot_airy_disk_calls_figure(self, mock_plt, calculator_with_mocked_methods):
        """Test that plot_airy_disk creates matplotlib figure"""
        calculator_with_mocked_methods.plot_airy_disk(focal_length=50.0, aperture_diameter=25.0)
        mock_plt.figure.assert_called_once()

    @patch('openlens.diffraction.plt')
    @patch('openlens.diffraction.MATPLOTLIB_AVAILABLE', True)
    def test_plot_airy_disk_calls_show_without_save_path(self, mock_plt, calculator_with_mocked_methods):
        """Test that plot_airy_disk calls show when save_path is None"""
        calculator_with_mocked_methods.plot_airy_disk(focal_length=50.0, aperture_diameter=25.0)
        mock_plt.show.assert_called_once()

    @patch('openlens.diffraction.plt')
    @patch('openlens.diffraction.MATPLOTLIB_AVAILABLE', True)
    def test_plot_airy_disk_calls_savefig_with_save_path(self, mock_plt, calculator_with_mocked_methods):
        """Test that plot_airy_disk saves to file when save_path is provided"""
        save_path = "/tmp/test_airy.png"
        calculator_with_mocked_methods.plot_airy_disk(
            focal_length=50.0, aperture_diameter=25.0, save_path=save_path
        )
        mock_plt.savefig.assert_called_once()
        args, kwargs = mock_plt.savefig.call_args
        assert save_path in args or save_path == args[0]

    @patch('openlens.diffraction.plt')
    @patch('openlens.diffraction.MATPLOTLIB_AVAILABLE', True)
    def test_plot_airy_disk_closes_figure(self, mock_plt, calculator_with_mocked_methods):
        """Test that plot_airy_disk closes the figure"""
        calculator_with_mocked_methods.plot_airy_disk(focal_length=50.0, aperture_diameter=25.0)
        mock_plt.close.assert_called_once()

    @patch('openlens.diffraction.logger')
    @patch('openlens.diffraction.MATPLOTLIB_AVAILABLE', False)
    def test_plot_airy_disk_matplotlib_unavailable(self, mock_logger, calculator_with_mocked_methods):
        """Test that plot_airy_disk logs warning when matplotlib is unavailable"""
        calculator_with_mocked_methods.plot_airy_disk(focal_length=50.0, aperture_diameter=25.0)
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "matplotlib" in warning_msg.lower()


class TestPlotEncircledEnergy:
    """Tests for DiffractionCalculator.plot_encircled_energy"""

    @pytest.fixture
    def calculator_with_mocked_methods(self):
        """Create calculator with mocked helper methods"""
        from openlens.diffraction import DiffractionCalculator
        calc = DiffractionCalculator()
        calc.airy_disk_radius = MagicMock(return_value=2.5)
        r = np.linspace(0, 10, 100)
        encircled = np.linspace(0, 1, 100)
        calc.encircled_energy = MagicMock(return_value=(r, encircled))
        calc.wavelength = 550e-9
        return calc

    @patch('openlens.diffraction.plt')
    @patch('openlens.diffraction.MATPLOTLIB_AVAILABLE', True)
    def test_plot_encircled_energy_creates_figure(self, mock_plt, calculator_with_mocked_methods):
        """Test that plot_encircled_energy creates figure"""
        calculator_with_mocked_methods.plot_encircled_energy(focal_length=50.0, aperture_diameter=25.0)
        mock_plt.figure.assert_called_once()

    @patch('openlens.diffraction.plt')
    @patch('openlens.diffraction.MATPLOTLIB_AVAILABLE', True)
    def test_plot_encircled_energy_marks_50_80_percent(self, mock_plt, calculator_with_mocked_methods):
        """Test that plot_encircled_energy marks 50% and 80% energy points"""
        calculator_with_mocked_methods.plot_encircled_energy(focal_length=50.0, aperture_diameter=25.0)
        
        # Should have multiple plot calls for main plot and markers
        plot_calls = mock_plt.plot.call_count
        assert plot_calls >= 2

    @patch('openlens.diffraction.plt')
    @patch('openlens.diffraction.MATPLOTLIB_AVAILABLE', True)
    def test_plot_encircled_energy_calls_savefig_with_path(self, mock_plt, calculator_with_mocked_methods):
        """Test that plot_encircled_energy saves to file when save_path provided"""
        save_path = "/tmp/test_encircled.png"
        calculator_with_mocked_methods.plot_encircled_energy(
            focal_length=50.0, aperture_diameter=25.0, save_path=save_path
        )
        mock_plt.savefig.assert_called_once()

    @patch('openlens.diffraction.plt')
    @patch('openlens.diffraction.MATPLOTLIB_AVAILABLE', True)
    def test_plot_encircled_energy_closes_figure(self, mock_plt, calculator_with_mocked_methods):
        """Test that plot_encircled_energy closes figure"""
        calculator_with_mocked_methods.plot_encircled_energy(focal_length=50.0, aperture_diameter=25.0)
        mock_plt.close.assert_called_once()

    @patch('openlens.diffraction.logger')
    @patch('openlens.diffraction.MATPLOTLIB_AVAILABLE', False)
    def test_plot_encircled_energy_matplotlib_unavailable(self, mock_logger, calculator_with_mocked_methods):
        """Test that plot_encircled_energy logs warning when matplotlib unavailable"""
        calculator_with_mocked_methods.plot_encircled_energy(focal_length=50.0, aperture_diameter=25.0)
        mock_logger.warning.assert_called_once()


class TestPlotPsf2d:
    """Tests for DiffractionCalculator.plot_psf_2d"""

    @pytest.fixture
    def calculator_with_mocked_methods(self):
        """Create calculator with mocked helper methods"""
        from openlens.diffraction import DiffractionCalculator
        calc = DiffractionCalculator()
        calc.airy_disk_radius = MagicMock(return_value=2.5)
        calc.point_spread_function_2d = MagicMock(return_value=np.random.rand(100, 100))
        calc.wavelength = 550e-9
        return calc

    @patch('openlens.diffraction.plt')
    @patch('openlens.diffraction.MATPLOTLIB_AVAILABLE', True)
    def test_plot_psf_2d_creates_subplots(self, mock_plt, calculator_with_mocked_methods):
        """Test that plot_psf_2d creates subplots"""
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_fig = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        
        calculator_with_mocked_methods.plot_psf_2d(focal_length=50.0, aperture_diameter=25.0)
        
        mock_plt.subplots.assert_called_once_with(1, 2, figsize=(14, 6))

    @patch('openlens.diffraction.plt')
    @patch('openlens.diffraction.MATPLOTLIB_AVAILABLE', True)
    def test_plot_psf_2d_calls_point_spread_function_with_correct_size(self, mock_plt, calculator_with_mocked_methods):
        """Test that plot_psf_2d calls point_spread_function_2d with correct size"""
        airy_radius = 2.5
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_fig = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        
        calculator_with_mocked_methods.plot_psf_2d(focal_length=50.0, aperture_diameter=25.0)
        
        # Check PSF was called with size = 10 * airy_radius
        call_args = calculator_with_mocked_methods.point_spread_function_2d.call_args
        expected_size = 10 * airy_radius
        assert call_args[0][0] == expected_size

    @patch('openlens.diffraction.plt')
    @patch('openlens.diffraction.MATPLOTLIB_AVAILABLE', True)
    def test_plot_psf_2d_saves_file(self, mock_plt, calculator_with_mocked_methods):
        """Test that plot_psf_2d saves to file when save_path provided"""
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_fig = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        
        save_path = "/tmp/test_psf.png"
        calculator_with_mocked_methods.plot_psf_2d(
            focal_length=50.0, aperture_diameter=25.0, save_path=save_path
        )
        
        mock_plt.savefig.assert_called_once()

    @patch('openlens.diffraction.plt')
    @patch('openlens.diffraction.MATPLOTLIB_AVAILABLE', True)
    def test_plot_psf_2d_closes_figure(self, mock_plt, calculator_with_mocked_methods):
        """Test that plot_psf_2d closes figure"""
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_fig = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        
        calculator_with_mocked_methods.plot_psf_2d(focal_length=50.0, aperture_diameter=25.0)
        mock_plt.close.assert_called_once()

    @patch('openlens.diffraction.logger')
    @patch('openlens.diffraction.MATPLOTLIB_AVAILABLE', False)
    def test_plot_psf_2d_matplotlib_unavailable(self, mock_logger, calculator_with_mocked_methods):
        """Test that plot_psf_2d logs warning when matplotlib unavailable"""
        calculator_with_mocked_methods.plot_psf_2d(focal_length=50.0, aperture_diameter=25.0)
        mock_logger.warning.assert_called_once()


class TestPlottingParameterHandling:
    """Tests for plotting function parameter handling"""

    @patch('openlens.diffraction.plt')
    @patch('openlens.diffraction.MATPLOTLIB_AVAILABLE', True)
    def test_plot_functions_handle_various_parameters(self, mock_plt):
        """Test plotting functions with various parameter values"""
        from openlens.diffraction import DiffractionCalculator
        
        calc = DiffractionCalculator()
        calc.airy_disk_radius = MagicMock(return_value=2.5)
        calc.airy_pattern = MagicMock(return_value=(np.linspace(0, 10, 10), np.ones(10)))
        calc.wavelength = 550e-9
        
        # Test with different focal lengths and aperture diameters
        test_params = [(10.0, 5.0), (50.0, 25.0), (100.0, 50.0)]
        for focal_length, aperture_diameter in test_params:
            mock_plt.reset_mock()
            calc.plot_airy_disk(focal_length=focal_length, aperture_diameter=aperture_diameter)
            assert mock_plt.figure.called

    @patch('openlens.diffraction.plt')
    @patch('openlens.diffraction.MATPLOTLIB_AVAILABLE', True)
    def test_plot_airy_disk_with_dpi_settings(self, mock_plt):
        """Test that plot_airy_disk saves with appropriate DPI"""
        from openlens.diffraction import DiffractionCalculator
        
        calc = DiffractionCalculator()
        calc.airy_disk_radius = MagicMock(return_value=2.5)
        calc.airy_pattern = MagicMock(return_value=(np.linspace(0, 10, 10), np.ones(10)))
        calc.wavelength = 550e-9
        
        save_path = "/tmp/test.png"
        calc.plot_airy_disk(focal_length=50.0, aperture_diameter=25.0, save_path=save_path)
        
        # Check savefig was called with dpi=300
        call_kwargs = mock_plt.savefig.call_args[1]
        assert call_kwargs.get('dpi') == 300

    @patch('openlens.diffraction.plt')
    @patch('openlens.diffraction.MATPLOTLIB_AVAILABLE', True)
    def test_plot_encircled_energy_ylim_set(self, mock_plt):
        """Test that plot_encircled_energy sets y-axis limits"""
        from openlens.diffraction import DiffractionCalculator
        
        calc = DiffractionCalculator()
        calc.airy_disk_radius = MagicMock(return_value=2.5)
        calc.encircled_energy = MagicMock(return_value=(
            np.linspace(0, 10, 100),
            np.linspace(0, 1, 100)
        ))
        calc.wavelength = 550e-9
        
        calc.plot_encircled_energy(focal_length=50.0, aperture_diameter=25.0)
        
        mock_plt.ylim.assert_called_once_with(0, 105)

    @patch('openlens.diffraction.plt')
    @patch('openlens.diffraction.MATPLOTLIB_AVAILABLE', True)
    def test_plot_psf_2d_calls_imshow(self, mock_plt):
        """Test that plot_psf_2d calls imshow for 2D visualization"""
        from openlens.diffraction import DiffractionCalculator
        
        calc = DiffractionCalculator()
        calc.airy_disk_radius = MagicMock(return_value=2.5)
        calc.point_spread_function_2d = MagicMock(return_value=np.random.rand(50, 50))
        calc.wavelength = 550e-9
        
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_fig = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        
        calc.plot_psf_2d(focal_length=50.0, aperture_diameter=25.0)
        
        # Check that imshow was called on the first axis
        assert mock_ax1.imshow.called

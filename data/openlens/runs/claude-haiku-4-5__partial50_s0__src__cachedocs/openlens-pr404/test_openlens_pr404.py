import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

try:
    from openlens.diffraction import DiffractionCalculator
except ImportError:
    pytest.skip("openlens not installed", allow_module_level=True)


class TestDiffractionCalculatorPlotting:
    """Test suite for DiffractionCalculator plotting methods."""

    @pytest.fixture
    def calculator(self):
        """Create a DiffractionCalculator instance."""
        return DiffractionCalculator(wavelength=0.55e-6)

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    # plot_airy_disk tests
    @patch("openlens.diffraction.plt")
    @patch("openlens.diffraction.MATPLOTLIB_AVAILABLE", True)
    def test_plot_airy_disk_calls_figure(self, mock_plt, calculator):
        """Test plot_airy_disk creates a figure."""
        with patch.object(
            calculator, "airy_disk_radius", return_value=10.0
        ), patch.object(
            calculator,
            "airy_pattern",
            return_value=(np.array([0, 10, 20]), np.array([1.0, 0.5, 0.1])),
        ):
            calculator.plot_airy_disk(focal_length=50.0, aperture_diameter=10.0)
            mock_plt.figure.assert_called_once()

    @patch("openlens.diffraction.plt")
    @patch("openlens.diffraction.MATPLOTLIB_AVAILABLE", True)
    def test_plot_airy_disk_shows_without_save_path(self, mock_plt, calculator):
        """Test plot_airy_disk calls show() when save_path is None."""
        with patch.object(
            calculator, "airy_disk_radius", return_value=10.0
        ), patch.object(
            calculator,
            "airy_pattern",
            return_value=(np.array([0, 10, 20]), np.array([1.0, 0.5, 0.1])),
        ):
            calculator.plot_airy_disk(focal_length=50.0, aperture_diameter=10.0)
            mock_plt.show.assert_called_once()
            mock_plt.savefig.assert_not_called()

    @patch("openlens.diffraction.plt")
    @patch("openlens.diffraction.MATPLOTLIB_AVAILABLE", True)
    def test_plot_airy_disk_saves_with_path(self, mock_plt, calculator, temp_dir):
        """Test plot_airy_disk saves file when save_path provided."""
        save_path = os.path.join(temp_dir, "airy.png")
        with patch.object(
            calculator, "airy_disk_radius", return_value=10.0
        ), patch.object(
            calculator,
            "airy_pattern",
            return_value=(np.array([0, 10, 20]), np.array([1.0, 0.5, 0.1])),
        ):
            calculator.plot_airy_disk(
                focal_length=50.0, aperture_diameter=10.0, save_path=save_path
            )
            mock_plt.savefig.assert_called_once()
            mock_plt.show.assert_not_called()

    @patch("openlens.diffraction.logger")
    @patch("openlens.diffraction.MATPLOTLIB_AVAILABLE", False)
    def test_plot_airy_disk_matplotlib_unavailable(self, mock_logger, calculator):
        """Test plot_airy_disk logs warning when matplotlib unavailable."""
        with patch.object(
            calculator, "airy_disk_radius", return_value=10.0
        ), patch.object(
            calculator,
            "airy_pattern",
            return_value=(np.array([0, 10, 20]), np.array([1.0, 0.5, 0.1])),
        ):
            calculator.plot_airy_disk(focal_length=50.0, aperture_diameter=10.0)
            mock_logger.warning.assert_called_once()

    @patch("openlens.diffraction.plt")
    @patch("openlens.diffraction.MATPLOTLIB_AVAILABLE", True)
    def test_plot_airy_disk_closes_figure(self, mock_plt, calculator):
        """Test plot_airy_disk closes figure after plotting."""
        with patch.object(
            calculator, "airy_disk_radius", return_value=10.0
        ), patch.object(
            calculator,
            "airy_pattern",
            return_value=(np.array([0, 10, 20]), np.array([1.0, 0.5, 0.1])),
        ):
            calculator.plot_airy_disk(focal_length=50.0, aperture_diameter=10.0)
            mock_plt.close.assert_called_once()

    # plot_encircled_energy tests
    @patch("openlens.diffraction.plt")
    @patch("openlens.diffraction.MATPLOTLIB_AVAILABLE", True)
    def test_plot_encircled_energy_basic(self, mock_plt, calculator):
        """Test plot_encircled_energy creates figure."""
        with patch.object(
            calculator, "airy_disk_radius", return_value=10.0
        ), patch.object(
            calculator,
            "encircled_energy",
            return_value=(np.linspace(0, 50, 10), np.linspace(0, 1, 10)),
        ):
            calculator.plot_encircled_energy(focal_length=50.0, aperture_diameter=10.0)
            mock_plt.figure.assert_called_once()

    @patch("openlens.diffraction.plt")
    @patch("openlens.diffraction.MATPLOTLIB_AVAILABLE", True)
    def test_plot_encircled_energy_saves_with_path(self, mock_plt, calculator, temp_dir):
        """Test plot_encircled_energy saves file when save_path provided."""
        save_path = os.path.join(temp_dir, "encircled.png")
        with patch.object(
            calculator, "airy_disk_radius", return_value=10.0
        ), patch.object(
            calculator,
            "encircled_energy",
            return_value=(np.linspace(0, 50, 10), np.linspace(0, 1, 10)),
        ):
            calculator.plot_encircled_energy(
                focal_length=50.0, aperture_diameter=10.0, save_path=save_path
            )
            mock_plt.savefig.assert_called_once()
            mock_plt.show.assert_not_called()

    @patch("openlens.diffraction.logger")
    @patch("openlens.diffraction.MATPLOTLIB_AVAILABLE", False)
    def test_plot_encircled_energy_matplotlib_unavailable(self, mock_logger, calculator):
        """Test plot_encircled_energy logs warning when matplotlib unavailable."""
        with patch.object(
            calculator, "airy_disk_radius", return_value=10.0
        ), patch.object(
            calculator,
            "encircled_energy",
            return_value=(np.linspace(0, 50, 10), np.linspace(0, 1, 10)),
        ):
            calculator.plot_encircled_energy(focal_length=50.0, aperture_diameter=10.0)
            mock_logger.warning.assert_called_once()

    # plot_psf_2d tests
    @patch("openlens.diffraction.plt")
    @patch("openlens.diffraction.MATPLOTLIB_AVAILABLE", True)
    def test_plot_psf_2d_basic(self, mock_plt, calculator):
        """Test plot_psf_2d creates figure and subplots."""
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, [mock_ax1, mock_ax2])

        with patch.object(
            calculator, "airy_disk_radius", return_value=10.0
        ), patch.object(
            calculator, "point_spread_function_2d", return_value=np.random.rand(50, 50)
        ):
            calculator.plot_psf_2d(focal_length=50.0, aperture_diameter=10.0)
            mock_plt.figure.assert_not_called()
            mock_plt.subplots.assert_called_once_with(1, 2, figsize=(14, 6))

    @patch("openlens.diffraction.plt")
    @patch("openlens.diffraction.MATPLOTLIB_AVAILABLE", True)
    def test_plot_psf_2d_saves_with_path(self, mock_plt, calculator, temp_dir):
        """Test plot_psf_2d saves file when save_path provided."""
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, [mock_ax1, mock_ax2])
        save_path = os.path.join(temp_dir, "psf2d.png")

        with patch.object(
            calculator, "airy_disk_radius", return_value=10.0
        ), patch.object(
            calculator, "point_spread_function_2d", return_value=np.random.rand(50, 50)
        ):
            calculator.plot_psf_2d(
                focal_length=50.0, aperture_diameter=10.0, save_path=save_path
            )
            mock_plt.savefig.assert_called_once()
            mock_plt.show.assert_not_called()

    @patch("openlens.diffraction.logger")
    @patch("openlens.diffraction.MATPLOTLIB_AVAILABLE", False)
    def test_plot_psf_2d_matplotlib_unavailable(self, mock_logger, calculator):
        """Test plot_psf_2d logs warning when matplotlib unavailable."""
        with patch.object(
            calculator, "airy_disk_radius", return_value=10.0
        ), patch.object(
            calculator, "point_spread_function_2d", return_value=np.random.rand(50, 50)
        ):
            calculator.plot_psf_2d(focal_length=50.0, aperture_diameter=10.0)
            mock_logger.warning.assert_called_once()

    @patch("openlens.diffraction.plt")
    @patch("openlens.diffraction.MATPLOTLIB_AVAILABLE", True)
    def test_plot_psf_2d_closes_figure(self, mock_plt, calculator):
        """Test plot_psf_2d closes figure after plotting."""
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, [mock_ax1, mock_ax2])

        with patch.object(
            calculator, "airy_disk_radius", return_value=10.0
        ), patch.object(
            calculator, "point_spread_function_2d", return_value=np.random.rand(50, 50)
        ):
            calculator.plot_psf_2d(focal_length=50.0, aperture_diameter=10.0)
            mock_plt.close.assert_called_once()

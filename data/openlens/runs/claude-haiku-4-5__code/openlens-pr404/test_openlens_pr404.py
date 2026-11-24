import pytest
import tempfile
import os
import matplotlib.pyplot as plt
from openlens.diffraction import DiffractionCalculator


@pytest.fixture
def temp_dir():
    """Create a temporary directory for saving plots."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def default_calculator():
    """Create a DiffractionCalculator with default wavelength (550 nm)."""
    return DiffractionCalculator(wavelength=550e-9)


class TestPlotAiryDisk:
    """Tests for DiffractionCalculator.plot_airy_disk"""

    def test_plot_airy_disk_basic_no_save(self, default_calculator):
        """Test plot_airy_disk runs without error when no save_path provided."""
        default_calculator.plot_airy_disk(focal_length=100.0, aperture_diameter=10.0)
        plt.close('all')

    def test_plot_airy_disk_with_save_path(self, default_calculator, temp_dir):
        """Test plot_airy_disk creates file when save_path is provided."""
        save_path = os.path.join(temp_dir, "airy_disk.png")
        default_calculator.plot_airy_disk(
            focal_length=100.0,
            aperture_diameter=10.0,
            save_path=save_path
        )
        assert os.path.exists(save_path), "Plot file should be created"
        assert os.path.getsize(save_path) > 0, "Plot file should not be empty"
        plt.close('all')

    def test_plot_airy_disk_small_aperture(self, default_calculator, temp_dir):
        """Test plot_airy_disk with small aperture diameter."""
        save_path = os.path.join(temp_dir, "airy_small.png")
        default_calculator.plot_airy_disk(
            focal_length=100.0,
            aperture_diameter=1.0,
            save_path=save_path
        )
        assert os.path.exists(save_path)
        plt.close('all')

    def test_plot_airy_disk_large_aperture(self, default_calculator, temp_dir):
        """Test plot_airy_disk with large aperture diameter."""
        save_path = os.path.join(temp_dir, "airy_large.png")
        default_calculator.plot_airy_disk(
            focal_length=100.0,
            aperture_diameter=100.0,
            save_path=save_path
        )
        assert os.path.exists(save_path)
        plt.close('all')


class TestPlotEncircledEnergy:
    """Tests for DiffractionCalculator.plot_encircled_energy"""

    def test_plot_encircled_energy_basic_no_save(self, default_calculator):
        """Test plot_encircled_energy runs without error when no save_path provided."""
        default_calculator.plot_encircled_energy(focal_length=100.0, aperture_diameter=10.0)
        plt.close('all')

    def test_plot_encircled_energy_with_save_path(self, default_calculator, temp_dir):
        """Test plot_encircled_energy creates file when save_path is provided."""
        save_path = os.path.join(temp_dir, "encircled_energy.png")
        default_calculator.plot_encircled_energy(
            focal_length=100.0,
            aperture_diameter=10.0,
            save_path=save_path
        )
        assert os.path.exists(save_path), "Plot file should be created"
        assert os.path.getsize(save_path) > 0, "Plot file should not be empty"
        plt.close('all')

    def test_plot_encircled_energy_short_focal_length(self, default_calculator, temp_dir):
        """Test plot_encircled_energy with short focal length."""
        save_path = os.path.join(temp_dir, "encircled_short.png")
        default_calculator.plot_encircled_energy(
            focal_length=50.0,
            aperture_diameter=10.0,
            save_path=save_path
        )
        assert os.path.exists(save_path)
        plt.close('all')

    def test_plot_encircled_energy_long_focal_length(self, default_calculator, temp_dir):
        """Test plot_encircled_energy with long focal length."""
        save_path = os.path.join(temp_dir, "encircled_long.png")
        default_calculator.plot_encircled_energy(
            focal_length=200.0,
            aperture_diameter=10.0,
            save_path=save_path
        )
        assert os.path.exists(save_path)
        plt.close('all')


class TestPlotPSF2D:
    """Tests for DiffractionCalculator.plot_psf_2d"""

    def test_plot_psf_2d_basic_no_save(self, default_calculator):
        """Test plot_psf_2d runs without error when no save_path provided."""
        default_calculator.plot_psf_2d(focal_length=100.0, aperture_diameter=10.0)
        plt.close('all')

    def test_plot_psf_2d_with_save_path(self, default_calculator, temp_dir):
        """Test plot_psf_2d creates file when save_path is provided."""
        save_path = os.path.join(temp_dir, "psf_2d.png")
        default_calculator.plot_psf_2d(
            focal_length=100.0,
            aperture_diameter=10.0,
            save_path=save_path
        )
        assert os.path.exists(save_path), "Plot file should be created"
        assert os.path.getsize(save_path) > 0, "Plot file should not be empty"
        plt.close('all')

    def test_plot_psf_2d_narrow_aperture(self, default_calculator, temp_dir):
        """Test plot_psf_2d with narrow aperture."""
        save_path = os.path.join(temp_dir, "psf_narrow.png")
        default_calculator.plot_psf_2d(
            focal_length=100.0,
            aperture_diameter=2.0,
            save_path=save_path
        )
        assert os.path.exists(save_path)
        plt.close('all')

    def test_plot_psf_2d_wide_aperture(self, default_calculator, temp_dir):
        """Test plot_psf_2d with wide aperture."""
        save_path = os.path.join(temp_dir, "psf_wide.png")
        default_calculator.plot_psf_2d(
            focal_length=100.0,
            aperture_diameter=50.0,
            save_path=save_path
        )
        assert os.path.exists(save_path)
        plt.close('all')


class TestPlottingWithDifferentWavelengths:
    """Tests with different wavelengths to ensure wavelength-independent parameter handling"""

    def test_all_plots_with_different_wavelengths(self, temp_dir):
        """Test all three plotting functions with different wavelengths."""
        wavelengths = [405e-9, 550e-9, 650e-9]

        for wavelength in wavelengths:
            calc = DiffractionCalculator(wavelength=wavelength)

            airy_path = os.path.join(temp_dir, f"airy_{wavelength*1e9:.0f}.png")
            encircled_path = os.path.join(temp_dir, f"encircled_{wavelength*1e9:.0f}.png")
            psf_path = os.path.join(temp_dir, f"psf_{wavelength*1e9:.0f}.png")

            calc.plot_airy_disk(focal_length=100.0, aperture_diameter=10.0, save_path=airy_path)
            calc.plot_encircled_energy(focal_length=100.0, aperture_diameter=10.0, save_path=encircled_path)
            calc.plot_psf_2d(focal_length=100.0, aperture_diameter=10.0, save_path=psf_path)

            assert os.path.exists(airy_path)
            assert os.path.exists(encircled_path)
            assert os.path.exists(psf_path)

        plt.close('all')

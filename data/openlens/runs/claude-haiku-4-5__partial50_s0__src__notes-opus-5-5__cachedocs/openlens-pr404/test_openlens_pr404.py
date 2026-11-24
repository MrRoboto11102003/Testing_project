import pytest
import numpy as np
import logging
from pathlib import Path
from unittest.mock import patch

# Set non-interactive backend before importing openlens
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openlens.diffraction import DiffractionCalculator
import openlens.diffraction as diffraction_module


class TestPlotAiryDisk:
    """Tests for DiffractionCalculator.plot_airy_disk"""

    def test_returns_none(self):
        """plot_airy_disk should always return None"""
        calc = DiffractionCalculator()
        with patch("openlens.diffraction.plt.show"):
            result = calc.plot_airy_disk(100, 10, save_path=None)
        assert result is None

    def test_raises_zero_division_error_on_zero_aperture(self):
        """plot_airy_disk should raise ZeroDivisionError when aperture_diameter=0"""
        calc = DiffractionCalculator()
        with pytest.raises(ZeroDivisionError, match="float division by zero"):
            calc.plot_airy_disk(100, 0)

    def test_saves_file_when_save_path_provided(self, tmp_path):
        """plot_airy_disk should save a file to the given path"""
        calc = DiffractionCalculator()
        save_path = tmp_path / "airy.png"
        calc.plot_airy_disk(100, 10, save_path=str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_calls_show_when_save_path_none(self):
        """plot_airy_disk should call plt.show() when save_path is None"""
        calc = DiffractionCalculator()
        with patch("openlens.diffraction.plt.show") as mock_show:
            calc.plot_airy_disk(100, 10, save_path=None)
        mock_show.assert_called_once()

    def test_calls_show_when_save_path_empty_string(self):
        """plot_airy_disk should call plt.show() when save_path is empty string"""
        calc = DiffractionCalculator()
        with patch("openlens.diffraction.plt.show") as mock_show:
            calc.plot_airy_disk(100, 10, save_path="")
        mock_show.assert_called_once()

    def test_does_not_call_show_when_save_path_provided(self, tmp_path):
        """plot_airy_disk should not call plt.show() when save_path is provided"""
        calc = DiffractionCalculator()
        with patch("openlens.diffraction.plt.show") as mock_show:
            calc.plot_airy_disk(100, 10, save_path=str(tmp_path / "test.png"))
        mock_show.assert_not_called()

    def test_logs_warning_when_matplotlib_unavailable(self, caplog):
        """plot_airy_disk should log warning when matplotlib is unavailable"""
        calc = DiffractionCalculator()
        with patch.object(diffraction_module, "MATPLOTLIB_AVAILABLE", False):
            with caplog.at_level(logging.WARNING, logger="openlens.diffraction"):
                calc.plot_airy_disk(100, 10, save_path=None)
        assert "plot_airy_disk unavailable: matplotlib is not installed" in caplog.text

    def test_figure_content_is_correct(self):
        """plot_airy_disk should create figure with correct labels and structure"""
        calc = DiffractionCalculator()
        captured = {}

        def capture_figure(path, **kw):
            fig = plt.gcf()
            ax = fig.axes[0]
            captured.update({
                "title": ax.get_title(),
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "lines_count": len(ax.get_lines()),
                "has_legend": ax.get_legend() is not None,
            })

        with patch("openlens.diffraction.plt.savefig", capture_figure):
            calc.plot_airy_disk(100, 10, save_path="dummy.png")

        assert "Airy Diffraction Pattern" in captured["title"]
        assert "f=100mm" in captured["title"]
        assert "D=10mm" in captured["title"]
        assert captured["xlabel"] == "Radius (μm)"
        assert captured["ylabel"] == "Normalized Intensity"
        assert captured["lines_count"] == 3
        assert captured["has_legend"] is True


class TestPlotEncircledEnergy:
    """Tests for DiffractionCalculator.plot_encircled_energy"""

    def test_returns_none(self):
        """plot_encircled_energy should always return None"""
        calc = DiffractionCalculator()
        with patch("openlens.diffraction.plt.show"):
            result = calc.plot_encircled_energy(100, 10, save_path=None)
        assert result is None

    def test_raises_zero_division_error_on_zero_aperture(self):
        """plot_encircled_energy should raise ZeroDivisionError for zero aperture"""
        calc = DiffractionCalculator()
        with pytest.raises(ZeroDivisionError):
            calc.plot_encircled_energy(100, 0)

    def test_saves_file_when_save_path_provided(self, tmp_path):
        """plot_encircled_energy should save a file to the given path"""
        calc = DiffractionCalculator()
        save_path = tmp_path / "ee.png"
        calc.plot_encircled_energy(100, 10, save_path=str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_calls_show_when_save_path_none(self):
        """plot_encircled_energy should call plt.show() when save_path is None"""
        calc = DiffractionCalculator()
        with patch("openlens.diffraction.plt.show") as mock_show:
            calc.plot_encircled_energy(100, 10, save_path=None)
        mock_show.assert_called_once()

    def test_logs_warning_when_matplotlib_unavailable(self, caplog):
        """plot_encircled_energy should log warning when matplotlib is unavailable"""
        calc = DiffractionCalculator()
        with patch.object(diffraction_module, "MATPLOTLIB_AVAILABLE", False):
            with caplog.at_level(logging.WARNING, logger="openlens.diffraction"):
                calc.plot_encircled_energy(100, 10, save_path=None)
        assert "plot_encircled_energy unavailable: matplotlib is not installed" in caplog.text

    def test_figure_content_is_correct(self):
        """plot_encircled_energy should have correct labels, ylim, and 6 lines"""
        calc = DiffractionCalculator()
        captured = {}

        def capture_figure(path, **kw):
            fig = plt.gcf()
            ax = fig.axes[0]
            captured.update({
                "title": ax.get_title(),
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "lines_count": len(ax.get_lines()),
                "ylim": ax.get_ylim(),
                "legend_count": len(ax.get_legend().get_texts()) if ax.get_legend() else 0,
            })

        with patch("openlens.diffraction.plt.savefig", capture_figure):
            calc.plot_encircled_energy(100, 10, save_path="dummy.png")

        assert "Encircled Energy Curve" in captured["title"]
        assert captured["xlabel"] == "Radius (μm)"
        assert captured["ylabel"] == "Encircled Energy (%)"
        assert captured["lines_count"] == 6
        assert captured["ylim"] == pytest.approx((0.0, 105.0))
        assert captured["legend_count"] == 3


class TestPlotPSF2D:
    """Tests for DiffractionCalculator.plot_psf_2d"""

    def test_returns_none(self):
        """plot_psf_2d should always return None"""
        calc = DiffractionCalculator()
        with patch("openlens.diffraction.plt.show"):
            result = calc.plot_psf_2d(100, 10, save_path=None)
        assert result is None

    def test_raises_zero_division_error_on_zero_aperture(self):
        """plot_psf_2d should raise ZeroDivisionError for zero aperture"""
        calc = DiffractionCalculator()
        with pytest.raises(ZeroDivisionError):
            calc.plot_psf_2d(100, 0)

    def test_saves_file_when_save_path_provided(self, tmp_path):
        """plot_psf_2d should save a file to the given path"""
        calc = DiffractionCalculator()
        save_path = tmp_path / "psf.png"
        calc.plot_psf_2d(100, 10, save_path=str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_calls_show_when_save_path_none(self):
        """plot_psf_2d should call plt.show() when save_path is None"""
        calc = DiffractionCalculator()
        with patch("openlens.diffraction.plt.show") as mock_show:
            calc.plot_psf_2d(100, 10, save_path=None)
        mock_show.assert_called_once()

    def test_logs_warning_when_matplotlib_unavailable(self, caplog):
        """plot_psf_2d should log warning when matplotlib is unavailable"""
        calc = DiffractionCalculator()
        with patch.object(diffraction_module, "MATPLOTLIB_AVAILABLE", False):
            with caplog.at_level(logging.WARNING, logger="openlens.diffraction"):
                calc.plot_psf_2d(100, 10, save_path=None)
        assert "plot_psf_2d unavailable: matplotlib is not installed" in caplog.text

    def test_figure_has_two_subplots_with_correct_titles(self):
        """plot_psf_2d should create two-panel figure with correct titles and labels"""
        calc = DiffractionCalculator()
        captured = {}

        def capture_figure(path, **kw):
            fig = plt.gcf()
            ax1, ax2 = fig.axes[0], fig.axes[1]
            captured.update({
                "suptitle": fig._suptitle.get_text() if fig._suptitle else "",
                "ax1_title": ax1.get_title(),
                "ax2_title": ax2.get_title(),
                "ax1_xlabel": ax1.get_xlabel(),
                "ax1_ylabel": ax1.get_ylabel(),
                "ax2_xlabel": ax2.get_xlabel(),
                "ax2_ylabel": ax2.get_ylabel(),
                "ax2_lines_count": len(ax2.get_lines()),
            })

        with patch("openlens.diffraction.plt.savefig", capture_figure):
            calc.plot_psf_2d(100, 10, save_path="dummy.png")

        assert "Point Spread Function Analysis" in captured["suptitle"]
        assert "f=100mm" in captured["suptitle"]
        assert captured["ax1_title"] == "2D Point Spread Function"
        assert captured["ax2_title"] == "PSF Cross-Section"
        assert captured["ax1_xlabel"] == "X (μm)"
        assert captured["ax1_ylabel"] == "Y (μm)"
        assert captured["ax2_xlabel"] == "Position (μm)"
        assert captured["ax2_ylabel"] == "Normalized Intensity"
        assert captured["ax2_lines_count"] == 3

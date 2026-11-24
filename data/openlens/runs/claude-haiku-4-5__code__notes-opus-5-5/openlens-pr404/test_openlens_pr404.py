import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import MagicMock

pytest.importorskip("matplotlib")

try:
    from openlens.diffraction import DiffractionCalculator
except ImportError:
    pytest.skip("openlens not installed", allow_module_level=True)


class TestPlotAiryDisk:
    """Tests for DiffractionCalculator.plot_airy_disk"""

    def test_returns_none(self):
        """plot_airy_disk always returns None"""
        calc = DiffractionCalculator()
        result = calc.plot_airy_disk(100, 10, save_path=None)
        assert result is None

    def test_saves_file_when_path_provided(self, tmp_path):
        """plot_airy_disk saves file when save_path is provided"""
        calc = DiffractionCalculator()
        save_path = tmp_path / "airy.png"
        calc.plot_airy_disk(100, 10, save_path=str(save_path))
        assert save_path.exists() and save_path.stat().st_size > 0

    def test_raises_on_zero_aperture(self):
        """plot_airy_disk raises ZeroDivisionError when aperture_diameter=0"""
        calc = DiffractionCalculator()
        with pytest.raises(ZeroDivisionError):
            calc.plot_airy_disk(100, 0)

    def test_matplotlib_unavailable_logs_warning(self, monkeypatch, caplog):
        """plot_airy_disk logs warning when matplotlib unavailable"""
        import openlens.diffraction as d
        monkeypatch.setattr(d, "MATPLOTLIB_AVAILABLE", False)
        calc = DiffractionCalculator()
        with caplog.at_level("WARNING", logger="openlens.diffraction"):
            calc.plot_airy_disk(100, 10)
        assert "plot_airy_disk unavailable: matplotlib is not installed" in caplog.text

    def test_calls_show_when_no_save_path(self, monkeypatch):
        """plot_airy_disk calls plt.show() when save_path is None"""
        import openlens.diffraction as d
        show_mock = MagicMock()
        monkeypatch.setattr(d.plt, "show", show_mock)
        calc = DiffractionCalculator()
        calc.plot_airy_disk(100, 10, save_path=None)
        show_mock.assert_called_once()

    def test_figure_contains_expected_elements(self, monkeypatch):
        """plot_airy_disk creates figure with correct plot elements"""
        import openlens.diffraction as d
        captured = {}

        def fake_savefig(path, **kw):
            ax = d.plt.gcf().axes[0]
            captured.update({
                "title": ax.get_title(),
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "num_lines": len(ax.get_lines()),
                "num_legend_entries": len(ax.get_legend().get_texts()),
            })

        monkeypatch.setattr(d.plt, "savefig", fake_savefig)
        calc = DiffractionCalculator()
        calc.plot_airy_disk(100, 10, save_path="x.png")

        assert captured["xlabel"] == "Radius (μm)"
        assert captured["ylabel"] == "Normalized Intensity"
        assert "Airy Diffraction Pattern" in captured["title"]
        assert captured["num_lines"] == 3
        assert captured["num_legend_entries"] == 1

    def test_title_includes_parameters_and_wavelength(self, monkeypatch):
        """plot_airy_disk title includes focal length, aperture, and wavelength"""
        import openlens.diffraction as d
        captured = {}

        def fake_savefig(path, **kw):
            captured["title"] = d.plt.gcf().axes[0].get_title()

        monkeypatch.setattr(d.plt, "savefig", fake_savefig)
        calc = DiffractionCalculator(wavelength=633e-9)
        calc.plot_airy_disk(100, 10, save_path="x.png")

        assert "f=100mm" in captured["title"]
        assert "D=10mm" in captured["title"]
        assert "λ=633nm" in captured["title"]


class TestPlotEncircledEnergy:
    """Tests for DiffractionCalculator.plot_encircled_energy"""

    def test_returns_none(self):
        """plot_encircled_energy always returns None"""
        calc = DiffractionCalculator()
        result = calc.plot_encircled_energy(100, 10, save_path=None)
        assert result is None

    def test_saves_file_when_path_provided(self, tmp_path):
        """plot_encircled_energy saves file when save_path is provided"""
        calc = DiffractionCalculator()
        save_path = tmp_path / "ee.png"
        calc.plot_encircled_energy(100, 10, save_path=str(save_path))
        assert save_path.exists() and save_path.stat().st_size > 0

    def test_raises_on_zero_aperture(self):
        """plot_encircled_energy raises ZeroDivisionError when aperture_diameter=0"""
        calc = DiffractionCalculator()
        with pytest.raises(ZeroDivisionError):
            calc.plot_encircled_energy(100, 0)

    def test_matplotlib_unavailable_logs_warning(self, monkeypatch, caplog):
        """plot_encircled_energy logs warning when matplotlib unavailable"""
        import openlens.diffraction as d
        monkeypatch.setattr(d, "MATPLOTLIB_AVAILABLE", False)
        calc = DiffractionCalculator()
        with caplog.at_level("WARNING", logger="openlens.diffraction"):
            calc.plot_encircled_energy(100, 10)
        assert "plot_encircled_energy unavailable: matplotlib is not installed" in caplog.text

    def test_figure_attributes(self, monkeypatch):
        """plot_encircled_energy creates figure with correct attributes"""
        import openlens.diffraction as d
        captured = {}

        def fake_savefig(path, **kw):
            ax = d.plt.gcf().axes[0]
            captured.update({
                "title": ax.get_title(),
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "ylim": ax.get_ylim(),
                "num_lines": len(ax.get_lines()),
                "legend_texts": [t.get_text() for t in ax.get_legend().get_texts()],
            })

        monkeypatch.setattr(d.plt, "savefig", fake_savefig)
        calc = DiffractionCalculator()
        calc.plot_encircled_energy(100, 10, save_path="x.png")

        assert captured["xlabel"] == "Radius (μm)"
        assert captured["ylabel"] == "Encircled Energy (%)"
        assert "Encircled Energy Curve" in captured["title"]
        assert captured["ylim"] == pytest.approx((0.0, 105.0))
        assert captured["num_lines"] == 6
        assert len(captured["legend_texts"]) == 3
        assert any("Airy disk radius" in t for t in captured["legend_texts"])
        assert any("50%" in t for t in captured["legend_texts"])
        assert any("80%" in t for t in captured["legend_texts"])

    def test_title_includes_parameters_and_wavelength(self, monkeypatch):
        """plot_encircled_energy title includes focal length, aperture, and wavelength"""
        import openlens.diffraction as d
        captured = {}

        def fake_savefig(path, **kw):
            captured["title"] = d.plt.gcf().axes[0].get_title()

        monkeypatch.setattr(d.plt, "savefig", fake_savefig)
        calc = DiffractionCalculator(wavelength=633e-9)
        calc.plot_encircled_energy(100, 10, save_path="x.png")

        assert "f=100mm" in captured["title"]
        assert "D=10mm" in captured["title"]
        assert "λ=633nm" in captured["title"]

    def test_calls_show_when_no_save_path(self, monkeypatch):
        """plot_encircled_energy calls plt.show() when save_path is None"""
        import openlens.diffraction as d
        show_mock = MagicMock()
        monkeypatch.setattr(d.plt, "show", show_mock)
        calc = DiffractionCalculator()
        calc.plot_encircled_energy(100, 10, save_path=None)
        show_mock.assert_called_once()


class TestPlotPSF2D:
    """Tests for DiffractionCalculator.plot_psf_2d"""

    def test_returns_none(self):
        """plot_psf_2d always returns None"""
        calc = DiffractionCalculator()
        result = calc.plot_psf_2d(100, 10, save_path=None)
        assert result is None

    def test_saves_file_when_path_provided(self, tmp_path):
        """plot_psf_2d saves file when save_path is provided"""
        calc = DiffractionCalculator()
        save_path = tmp_path / "psf.png"
        calc.plot_psf_2d(100, 10, save_path=str(save_path))
        assert save_path.exists() and save_path.stat().st_size > 0

    def test_raises_on_zero_aperture(self):
        """plot_psf_2d raises ZeroDivisionError when aperture_diameter=0"""
        calc = DiffractionCalculator()
        with pytest.raises(ZeroDivisionError):
            calc.plot_psf_2d(100, 0)

    def test_matplotlib_unavailable_logs_warning(self, monkeypatch, caplog):
        """plot_psf_2d logs warning when matplotlib unavailable"""
        import openlens.diffraction as d
        monkeypatch.setattr(d, "MATPLOTLIB_AVAILABLE", False)
        calc = DiffractionCalculator()
        with caplog.at_level("WARNING", logger="openlens.diffraction"):
            calc.plot_psf_2d(100, 10)
        assert "plot_psf_2d unavailable: matplotlib is not installed" in caplog.text

    def test_figure_structure_and_suptitle(self, monkeypatch):
        """plot_psf_2d creates two-subplot figure with correct structure and suptitle"""
        import openlens.diffraction as d
        captured = {}

        def fake_savefig(path, **kw):
            fig = d.plt.gcf()
            captured.update({
                "num_axes": len(fig.axes),
                "suptitle": fig._suptitle.get_text() if fig._suptitle else None,
                "kw": kw,
            })

        monkeypatch.setattr(d.plt, "savefig", fake_savefig)
        calc = DiffractionCalculator()
        calc.plot_psf_2d(100, 10, save_path="x.png")

        assert captured["num_axes"] >= 2
        assert "Point Spread Function Analysis" in captured["suptitle"]
        assert captured["kw"] == {"dpi": 300, "bbox_inches": "tight"}

    def test_subplot_titles_and_labels(self, monkeypatch):
        """plot_psf_2d subplots have correct titles and axis labels"""
        import openlens.diffraction as d
        captured = {}

        def fake_savefig(path, **kw):
            fig = d.plt.gcf()
            ax1, ax2 = fig.axes[:2]
            captured.update({
                "ax1_title": ax1.get_title(),
                "ax1_xlabel": ax1.get_xlabel(),
                "ax1_ylabel": ax1.get_ylabel(),
                "ax2_title": ax2.get_title(),
                "ax2_xlabel": ax2.get_xlabel(),
                "ax2_ylabel": ax2.get_ylabel(),
            })

        monkeypatch.setattr(d.plt, "savefig", fake_savefig)
        calc = DiffractionCalculator()
        calc.plot_psf_2d(100, 10, save_path="x.png")

        assert "2D Point Spread Function" in captured["ax1_title"]
        assert captured["ax1_xlabel"] == "X (μm)"
        assert captured["ax1_ylabel"] == "Y (μm)"
        assert "PSF Cross-Section" in captured["ax2_title"]
        assert captured["ax2_xlabel"] == "Position (μm)"
        assert captured["ax2_ylabel"] == "Normalized Intensity"

    def test_suptitle_includes_parameters_and_wavelength(self, monkeypatch):
        """plot_psf_2d suptitle includes focal length, aperture, and wavelength"""
        import openlens.diffraction as d
        captured = {}

        def fake_savefig(path, **kw):
            captured["suptitle"] = d.plt.gcf()._suptitle.get_text()

        monkeypatch.setattr(d.plt, "savefig", fake_savefig)
        calc = DiffractionCalculator(wavelength=633e-9)
        calc.plot_psf_2d(100, 10, save_path="x.png")

        assert "f=100mm" in captured["suptitle"]
        assert "D=10mm" in captured["suptitle"]
        assert "λ=633nm" in captured["suptitle"]

    def test_calls_show_when_no_save_path(self, monkeypatch):
        """plot_psf_2d calls plt.show() when save_path is None"""
        import openlens.diffraction as d
        show_mock = MagicMock()
        monkeypatch.setattr(d.plt, "show", show_mock)
        calc = DiffractionCalculator()
        calc.plot_psf_2d(100, 10, save_path=None)
        show_mock.assert_called_once()

    def test_no_savefig_when_show(self, monkeypatch):
        """plot_psf_2d does not call savefig when save_path is None"""
        import openlens.diffraction as d
        savefig_mock = MagicMock()
        show_mock = MagicMock()
        monkeypatch.setattr(d.plt, "savefig", savefig_mock)
        monkeypatch.setattr(d.plt, "show", show_mock)
        calc = DiffractionCalculator()
        calc.plot_psf_2d(100, 10, save_path=None)
        savefig_mock.assert_not_called()

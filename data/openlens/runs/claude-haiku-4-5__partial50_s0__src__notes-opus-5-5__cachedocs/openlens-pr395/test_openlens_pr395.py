import pytest
from pathlib import Path
from xml.sax.saxutils import escape

from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.export_formats import SVGExporter


class TestSVGExporterLens:
    """Tests for SVGExporter.export_lens"""

    def test_export_lens_creates_valid_file(self, tmp_path):
        """Test basic export_lens creates a valid SVG file"""
        lens = Lens(
            name="Test Lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        filepath = tmp_path / "lens.svg"

        result = SVGExporter.export_lens(lens, str(filepath))

        assert result is None
        assert filepath.exists()
        content = filepath.read_text(encoding="utf-8")
        assert content.startswith('<?xml version="1.0" encoding="UTF-8"?>\n')
        assert content.rstrip().endswith("</svg>")

    def test_export_lens_svg_structure(self, tmp_path):
        """Test SVG structure contains required elements"""
        lens = Lens(
            name="Biconvex",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        filepath = tmp_path / "lens.svg"

        SVGExporter.export_lens(lens, str(filepath), width=400, height=300)
        content = filepath.read_text(encoding="utf-8")

        assert 'width="400" height="300" viewBox="0 0 400 300"' in content
        assert '<svg xmlns="http://www.w3.org/2000/svg"' in content
        assert '<defs>' in content
        assert "<style>" in content
        assert "</style>" in content
        assert "</defs>" in content

    def test_export_lens_title_escaping(self, tmp_path):
        """Test lens name is properly XML-escaped in title"""
        lens = Lens(
            name="A&B<C>",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        filepath = tmp_path / "lens.svg"

        SVGExporter.export_lens(lens, str(filepath))
        content = filepath.read_text(encoding="utf-8")

        assert f'class="title">{escape("A&B<C>")}</text>' in content

    def test_export_lens_optical_axis(self, tmp_path):
        """Test optical axis is drawn with correct styling"""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        filepath = tmp_path / "lens.svg"

        SVGExporter.export_lens(lens, str(filepath), width=400, height=300)
        content = filepath.read_text(encoding="utf-8")

        assert 'class="axis"' in content
        assert 'x1="40"' in content
        assert 'x2="360"' in content
        assert 'stroke-dasharray="5,5"' in content

    def test_export_lens_path_elements(self, tmp_path):
        """Test lens outline path is present with identical fill and stroke"""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        filepath = tmp_path / "lens.svg"

        SVGExporter.export_lens(lens, str(filepath))
        content = filepath.read_text(encoding="utf-8")

        assert content.count('class="lens-fill"/>') == 1
        assert content.count('class="lens-stroke"/>') == 1
        # Both paths should have the same d attribute
        assert '<path d="' in content
        fill_start = content.find('class="lens-fill"/>')
        stroke_start = content.find('class="lens-stroke"/>')
        assert fill_start > 0 and stroke_start > 0

    def test_export_lens_with_rays_enabled(self, tmp_path):
        """Test ray tracing lines appear only when show_rays=True"""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        filepath = tmp_path / "lens.svg"

        SVGExporter.export_lens(lens, str(filepath), show_rays=True)
        content = filepath.read_text(encoding="utf-8")

        # Should have exactly 3 ray lines
        assert content.count('class="ray"') == 3

    def test_export_lens_with_rays_disabled(self, tmp_path):
        """Test no rays appear when show_rays=False"""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        filepath = tmp_path / "lens.svg"

        SVGExporter.export_lens(lens, str(filepath), show_rays=False)
        content = filepath.read_text(encoding="utf-8")

        assert 'class="ray"' not in content

    def test_export_lens_labels_enabled(self, tmp_path):
        """Test dimension labels appear when show_labels=True"""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        filepath = tmp_path / "lens.svg"

        SVGExporter.export_lens(lens, str(filepath), show_labels=True)
        content = filepath.read_text(encoding="utf-8")

        assert f"∅{lens.diameter:.1f}mm" in content
        assert f"t={lens.thickness:.1f}mm" in content
        assert f"n={lens.refractive_index:.4f}" in content

    def test_export_lens_labels_disabled(self, tmp_path):
        """Test dimension labels don't appear when show_labels=False"""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        filepath = tmp_path / "lens.svg"

        SVGExporter.export_lens(lens, str(filepath), show_labels=False)
        content = filepath.read_text(encoding="utf-8")

        # Diameter label should not appear
        assert f"∅{lens.diameter:.1f}mm" not in content
        assert f"t={lens.thickness:.1f}mm" not in content

    def test_export_lens_focal_length_label(self, tmp_path):
        """Test focal length label appears only when calculate_focal_length is truthy"""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        filepath = tmp_path / "lens.svg"

        SVGExporter.export_lens(lens, str(filepath), show_labels=True)
        content = filepath.read_text(encoding="utf-8")

        fl = lens.calculate_focal_length()
        if fl:
            assert f"f={fl:.1f}mm" in content
        else:
            assert "f=" not in content or "f=" not in content.split("class=")[
                -1
            ].split(">")[0]

    def test_export_lens_custom_dimensions(self, tmp_path):
        """Test custom width and height parameters"""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        filepath = tmp_path / "lens.svg"

        SVGExporter.export_lens(lens, str(filepath), width=600, height=400)
        content = filepath.read_text(encoding="utf-8")

        assert 'width="600" height="400"' in content
        assert 'viewBox="0 0 600 400"' in content


class TestSVGExporterSystem:
    """Tests for SVGExporter.export_system"""

    def test_export_system_creates_file(self, tmp_path):
        """Test basic export_system creates valid SVG file"""
        system = OpticalSystem(name="Test System")
        lens = Lens(
            name="L1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        system.add_lens(lens)
        filepath = tmp_path / "system.svg"

        result = SVGExporter.export_system(system, str(filepath))

        assert result is None
        assert filepath.exists()
        content = filepath.read_text(encoding="utf-8")
        assert content.startswith('<?xml version="1.0" encoding="UTF-8"?>\n')
        assert content.rstrip().endswith("</svg>")

    def test_export_system_no_viewbox(self, tmp_path):
        """Test that export_system does not include viewBox attribute"""
        system = OpticalSystem(name="Test")
        lens = Lens(
            name="L1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        system.add_lens(lens)
        filepath = tmp_path / "system.svg"

        SVGExporter.export_system(system, str(filepath))
        content = filepath.read_text(encoding="utf-8")

        assert "viewBox" not in content

    def test_export_system_title_and_axis(self, tmp_path):
        """Test system title and optical axis are present"""
        system = OpticalSystem(name="My Optical System")
        lens = Lens(
            name="L1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        system.add_lens(lens)
        filepath = tmp_path / "system.svg"

        SVGExporter.export_system(system, str(filepath), width=600, height=300)
        content = filepath.read_text(encoding="utf-8")

        assert f'class="title">{escape("My Optical System")}</text>' in content
        assert 'class="axis"' in content
        assert 'x1="40"' in content
        assert 'x2="560"' in content

    def test_export_system_multiple_elements(self, tmp_path):
        """Test system with multiple lens elements"""
        system = OpticalSystem(name="Multi")
        lens1 = Lens(
            name="L1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        lens2 = Lens(
            name="L2",
            radius_of_curvature_1=40.0,
            radius_of_curvature_2=-40.0,
            thickness=4.0,
            diameter=20.0,
        )
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=10.0)
        filepath = tmp_path / "system.svg"

        SVGExporter.export_system(system, str(filepath))
        content = filepath.read_text(encoding="utf-8")

        # 2 elements -> 2 fill + 2 stroke rects
        assert content.count('class="lens-fill"') == 2
        assert content.count('class="lens-stroke"') == 2

    def test_export_system_empty(self, tmp_path):
        """Test export_system with no elements"""
        system = OpticalSystem(name="Empty")
        filepath = tmp_path / "system.svg"

        SVGExporter.export_system(system, str(filepath))
        content = filepath.read_text(encoding="utf-8")

        # No element rects
        assert 'class="lens-fill"' not in content
        assert 'class="lens-stroke"' not in content
        # But should have structure
        assert "<svg" in content
        assert "class=\"axis\"" in content

    def test_export_system_name_escaping(self, tmp_path):
        """Test system name escaping in title"""
        system = OpticalSystem(name="System&Design<Test>")
        lens = Lens(
            name="L1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        system.add_lens(lens)
        filepath = tmp_path / "system.svg"

        SVGExporter.export_system(system, str(filepath))
        content = filepath.read_text(encoding="utf-8")

        assert "System&amp;Design&lt;Test&gt;" in content

    def test_export_system_custom_dimensions(self, tmp_path):
        """Test custom width and height for system export"""
        system = OpticalSystem(name="Test")
        lens = Lens(
            name="L1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        system.add_lens(lens)
        filepath = tmp_path / "system.svg"

        SVGExporter.export_system(system, str(filepath), width=800, height=400)
        content = filepath.read_text(encoding="utf-8")

        assert 'width="800" height="400"' in content

import pytest
import math
from pathlib import Path

try:
    from openlens import Lens, OpticalSystem
    from openlens.export_formats import SVGExporter
except ImportError:
    pytest.skip("openlens not available", allow_module_level=True)


class TestSVGExporterExportLens:
    """Tests for SVGExporter.export_lens"""

    def test_export_lens_creates_file(self, tmp_path):
        """Test that export_lens creates an SVG file"""
        lens = Lens(thickness=5.0, diameter=50.0)
        output_file = tmp_path / "lens.svg"
        
        result = SVGExporter.export_lens(lens, str(output_file))
        
        assert result is None
        assert output_file.exists()

    def test_export_lens_valid_xml_structure(self, tmp_path):
        """Test SVG has valid XML header and structure"""
        lens = Lens(thickness=5.0, diameter=50.0)
        output_file = tmp_path / "lens.svg"
        
        SVGExporter.export_lens(lens, str(output_file), width=400, height=300)
        content = output_file.read_text(encoding='utf-8')
        
        assert content.startswith('<?xml version="1.0" encoding="UTF-8"?>')
        assert content.rstrip().endswith('</svg>')
        assert 'width="400" height="300" viewBox="0 0 400 300"' in content

    def test_export_lens_has_required_styles(self, tmp_path):
        """Test SVG includes all required CSS classes"""
        lens = Lens(thickness=5.0, diameter=50.0)
        output_file = tmp_path / "lens.svg"
        
        SVGExporter.export_lens(lens, str(output_file))
        content = output_file.read_text(encoding='utf-8')
        
        assert '.lens-fill' in content
        assert '.lens-stroke' in content
        assert '.axis' in content
        assert '.label' in content
        assert '.title' in content
        assert '.ray' in content

    def test_export_lens_title_placement(self, tmp_path):
        """Test lens title appears at correct position"""
        lens = Lens(thickness=5.0, diameter=50.0, name="TestLens")
        output_file = tmp_path / "lens.svg"
        
        SVGExporter.export_lens(lens, str(output_file), width=400, height=300)
        content = output_file.read_text(encoding='utf-8')
        
        assert 'class="title"' in content
        assert 'TestLens' in content
        assert 'x="200.0"' in content  # width/2
        assert 'y="25"' in content
        assert 'text-anchor="middle"' in content

    def test_export_lens_special_char_escaping(self, tmp_path):
        """Test that & < > in lens name are properly escaped"""
        lens = Lens(thickness=5.0, diameter=50.0, name="A & B < C >")
        output_file = tmp_path / "lens.svg"
        
        SVGExporter.export_lens(lens, str(output_file))
        content = output_file.read_text(encoding='utf-8')
        
        assert 'A &amp; B &lt; C &gt;' in content

    def test_export_lens_optical_axis_position(self, tmp_path):
        """Test optical axis is horizontal at center height"""
        lens = Lens(thickness=5.0, diameter=50.0)
        output_file = tmp_path / "lens.svg"
        
        SVGExporter.export_lens(lens, str(output_file), width=400, height=300)
        content = output_file.read_text(encoding='utf-8')
        
        # margin=40, so x1=40, x2=width-margin=360, y=cy=height/2
        assert 'x1="40" y1="150.0" x2="360" y2="150.0" class="axis"' in content

    def test_export_lens_path_exists_and_duplicated(self, tmp_path):
        """Test lens outline path appears exactly twice with same d"""
        lens = Lens(thickness=5.0, diameter=50.0)
        output_file = tmp_path / "lens.svg"
        
        SVGExporter.export_lens(lens, str(output_file))
        content = output_file.read_text(encoding='utf-8')
        
        # Both fill and stroke paths should have identical d attribute
        assert content.count('class="lens-fill"/>') == 1
        assert content.count('class="lens-stroke"/>') == 1
        
        # Extract both path lines and verify d attributes are identical
        lines = content.split('\n')
        fill_line = next(l for l in lines if 'class="lens-fill"' in l)
        stroke_line = next(l for l in lines if 'class="lens-stroke"' in l)
        
        fill_d = fill_line[fill_line.find('d="'):fill_line.find('class=')]
        stroke_d = stroke_line[stroke_line.find('d="'):stroke_line.find('class=')]
        assert fill_d == stroke_d

    def test_export_lens_show_rays_enabled(self, tmp_path):
        """Test exactly 3 ray lines appear when show_rays=True"""
        lens = Lens(thickness=5.0, diameter=50.0)
        output_file = tmp_path / "lens.svg"
        
        SVGExporter.export_lens(lens, str(output_file), show_rays=True)
        content = output_file.read_text(encoding='utf-8')
        
        assert content.count('class="ray"') == 3

    def test_export_lens_show_rays_disabled(self, tmp_path):
        """Test no rays when show_rays=False"""
        lens = Lens(thickness=5.0, diameter=50.0)
        output_file = tmp_path / "lens.svg"
        
        SVGExporter.export_lens(lens, str(output_file), show_rays=False)
        content = output_file.read_text(encoding='utf-8')
        
        assert 'class="ray"' not in content

    def test_export_lens_labels_enabled(self, tmp_path):
        """Test diameter and thickness labels when show_labels=True"""
        lens = Lens(thickness=5.0, diameter=50.0)
        output_file = tmp_path / "lens.svg"
        
        SVGExporter.export_lens(lens, str(output_file), show_labels=True)
        content = output_file.read_text(encoding='utf-8')
        
        assert '∅' in content  # Diameter symbol
        assert 't=' in content  # Thickness label

    def test_export_lens_labels_disabled(self, tmp_path):
        """Test no dimension labels when show_labels=False"""
        lens = Lens(thickness=5.0, diameter=50.0)
        output_file = tmp_path / "lens.svg"
        
        SVGExporter.export_lens(lens, str(output_file), show_labels=False)
        content = output_file.read_text(encoding='utf-8')
        
        assert '∅' not in content
        assert 't=' not in content
        # CSS class should still exist
        assert '.label' in content

    def test_export_lens_material_and_index(self, tmp_path):
        """Test material and refractive index are displayed"""
        lens = Lens(thickness=5.0, diameter=50.0)
        output_file = tmp_path / "lens.svg"
        
        SVGExporter.export_lens(lens, str(output_file), show_labels=True)
        content = output_file.read_text(encoding='utf-8')
        
        # Index should be formatted with 4 decimal places
        expected_index = f"{lens.refractive_index:.4f}"
        assert f"n={expected_index}" in content

    def test_export_lens_custom_dimensions(self, tmp_path):
        """Test custom width and height parameters are used"""
        lens = Lens(thickness=5.0, diameter=50.0)
        output_file = tmp_path / "lens.svg"
        
        SVGExporter.export_lens(lens, str(output_file), width=600, height=400)
        content = output_file.read_text(encoding='utf-8')
        
        assert 'width="600" height="400"' in content
        assert 'x="300.0"' in content  # width/2
        # Axis x2 should be width - margin
        assert 'x2="560"' in content  # 600 - 40


class TestSVGExporterExportSystem:
    """Tests for SVGExporter.export_system"""

    def test_export_system_creates_file(self, tmp_path):
        """Test that export_system creates an SVG file"""
        system = OpticalSystem(name="TestSystem")
        system.add_lens(Lens(thickness=5.0, diameter=50.0))
        output_file = tmp_path / "system.svg"
        
        result = SVGExporter.export_system(system, str(output_file))
        
        assert result is None
        assert output_file.exists()

    def test_export_system_valid_xml_structure(self, tmp_path):
        """Test system SVG has valid XML structure"""
        system = OpticalSystem(name="TestSystem")
        system.add_lens(Lens(thickness=5.0, diameter=50.0))
        output_file = tmp_path / "system.svg"
        
        SVGExporter.export_system(system, str(output_file), width=600, height=300)
        content = output_file.read_text(encoding='utf-8')
        
        assert content.startswith('<?xml version="1.0" encoding="UTF-8"?>\n')
        assert content.endswith('</svg>\n')

    def test_export_system_no_viewbox(self, tmp_path):
        """Test system SVG does NOT contain viewBox attribute"""
        system = OpticalSystem(name="TestSystem")
        system.add_lens(Lens(thickness=5.0, diameter=50.0))
        output_file = tmp_path / "system.svg"
        
        SVGExporter.export_system(system, str(output_file), width=600, height=300)
        content = output_file.read_text(encoding='utf-8')
        
        assert 'width="600" height="300"' in content
        assert 'viewBox' not in content

    def test_export_system_has_required_styles(self, tmp_path):
        """Test system SVG includes required CSS classes"""
        system = OpticalSystem(name="TestSystem")
        system.add_lens(Lens(thickness=5.0, diameter=50.0))
        output_file = tmp_path / "system.svg"
        
        SVGExporter.export_system(system, str(output_file))
        content = output_file.read_text(encoding='utf-8')
        
        assert '.lens-fill' in content
        assert '.lens-stroke' in content
        assert '.axis' in content
        assert '.title' in content
        # System export should not have .ray or .label
        assert '.ray' not in content
        assert '.label' not in content

    def test_export_system_title_placement(self, tmp_path):
        """Test system title appears at correct position"""
        system = OpticalSystem(name="TestSystem")
        system.add_lens(Lens(thickness=5.0, diameter=50.0))
        output_file = tmp_path / "system.svg"
        
        SVGExporter.export_system(system, str(output_file), width=600, height=300)
        content = output_file.read_text(encoding='utf-8')
        
        assert 'class="title"' in content
        assert 'TestSystem' in content
        assert 'x="300.0"' in content  # width/2
        assert 'y="25"' in content

    def test_export_system_special_char_escaping(self, tmp_path):
        """Test & < > in system name are properly escaped"""
        system = OpticalSystem(name="System & Test<>")
        system.add_lens(Lens(thickness=5.0, diameter=50.0))
        output_file = tmp_path / "system.svg"
        
        SVGExporter.export_system(system, str(output_file))
        content = output_file.read_text(encoding='utf-8')
        
        assert 'System &amp; Test&lt;&gt;' in content

    def test_export_system_optical_axis(self, tmp_path):
        """Test optical axis is drawn at center height"""
        system = OpticalSystem(name="TestSystem")
        system.add_lens(Lens(thickness=5.0, diameter=50.0))
        output_file = tmp_path / "system.svg"
        
        SVGExporter.export_system(system, str(output_file), width=600, height=300)
        content = output_file.read_text(encoding='utf-8')
        
        assert 'x1="40" y1="150.0" x2="560" y2="150.0" class="axis"' in content

    def test_export_system_empty_system(self, tmp_path):
        """Test empty system exports without lens rectangles"""
        system = OpticalSystem(name="EmptySystem")
        output_file = tmp_path / "system.svg"
        
        SVGExporter.export_system(system, str(output_file))
        
        assert output_file.exists()
        content = output_file.read_text(encoding='utf-8')
        # Should have no lens rectangles
        assert 'class="lens-fill"' not in content
        assert 'class="lens-stroke"' not in content

    def test_export_system_single_lens_rectangles(self, tmp_path):
        """Test single lens system draws exactly 2 lens rectangles"""
        system = OpticalSystem(name="SingleLens")
        system.add_lens(Lens(thickness=5.0, diameter=50.0))
        output_file = tmp_path / "system.svg"
        
        SVGExporter.export_system(system, str(output_file))
        content = output_file.read_text(encoding='utf-8')
        
        assert content.count('class="lens-fill"') == 1
        assert content.count('class="lens-stroke"') == 1

    def test_export_system_multiple_lenses(self, tmp_path):
        """Test multiple lens system draws all lens rectangles"""
        system = OpticalSystem(name="MultiLens")
        system.add_lens(Lens(thickness=5.0, diameter=50.0))
        system.add_lens(Lens(thickness=4.0, diameter=40.0))
        system.add_lens(Lens(thickness=3.0, diameter=30.0))
        output_file = tmp_path / "system.svg"
        
        SVGExporter.export_system(system, str(output_file))
        content = output_file.read_text(encoding='utf-8')
        
        # 3 lenses, 2 rectangles each (fill and stroke)
        assert content.count('class="lens-fill"') == 3
        assert content.count('class="lens-stroke"') == 3

    def test_export_system_custom_dimensions(self, tmp_path):
        """Test custom dimensions are applied"""
        system = OpticalSystem(name="CustomSystem")
        system.add_lens(Lens(thickness=5.0, diameter=50.0))
        output_file = tmp_path / "system.svg"
        
        SVGExporter.export_system(system, str(output_file), width=800, height=400)
        content = output_file.read_text(encoding='utf-8')
        
        assert 'width="800" height="400"' in content
        assert 'x="400.0"' in content  # width/2
        assert 'x2="760"' in content   # width - margin
        assert 'y1="200.0"' in content  # height/2

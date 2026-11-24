import pytest
import tempfile
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock, patch

from openlens.export_formats import SVGExporter


class TestSVGExporterLens:
    """Test SVGExporter.export_lens method"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for SVG files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def biconvex_lens(self):
        """Create a mock biconvex lens"""
        lens = Mock()
        lens.name = "Biconvex Demo"
        lens.radius_of_curvature_1 = 50.0
        lens.radius_of_curvature_2 = -50.0
        lens.thickness = 5.0
        lens.diameter = 25.0
        lens.material = "BK7"
        lens.refractive_index = 1.5168
        lens.calculate_focal_length.return_value = 97.58
        return lens

    @pytest.fixture
    def plano_convex_lens(self):
        """Create a mock plano-convex lens"""
        lens = Mock()
        lens.name = "Plano-Convex"
        lens.radius_of_curvature_1 = 50.0
        lens.radius_of_curvature_2 = 1e10  # Flat surface
        lens.thickness = 4.0
        lens.diameter = 20.0
        lens.material = "Fused Silica"
        lens.refractive_index = 1.4585
        lens.calculate_focal_length.return_value = 109.0
        return lens

    @pytest.fixture
    def afocal_lens(self):
        """Create a mock afocal lens (no focal length)"""
        lens = Mock()
        lens.name = "Afocal Element"
        lens.radius_of_curvature_1 = 50.0
        lens.radius_of_curvature_2 = 50.0
        lens.thickness = 5.0
        lens.diameter = 25.0
        lens.material = "BK7"
        lens.refractive_index = 1.5168
        lens.calculate_focal_length.return_value = None
        return lens

    def test_export_lens_creates_file(self, temp_dir, biconvex_lens):
        """Test that export_lens creates an SVG file"""
        filepath = os.path.join(temp_dir, "test.svg")
        SVGExporter.export_lens(biconvex_lens, filepath)
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0

    def test_export_lens_valid_xml(self, temp_dir, biconvex_lens):
        """Test that exported SVG is valid XML"""
        filepath = os.path.join(temp_dir, "test.svg")
        SVGExporter.export_lens(biconvex_lens, filepath)
        tree = ET.parse(filepath)
        root = tree.getroot()
        assert "svg" in root.tag

    def test_export_lens_contains_title(self, temp_dir, biconvex_lens):
        """Test that SVG contains lens name as title"""
        filepath = os.path.join(temp_dir, "test.svg")
        SVGExporter.export_lens(biconvex_lens, filepath)
        with open(filepath, "r") as f:
            content = f.read()
        assert biconvex_lens.name in content
        assert "<text" in content

    def test_export_lens_contains_optical_axis(self, temp_dir, biconvex_lens):
        """Test that SVG contains optical axis line"""
        filepath = os.path.join(temp_dir, "test.svg")
        SVGExporter.export_lens(biconvex_lens, filepath)
        with open(filepath, "r") as f:
            content = f.read()
        assert 'class="axis"' in content

    def test_export_lens_with_labels(self, temp_dir, biconvex_lens):
        """Test that SVG contains dimension labels when show_labels=True"""
        filepath = os.path.join(temp_dir, "test.svg")
        SVGExporter.export_lens(biconvex_lens, filepath, show_labels=True)
        with open(filepath, "r") as f:
            content = f.read()
        # Should contain material, dimensions, and other labels
        assert "BK7" in content
        assert "mm" in content
        assert 'class="label"' in content

    def test_export_lens_without_labels(self, temp_dir, biconvex_lens):
        """Test that SVG has no label class when show_labels=False"""
        filepath = os.path.join(temp_dir, "test.svg")
        SVGExporter.export_lens(biconvex_lens, filepath, show_labels=False)
        with open(filepath, "r") as f:
            content = f.read()
        # Still has title but no dimension labels
        assert biconvex_lens.name in content
        assert 'class="label"' not in content

    def test_export_lens_with_rays(self, temp_dir, biconvex_lens):
        """Test that SVG contains ray paths when show_rays=True"""
        filepath = os.path.join(temp_dir, "test.svg")
        SVGExporter.export_lens(biconvex_lens, filepath, show_rays=True)
        with open(filepath, "r") as f:
            content = f.read()
        assert 'class="ray"' in content

    def test_export_lens_without_rays(self, temp_dir, biconvex_lens):
        """Test that SVG has no ray paths when show_rays=False"""
        filepath = os.path.join(temp_dir, "test.svg")
        SVGExporter.export_lens(biconvex_lens, filepath, show_rays=False)
        with open(filepath, "r") as f:
            content = f.read()
        assert 'class="ray"' not in content

    def test_export_lens_custom_dimensions(self, temp_dir, biconvex_lens):
        """Test that custom width and height are applied"""
        filepath = os.path.join(temp_dir, "test.svg")
        width, height = 800, 600
        SVGExporter.export_lens(biconvex_lens, filepath, width=width, height=height)
        tree = ET.parse(filepath)
        root = tree.getroot()
        assert int(root.get("width")) == width
        assert int(root.get("height")) == height

    def test_export_lens_plano_convex_flat_surface(self, temp_dir, plano_convex_lens):
        """Test export of plano-convex lens with flat surface"""
        filepath = os.path.join(temp_dir, "test.svg")
        SVGExporter.export_lens(plano_convex_lens, filepath)
        assert os.path.exists(filepath)
        with open(filepath, "r") as f:
            content = f.read()
        assert plano_convex_lens.name in content
        # Should have lens path
        assert "<path" in content

    def test_export_lens_afocal_no_error(self, temp_dir, afocal_lens):
        """Test that afocal lens with None focal_length exports without error"""
        filepath = os.path.join(temp_dir, "test.svg")
        SVGExporter.export_lens(afocal_lens, filepath, show_labels=True)
        assert os.path.exists(filepath)
        with open(filepath, "r") as f:
            content = f.read()
        # Should contain lens info but handle None focal length gracefully
        assert afocal_lens.name in content


class TestSVGExporterSystem:
    """Test SVGExporter.export_system method"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for SVG files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_lens(self):
        """Create a mock lens"""
        lens = Mock()
        lens.name = "L1"
        lens.diameter = 25.0
        lens.thickness = 5.0
        lens.material = "BK7"
        lens.refractive_index = 1.5168
        return lens

    @pytest.fixture
    def mock_system_single(self, mock_lens):
        """Create a mock optical system with single lens"""
        system = Mock()
        system.name = "Single Lens System"
        
        elem = Mock()
        elem.lens = mock_lens
        
        system.elements = [elem]
        system.get_total_length.return_value = 50.0
        return system

    @pytest.fixture
    def mock_system_multi(self):
        """Create a mock optical system with multiple lenses"""
        system = Mock()
        system.name = "Multi-Element System"
        
        lens1 = Mock()
        lens1.diameter = 25.0
        lens1.thickness = 5.0
        
        lens2 = Mock()
        lens2.diameter = 20.0
        lens2.thickness = 4.0
        
        elem1 = Mock()
        elem1.lens = lens1
        
        elem2 = Mock()
        elem2.lens = lens2
        
        system.elements = [elem1, elem2]
        system.get_total_length.return_value = 100.0
        return system

    def test_export_system_creates_file(self, temp_dir, mock_system_single):
        """Test that export_system creates an SVG file"""
        filepath = os.path.join(temp_dir, "system.svg")
        SVGExporter.export_system(mock_system_single, filepath)
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0

    def test_export_system_valid_xml(self, temp_dir, mock_system_single):
        """Test that exported system SVG is valid XML"""
        filepath = os.path.join(temp_dir, "system.svg")
        SVGExporter.export_system(mock_system_single, filepath)
        tree = ET.parse(filepath)
        root = tree.getroot()
        assert "svg" in root.tag

    def test_export_system_contains_title(self, temp_dir, mock_system_single):
        """Test that SVG contains system name as title"""
        filepath = os.path.join(temp_dir, "system.svg")
        SVGExporter.export_system(mock_system_single, filepath)
        with open(filepath, "r") as f:
            content = f.read()
        assert mock_system_single.name in content

    def test_export_system_contains_axis(self, temp_dir, mock_system_single):
        """Test that SVG contains optical axis"""
        filepath = os.path.join(temp_dir, "system.svg")
        SVGExporter.export_system(mock_system_single, filepath)
        with open(filepath, "r") as f:
            content = f.read()
        assert 'class="axis"' in content

    def test_export_system_multi_element(self, temp_dir, mock_system_multi):
        """Test export of multi-element optical system"""
        filepath = os.path.join(temp_dir, "system.svg")
        SVGExporter.export_system(mock_system_multi, filepath)
        assert os.path.exists(filepath)
        tree = ET.parse(filepath)
        root = tree.getroot()
        # Should contain lens rectangles
        assert "<rect" in ET.tostring(root, encoding="unicode")

    def test_export_system_custom_dimensions(self, temp_dir, mock_system_single):
        """Test that custom width and height are applied to system"""
        filepath = os.path.join(temp_dir, "system.svg")
        width, height = 1000, 400
        SVGExporter.export_system(mock_system_single, filepath, width=width, height=height)
        tree = ET.parse(filepath)
        root = tree.getroot()
        assert int(root.get("width")) == width
        assert int(root.get("height")) == height

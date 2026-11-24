import pytest
import tempfile
import os
from xml.etree import ElementTree as ET

from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.export_formats import SVGExporter


class TestSVGExporterLens:
    """Test cases for SVGExporter.export_lens"""

    def test_export_lens_basic(self):
        """Test basic lens export with default parameters"""
        lens = Lens(
            name="Test Lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_lens.svg")
            SVGExporter.export_lens(lens, filepath)
            
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0

    def test_export_lens_valid_svg_format(self):
        """Test that exported file is valid SVG XML"""
        lens = Lens(
            name="Valid SVG Test",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=10.0,
            diameter=50.0,
            material="BK7"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "valid.svg")
            SVGExporter.export_lens(lens, filepath)
            
            tree = ET.parse(filepath)
            root = tree.getroot()
            assert "svg" in root.tag.lower()

    def test_export_lens_contains_lens_name(self):
        """Test that exported SVG contains lens name in title"""
        lens = Lens(
            name="My Special Lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "lens_title.svg")
            SVGExporter.export_lens(lens, filepath)
            
            with open(filepath, 'r') as f:
                content = f.read()
                assert "My Special Lens" in content

    def test_export_lens_custom_dimensions(self):
        """Test export with custom width and height"""
        lens = Lens(
            name="Custom Size",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "custom_size.svg")
            SVGExporter.export_lens(lens, filepath, width=800, height=600)
            
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            assert root.get('width') == '800'
            assert root.get('height') == '600'

    def test_export_lens_with_rays_enabled(self):
        """Test export with show_rays=True includes ray paths"""
        lens = Lens(
            name="With Rays",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "with_rays.svg")
            SVGExporter.export_lens(lens, filepath, show_rays=True)
            
            with open(filepath, 'r') as f:
                content = f.read()
                assert 'class="ray"' in content

    def test_export_lens_with_rays_disabled(self):
        """Test that show_rays=False produces smaller output"""
        lens = Lens(
            name="No Rays",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath_with = os.path.join(tmpdir, "with_rays.svg")
            filepath_without = os.path.join(tmpdir, "without_rays.svg")
            
            SVGExporter.export_lens(lens, filepath_with, show_rays=True)
            SVGExporter.export_lens(lens, filepath_without, show_rays=False)
            
            with open(filepath_with, 'r') as f:
                content_with = f.read()
            with open(filepath_without, 'r') as f:
                content_without = f.read()
            
            with_ray_lines = content_with.count('class="ray"')
            without_ray_lines = content_without.count('class="ray"')
            assert with_ray_lines > without_ray_lines

    def test_export_lens_with_labels_enabled(self):
        """Test export with show_labels=True includes dimension labels"""
        lens = Lens(
            name="With Labels",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "with_labels.svg")
            SVGExporter.export_lens(lens, filepath, show_labels=True)
            
            with open(filepath, 'r') as f:
                content = f.read()
                assert 'class="label"' in content
                assert "mm" in content

    def test_export_lens_with_labels_disabled(self):
        """Test that show_labels=False produces fewer labels"""
        lens = Lens(
            name="No Labels",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath_with = os.path.join(tmpdir, "with_labels.svg")
            filepath_without = os.path.join(tmpdir, "without_labels.svg")
            
            SVGExporter.export_lens(lens, filepath_with, show_labels=True)
            SVGExporter.export_lens(lens, filepath_without, show_labels=False)
            
            with open(filepath_with, 'r') as f:
                content_with = f.read()
            with open(filepath_without, 'r') as f:
                content_without = f.read()
            
            assert content_with.count('class="label"') > content_without.count('class="label"')

    def test_export_lens_plano_convex(self):
        """Test export with flat (plano) surface"""
        lens = Lens(
            name="Plano-Convex",
            radius_of_curvature_1=1e10,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "plano_convex.svg")
            SVGExporter.export_lens(lens, filepath)
            
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0

    def test_export_lens_contains_optical_axis(self):
        """Test that SVG contains optical axis line"""
        lens = Lens(
            name="Axis Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "axis.svg")
            SVGExporter.export_lens(lens, filepath)
            
            with open(filepath, 'r') as f:
                content = f.read()
                assert 'class="axis"' in content


class TestSVGExporterSystem:
    """Test cases for SVGExporter.export_system"""

    def test_export_system_single_element(self):
        """Test basic system export with single element"""
        lens = Lens(
            name="L1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        
        system = OpticalSystem(name="Single Element System")
        system.add_element(lens)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "system.svg")
            SVGExporter.export_system(system, filepath)
            
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0

    def test_export_system_valid_svg_format(self):
        """Test that exported system file is valid SVG"""
        lens = Lens(
            name="L1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        
        system = OpticalSystem(name="Valid System")
        system.add_element(lens)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "valid_system.svg")
            SVGExporter.export_system(system, filepath)
            
            tree = ET.parse(filepath)
            root = tree.getroot()
            assert "svg" in root.tag.lower()

    def test_export_system_custom_dimensions(self):
        """Test system export with custom width and height"""
        lens = Lens(
            name="L1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        
        system = OpticalSystem(name="Custom System")
        system.add_element(lens)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "custom_system.svg")
            SVGExporter.export_system(system, filepath, width=1000, height=400)
            
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            assert root.get('width') == '1000'
            assert root.get('height') == '400'

    def test_export_system_contains_system_name(self):
        """Test that system SVG contains system name"""
        system = OpticalSystem(name="My Optical System")
        lens = Lens(
            name="L1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system.add_element(lens)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "system_title.svg")
            SVGExporter.export_system(system, filepath)
            
            with open(filepath, 'r') as f:
                content = f.read()
                assert "My Optical System" in content

    def test_export_system_multi_element_with_gaps(self):
        """Test export of multi-element system with air gaps"""
        system = OpticalSystem(name="Multi-Element")
        
        for i in range(3):
            lens = Lens(
                name=f"L{i+1}",
                radius_of_curvature_1=50.0 + i*10,
                radius_of_curvature_2=-(50.0 + i*10),
                thickness=5.0,
                diameter=25.0 - i*2,
                material="BK7"
            )
            system.add_element(lens, air_gap_before=5.0 if i > 0 else 0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "multi.svg")
            SVGExporter.export_system(system, filepath)
            
            assert os.path.exists(filepath)
            with open(filepath, 'r') as f:
                content = f.read()
                assert "rect" in content

    def test_export_system_contains_optical_axis(self):
        """Test that system SVG contains optical axis"""
        system = OpticalSystem(name="Axis System")
        lens = Lens(
            name="L1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system.add_element(lens)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "axis_system.svg")
            SVGExporter.export_system(system, filepath)
            
            with open(filepath, 'r') as f:
                content = f.read()
                assert 'class="axis"' in content

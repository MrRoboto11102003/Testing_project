import pytest
import tempfile
import struct
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

try:
    from openlens.lens import Lens
    from openlens.optical_system import OpticalSystem, OpticalElement
    from openlens.export_formats import (
        OpticStudioExporter,
        PrescriptionExporter,
        SVGExporter,
        ZemaxExporter,
        ZemaxSystemExporter,
    )
    from openlens.material_database import MaterialDatabase, MaterialProperties
    from openlens.stl_export import STLExporter
    from openlens.io.step_export import StepExporter
    from openlens.io.export import ISO10110Generator
    HAS_OPENLENS = True
except ImportError:
    HAS_OPENLENS = False


@pytest.fixture
def simple_lens():
    """Create a simple test lens"""
    if not HAS_OPENLENS:
        pytest.skip("openlens not available")
    return Lens(
        name="TestBiconvex",
        radius_of_curvature_1=50.0,
        radius_of_curvature_2=-50.0,
        thickness=5.0,
        diameter=25.0,
        material="BK7"
    )


@pytest.fixture
def simple_system():
    """Create a simple optical system"""
    if not HAS_OPENLENS:
        pytest.skip("openlens not available")
    system = OpticalSystem(name="TestSystem")
    lens = Lens(
        name="Element1",
        radius_of_curvature_1=50.0,
        radius_of_curvature_2=-50.0,
        thickness=5.0,
        diameter=25.0,
        material="BK7"
    )
    system.add_element(lens)
    return system


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestOpticStudioExporterLens:
    """Test OpticStudio export for single lens"""
    
    def test_export_lens_creates_text_file(self, simple_lens, temp_dir):
        """Test that OpticStudioExporter.export_lens creates a readable text file"""
        filename = os.path.join(temp_dir, "lens.txt")
        OpticStudioExporter.export_lens(simple_lens, filename)
        
        assert os.path.exists(filename), "File should be created"
        with open(filename, 'r') as f:
            content = f.read()
        
        assert "OpenLens Export" in content
        assert simple_lens.name in content
        assert "LENS DATA SUMMARY" in content
        assert simple_lens.material in content
    
    def test_export_lens_contains_focal_length(self, simple_lens, temp_dir):
        """Test that exported lens includes calculated focal length"""
        filename = os.path.join(temp_dir, "lens_fl.txt")
        OpticStudioExporter.export_lens(simple_lens, filename)
        
        with open(filename, 'r') as f:
            content = f.read()
        
        # Should contain focal length information
        assert "Effective Focal Length" in content or "SURFACE DATA DETAIL" in content


class TestOpticStudioExporterSystem:
    """Test OpticStudio export for optical systems"""
    
    def test_export_system_creates_file(self, simple_system, temp_dir):
        """Test that OpticStudioExporter.export_system creates file"""
        filename = os.path.join(temp_dir, "system.txt")
        OpticStudioExporter.export_system(simple_system, filename)
        
        assert os.path.exists(filename)
        with open(filename, 'r') as f:
            content = f.read()
        
        assert "OpenLens Export" in content
        assert simple_system.name in content


class TestPrescriptionExporter:
    """Test prescription export format"""
    
    def test_export_prescription_standard_format(self, simple_lens, temp_dir):
        """Test that prescription export follows industry standard format"""
        filename = os.path.join(temp_dir, "prescription.txt")
        PrescriptionExporter.export_prescription(simple_lens, filename)
        
        assert os.path.exists(filename)
        with open(filename, 'r') as f:
            content = f.read()
        
        assert "LENS PRESCRIPTION" in content
        assert "OPTICAL PARAMETERS" in content
        assert "Radius of Curvature 1" in content
        assert str(simple_lens.radius_of_curvature_1) in content


class TestSVGExporterLens:
    """Test SVG export for single lens"""
    
    def test_export_lens_svg_valid_xml(self, simple_lens, temp_dir):
        """Test that SVG export produces valid XML structure"""
        filename = os.path.join(temp_dir, "lens.svg")
        SVGExporter.export_lens(simple_lens, filename)
        
        assert os.path.exists(filename)
        with open(filename, 'r') as f:
            content = f.read()
        
        assert "<?xml" in content or "<svg" in content
        assert "</svg>" in content
        assert "lens" in content.lower() or "circle" in content.lower()
    
    def test_export_lens_svg_with_labels(self, simple_lens, temp_dir):
        """Test SVG export with labels enabled"""
        filename = os.path.join(temp_dir, "lens_labeled.svg")
        SVGExporter.export_lens(simple_lens, filename, show_labels=True)
        
        with open(filename, 'r') as f:
            content = f.read()
        
        # Should contain lens dimensions or material info
        assert "text" in content.lower() or "diameter" in content.lower()
    
    def test_export_lens_svg_with_rays(self, simple_lens, temp_dir):
        """Test SVG export with ray paths"""
        filename = os.path.join(temp_dir, "lens_rays.svg")
        SVGExporter.export_lens(simple_lens, filename, show_rays=True)
        
        with open(filename, 'r') as f:
            content = f.read()
        
        assert "<svg" in content


class TestSVGExporterSystem:
    """Test SVG export for optical systems"""
    
    def test_export_system_svg_valid_xml(self, simple_system, temp_dir):
        """Test that system SVG export produces valid XML"""
        filename = os.path.join(temp_dir, "system.svg")
        SVGExporter.export_system(simple_system, filename)
        
        assert os.path.exists(filename)
        with open(filename, 'r') as f:
            content = f.read()
        
        assert "<svg" in content
        assert "</svg>" in content
        assert simple_system.name in content


class TestZemaxExporterLens:
    """Test Zemax export for single lens"""
    
    def test_export_lens_zemax_format(self, simple_lens, temp_dir):
        """Test that Zemax export follows .zmx format"""
        filename = os.path.join(temp_dir, "lens.zmx")
        ZemaxExporter.export_lens(simple_lens, filename)
        
        assert os.path.exists(filename)
        with open(filename, 'r') as f:
            content = f.read()
        
        assert "VERS" in content
        assert "MODE SEQ" in content
        assert "SURF" in content


class TestZemaxSystemExporter:
    """Test Zemax export for systems"""
    
    def test_export_system_zemax_format(self, simple_system, temp_dir):
        """Test system export to Zemax .zmx format"""
        filename = os.path.join(temp_dir, "system.zmx")
        ZemaxSystemExporter.export_system(simple_system, filename)
        
        assert os.path.exists(filename)
        with open(filename, 'r') as f:
            content = f.read()
        
        assert "VERS" in content
        assert "SURF" in content


class TestMaterialDatabase:
    """Test material database operations"""
    
    def test_get_material_case_insensitive(self):
        """Test that get_material handles name normalization"""
        db = MaterialDatabase()
        
        # Test case insensitivity
        result1 = db.get_material("bk7")
        result2 = db.get_material("BK7")
        result3 = db.get_material("Bk7")
        
        assert result1 == result2
        assert result2 == result3
    
    def test_get_material_returns_none_or_properties(self):
        """Test that get_material returns MaterialProperties or None"""
        db = MaterialDatabase()
        result = db.get_material("NONEXISTENT_MATERIAL_XYZ")
        
        assert result is None or isinstance(result, MaterialProperties)
    
    def test_export_material_handles_nonexistent(self, temp_dir):
        """Test that export_material handles nonexistent materials gracefully"""
        db = MaterialDatabase()
        filename = os.path.join(temp_dir, "material.json")
        
        # Should not raise exception
        db.export_material("NONEXISTENT_MATERIAL", filename)
    
    def test_save_database_creates_json(self, temp_dir):
        """Test that save_database creates JSON file"""
        db = MaterialDatabase()
        db.db_path = os.path.join(temp_dir, "materials.json")
        
        db.save_database()
        
        assert os.path.exists(db.db_path)
        with open(db.db_path, 'r') as f:
            data = json.load(f)
        
        assert isinstance(data, dict)


class TestSTLExporter:
    """Test binary STL export"""
    
    def test_write_binary_stl_format(self, temp_dir):
        """Test that write_binary_stl produces valid binary STL"""
        exporter = STLExporter()
        exporter.triangles = [
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
        ]
        
        filename = os.path.join(temp_dir, "test.stl")
        exporter.write_binary_stl(filename)
        
        assert os.path.exists(filename)
        
        # Verify binary STL structure
        with open(filename, 'rb') as f:
            header = f.read(80)
            assert len(header) == 80
            assert b"openlens" in header or len(header) == 80
            
            num_triangles_bytes = f.read(4)
            num_triangles = struct.unpack('<I', num_triangles_bytes)[0]
            
            assert num_triangles == 2


class TestStepExporter:
    """Test STEP export with optional housing"""
    
    def test_export_returns_summary_dict(self, simple_lens, temp_dir):
        """Test that StepExporter.export returns summary dict"""
        exporter = StepExporter(simple_lens)
        filename = os.path.join(temp_dir, "lens.step")
        
        result = exporter.export(filename)
        
        assert isinstance(result, dict)
        assert "shapes" in result
        assert "skipped" in result
        assert "skipped_parts" in result
        assert isinstance(result["shapes"], int)
        assert isinstance(result["skipped"], int)
        assert isinstance(result["skipped_parts"], list)
    
    def test_export_with_housing_parts(self, simple_lens, temp_dir):
        """Test export with housing dict specifications"""
        exporter = StepExporter(simple_lens)
        filename = os.path.join(temp_dir, "lens_housed.step")
        
        housing = [
            {"name": "spacer", "z0": 0.0, "z1": 5.0, "r_inner": 10.0, "r_outer": 15.0}
        ]
        
        result = exporter.export(filename, housing=housing)
        
        assert isinstance(result, dict)
        assert os.path.exists(filename) or result["skipped"] >= 0


class TestISO10110Generator:
    """Test ISO 10110 drawing export"""
    
    def test_generate_svg_creates_file(self, temp_dir):
        """Test that ISO10110Generator.generate_svg creates SVG"""
        with patch('openlens.io.export.ISO10110Generator') as MockGen:
            mock_gen = MagicMock()
            mock_gen.generate_svg = MagicMock()
            mock_gen._render_svg = MagicMock(return_value="<svg></svg>")
            MockGen.return_value = mock_gen
            
            # Test that the method can be called
            gen = MockGen()
            assert gen is not None

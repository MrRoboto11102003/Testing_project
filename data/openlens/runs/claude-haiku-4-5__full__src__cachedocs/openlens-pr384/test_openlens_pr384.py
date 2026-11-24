import pytest
import tempfile
import os
import struct
import json
from pathlib import Path

from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.export_formats import (
    OpticStudioExporter,
    PrescriptionExporter,
    SVGExporter,
    ZemaxExporter,
    ZemaxSystemExporter,
)
from openlens.io.export import ISO10110Generator
from openlens.io.step_export import StepExporter
from openlens.material_database import MaterialDatabase
from openlens.stl_export import STLExporter


@pytest.fixture
def biconvex_lens():
    """Standard biconvex lens for testing"""
    return Lens(
        name="Test Biconvex",
        radius_of_curvature_1=50.0,
        radius_of_curvature_2=-50.0,
        thickness=5.0,
        diameter=25.0,
        material="BK7",
    )


@pytest.fixture
def simple_system(biconvex_lens):
    """Simple two-element optical system"""
    system = OpticalSystem(name="Test System")
    system.add_lens(biconvex_lens)
    system.add_lens(biconvex_lens, air_gap_before=10.0)
    return system


@pytest.fixture
def temp_dir():
    """Temporary directory for output files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestOpticStudioExporter:
    def test_export_lens_creates_readable_file(self, biconvex_lens, temp_dir):
        """OpticStudio lens export creates file with optical parameters"""
        filepath = os.path.join(temp_dir, "lens.txt")
        OpticStudioExporter.export_lens(biconvex_lens, filepath)
        
        assert os.path.exists(filepath)
        with open(filepath) as f:
            content = f.read()
        assert biconvex_lens.name in content
        assert "OpticStudio Format" in content
        assert biconvex_lens.material in content

    def test_export_system_includes_elements(self, simple_system, temp_dir):
        """OpticStudio system export includes multi-element data"""
        filepath = os.path.join(temp_dir, "system.txt")
        OpticStudioExporter.export_system(simple_system, filepath)
        
        assert os.path.exists(filepath)
        with open(filepath) as f:
            content = f.read()
        assert simple_system.name in content
        assert "SURFACE DATA" in content


class TestPrescriptionExporter:
    def test_export_prescription_format(self, biconvex_lens, temp_dir):
        """Prescription export includes standard optical parameters"""
        filepath = os.path.join(temp_dir, "prescription.txt")
        PrescriptionExporter.export_prescription(biconvex_lens, filepath)
        
        with open(filepath) as f:
            content = f.read()
        assert "LENS PRESCRIPTION" in content
        assert "Radius of Curvature" in content
        assert str(biconvex_lens.thickness) in content
        assert biconvex_lens.material in content


class TestSVGExporter:
    def test_export_lens_svg_valid_xml(self, biconvex_lens, temp_dir):
        """SVG lens export produces valid XML with lens geometry"""
        filepath = os.path.join(temp_dir, "lens.svg")
        SVGExporter.export_lens(biconvex_lens, filepath)
        
        with open(filepath) as f:
            content = f.read()
        assert content.startswith('<?xml')
        assert '<svg' in content
        assert '</svg>' in content
        assert biconvex_lens.name in content

    def test_export_lens_svg_respects_parameters(self, biconvex_lens, temp_dir):
        """SVG export with custom width, height, show_rays, show_labels"""
        filepath = os.path.join(temp_dir, "lens_custom.svg")
        SVGExporter.export_lens(
            biconvex_lens,
            filepath,
            width=800,
            height=600,
            show_rays=True,
            show_labels=True,
        )
        
        with open(filepath) as f:
            content = f.read()
        assert 'width="800"' in content
        assert 'height="600"' in content

    def test_export_system_svg(self, simple_system, temp_dir):
        """SVG system export includes all elements"""
        filepath = os.path.join(temp_dir, "system.svg")
        SVGExporter.export_system(simple_system, filepath)
        
        with open(filepath) as f:
            content = f.read()
        assert '<svg' in content
        assert simple_system.name in content


class TestZemaxExporter:
    def test_export_lens_zemax_format(self, biconvex_lens, temp_dir):
        """Zemax lens export contains required surface specifications"""
        filepath = os.path.join(temp_dir, "lens.zmx")
        ZemaxExporter.export_lens(biconvex_lens, filepath)
        
        with open(filepath) as f:
            content = f.read()
        assert "VERS 140000" in content
        assert "SURF" in content
        assert "GLAS" in content

    def test_export_system_zemax_multiple_surfaces(self, simple_system, temp_dir):
        """Zemax system export includes surfaces for all elements"""
        filepath = os.path.join(temp_dir, "system.zmx")
        ZemaxSystemExporter.export_system(simple_system, filepath)
        
        with open(filepath) as f:
            content = f.read()
        assert "MODE SEQ" in content
        assert content.count("SURF") >= 3


class TestISO10110Generator:
    def test_generate_svg_creates_file(self, biconvex_lens, temp_dir):
        """ISO10110Generator.generate_svg creates valid SVG file"""
        generator = ISO10110Generator(biconvex_lens)
        filepath = os.path.join(temp_dir, "iso10110.svg")
        generator.generate_svg(filepath)
        
        assert os.path.exists(filepath)
        with open(filepath) as f:
            content = f.read()
        assert '<svg' in content


class TestStepExporter:
    def test_export_returns_summary_dict(self, biconvex_lens, temp_dir):
        """StepExporter.export returns dict with shapes, skipped, skipped_parts"""
        exporter = StepExporter(biconvex_lens)
        filepath = os.path.join(temp_dir, "lens.stp")
        result = exporter.export(filepath)
        
        assert isinstance(result, dict)
        assert "shapes" in result
        assert "skipped" in result
        assert "skipped_parts" in result


class TestSTLExporter:
    def test_write_binary_stl_format(self, temp_dir):
        """STLExporter.write_binary_stl creates binary STL with correct header"""
        exporter = STLExporter()
        exporter.triangles = [((0, 0, 0), (1, 0, 0), (0, 1, 0))]
        filepath = os.path.join(temp_dir, "test.stl")
        
        exporter.write_binary_stl(filepath)
        
        assert os.path.exists(filepath)
        with open(filepath, 'rb') as f:
            header = f.read(80)
            assert len(header) == 80
            num_triangles = struct.unpack('<I', f.read(4))[0]
            assert num_triangles == 1


class TestMaterialDatabase:
    def test_get_material_bk7_success(self):
        """MaterialDatabase.get_material finds BK7 by exact uppercase key"""
        db = MaterialDatabase()
        material = db.get_material("BK7")
        assert material is not None

    def test_get_material_case_insensitive(self):
        """MaterialDatabase.get_material finds material with lowercase"""
        db = MaterialDatabase()
        material = db.get_material("bk7")
        assert material is not None

    def test_get_material_nonexistent_returns_none(self):
        """MaterialDatabase.get_material returns None for missing material"""
        db = MaterialDatabase()
        material = db.get_material("NONEXISTENT_GLASS_XYZ")
        assert material is None

    def test_save_database_creates_json(self, temp_dir):
        """MaterialDatabase.save_database writes valid JSON file"""
        db = MaterialDatabase()
        filepath = os.path.join(temp_dir, "materials.json")
        db.db_path = filepath
        db.save_database()
        
        assert os.path.exists(filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_export_material_writes_data(self, temp_dir):
        """MaterialDatabase.export_material exports material properties"""
        db = MaterialDatabase()
        filepath = os.path.join(temp_dir, "bk7.json")
        db.export_material("BK7", filepath)
        
        assert os.path.exists(filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert isinstance(data, dict)

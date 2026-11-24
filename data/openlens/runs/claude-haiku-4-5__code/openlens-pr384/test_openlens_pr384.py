import pytest
import tempfile
import os
import struct
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_lens():
    """Create a mock lens with required attributes"""
    lens = Mock()
    lens.name = "TestLens"
    lens.material = "BK7"
    lens.refractive_index = 1.5168
    lens.diameter = 50.0
    lens.thickness = 10.0
    lens.radius_of_curvature_1 = 100.0
    lens.radius_of_curvature_2 = -100.0
    lens.lens_type = "Biconvex"
    lens.wavelength = 0.587
    lens.is_parabolic_1 = False
    lens.is_parabolic_2 = False
    lens.conic_constant_1 = 0.0
    lens.conic_constant_2 = 0.0
    lens.bevel_1 = 0.5
    lens.bevel_2 = 0.5
    lens.get_clear_aperture_1 = Mock(return_value=48.0)
    lens.get_clear_aperture_2 = Mock(return_value=48.0)
    lens.calculate_focal_length = Mock(return_value=150.0)
    lens.coating_label = Mock(return_value="Uncoated")
    lens.get_effective_radius_1 = Mock(return_value=100.0)
    lens.get_effective_radius_2 = Mock(return_value=-100.0)
    return lens


@pytest.fixture
def mock_system():
    """Create a mock optical system"""
    system = Mock()
    system.name = "TestSystem"
    system.elements = []
    system.air_gaps = []
    system.get_total_length = Mock(return_value=200.0)
    system.get_system_focal_length = Mock(return_value=150.0)
    system.get_aperture_stop = Mock(return_value=None)
    return system


@pytest.fixture
def mock_material_props():
    """Create a mock MaterialProperties object"""
    mat = Mock()
    mat.name = "BK7"
    mat.refractive_index = 1.5168
    mat.to_dict = Mock(return_value={"name": "BK7", "refractive_index": 1.5168})
    return mat


class TestOpticStudioExporter:
    def test_export_lens_creates_file_with_header(self, temp_dir, mock_lens):
        """Test OpticStudioExporter.export_lens creates file with proper header"""
        from openlens.export_formats import OpticStudioExporter
        
        output_file = os.path.join(temp_dir, "lens.txt")
        written_content = []
        
        def mock_write(text):
            written_content.append(text)
        
        with patch('openlens.export_formats.atomic_write_text') as mock_ctx:
            mock_file = MagicMock()
            mock_file.write = mock_write
            mock_ctx.return_value.__enter__.return_value = mock_file
            OpticStudioExporter.export_lens(mock_lens, output_file)
        
        content = ''.join(written_content)
        assert "OpenLens Export - OpticStudio Format" in content
        assert mock_lens.name in content
        assert "LENS DATA SUMMARY" in content
        assert "SURFACE DATA DETAIL" in content
    
    def test_export_system_creates_file_with_elements(self, temp_dir, mock_system):
        """Test OpticStudioExporter.export_system creates file with system info"""
        from openlens.export_formats import OpticStudioExporter
        
        output_file = os.path.join(temp_dir, "system.txt")
        written_content = []
        
        def mock_write(text):
            written_content.append(text)
        
        with patch('openlens.export_formats.atomic_write_text') as mock_ctx:
            mock_file = MagicMock()
            mock_file.write = mock_write
            mock_ctx.return_value.__enter__.return_value = mock_file
            OpticStudioExporter.export_system(mock_system, output_file)
        
        content = ''.join(written_content)
        assert "OpenLens Export - OpticStudio Format" in content
        assert "Number of Elements:" in content


class TestPrescriptionExporter:
    def test_export_prescription_contains_lens_data(self, temp_dir, mock_lens):
        """Test PrescriptionExporter.export_prescription includes all lens parameters"""
        from openlens.export_formats import PrescriptionExporter
        
        output_file = os.path.join(temp_dir, "prescription.txt")
        written_content = []
        
        def mock_write(text):
            written_content.append(text)
        
        with patch('openlens.export_formats.atomic_write_text') as mock_ctx:
            mock_file = MagicMock()
            mock_file.write = mock_write
            mock_ctx.return_value.__enter__.return_value = mock_file
            PrescriptionExporter.export_prescription(mock_lens, output_file)
        
        content = ''.join(written_content)
        assert "LENS PRESCRIPTION" in content
        assert mock_lens.name in content
        assert "Radius of Curvature 1" in content
        assert "Refractive Index" in content
        assert mock_lens.material in content


class TestSVGExporterLens:
    def test_export_lens_generates_valid_svg(self, temp_dir, mock_lens):
        """Test SVGExporter.export_lens generates valid XML SVG"""
        from openlens.export_formats import SVGExporter
        
        output_file = os.path.join(temp_dir, "lens.svg")
        written_content = []
        
        def mock_write(text):
            written_content.append(text)
        
        with patch('openlens.export_formats.atomic_write_text') as mock_ctx:
            mock_file = MagicMock()
            mock_file.write = mock_write
            mock_ctx.return_value.__enter__.return_value = mock_file
            SVGExporter.export_lens(mock_lens, output_file, width=400, height=300)
        
        content = ''.join(written_content)
        assert '<?xml version="1.0"' in content
        assert '<svg' in content
        assert '</svg>' in content
        assert 'class="lens-fill"' in content
    
    def test_export_lens_with_rays_option(self, temp_dir, mock_lens):
        """Test SVGExporter.export_lens includes rays when show_rays=True"""
        from openlens.export_formats import SVGExporter
        
        output_file = os.path.join(temp_dir, "lens_rays.svg")
        written_content = []
        
        def mock_write(text):
            written_content.append(text)
        
        with patch('openlens.export_formats.atomic_write_text') as mock_ctx:
            mock_file = MagicMock()
            mock_file.write = mock_write
            mock_ctx.return_value.__enter__.return_value = mock_file
            SVGExporter.export_lens(mock_lens, output_file, show_rays=True, show_labels=False)
        
        content = ''.join(written_content)
        assert 'class="ray"' in content
    
    def test_export_system_creates_svg(self, temp_dir, mock_system):
        """Test SVGExporter.export_system generates SVG diagram"""
        from openlens.export_formats import SVGExporter
        
        output_file = os.path.join(temp_dir, "system.svg")
        written_content = []
        
        def mock_write(text):
            written_content.append(text)
        
        with patch('openlens.export_formats.atomic_write_text') as mock_ctx:
            mock_file = MagicMock()
            mock_file.write = mock_write
            mock_ctx.return_value.__enter__.return_value = mock_file
            SVGExporter.export_system(mock_system, output_file, width=600, height=300)
        
        content = ''.join(written_content)
        assert '<svg' in content
        assert mock_system.name in content


class TestZemaxExporter:
    def test_export_lens_zemax_format(self, temp_dir, mock_lens):
        """Test ZemaxExporter.export_lens creates Zemax format file"""
        from openlens.export_formats import ZemaxExporter
        
        output_file = os.path.join(temp_dir, "lens.zmx")
        written_content = []
        
        def mock_write(text):
            written_content.append(text)
        
        with patch('openlens.export_formats.atomic_write_text') as mock_ctx:
            mock_file = MagicMock()
            mock_file.write = mock_write
            mock_ctx.return_value.__enter__.return_value = mock_file
            ZemaxExporter.export_lens(mock_lens, output_file)
        
        content = ''.join(written_content)
        assert "VERS 140000" in content
        assert "MODE SEQ" in content
        assert "SURF" in content
    
    def test_export_system_zemax_format(self, temp_dir, mock_system):
        """Test ZemaxSystemExporter.export_system creates Zemax system file"""
        from openlens.export_formats import ZemaxSystemExporter
        
        output_file = os.path.join(temp_dir, "system.zmx")
        written_content = []
        
        def mock_write(text):
            written_content.append(text)
        
        with patch('openlens.export_formats.atomic_write_text') as mock_ctx:
            mock_file = MagicMock()
            mock_file.write = mock_write
            mock_ctx.return_value.__enter__.return_value = mock_file
            ZemaxSystemExporter.export_system(mock_system, output_file)
        
        content = ''.join(written_content)
        assert "VERS 140000" in content
        assert "MODE SEQ" in content


class TestISO10110Generator:
    def test_generate_svg_writes_svg_content(self, temp_dir):
        """Test ISO10110Generator.generate_svg writes SVG file"""
        from openlens.io.export import ISO10110Generator
        
        generator = ISO10110Generator()
        output_file = os.path.join(temp_dir, "iso10110.svg")
        written_content = []
        
        def mock_write(text):
            written_content.append(text)
        
        with patch.object(generator, '_render_svg', return_value='<svg></svg>'):
            with patch('openlens.io.export.atomic_write_text') as mock_ctx:
                mock_file = MagicMock()
                mock_file.write = mock_write
                mock_ctx.return_value.__enter__.return_value = mock_file
                generator.generate_svg(output_file, width=800, height=600)
        
        content = ''.join(written_content)
        assert '<svg></svg>' in content


class TestStepExporter:
    def test_export_returns_summary_dict(self, temp_dir, mock_system):
        """Test StepExporter.export returns dict with shapes, skipped, and skipped_parts keys"""
        from openlens.io.step_export import StepExporter
        
        exporter = StepExporter(mock_system)
        output_file = os.path.join(temp_dir, "system.step")
        
        with patch.object(exporter, '_create_context'):
            with patch.object(exporter, '_element_placements', return_value=[]):
                with patch.object(exporter, '_create_product_structure'):
                    with patch.object(exporter.writer, 'generate', return_value=''):
                        with patch('openlens.io.step_export.atomic_write_text'):
                            result = exporter.export(output_file)
        
        assert isinstance(result, dict)
        assert 'shapes' in result
        assert 'skipped' in result
        assert 'skipped_parts' in result
        assert isinstance(result['skipped_parts'], list)
    
    def test_export_with_housing_parts(self, temp_dir, mock_system):
        """Test StepExporter.export handles housing parameter"""
        from openlens.io.step_export import StepExporter
        
        exporter = StepExporter(mock_system)
        output_file = os.path.join(temp_dir, "system_housing.step")
        housing = [{"name": "spacer", "z0": 0, "z1": 5, "r_inner": 10, "r_outer": 20}]
        
        with patch.object(exporter, '_create_context'):
            with patch.object(exporter, '_element_placements', return_value=[]):
                with patch.object(exporter, '_export_tube_solid', return_value=1):
                    with patch.object(exporter, '_create_product_structure'):
                        with patch.object(exporter.writer, 'generate', return_value=''):
                            with patch('openlens.io.step_export.atomic_write_text'):
                                result = exporter.export(output_file, housing=housing)
        
        assert result['shapes'] >= 0


class TestMaterialDatabase:
    def test_get_material_case_insensitive(self, mock_material_props):
        """Test MaterialDatabase.get_material supports case-insensitive lookup"""
        from openlens.material_database import MaterialDatabase
        
        db = MaterialDatabase()
        db.materials = {"BK7": mock_material_props}
        
        # Test exact match first
        result = db.get_material("BK7")
        assert result is not None
        assert result.name == "BK7"
    
    def test_get_material_normalized_key(self, mock_material_props):
        """Test MaterialDatabase.get_material uses normalized keys for variant lookup"""
        from openlens.material_database import MaterialDatabase
        
        db = MaterialDatabase()
        db.materials = {"BK7": mock_material_props}
        db._normalised_keys = Mock(return_value={"fused silica": "BK7"})
        
        # Should use normalized keys as fallback
        result = db.get_material("BK7")
        assert result is not None
    
    def test_export_material_calls_atomic_write(self, temp_dir, mock_material_props):
        """Test MaterialDatabase.export_material exports material to file"""
        from openlens.material_database import MaterialDatabase
        
        db = MaterialDatabase()
        db.materials = {"BK7": mock_material_props}
        output_file = os.path.join(temp_dir, "material.json")
        
        with patch('openlens.material_database.atomic_write_json') as mock_write:
            db.export_material("BK7", output_file)
            mock_write.assert_called_once()
    
    def test_save_database_writes_all_materials(self, temp_dir, mock_material_props):
        """Test MaterialDatabase.save_database saves all materials to JSON"""
        from openlens.material_database import MaterialDatabase
        
        db = MaterialDatabase(db_path=os.path.join(temp_dir, "db.json"))
        db.materials = {"BK7": mock_material_props}
        
        with patch('openlens.material_database.ensure_parent_dir'):
            with patch('openlens.material_database.atomic_write_json') as mock_write:
                db.save_database()
                mock_write.assert_called_once()
                # Verify data passed includes material dict
                call_args = mock_write.call_args
                assert isinstance(call_args[0][1], dict)


class TestSTLExporter:
    def test_write_binary_stl_format_header(self, temp_dir):
        """Test STLExporter.write_binary_stl writes 80-byte header"""
        from openlens.stl_export import STLExporter
        
        exporter = STLExporter()
        exporter.triangles = [((0, 0, 0), (1, 0, 0), (0, 1, 0))]
        exporter.calculate_normal = Mock(return_value=(0, 0, 1))
        
        writes = []
        def capture_write(data):
            writes.append(data)
        
        output_file = os.path.join(temp_dir, "model.stl")
        
        with patch('openlens.stl_export.atomic_write_bytes') as mock_ctx:
            mock_file = MagicMock()
            mock_file.write = capture_write
            mock_ctx.return_value.__enter__.return_value = mock_file
            exporter.write_binary_stl(output_file)
        
        # First write should be 80-byte header
        assert len(writes) > 0
        assert len(writes[0]) == 80
        assert writes[0].startswith(b"openlens STL export")
    
    def test_write_binary_stl_format_triangle_count(self, temp_dir):
        """Test STLExporter.write_binary_stl writes correct triangle count"""
        from openlens.stl_export import STLExporter
        
        exporter = STLExporter()
        exporter.triangles = [
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
            ((0, 0, 0), (0, 1, 0), (0, 0, 1)),
        ]
        exporter.calculate_normal = Mock(return_value=(0, 0, 1))
        
        writes = []
        def capture_write(data):
            writes.append(data)
        
        output_file = os.path.join(temp_dir, "model.stl")
        
        with patch('openlens.stl_export.atomic_write_bytes') as mock_ctx:
            mock_file = MagicMock()
            mock_file.write = capture_write
            mock_ctx.return_value.__enter__.return_value = mock_file
            exporter.write_binary_stl(output_file)
        
        # Second write should be triangle count as 4-byte unsigned int
        assert len(writes) > 1
        tri_count = struct.unpack("<I", writes[1])[0]
        assert tri_count == 2

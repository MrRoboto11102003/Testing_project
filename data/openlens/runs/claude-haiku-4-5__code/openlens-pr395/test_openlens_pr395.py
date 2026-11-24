import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from xml.etree import ElementTree as ET

# Import the module under test
try:
    from openlens.export_formats import SVGExporter
except ImportError:
    pytest.skip("openlens not installed", allow_module_level=True)


class TestSVGExporterExportLens:
    """Tests for SVGExporter.export_lens static method"""
    
    @pytest.fixture
    def temp_svg_file(self):
        """Provide a temporary SVG file path"""
        fd, path = tempfile.mkstemp(suffix='.svg')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)
    
    @pytest.fixture
    def mock_lens_basic(self):
        """Mock a basic lens with standard properties"""
        lens = Mock()
        lens.name = "Test Lens"
        lens.thickness = 5.0
        lens.diameter = 20.0
        lens.radius_of_curvature_1 = 100.0
        lens.radius_of_curvature_2 = -100.0
        lens.material = "BK7"
        lens.refractive_index = 1.517
        lens.calculate_focal_length.return_value = 50.0
        return lens
    
    @patch('openlens.export_formats.atomic_write_text')
    @patch('openlens.export_formats.escape')
    def test_export_lens_writes_svg_header(self, mock_escape, mock_atomic_write, mock_lens_basic):
        """Test that export_lens writes valid SVG header"""
        mock_escape.side_effect = lambda x: str(x)
        mock_file = MagicMock()
        mock_atomic_write.return_value.__enter__.return_value = mock_file
        
        SVGExporter.export_lens(mock_lens_basic, "output.svg")
        
        # Verify file was opened for writing
        mock_atomic_write.assert_called_once_with("output.svg")
        # Verify SVG header was written
        written_calls = [str(call[0][0]) for call in mock_file.write.call_args_list]
        assert any('<?xml' in call for call in written_calls)
        assert any('<svg' in call for call in written_calls)
    
    @patch('openlens.export_formats.atomic_write_text')
    @patch('openlens.export_formats.escape')
    def test_export_lens_includes_lens_name(self, mock_escape, mock_atomic_write, mock_lens_basic):
        """Test that lens name is included in SVG via escape function"""
        mock_escape.side_effect = lambda x: str(x)
        mock_file = MagicMock()
        mock_atomic_write.return_value.__enter__.return_value = mock_file
        
        SVGExporter.export_lens(mock_lens_basic, "output.svg")
        
        # Verify lens name was passed to escape (for XML safety)
        mock_escape.assert_any_call(mock_lens_basic.name)
    
    @patch('openlens.export_formats.atomic_write_text')
    @patch('openlens.export_formats.escape')
    def test_export_lens_custom_dimensions(self, mock_escape, mock_atomic_write, mock_lens_basic):
        """Test that custom width and height parameters are used in SVG"""
        mock_escape.side_effect = lambda x: str(x)
        mock_file = MagicMock()
        mock_atomic_write.return_value.__enter__.return_value = mock_file
        
        SVGExporter.export_lens(mock_lens_basic, "output.svg", width=800, height=600)
        
        written_text = ''.join(str(call[0][0]) for call in mock_file.write.call_args_list)
        assert 'width="800"' in written_text
        assert 'height="600"' in written_text
    
    @patch('openlens.export_formats.atomic_write_text')
    @patch('openlens.export_formats.escape')
    def test_export_lens_show_rays_true(self, mock_escape, mock_atomic_write, mock_lens_basic):
        """Test that ray paths are included when show_rays=True"""
        mock_escape.side_effect = lambda x: str(x)
        mock_file = MagicMock()
        mock_atomic_write.return_value.__enter__.return_value = mock_file
        
        SVGExporter.export_lens(mock_lens_basic, "output.svg", show_rays=True)
        
        written_text = ''.join(str(call[0][0]) for call in mock_file.write.call_args_list)
        assert 'class="ray"' in written_text
        assert '<line' in written_text and 'x1=' in written_text
    
    @patch('openlens.export_formats.atomic_write_text')
    @patch('openlens.export_formats.escape')
    def test_export_lens_show_rays_false(self, mock_escape, mock_atomic_write, mock_lens_basic):
        """Test that ray paths are excluded when show_rays=False"""
        mock_escape.side_effect = lambda x: str(x)
        mock_file = MagicMock()
        mock_atomic_write.return_value.__enter__.return_value = mock_file
        
        SVGExporter.export_lens(mock_lens_basic, "output.svg", show_rays=False)
        
        written_text = ''.join(str(call[0][0]) for call in mock_file.write.call_args_list)
        assert 'class="ray"' not in written_text
    
    @patch('openlens.export_formats.atomic_write_text')
    @patch('openlens.export_formats.escape')
    def test_export_lens_show_labels_true(self, mock_escape, mock_atomic_write, mock_lens_basic):
        """Test that dimension labels are included when show_labels=True"""
        mock_escape.side_effect = lambda x: str(x)
        mock_file = MagicMock()
        mock_atomic_write.return_value.__enter__.return_value = mock_file
        
        SVGExporter.export_lens(mock_lens_basic, "output.svg", show_labels=True)
        
        written_text = ''.join(str(call[0][0]) for call in mock_file.write.call_args_list)
        # Should have multiple text elements (title + dimension labels)
        text_count = written_text.count('<text')
        assert text_count >= 3  # title, diameter, thickness, material
    
    @patch('openlens.export_formats.atomic_write_text')
    @patch('openlens.export_formats.escape')
    def test_export_lens_show_labels_false(self, mock_escape, mock_atomic_write, mock_lens_basic):
        """Test that labels are excluded when show_labels=False"""
        mock_escape.side_effect = lambda x: str(x)
        mock_file = MagicMock()
        mock_atomic_write.return_value.__enter__.return_value = mock_file
        
        SVGExporter.export_lens(mock_lens_basic, "output.svg", show_labels=False)
        
        written_text = ''.join(str(call[0][0]) for call in mock_file.write.call_args_list)
        # Only title text should be present
        text_count = written_text.count('<text')
        assert text_count == 1
    
    @patch('openlens.export_formats.atomic_write_text')
    @patch('openlens.export_formats.escape')
    def test_export_lens_flat_surfaces(self, mock_escape, mock_atomic_write):
        """Test lens with flat surfaces (infinite radius of curvature)"""
        mock_escape.side_effect = lambda x: str(x)
        lens = Mock()
        lens.name = "Flat Lens"
        lens.thickness = 3.0
        lens.diameter = 15.0
        lens.radius_of_curvature_1 = float('inf')
        lens.radius_of_curvature_2 = float('inf')
        lens.material = "Glass"
        lens.refractive_index = 1.5
        lens.calculate_focal_length.return_value = None
        
        mock_file = MagicMock()
        mock_atomic_write.return_value.__enter__.return_value = mock_file
        
        # Should handle flat surfaces without error
        SVGExporter.export_lens(lens, "output.svg")
        assert mock_file.write.called
    
    @patch('openlens.export_formats.atomic_write_text')
    @patch('openlens.export_formats.escape')
    def test_export_lens_includes_material_label(self, mock_escape, mock_atomic_write):
        """Test that material and refractive index labels are included"""
        mock_escape.side_effect = lambda x: str(x)
        lens = Mock()
        lens.name = "Material Test"
        lens.thickness = 5.0
        lens.diameter = 20.0
        lens.radius_of_curvature_1 = 100.0
        lens.radius_of_curvature_2 = -100.0
        lens.material = "SF2"
        lens.refractive_index = 1.647
        lens.calculate_focal_length.return_value = 40.0
        
        mock_file = MagicMock()
        mock_atomic_write.return_value.__enter__.return_value = mock_file
        
        SVGExporter.export_lens(lens, "output.svg", show_labels=True)
        
        # Material should be passed to escape for XML safety
        mock_escape.assert_any_call(lens.material)
    
    @patch('openlens.export_formats.atomic_write_text')
    @patch('openlens.export_formats.escape')
    def test_export_lens_handles_none_focal_length(self, mock_escape, mock_atomic_write):
        """Test that None focal length is handled gracefully"""
        mock_escape.side_effect = lambda x: str(x)
        lens = Mock()
        lens.name = "No Focal Length"
        lens.thickness = 5.0
        lens.diameter = 20.0
        lens.radius_of_curvature_1 = float('inf')
        lens.radius_of_curvature_2 = float('inf')
        lens.material = "Glass"
        lens.refractive_index = 1.5
        lens.calculate_focal_length.return_value = None
        
        mock_file = MagicMock()
        mock_atomic_write.return_value.__enter__.return_value = mock_file
        
        SVGExporter.export_lens(lens, "output.svg", show_labels=True)
        
        written_text = ''.join(str(call[0][0]) for call in mock_file.write.call_args_list)
        # Should complete without error
        assert '<svg' in written_text


class TestSVGExporterExportSystem:
    """Tests for SVGExporter.export_system static method"""
    
    @pytest.fixture
    def mock_lens(self):
        """Mock a lens for system elements"""
        lens = Mock()
        lens.thickness = 5.0
        lens.diameter = 20.0
        return lens
    
    @pytest.fixture
    def mock_element(self, mock_lens):
        """Mock an optical element"""
        elem = Mock()
        elem.lens = mock_lens
        elem.distance = 0.0
        return elem
    
    @pytest.fixture
    def mock_system_single(self, mock_element):
        """Mock an optical system with one element"""
        system = Mock()
        system.name = "Single Element System"
        system.elements = [mock_element]
        system.get_total_length.return_value = 100.0
        return system
    
    @patch('openlens.export_formats.atomic_write_text')
    @patch('openlens.export_formats.escape')
    def test_export_system_writes_svg_header(self, mock_escape, mock_atomic_write, mock_system_single):
        """Test that export_system writes valid SVG header"""
        mock_escape.side_effect = lambda x: str(x)
        mock_file = MagicMock()
        mock_atomic_write.return_value.__enter__.return_value = mock_file
        
        SVGExporter.export_system(mock_system_single, "output.svg")
        
        mock_atomic_write.assert_called_once_with("output.svg")
        written_calls = [str(call[0][0]) for call in mock_file.write.call_args_list]
        assert any('<?xml' in call for call in written_calls)
        assert any('<svg' in call for call in written_calls)
    
    @patch('openlens.export_formats.atomic_write_text')
    @patch('openlens.export_formats.escape')
    def test_export_system_includes_system_name(self, mock_escape, mock_atomic_write, mock_system_single):
        """Test that system name is included in SVG"""
        mock_escape.side_effect = lambda x: str(x)
        mock_file = MagicMock()
        mock_atomic_write.return_value.__enter__.return_value = mock_file
        
        SVGExporter.export_system(mock_system_single, "output.svg")
        
        mock_escape.assert_any_call(mock_system_single.name)
    
    @patch('openlens.export_formats.atomic_write_text')
    @patch('openlens.export_formats.escape')
    def test_export_system_custom_dimensions(self, mock_escape, mock_atomic_write, mock_system_single):
        """Test that custom width and height are used in SVG"""
        mock_escape.side_effect = lambda x: str(x)
        mock_file = MagicMock()
        mock_atomic_write.return_value.__enter__.return_value = mock_file
        
        SVGExporter.export_system(mock_system_single, "output.svg", width=1000, height=400)
        
        written_text = ''.join(str(call[0][0]) for call in mock_file.write.call_args_list)
        assert 'width="1000"' in written_text
        assert 'height="400"' in written_text
    
    @patch('openlens.export_formats.atomic_write_text')
    @patch('openlens.export_formats.escape')
    def test_export_system_uses_default_dimensions(self, mock_escape, mock_atomic_write, mock_system_single):
        """Test that default dimensions (600x300) are used when not specified"""
        mock_escape.side_effect = lambda x: str(x)
        mock_file = MagicMock()
        mock_atomic_write.return_value.__enter__.return_value = mock_file
        
        SVGExporter.export_system(mock_system_single, "output.svg")
        
        written_text = ''.join(str(call[0][0]) for call in mock_file.write.call_args_list)
        assert 'width="600"' in written_text
        assert 'height="300"' in written_text
    
    @patch('openlens.export_formats.atomic_write_text')
    @patch('openlens.export_formats.escape')
    @patch('openlens.export_formats._gap_before')
    def test_export_system_draws_lens_elements(self, mock_gap_before, mock_escape, mock_atomic_write, mock_system_single):
        """Test that lens elements are drawn as rectangles"""
        mock_escape.side_effect = lambda x: str(x)
        mock_gap_before.return_value = 5.0
        mock_file = MagicMock()
        mock_atomic_write.return_value.__enter__.return_value = mock_file
        
        SVGExporter.export_system(mock_system_single, "output.svg")
        
        written_text = ''.join(str(call[0][0]) for call in mock_file.write.call_args_list)
        assert 'class="lens-fill"' in written_text
        assert 'class="lens-stroke"' in written_text
        assert '<rect' in written_text
    
    @patch('openlens.export_formats.atomic_write_text')
    @patch('openlens.export_formats.escape')
    @patch('openlens.export_formats._gap_before')
    def test_export_system_multiple_elements(self, mock_gap_before, mock_escape, mock_atomic_write, mock_lens):
        """Test system with multiple lens elements draws all of them"""
        mock_escape.side_effect = lambda x: str(x)
        mock_gap_before.return_value = 5.0
        
        system = Mock()
        system.name = "Multi-Element System"
        elements = [Mock(lens=mock_lens, distance=10.0) for _ in range(3)]
        system.elements = elements
        system.get_total_length.return_value = 150.0
        
        mock_file = MagicMock()
        mock_atomic_write.return_value.__enter__.return_value = mock_file
        
        SVGExporter.export_system(system, "output.svg")
        
        written_text = ''.join(str(call[0][0]) for call in mock_file.write.call_args_list)
        # At least 2 rects per lens (fill + stroke)
        rect_count = written_text.count('<rect')
        assert rect_count >= 6
    
    @patch('openlens.export_formats.atomic_write_text')
    @patch('openlens.export_formats.escape')
    def test_export_system_includes_optical_axis(self, mock_escape, mock_atomic_write, mock_system_single):
        """Test that optical axis is included in SVG"""
        mock_escape.side_effect = lambda x: str(x)
        mock_file = MagicMock()
        mock_atomic_write.return_value.__enter__.return_value = mock_file
        
        SVGExporter.export_system(mock_system_single, "output.svg")
        
        written_text = ''.join(str(call[0][0]) for call in mock_file.write.call_args_list)
        assert 'class="axis"' in written_text
        assert '<line' in written_text
    
    @patch('openlens.export_formats.atomic_write_text')
    @patch('openlens.export_formats.escape')
    def test_export_system_empty_system(self, mock_escape, mock_atomic_write):
        """Test system with no elements"""
        mock_escape.side_effect = lambda x: str(x)
        
        system = Mock()
        system.name = "Empty System"
        system.elements = []
        system.get_total_length.return_value = 100.0
        
        mock_file = MagicMock()
        mock_atomic_write.return_value.__enter__.return_value = mock_file
        
        # Should complete without error
        SVGExporter.export_system(system, "output.svg")
        assert mock_file.write.called

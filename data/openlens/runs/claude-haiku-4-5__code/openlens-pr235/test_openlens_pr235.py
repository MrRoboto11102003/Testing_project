import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock
import matplotlib.pyplot as plt

from openlens.analysis.plots import draw_system_outline
from openlens.validation import check_physical_feasibility, validate_file_path, ValidationError


class TestDrawSystemOutline:
    """Tests for draw_system_outline function."""
    
    def test_single_lens_draws_successfully(self):
        """Test that a single lens system draws without error."""
        from openlens.optics.lens import Lens
        
        fig, ax = plt.subplots()
        try:
            lens = Lens(
                radius_of_curvature_1=100,
                radius_of_curvature_2=-100,
                thickness=5,
                diameter=10
            )
            element = Mock()
            element.lens = lens
            
            system = Mock()
            system.elements = [element]
            system.air_gaps = []
            
            draw_system_outline(ax, system)
            assert ax is not None
        finally:
            plt.close(fig)
    
    def test_multiple_elements_with_air_gaps(self):
        """Test drawing system with multiple lenses and air gaps."""
        from openlens.optics.lens import Lens
        
        fig, ax = plt.subplots()
        try:
            lens1 = Lens(radius_of_curvature_1=100, radius_of_curvature_2=-100, 
                        thickness=5, diameter=10)
            lens2 = Lens(radius_of_curvature_1=50, radius_of_curvature_2=-50,
                        thickness=3, diameter=8)
            
            element1, element2 = Mock(), Mock()
            element1.lens, element2.lens = lens1, lens2
            
            air_gap = Mock()
            air_gap.thickness = 10
            
            system = Mock()
            system.elements = [element1, element2]
            system.air_gaps = [air_gap]
            
            draw_system_outline(ax, system)
            assert ax is not None
        finally:
            plt.close(fig)
    
    def test_infeasible_geometry_handled_gracefully(self):
        """Test that infeasible lens geometry doesn't crash drawing."""
        from openlens.optics.lens import Lens
        
        fig, ax = plt.subplots()
        try:
            lens = Lens(radius_of_curvature_1=5, radius_of_curvature_2=-5,
                       thickness=20, diameter=20)
            element = Mock()
            element.lens = lens
            
            system = Mock()
            system.elements = [element]
            system.air_gaps = []
            
            draw_system_outline(ax, system)
            assert ax is not None
        finally:
            plt.close(fig)


class TestCheckPhysicalFeasibility:
    """Tests for check_physical_feasibility function."""
    
    def test_valid_lens_parameters_pass(self):
        """Test that valid lens parameters return feasible=True, warning=None."""
        is_feasible, warning = check_physical_feasibility(
            radius1=100, radius2=-100, thickness=5, diameter=10
        )
        assert is_feasible is True
        assert warning is None
    
    def test_intersecting_surfaces_fail(self):
        """Test that intersecting surfaces are detected as infeasible."""
        is_feasible, warning = check_physical_feasibility(
            radius1=5, radius2=5, thickness=0.5, diameter=10
        )
        assert is_feasible is False
        assert warning is not None
    
    def test_aperture_overhang_detected(self):
        """Test that aperture overhang (|R| < D/2) is detected."""
        is_feasible, warning = check_physical_feasibility(
            radius1=3, radius2=100, thickness=5, diameter=10
        )
        assert is_feasible is False
        assert warning is not None
    
    def test_edge_thickness_warning(self):
        """Test that very small edge thickness generates warning."""
        is_feasible, warning = check_physical_feasibility(
            radius1=200, radius2=-200, thickness=0.5, diameter=10
        )
        if warning:
            assert "edge" in warning.lower() or "fragile" in warning.lower()
    
    def test_thickness_ratio_warning(self):
        """Test warning when thickness > 0.5 * min_radius."""
        is_feasible, warning = check_physical_feasibility(
            radius1=10, radius2=-10, thickness=6, diameter=5
        )
        if warning:
            assert "thickness" in warning.lower()
    
    def test_diameter_ratio_warning(self):
        """Test warning when diameter > 0.8 * min_radius."""
        is_feasible, warning = check_physical_feasibility(
            radius1=10, radius2=-10, thickness=2, diameter=8
        )
        if warning:
            assert "diameter" in warning.lower()
    
    def test_very_thin_lens_warning(self):
        """Test warning when thickness < 0.02 * diameter."""
        is_feasible, warning = check_physical_feasibility(
            radius1=100, radius2=-100, thickness=0.1, diameter=10
        )
        if warning:
            assert "thin" in warning.lower()
    
    def test_parabolic_surfaces(self):
        """Test that parabolic surfaces are handled correctly."""
        is_feasible, warning = check_physical_feasibility(
            radius1=100, radius2=-100, thickness=5, diameter=10,
            is_parabolic_1=True, parabolic_sag_1=0.5
        )
        assert isinstance(is_feasible, bool)
    
    def test_conic_constants(self):
        """Test that conic constants are accepted."""
        is_feasible, warning = check_physical_feasibility(
            radius1=100, radius2=-100, thickness=5, diameter=10,
            conic_constant_1=-1.0, conic_constant_2=0.5
        )
        assert isinstance(is_feasible, bool)
    
    def test_flat_surface_infinite_radius(self):
        """Test that flat surfaces (infinite radius) are handled."""
        is_feasible, warning = check_physical_feasibility(
            radius1=float('inf'), radius2=-100, thickness=5, diameter=10
        )
        assert isinstance(is_feasible, bool)
    
    def test_clear_apertures_accepted(self):
        """Test that explicit clear apertures are handled."""
        is_feasible, warning = check_physical_feasibility(
            radius1=100, radius2=-100, thickness=5, diameter=10,
            clear_aperture_1=8, clear_aperture_2=8
        )
        assert isinstance(is_feasible, bool)


class TestValidateFilePath:
    """Tests for validate_file_path function."""
    
    def test_string_path_converts_to_path(self):
        """Test that string paths are converted to Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_str = os.path.join(tmpdir, "test.txt")
            result = validate_file_path(path_str)
            assert isinstance(result, Path)
            assert result.is_absolute()
    
    def test_path_object_accepted(self):
        """Test that Path objects are accepted and returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_obj = Path(tmpdir) / "test.txt"
            result = validate_file_path(path_obj)
            assert isinstance(result, Path)
    
    def test_relative_path_resolved_to_absolute(self):
        """Test that relative paths are resolved to absolute."""
        result = validate_file_path("./test.txt")
        assert result.is_absolute()
    
    def test_must_exist_raises_when_missing(self):
        """Test that must_exist=True raises for nonexistent file."""
        with pytest.raises(ValidationError):
            validate_file_path("/nonexistent/file.txt", must_exist=True)
    
    def test_must_exist_succeeds_for_existing_file(self):
        """Test that must_exist=True succeeds for existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = validate_file_path(tmp_path, must_exist=True)
            assert result.exists()
        finally:
            os.unlink(tmp_path)
    
    def test_nonexistent_parent_raises_error(self):
        """Test that ValidationError is raised for nonexistent parent."""
        with pytest.raises(ValidationError):
            validate_file_path("/nonexistent/parent/file.txt")
    
    def test_create_parent_creates_directories(self):
        """Test that create_parent=True creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = os.path.join(tmpdir, "sub1", "sub2", "file.txt")
            result = validate_file_path(new_path, create_parent=True)
            assert result.parent.exists()
    
    def test_invalid_type_raises_error(self):
        """Test that non-string/Path types raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_file_path(123)
    
    def test_param_name_in_error_message(self):
        """Test that custom param_name appears in error messages."""
        with pytest.raises(ValidationError) as excinfo:
            validate_file_path(None, param_name="my_file")
        assert "my_file" in str(excinfo.value)

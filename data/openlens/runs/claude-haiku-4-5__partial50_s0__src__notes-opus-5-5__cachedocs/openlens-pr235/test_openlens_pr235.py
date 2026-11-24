import pytest
from pathlib import Path
from unittest.mock import MagicMock

from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.analysis.plots import draw_system_outline
from openlens.validation import check_physical_feasibility, validate_file_path, ValidationError


class TestDrawSystemOutline:
    """Tests for openlens.analysis.plots.draw_system_outline"""
    
    def test_empty_system_no_calls(self):
        """Empty system should make no plot or add_patch calls"""
        ax = MagicMock()
        system = OpticalSystem()
        draw_system_outline(ax, system)
        assert ax.plot.call_count == 0
        assert ax.add_patch.call_count == 0
    
    def test_single_lens_produces_patch_and_lines(self):
        """Single lens produces 1 add_patch and 4 plot calls"""
        ax = MagicMock()
        system = OpticalSystem()
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                    thickness=5.0, diameter=25.0)
        system.add_lens(lens)
        draw_system_outline(ax, system)
        assert ax.add_patch.call_count == 1
        assert ax.plot.call_count == 4
    
    def test_two_lenses_correct_call_counts(self):
        """Two lenses produces 2 patches and 8 plot calls"""
        ax = MagicMock()
        system = OpticalSystem()
        lens1 = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                     thickness=5.0, diameter=25.0)
        lens2 = Lens(radius_of_curvature_1=60.0, radius_of_curvature_2=-60.0,
                     thickness=4.0, diameter=20.0)
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=10.0)
        draw_system_outline(ax, system)
        assert ax.add_patch.call_count == 2
        assert ax.plot.call_count == 8
    
    def test_fill_error_still_draws_outlines(self):
        """ValueError in polygon fill still allows outline lines to be drawn"""
        ax = MagicMock()
        ax.add_patch.side_effect = ValueError("Degenerate polygon")
        system = OpticalSystem()
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                    thickness=5.0, diameter=25.0)
        system.add_lens(lens)
        draw_system_outline(ax, system)
        # Should not raise and should still plot 4 lines
        assert ax.plot.call_count == 4
    
    def test_import_error_in_fill_still_draws_outlines(self):
        """ImportError for Polygon import still allows outlines"""
        ax = MagicMock()
        ax.add_patch.side_effect = ImportError("No module named matplotlib.patches")
        system = OpticalSystem()
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                    thickness=5.0, diameter=25.0)
        system.add_lens(lens)
        draw_system_outline(ax, system)
        assert ax.plot.call_count == 4


class TestCheckPhysicalFeasibility:
    """Tests for openlens.validation.check_physical_feasibility"""
    
    def test_valid_biconvex_lens(self):
        """Valid biconvex lens returns (True, None)"""
        feasible, msg = check_physical_feasibility(
            radius1=200.0, radius2=-200.0, thickness=10.0, diameter=25.0
        )
        assert feasible is True
        assert msg is None
    
    def test_aperture_overhang_hard_error(self):
        """Aperture overhang generates hard error"""
        feasible, msg = check_physical_feasibility(
            radius1=10.0, radius2=-10.0, thickness=5.0, diameter=30.0
        )
        assert feasible is False
        assert "Aperture overhangs" in msg
        assert "10.0mm" in msg
    
    def test_intersecting_surfaces_hard_error(self):
        """Surfaces intersecting generates hard error"""
        feasible, msg = check_physical_feasibility(
            radius1=50.0, radius2=-50.0, thickness=1.0, diameter=40.0
        )
        assert feasible is False
        assert "intersect" in msg.lower()
    
    def test_thickness_warning_soft_error(self):
        """Thickness > 0.5*min_radius generates soft warning"""
        feasible, msg = check_physical_feasibility(
            radius1=float('inf'), radius2=-20.0, thickness=15.0, diameter=10.0
        )
        assert feasible is False
        assert "Thickness" in msg
        assert "very large" in msg
    
    def test_diameter_warning_soft_error(self):
        """Diameter > 0.8*min_radius generates soft warning"""
        feasible, msg = check_physical_feasibility(
            radius1=float('inf'), radius2=-100.0, thickness=5.0, diameter=85.0
        )
        assert feasible is False
        assert "Diameter" in msg
        assert "large compared" in msg
    
    def test_thin_lens_warning_soft_error(self):
        """Thickness < 0.02*diameter generates soft warning"""
        feasible, msg = check_physical_feasibility(
            radius1=float('inf'), radius2=-1000.0, thickness=0.5, diameter=50.0
        )
        assert feasible is False
        assert "thin" in msg.lower()
    
    def test_flat_flat_lens_valid(self):
        """Flat/flat (plano-plano) lens is valid"""
        feasible, msg = check_physical_feasibility(
            radius1=float('inf'), radius2=float('inf'), thickness=5.0, diameter=25.0
        )
        assert feasible is True
        assert msg is None
    
    def test_non_numeric_radius_raises_type_error(self):
        """Non-numeric radius raises TypeError"""
        with pytest.raises(TypeError):
            check_physical_feasibility(
                radius1=None, radius2=-50.0, thickness=5.0, diameter=25.0
            )


class TestValidateFilePath:
    """Tests for openlens.validation.validate_file_path"""
    
    def test_valid_new_file_path(self, tmp_path):
        """Valid new file path returns resolved Path"""
        file_path = tmp_path / "newfile.txt"
        result = validate_file_path(file_path)
        assert isinstance(result, Path)
        assert result.is_absolute()
        assert result == file_path.resolve()
        assert not file_path.exists()
    
    def test_valid_existing_file_with_must_exist(self, tmp_path):
        """Valid existing file with must_exist=True returns Path"""
        file_path = tmp_path / "existing.txt"
        file_path.write_text("test content")
        result = validate_file_path(file_path, must_exist=True)
        assert isinstance(result, Path)
        assert result == file_path.resolve()
    
    def test_string_path_resolved_to_absolute(self, tmp_path):
        """String path is resolved to absolute Path"""
        result = validate_file_path(str(tmp_path / "file.txt"))
        assert isinstance(result, Path)
        assert result.is_absolute()
    
    def test_invalid_type_raises_validation_error(self):
        """Non-string/Path type raises ValidationError"""
        with pytest.raises(ValidationError, match="must be a string or Path object"):
            validate_file_path(12345)
    
    def test_bytes_type_raises_validation_error(self):
        """bytes type raises ValidationError"""
        with pytest.raises(ValidationError, match="must be a string or Path object"):
            validate_file_path(b"path.txt")
    
    def test_missing_file_with_must_exist_raises(self, tmp_path):
        """Missing file with must_exist=True raises ValidationError"""
        with pytest.raises(ValidationError, match="does not exist"):
            validate_file_path(tmp_path / "nonexistent.txt", must_exist=True)
    
    def test_missing_parent_without_create_raises(self, tmp_path):
        """Missing parent directory without create_parent raises"""
        with pytest.raises(ValidationError, match="Parent directory does not exist"):
            validate_file_path(tmp_path / "missing" / "file.txt")
    
    def test_create_parent_true_creates_directories(self, tmp_path):
        """create_parent=True creates missing parent directories"""
        file_path = tmp_path / "sub1" / "sub2" / "file.txt"
        result = validate_file_path(file_path, create_parent=True)
        assert result.parent.exists()
        assert result.parent.is_dir()
    
    def test_custom_param_name_in_error_message(self, tmp_path):
        """Custom param_name appears in error messages"""
        with pytest.raises(ValidationError, match="my_output"):
            validate_file_path(tmp_path / "missing_dir" / "file.txt", param_name="my_output")
    
    def test_type_check_before_must_exist(self):
        """Type validation runs before must_exist check"""
        with pytest.raises(ValidationError, match="must be a string or Path object"):
            validate_file_path(None, must_exist=True)

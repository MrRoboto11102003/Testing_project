import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from openlens.analysis.plots import draw_system_outline
from openlens.validation import check_physical_feasibility, validate_file_path, ValidationError
from openlens.models import OpticalSystem, Lens


class TestDrawSystemOutline:
    """Tests for draw_system_outline(ax, system)."""

    def test_empty_system_no_calls(self):
        """Empty system should produce no plot or patch calls."""
        ax = MagicMock()
        system = OpticalSystem()

        draw_system_outline(ax, system)

        ax.plot.assert_not_called()
        ax.add_patch.assert_not_called()

    def test_single_lens_generates_patch_and_lines(self):
        """Single lens should generate 1 patch and 4 plot calls."""
        ax = MagicMock()
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=20.0
        )
        system.add_lens(lens)

        draw_system_outline(ax, system)

        assert ax.add_patch.call_count == 1
        assert ax.plot.call_count == 4

    def test_two_lenses_generates_patch_and_lines_per_lens(self):
        """Two lenses should generate 2 patches and 8 plot calls."""
        ax = MagicMock()
        system = OpticalSystem()
        lens1 = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=20.0
        )
        lens2 = Lens(
            radius_of_curvature_1=150.0,
            radius_of_curvature_2=-150.0,
            thickness=8.0,
            diameter=25.0
        )
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=3.0)

        draw_system_outline(ax, system)

        assert ax.add_patch.call_count == 2
        assert ax.plot.call_count == 8

    def test_polygon_error_caught_lines_still_drawn(self):
        """ValueError or TypeError in Polygon should be caught; lines still drawn."""
        ax = MagicMock()
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=20.0
        )
        system.add_lens(lens)
        ax.add_patch.side_effect = ValueError("Degenerate polygon")

        draw_system_outline(ax, system)

        assert ax.plot.call_count == 4

    def test_non_polygon_error_propagates(self):
        """RuntimeError from other source should propagate."""
        ax = MagicMock()
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=20.0
        )
        system.add_lens(lens)
        ax.plot.side_effect = RuntimeError("Plot error")

        with pytest.raises(RuntimeError):
            draw_system_outline(ax, system)


class TestCheckPhysicalFeasibility:
    """Tests for check_physical_feasibility(...)."""

    def test_standard_biconvex_feasible(self):
        """Standard biconvex lens should be feasible with no warnings."""
        is_feasible, msg = check_physical_feasibility(
            radius1=200.0, radius2=-200.0, thickness=10.0, diameter=25.0
        )

        assert is_feasible is True
        assert msg is None

    def test_flat_flat_lens_feasible(self):
        """Flat lens (infinite radii) should be feasible."""
        is_feasible, msg = check_physical_feasibility(
            radius1=float('inf'), radius2=float('inf'), thickness=5.0, diameter=25.0
        )

        assert is_feasible is True
        assert msg is None

    def test_thickness_warning_returned_as_false(self):
        """Soft warning should return (False, message)."""
        is_feasible, msg = check_physical_feasibility(
            radius1=float('inf'), radius2=-20.0, thickness=15.0, diameter=10.0
        )

        assert is_feasible is False
        assert msg is not None
        assert "Thickness" in msg

    def test_intersecting_surfaces_hard_error(self):
        """Edge <= 0 should produce hard error."""
        is_feasible, msg = check_physical_feasibility(
            radius1=50.0, radius2=-50.0, thickness=1.0, diameter=40.0
        )

        assert is_feasible is False
        assert msg is not None
        assert "Surfaces intersect" in msg

    def test_aperture_overhang_hard_error(self):
        """Aperture overhanging surface (|R| < D/2) should produce hard error."""
        is_feasible, msg = check_physical_feasibility(
            radius1=10.0, radius2=-10.0, thickness=5.0, diameter=30.0
        )

        assert is_feasible is False
        assert msg is not None
        assert "Aperture overhangs" in msg

    def test_non_numeric_radius_raises_type_error(self):
        """Non-numeric radius should raise TypeError."""
        with pytest.raises(TypeError):
            check_physical_feasibility(
                radius1=None, radius2=100.0, thickness=5.0, diameter=25.0
            )


class TestValidateFilePath:
    """Tests for validate_file_path(...)."""

    def test_string_path_returns_resolved_path(self, tmp_path):
        """String path should return resolved Path object."""
        result = validate_file_path(str(tmp_path / "test.txt"))

        assert isinstance(result, Path)
        assert result.is_absolute()

    def test_path_object_returns_resolved_path(self, tmp_path):
        """Path object should return resolved Path object."""
        path = tmp_path / "test.txt"
        result = validate_file_path(path)

        assert isinstance(result, Path)
        assert result == path.resolve()

    def test_non_path_type_raises_validation_error(self):
        """Non-string, non-Path input should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_file_path(123)

        assert "must be a string or Path object" in str(exc_info.value)

    def test_must_exist_nonexistent_raises(self, tmp_path):
        """must_exist=True with nonexistent path should raise."""
        with pytest.raises(ValidationError) as exc_info:
            validate_file_path(tmp_path / "nonexistent.txt", must_exist=True)

        assert "does not exist" in str(exc_info.value)

    def test_must_exist_existing_file_succeeds(self, tmp_path):
        """must_exist=True with existing file should succeed."""
        file = tmp_path / "test.txt"
        file.write_text("content")

        result = validate_file_path(file, must_exist=True)

        assert result.exists()

    def test_missing_parent_raises_without_create_flag(self, tmp_path):
        """Missing parent with create_parent=False should raise."""
        with pytest.raises(ValidationError) as exc_info:
            validate_file_path(tmp_path / "missing" / "file.txt")

        assert "Parent directory does not exist" in str(exc_info.value)

    def test_create_parent_creates_missing_directories(self, tmp_path):
        """create_parent=True should create missing parent chain."""
        path = tmp_path / "a" / "b" / "c" / "file.txt"

        result = validate_file_path(path, create_parent=True)

        assert result.parent.exists()
        assert result == path.resolve()

    def test_custom_param_name_in_error(self):
        """Custom param_name should appear in error messages."""
        with pytest.raises(ValidationError) as exc_info:
            validate_file_path(None, param_name="output")

        assert "output" in str(exc_info.value)

    def test_type_check_before_must_exist(self):
        """Type check should precede must_exist check."""
        with pytest.raises(ValidationError) as exc_info:
            validate_file_path(None, must_exist=True)

        assert "must be a string or Path object" in str(exc_info.value)

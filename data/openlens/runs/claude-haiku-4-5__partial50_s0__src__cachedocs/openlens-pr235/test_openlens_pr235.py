import pytest
import tempfile
import os
from pathlib import Path

# Import openlens modules
from openlens.validation import check_physical_feasibility, validate_file_path, ValidationError
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem, AirGap

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TestCheckPhysicalFeasibility:
    """Tests for check_physical_feasibility validation function."""

    def test_valid_biconvex_lens(self):
        """Valid biconvex lens geometry should return feasible=True."""
        is_feasible, warning = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        assert is_feasible is True
        assert warning is None

    def test_surfaces_intersect_infeasible(self):
        """Surfaces intersecting within clear aperture should be infeasible."""
        is_feasible, warning = check_physical_feasibility(
            radius1=10.0,
            radius2=-10.0,
            thickness=50.0,
            diameter=25.0,
        )
        assert is_feasible is False
        assert warning is not None
        assert "intersect" in warning.lower() or "edge thickness" in warning.lower()

    def test_aperture_overhang_infeasible(self):
        """Aperture overhanging surface (|R| < D/2) should fail feasibility check."""
        is_feasible, warning = check_physical_feasibility(
            radius1=5.0,
            radius2=-5.0,
            thickness=1.0,
            diameter=20.0,
        )
        assert is_feasible is False
        assert warning is not None
        assert "aperture" in warning.lower() or "overhang" in warning.lower()

    def test_very_thin_lens_generates_warning(self):
        """Very thin lens relative to diameter should generate warning."""
        is_feasible, warning = check_physical_feasibility(
            radius1=200.0,
            radius2=-200.0,
            thickness=0.05,  # 0.05/50 = 0.001 << 0.02
            diameter=50.0,
        )
        assert is_feasible is False
        assert warning is not None
        assert "thin" in warning.lower()

    def test_very_thick_lens_warning(self):
        """Very thick lens relative to radius should generate warning."""
        is_feasible, warning = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=30.0,  # 30/50 > 0.5
            diameter=25.0,
        )
        assert is_feasible is False
        assert warning is not None

    def test_large_diameter_vs_radius_warning(self):
        """Large diameter relative to radius should generate warning."""
        is_feasible, warning = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=5.0,
            diameter=42.0,  # 42/50 > 0.8
        )
        assert is_feasible is False
        assert warning is not None

    def test_parabolic_and_conic_surfaces(self):
        """Parabolic and conic surfaces should be validated correctly."""
        is_feasible, warning = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=5.0,
            diameter=25.0,
            is_parabolic_1=True,
            parabolic_sag_1=1.0,
            conic_constant_2=-0.5,
        )
        assert isinstance(is_feasible, bool)
        assert warning is None or isinstance(warning, str)

    def test_clear_apertures_explicit(self):
        """Clear apertures can be explicitly set and validated."""
        is_feasible, warning = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=5.0,
            diameter=25.0,
            clear_aperture_1=15.0,
            clear_aperture_2=15.0,
        )
        assert isinstance(is_feasible, bool)


class TestValidateFilePath:
    """Tests for validate_file_path function."""

    def test_valid_existing_file_path(self):
        """Valid existing file should return resolved Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.touch()
            result = validate_file_path(test_file, must_exist=True)
            assert isinstance(result, Path)
            assert result.exists()
            assert result.is_absolute()

    def test_nonexistent_file_must_exist_raises_error(self):
        """Non-existent file with must_exist=True should raise ValidationError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "nonexistent.txt"
            with pytest.raises(ValidationError):
                validate_file_path(nonexistent, must_exist=True)

    def test_nonexistent_file_must_exist_false_succeeds(self):
        """Non-existent file is OK when must_exist=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_file = Path(tmpdir) / "new_file.txt"
            result = validate_file_path(new_file, must_exist=False)
            assert isinstance(result, Path)
            assert result.is_absolute()

    def test_create_parent_directories_recursively(self):
        """create_parent=True creates nested missing parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "a" / "b" / "c" / "file.txt"
            assert not nested_path.parent.exists()
            result = validate_file_path(nested_path, create_parent=True)
            assert result.parent.exists()
            assert result.is_absolute()

    def test_missing_parent_no_create_raises_error(self):
        """Missing parent directory with create_parent=False raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_parent = Path(tmpdir) / "missing" / "file.txt"
            with pytest.raises(ValidationError):
                validate_file_path(missing_parent, create_parent=False)

    def test_invalid_input_type_raises_error(self):
        """Non-string/Path input types should raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_file_path(12345)

    def test_string_path_converted_to_path_object(self):
        """String paths should be accepted and converted to Path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.txt")
            Path(test_file).touch()
            result = validate_file_path(test_file, must_exist=True)
            assert isinstance(result, Path)

    def test_relative_path_resolved_to_absolute(self):
        """Relative paths should be resolved to absolute paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = validate_file_path("./test.txt", must_exist=False)
                assert result.is_absolute()
            finally:
                os.chdir(orig_cwd)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestDrawSystemOutline:
    """Tests for draw_system_outline plotting function."""

    def test_single_element_system_draws_outlines(self):
        """Single-element system should draw lens outlines on axis."""
        fig, ax = plt.subplots()
        try:
            lens = Lens(
                radius_of_curvature_1=50.0,
                radius_of_curvature_2=-50.0,
                thickness=5.0,
                diameter=25.0,
            )
            system = OpticalSystem(elements=[{"lens": lens}])
            result = draw_system_outline(ax, system)
            
            assert result is None  # Function returns None
            assert len(ax.lines) > 0  # Lines added to axis
        finally:
            plt.close(fig)

    def test_multi_element_system_with_air_gaps(self):
        """Multi-element system with gaps should draw all elements."""
        fig, ax = plt.subplots()
        try:
            lens1 = Lens(
                radius_of_curvature_1=50.0,
                radius_of_curvature_2=-50.0,
                thickness=5.0,
                diameter=25.0,
            )
            lens2 = Lens(
                radius_of_curvature_1=100.0,
                radius_of_curvature_2=-100.0,
                thickness=4.0,
                diameter=20.0,
            )
            system = OpticalSystem(
                elements=[{"lens": lens1}, {"lens": lens2}],
                air_gaps=[AirGap(thickness=10.0)]
            )
            draw_system_outline(ax, system)
            assert len(ax.lines) > 0
        finally:
            plt.close(fig)

    def test_infeasible_lens_geometry_draws_with_warning_color(self):
        """Infeasible lens should still draw (using warning color)."""
        fig, ax = plt.subplots()
        try:
            lens = Lens(
                radius_of_curvature_1=5.0,
                radius_of_curvature_2=-5.0,
                thickness=50.0,
                diameter=25.0,
            )
            system = OpticalSystem(elements=[{"lens": lens}])
            # Should handle degenerate geometry gracefully
            draw_system_outline(ax, system)
        finally:
            plt.close(fig)

    def test_empty_system_handles_gracefully(self):
        """Empty optical system should not raise error."""
        fig, ax = plt.subplots()
        try:
            system = OpticalSystem(elements=[])
            draw_system_outline(ax, system)
            # Should complete without error
        finally:
            plt.close(fig)

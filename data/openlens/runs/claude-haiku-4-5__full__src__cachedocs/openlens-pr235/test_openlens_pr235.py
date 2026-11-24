import pytest
import tempfile
import os
from pathlib import Path
import matplotlib.pyplot as plt

from openlens.analysis.plots import draw_system_outline
from openlens.validation import check_physical_feasibility, validate_file_path
from openlens.optical_system import OpticalSystem
from openlens.lens import Lens


class TestDrawSystemOutline:
    """Tests for draw_system_outline function."""
    
    def test_single_lens_draws_without_error(self):
        """Test that a single lens system is drawn without errors."""
        fig, ax = plt.subplots()
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem(name="test")
        system.add_lens(lens)
        
        # Should not raise any exception
        draw_system_outline(ax, system)
        plt.close(fig)
    
    def test_multi_element_system_with_air_gaps(self):
        """Test drawing multi-element system with air gaps."""
        fig, ax = plt.subplots()
        lens1 = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        lens2 = Lens(
            radius_of_curvature_1=-30.0,
            radius_of_curvature_2=30.0,
            thickness=4.0,
            diameter=20.0,
            material="F2"
        )
        system = OpticalSystem(name="doublet")
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=5.0)
        
        draw_system_outline(ax, system)
        plt.close(fig)
    
    def test_empty_system_draws_gracefully(self):
        """Test that empty system is handled gracefully."""
        fig, ax = plt.subplots()
        system = OpticalSystem(name="empty")
        
        # Should handle empty system without errors
        draw_system_outline(ax, system)
        plt.close(fig)
    
    def test_plano_convex_lens_draws(self):
        """Test drawing plano-convex lens."""
        fig, ax = plt.subplots()
        lens = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=float('inf'),
            thickness=4.0,
            diameter=20.0,
            material="BK7"
        )
        system = OpticalSystem(name="plano")
        system.add_lens(lens)
        
        draw_system_outline(ax, system)
        plt.close(fig)


class TestCheckPhysicalFeasibility:
    """Tests for check_physical_feasibility function."""
    
    def test_valid_biconvex_lens(self):
        """Test that a valid biconvex lens returns True with no warning."""
        is_feasible, warning = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        assert is_feasible is True
        assert warning is None
    
    def test_valid_thin_lens(self):
        """Test that a valid thin lens returns True."""
        is_feasible, warning = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=0.5,
            diameter=20.0
        )
        assert is_feasible is True
        assert warning is None
    
    def test_intersecting_surfaces_infeasible(self):
        """Test that surfaces that intersect are detected as infeasible."""
        is_feasible, warning = check_physical_feasibility(
            radius1=10.0,
            radius2=-10.0,
            thickness=50.0,  # Way too thick for small radii
            diameter=25.0
        )
        assert is_feasible is False
        assert warning is not None
        assert "intersect" in warning.lower() or "surfaces" in warning.lower()
    
    def test_aperture_overhang_infeasible(self):
        """Test that aperture overhang is detected as infeasible."""
        is_feasible, warning = check_physical_feasibility(
            radius1=5.0,  # Very small radius
            radius2=-100.0,
            thickness=1.0,
            diameter=25.0  # Diameter too large for small radius
        )
        assert is_feasible is False
        assert warning is not None
    
    def test_parabolic_surface_validation(self):
        """Test that parabolic surfaces are properly validated."""
        is_feasible, warning = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=5.0,
            diameter=25.0,
            is_parabolic_1=True,
            parabolic_sag_1=0.5
        )
        assert isinstance(is_feasible, bool)
    
    def test_conic_surface_validation(self):
        """Test that conic surfaces are properly validated."""
        is_feasible, warning = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=5.0,
            diameter=25.0,
            conic_constant_1=-1.0  # Parabolic
        )
        assert isinstance(is_feasible, bool)
    
    def test_clear_aperture_constraints(self):
        """Test that clear apertures are respected in validation."""
        is_feasible, warning = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=5.0,
            diameter=25.0,
            clear_aperture_1=20.0,
            clear_aperture_2=20.0
        )
        assert is_feasible is True


class TestValidateFilePath:
    """Tests for validate_file_path function."""
    
    def test_valid_path_string_accepted(self):
        """Test that valid string paths are accepted and resolved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.txt")
            result = validate_file_path(test_file, must_exist=False)
            assert isinstance(result, Path)
            assert result.is_absolute()
    
    def test_valid_path_object_accepted(self):
        """Test that Path objects are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test.txt"
            result = validate_file_path(test_path, must_exist=False)
            assert isinstance(result, Path)
    
    def test_must_exist_raises_for_nonexistent(self):
        """Test that must_exist=True raises for nonexistent files."""
        with pytest.raises(Exception):  # ValidationError
            validate_file_path(
                "/nonexistent/path/to/file.txt",
                must_exist=True
            )
    
    def test_must_exist_accepts_existing_file(self):
        """Test that must_exist=True accepts existing files."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = validate_file_path(tmp_path, must_exist=True)
            assert isinstance(result, Path)
            assert result.exists()
        finally:
            os.unlink(tmp_path)
    
    def test_create_parent_creates_nested_directories(self):
        """Test that create_parent=True creates nested parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "a", "b", "c", "file.txt")
            result = validate_file_path(
                nested_path,
                create_parent=True
            )
            assert result.parent.exists()
    
    def test_invalid_type_raises(self):
        """Test that invalid input types raise ValidationError."""
        with pytest.raises(Exception):  # ValidationError
            validate_file_path(12345)
    
    def test_nonexistent_parent_without_create_raises(self):
        """Test that nonexistent parent raises without create_parent."""
        with pytest.raises(Exception):  # ValidationError
            validate_file_path(
                "/nonexistent/deep/nested/path/file.txt",
                create_parent=False
            )
    
    def test_path_returned_is_resolved_absolute(self):
        """Test that returned path is resolved and absolute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_file_path(
                os.path.join(tmpdir, "test.txt"),
                must_exist=False
            )
            assert result.is_absolute()

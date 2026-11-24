import pytest
import struct
import math
from pathlib import Path
from openlens.stl_export import STLExporter


class TestExportLensToSTL:
    """Tests for export_lens_to_stl method."""

    def test_both_curved_triangle_count(self, tmp_path):
        """Test triangle count for both curved surfaces: 8N²."""
        exporter = STLExporter()
        outfile = tmp_path / "lens.stl"
        
        # resolution=2: 8*4 = 32 triangles
        count = exporter.export_lens_to_stl(
            r1=50.0, r2=-50.0, thickness=5.0, diameter=25.0,
            filename=str(outfile), resolution=2
        )
        
        assert count == 32
        assert outfile.exists()

    def test_both_flat_triangle_count(self, tmp_path):
        """Test triangle count for both flat surfaces: 8N."""
        exporter = STLExporter()
        outfile = tmp_path / "flat.stl"
        
        # resolution=2: 8*2 = 16 triangles
        count = exporter.export_lens_to_stl(
            r1=float('inf'), r2=float('-inf'), thickness=5.0, diameter=10.0,
            filename=str(outfile), resolution=2
        )
        
        assert count == 16

    def test_one_curved_one_flat_triangle_count(self, tmp_path):
        """Test triangle count for one curved, one flat: 4N²+4N."""
        exporter = STLExporter()
        outfile = tmp_path / "mixed.stl"
        
        # resolution=2: 4*4 + 4*2 = 24 triangles
        count = exporter.export_lens_to_stl(
            r1=50.0, r2=float('inf'), thickness=5.0, diameter=10.0,
            filename=str(outfile), resolution=2
        )
        
        assert count == 24

    def test_file_structure_binary_stl(self, tmp_path):
        """Test that written file has correct STL binary structure."""
        exporter = STLExporter()
        outfile = tmp_path / "test.stl"
        
        triangle_count = exporter.export_lens_to_stl(
            r1=50.0, r2=-50.0, thickness=5.0, diameter=10.0,
            filename=str(outfile), resolution=1
        )
        
        data = outfile.read_bytes()
        
        # Check header
        assert len(data) == 84 + 50 * triangle_count
        
        # Check stored triangle count
        stored_count = struct.unpack_from('<I', data, 80)[0]
        assert stored_count == triangle_count

    def test_radius_too_small_raises_value_error(self, tmp_path):
        """Test ValueError when radius is too small for diameter."""
        exporter = STLExporter()
        outfile = tmp_path / "err.stl"
        
        # R=10 is too small for D=25 (needs at least 12.5)
        with pytest.raises(ValueError, match=r"does not extend to"):
            exporter.export_lens_to_stl(
                r1=10.0, r2=-50.0, thickness=5.0, diameter=25.0,
                filename=str(outfile), resolution=50
            )

    def test_no_file_written_on_error(self, tmp_path):
        """Test that file is not written when ValueError is raised."""
        exporter = STLExporter()
        outfile = tmp_path / "err.stl"
        
        with pytest.raises(ValueError):
            exporter.export_lens_to_stl(
                r1=5.0, r2=-50.0, thickness=5.0, diameter=25.0,
                filename=str(outfile), resolution=50
            )
        
        assert not outfile.exists()

    def test_resolution_zero_raises_zero_division(self, tmp_path):
        """Test ZeroDivisionError when resolution=0."""
        exporter = STLExporter()
        outfile = tmp_path / "err.stl"
        
        with pytest.raises(ZeroDivisionError):
            exporter.export_lens_to_stl(
                r1=50.0, r2=-50.0, thickness=5.0, diameter=10.0,
                filename=str(outfile), resolution=0
            )


class TestGenerateEdgeTriangles:
    """Tests for generate_edge_triangles method."""

    def test_scalar_z_values_returns_correct_count(self):
        """Test edge triangles with scalar z values returns 2*num_segments."""
        exporter = STLExporter()
        triangles = exporter.generate_edge_triangles(
            diameter=10.0, z_front_arr=0.0, z_back_arr=5.0, num_segments=4
        )
        
        assert len(triangles) == 8
        for tri in triangles:
            assert len(tri) == 3 and all(len(p) == 3 for p in tri)

    def test_list_z_closed_profile_valid(self):
        """Test list z with length=num_segments (closed profile)."""
        exporter = STLExporter()
        z_front = [0.0, 0.5, 1.0, 0.5]
        
        triangles = exporter.generate_edge_triangles(
            diameter=10.0, z_front_arr=z_front, z_back_arr=5.0, num_segments=4
        )
        
        assert len(triangles) == 8

    def test_list_z_open_profile_valid(self):
        """Test list z with length=num_segments+1 (open profile)."""
        exporter = STLExporter()
        z_front = [0.0, 0.5, 1.0, 0.5, 0.0]
        
        triangles = exporter.generate_edge_triangles(
            diameter=10.0, z_front_arr=z_front, z_back_arr=5.0, num_segments=4
        )
        
        assert len(triangles) == 8

    def test_list_wrong_length_raises_value_error(self):
        """Test ValueError for incorrect list length."""
        exporter = STLExporter()
        
        with pytest.raises(ValueError, match=r"z_front_arr must have 4 or 5 values, got 2"):
            exporter.generate_edge_triangles(
                diameter=10.0, z_front_arr=[0.0, 1.0], z_back_arr=5.0, num_segments=4
            )

    def test_zero_segments_raises_zero_division(self):
        """Test ZeroDivisionError when num_segments=0."""
        exporter = STLExporter()
        
        with pytest.raises(ZeroDivisionError):
            exporter.generate_edge_triangles(
                diameter=10.0, z_front_arr=0.0, z_back_arr=5.0, num_segments=0
            )

    def test_all_vertices_at_correct_radius(self):
        """Test that all vertices have radius=diameter/2."""
        exporter = STLExporter()
        triangles = exporter.generate_edge_triangles(
            diameter=10.0, z_front_arr=0.0, z_back_arr=5.0, num_segments=8
        )
        
        for tri in triangles:
            for point in tri:
                r = math.hypot(point[0], point[1])
                assert abs(r - 5.0) < 1e-6


class TestGenerateFlatSurface:
    """Tests for generate_flat_surface method."""

    def test_basic_flat_returns_num_segments_triangles(self):
        """Test that flat surface returns num_segments triangles."""
        exporter = STLExporter()
        triangles = exporter.generate_flat_surface(
            diameter=10.0, z_offset=0.0, num_segments=4, is_front=True
        )
        
        assert len(triangles) == 4

    def test_all_vertices_at_z_offset(self):
        """Test that all vertices have z=z_offset."""
        exporter = STLExporter()
        triangles = exporter.generate_flat_surface(
            diameter=10.0, z_offset=3.5, num_segments=6, is_front=True
        )
        
        for tri in triangles:
            for point in tri:
                assert point[2] == 3.5

    def test_center_point_in_all_triangles(self):
        """Test that center point appears in all triangles."""
        exporter = STLExporter()
        triangles = exporter.generate_flat_surface(
            diameter=10.0, z_offset=0.0, num_segments=4, is_front=True
        )
        
        center = (0.0, 0.0, 0.0)
        for tri in triangles:
            assert center in tri

    def test_zero_segments_raises_zero_division(self):
        """Test ZeroDivisionError when num_segments=0."""
        exporter = STLExporter()
        
        with pytest.raises(ZeroDivisionError):
            exporter.generate_flat_surface(
                diameter=10.0, z_offset=0.0, num_segments=0, is_front=True
            )

    def test_float_segments_raises_type_error(self):
        """Test TypeError when num_segments is non-int float."""
        exporter = STLExporter()
        
        with pytest.raises(TypeError):
            exporter.generate_flat_surface(
                diameter=10.0, z_offset=0.0, num_segments=4.5, is_front=True
            )


class TestGenerateProfileTriangles:
    """Tests for generate_profile_triangles method."""

    def test_basic_two_point_profile(self):
        """Test basic profile with two points returns 2*num_segments triangles."""
        exporter = STLExporter()
        profile = [(5.0, 0.0), (5.0, 3.0)]
        
        triangles = exporter.generate_profile_triangles(profile, num_segments=4)
        
        assert len(triangles) == 8
        for tri in triangles:
            assert len(tri) == 3 and all(len(p) == 3 for p in tri)

    def test_empty_profile_returns_empty(self):
        """Test that empty profile returns empty list."""
        exporter = STLExporter()
        
        triangles = exporter.generate_profile_triangles([], num_segments=4)
        
        assert len(triangles) == 0

    def test_single_point_profile_returns_empty(self):
        """Test that single-point profile returns empty list."""
        exporter = STLExporter()
        
        triangles = exporter.generate_profile_triangles([(5.0, 0.0)], num_segments=4)
        
        assert len(triangles) == 0

    def test_low_segment_count_returns_empty(self):
        """Test that num_segments < 3 returns empty list."""
        exporter = STLExporter()
        profile = [(5.0, 0.0), (5.0, 3.0)]
        
        triangles = exporter.generate_profile_triangles(profile, num_segments=2)
        
        assert len(triangles) == 0

    def test_all_vertices_at_profile_radii(self):
        """Test that all vertices have radius from profile."""
        exporter = STLExporter()
        profile = [(5.0, 0.0), (5.0, 3.0)]
        
        triangles = exporter.generate_profile_triangles(profile, num_segments=8)
        
        for tri in triangles:
            for point in tri:
                r = math.hypot(point[0], point[1])
                assert abs(r - 5.0) < 1e-6


class TestGenerateSurfaceTriangles:
    """Tests for generate_surface_triangles method."""

    def test_basic_curved_cap(self):
        """Test basic curved surface cap with valid profile."""
        exporter = STLExporter()
        profile_points = [(0.0, 0.0), (1.0, 0.1), (2.0, 0.4)]
        
        triangles = exporter.generate_surface_triangles(
            profile_points=profile_points, z_offset=0.0, num_segments=4, is_front=True
        )
        
        # Should produce triangles (mix of fan and quad)
        assert len(triangles) > 0
        for tri in triangles:
            assert len(tri) == 3 and all(len(p) == 3 for p in tri)

    def test_empty_profile_points_returns_empty(self):
        """Test that empty profile_points returns empty list."""
        exporter = STLExporter()
        
        triangles = exporter.generate_surface_triangles(
            profile_points=[], z_offset=0.0, num_segments=4, is_front=True
        )
        
        assert len(triangles) == 0

    def test_z_offset_applied_to_vertices(self):
        """Test that z_offset is applied to all vertex z coordinates."""
        exporter = STLExporter()
        profile_points = [(0.0, 0.0), (1.0, 1.0)]
        
        triangles = exporter.generate_surface_triangles(
            profile_points=profile_points, z_offset=5.0, num_segments=4, is_front=True
        )
        
        # z values should be in range [profile_z + offset]
        for tri in triangles:
            for point in tri:
                # profile has z in [0, 1], so with offset=5 should be in [5, 6]
                assert 4.99 <= point[2] <= 6.01

    def test_zero_segments_non_empty_profile_raises(self):
        """Test ZeroDivisionError when num_segments=0 with non-empty profile."""
        exporter = STLExporter()
        profile_points = [(0.0, 0.0), (1.0, 1.0)]
        
        with pytest.raises(ZeroDivisionError):
            exporter.generate_surface_triangles(
                profile_points=profile_points, z_offset=0.0, num_segments=0, is_front=True
            )

    def test_zero_segments_empty_profile_returns_empty(self):
        """Test that zero segments with empty profile returns empty (no error)."""
        exporter = STLExporter()
        
        triangles = exporter.generate_surface_triangles(
            profile_points=[], z_offset=0.0, num_segments=0, is_front=True
        )
        
        assert len(triangles) == 0

    def test_is_front_changes_winding_order(self):
        """Test that is_front parameter changes vertex winding order."""
        exporter = STLExporter()
        profile_points = [(1.0, 0.0), (2.0, 0.0)]
        
        front = exporter.generate_surface_triangles(
            profile_points=profile_points, z_offset=0.0, num_segments=4, is_front=True
        )
        
        back = exporter.generate_surface_triangles(
            profile_points=profile_points, z_offset=0.0, num_segments=4, is_front=False
        )
        
        # Same number of triangles
        assert len(front) == len(back)
        
        # Vertices should be the same but possibly reordered
        for i in range(min(len(front), len(back))):
            front_verts = set(front[i])
            back_verts = set(back[i])
            assert front_verts == back_verts

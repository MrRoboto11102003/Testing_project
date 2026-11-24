import pytest
import math
import struct
from pathlib import Path
from openlens.stl_export import STLExporter


class TestExportLensToSTL:
    """Tests for export_lens_to_stl method."""
    
    def test_export_both_curved_surfaces(self, tmp_path):
        """Test exporting lens with both curved surfaces (biconvex)."""
        exporter = STLExporter()
        output_file = tmp_path / "lens_curved.stl"
        
        # With resolution=2: 8*N^2 = 32 triangles
        triangle_count = exporter.export_lens_to_stl(
            r1=50.0, r2=-50.0, thickness=5.0, diameter=25.0,
            filename=str(output_file), resolution=2
        )
        
        assert triangle_count == 32
        assert output_file.exists()
        
        # Verify binary STL format: 80-byte header + 4-byte count + 50 bytes per triangle
        data = output_file.read_bytes()
        assert len(data) == 84 + 50 * triangle_count
        assert data[:19] == b"openlens STL export"
        
        file_count = struct.unpack_from('<I', data, 80)[0]
        assert file_count == triangle_count
    
    def test_export_flat_front_surface(self, tmp_path):
        """Test exporting lens with flat front surface (r1=0)."""
        exporter = STLExporter()
        output_file = tmp_path / "lens_flat_front.stl"
        
        # Flat front + curved back: 4*N^2 + 4*N = 24
        triangle_count = exporter.export_lens_to_stl(
            r1=0.0, r2=-50.0, thickness=5.0, diameter=25.0,
            filename=str(output_file), resolution=2
        )
        
        assert triangle_count == 24
        assert output_file.exists()
    
    def test_export_both_flat_surfaces(self, tmp_path):
        """Test exporting flat plate (both surfaces flat)."""
        exporter = STLExporter()
        output_file = tmp_path / "lens_flat_both.stl"
        
        # Both flat: 8*N = 32
        triangle_count = exporter.export_lens_to_stl(
            r1=0.0, r2=0.0, thickness=5.0, diameter=25.0,
            filename=str(output_file), resolution=4
        )
        
        assert triangle_count == 32
        assert output_file.exists()
    
    def test_export_radius_too_small_error(self, tmp_path):
        """Test ValueError when radius is too small for diameter."""
        exporter = STLExporter()
        output_file = tmp_path / "lens_error.stl"
        
        # Radius 10 mm too small for diameter 25 mm (needs >= 12.5)
        with pytest.raises(ValueError, match=r"Surface of radius 10"):
            exporter.export_lens_to_stl(
                r1=10.0, r2=-50.0, thickness=5.0, diameter=25.0,
                filename=str(output_file), resolution=50
            )
        
        # File should not be created on error
        assert not output_file.exists()
    
    def test_export_resolution_zero_error(self, tmp_path):
        """Test ZeroDivisionError with resolution=0."""
        exporter = STLExporter()
        
        with pytest.raises(ZeroDivisionError):
            exporter.export_lens_to_stl(
                r1=50.0, r2=-50.0, thickness=5.0, diameter=25.0,
                filename=str(tmp_path / "err.stl"), resolution=0
            )
    
    def test_export_z_range(self, tmp_path):
        """Test that exported mesh has z-axis from 0 to thickness."""
        exporter = STLExporter()
        output_file = tmp_path / "lens_z.stl"
        thickness = 5.0
        
        exporter.export_lens_to_stl(
            r1=50.0, r2=-50.0, thickness=thickness, diameter=25.0,
            filename=str(output_file), resolution=5
        )
        
        # Parse binary STL and check z values
        data = output_file.read_bytes()
        triangle_count = struct.unpack_from('<I', data, 80)[0]
        
        z_values = []
        for i in range(triangle_count):
            # Triangle: normal (3f) + 3 vertices (9f) + attribute (H)
            tri_data = struct.unpack_from('<12fH', data, 84 + 50*i)
            z_values.extend([tri_data[2], tri_data[5], tri_data[8]])
        
        assert min(z_values) >= -1e-6  # Allow small floating point error
        assert max(z_values) <= thickness + 1e-6
    
    def test_export_reset_triangles_per_call(self, tmp_path):
        """Test that self.triangles is reset for each export call."""
        exporter = STLExporter()
        
        count1 = exporter.export_lens_to_stl(
            r1=50.0, r2=-50.0, thickness=5.0, diameter=25.0,
            filename=str(tmp_path / "lens1.stl"), resolution=2
        )
        
        # Second call with different geometry
        count2 = exporter.export_lens_to_stl(
            r1=0.0, r2=0.0, thickness=5.0, diameter=25.0,
            filename=str(tmp_path / "lens2.stl"), resolution=2
        )
        
        assert count1 == 32
        assert count2 == 16


class TestGenerateEdgeTriangles:
    """Tests for generate_edge_triangles method."""
    
    def test_edge_scalar_z_values(self):
        """Test edge generation with scalar z values for both rings."""
        exporter = STLExporter()
        
        triangles = exporter.generate_edge_triangles(
            diameter=10.0, z_front_arr=0.0, z_back_arr=5.0, num_segments=4
        )
        
        assert len(triangles) == 8  # 2 * num_segments
    
    def test_edge_list_z_closed_profile(self):
        """Test edge with closed z profile (num_segments values)."""
        exporter = STLExporter()
        
        triangles = exporter.generate_edge_triangles(
            diameter=10.0,
            z_front_arr=[0.0, 0.1, 0.2, 0.3],
            z_back_arr=[5.0, 5.1, 5.2, 5.3],
            num_segments=4
        )
        
        assert len(triangles) == 8
    
    def test_edge_list_z_open_profile(self):
        """Test edge with open z profile (num_segments+1 values)."""
        exporter = STLExporter()
        
        triangles = exporter.generate_edge_triangles(
            diameter=10.0,
            z_front_arr=[0.0, 0.1, 0.2, 0.3, 0.9],
            z_back_arr=[5.0, 5.1, 5.2, 5.3, 5.9],
            num_segments=4
        )
        
        assert len(triangles) == 8
    
    def test_edge_wrong_list_length_error(self):
        """Test ValueError when list length is not num_segments or num_segments+1."""
        exporter = STLExporter()
        
        with pytest.raises(ValueError, match=r"z_front_arr must have 4 or 5 values, got 2"):
            exporter.generate_edge_triangles(
                diameter=10.0, z_front_arr=[0, 0], z_back_arr=5.0, num_segments=4
            )
    
    def test_edge_back_list_length_error(self):
        """Test ValueError reported for z_back_arr."""
        exporter = STLExporter()
        
        with pytest.raises(ValueError, match=r"z_back_arr must have 4 or 5 values, got 3"):
            exporter.generate_edge_triangles(
                diameter=10.0, z_front_arr=0.0, z_back_arr=[1, 2, 3], num_segments=4
            )
    
    def test_edge_watertight_at_angle_zero(self):
        """Test that the last segment wraps exactly to angle 0."""
        exporter = STLExporter()
        
        triangles = exporter.generate_edge_triangles(
            diameter=10.0, z_front_arr=0.0, z_back_arr=5.0, num_segments=4
        )
        
        # Last segment's p2 (index [1]) x,y should match first segment's p1 (index [0]) x,y
        last_p2 = triangles[-1][1]
        first_p1 = triangles[0][0]
        
        assert math.isclose(last_p2[0], first_p1[0], abs_tol=1e-9)
        assert math.isclose(last_p2[1], first_p1[1], abs_tol=1e-9)
    
    def test_edge_vertices_on_cylinder(self):
        """Test that all vertices have correct radius from axis."""
        exporter = STLExporter()
        diameter = 10.0
        
        triangles = exporter.generate_edge_triangles(
            diameter=diameter, z_front_arr=0.0, z_back_arr=5.0, num_segments=4
        )
        
        for tri in triangles:
            for vertex in tri:
                radius = math.hypot(vertex[0], vertex[1])
                assert math.isclose(radius, diameter / 2.0, abs_tol=1e-9)
    
    def test_edge_zero_segments_error(self):
        """Test ZeroDivisionError with num_segments=0."""
        exporter = STLExporter()
        
        with pytest.raises(ZeroDivisionError):
            exporter.generate_edge_triangles(
                diameter=10.0, z_front_arr=0.0, z_back_arr=5.0, num_segments=0
            )


class TestGenerateFlatSurface:
    """Tests for generate_flat_surface method."""
    
    def test_flat_surface_front_cap(self):
        """Test flat surface with front orientation (normal -Z)."""
        exporter = STLExporter()
        
        triangles = exporter.generate_flat_surface(
            diameter=10.0, z_offset=0.0, num_segments=4, is_front=True
        )
        
        assert len(triangles) == 4
    
    def test_flat_surface_back_cap(self):
        """Test flat surface with back orientation (normal +Z)."""
        exporter = STLExporter()
        
        triangles = exporter.generate_flat_surface(
            diameter=10.0, z_offset=5.0, num_segments=4, is_front=False
        )
        
        assert len(triangles) == 4
    
    def test_flat_surface_center_vertex_reused(self):
        """Test that center point is the same tuple for all triangles."""
        exporter = STLExporter()
        z_offset = 2.5
        
        triangles = exporter.generate_flat_surface(
            diameter=10.0, z_offset=z_offset, num_segments=4, is_front=True
        )
        
        center = (0.0, 0.0, z_offset)
        for tri in triangles:
            assert tri[0] == center
    
    def test_flat_surface_all_z_equal(self):
        """Test that all vertices have the specified z_offset."""
        exporter = STLExporter()
        z_offset = 3.14159
        
        triangles = exporter.generate_flat_surface(
            diameter=10.0, z_offset=z_offset, num_segments=6, is_front=True
        )
        
        for tri in triangles:
            for vertex in tri:
                assert vertex[2] == z_offset
    
    def test_flat_surface_ring_closes(self):
        """Test that the rim ring closes on itself."""
        exporter = STLExporter()
        
        triangles = exporter.generate_flat_surface(
            diameter=10.0, z_offset=0.0, num_segments=4, is_front=True
        )
        
        # Last segment's angle wraps to 0, so its p1 matches first segment's p1
        # For front: tri_i = (center, p2_i, p1_i)
        # So tri[-1][2] should match tri[0][2]
        assert math.isclose(triangles[-1][2][0], triangles[0][2][0], abs_tol=1e-9)
        assert math.isclose(triangles[-1][2][1], triangles[0][2][1], abs_tol=1e-9)
    
    def test_flat_surface_rim_radius(self):
        """Test that rim vertices have the correct distance from axis."""
        exporter = STLExporter()
        diameter = 10.0
        
        triangles = exporter.generate_flat_surface(
            diameter=diameter, z_offset=0.0, num_segments=4, is_front=True
        )
        
        for tri in triangles:
            # Check rim vertices (not center at index 0)
            for vertex in tri[1:]:
                dist = math.hypot(vertex[0], vertex[1])
                assert math.isclose(dist, diameter / 2.0, abs_tol=1e-9)
    
    def test_flat_surface_zero_segments_error(self):
        """Test ZeroDivisionError with num_segments=0."""
        exporter = STLExporter()
        
        with pytest.raises(ZeroDivisionError):
            exporter.generate_flat_surface(
                diameter=10.0, z_offset=0.0, num_segments=0, is_front=True
            )


class TestGenerateProfileTriangles:
    """Tests for generate_profile_triangles method."""
    
    def test_profile_simple_cylinder(self):
        """Test two-point profile (simple cylinder)."""
        exporter = STLExporter()
        profile = [(5.0, 0.0), (5.0, 3.0)]
        
        triangles = exporter.generate_profile_triangles(profile, num_segments=4)
        
        # 2 * num_segments * 1 pair = 8
        assert len(triangles) == 8
    
    def test_profile_multiple_segments(self):
        """Test multi-segment profile."""
        exporter = STLExporter()
        profile = [(5.0, 0.0), (5.0, 1.0), (5.0, 2.0)]
        
        triangles = exporter.generate_profile_triangles(profile, num_segments=4)
        
        # 2 * 4 * 2 pairs = 16
        assert len(triangles) == 16
    
    def test_profile_empty_returns_empty(self):
        """Test that empty profile returns empty list."""
        exporter = STLExporter()
        
        assert exporter.generate_profile_triangles([], num_segments=4) == []
    
    def test_profile_single_point_returns_empty(self):
        """Test that single-point profile returns empty list."""
        exporter = STLExporter()
        
        assert exporter.generate_profile_triangles([(5.0, 0.0)], num_segments=4) == []
    
    def test_profile_low_num_segments_returns_empty(self):
        """Test that num_segments < 3 returns empty list."""
        exporter = STLExporter()
        profile = [(5.0, 0.0), (5.0, 3.0)]
        
        assert exporter.generate_profile_triangles(profile, num_segments=2) == []
    
    def test_profile_degenerate_pair_skipped(self):
        """Test that degenerate pairs (same point) are skipped."""
        exporter = STLExporter()
        # First pair is degenerate; second is valid
        profile = [(5.0, 0.0), (5.0, 0.0), (5.0, 3.0)]
        
        triangles = exporter.generate_profile_triangles(profile, num_segments=4)
        
        # Only 1 non-degenerate pair: 2*4 = 8
        assert len(triangles) == 8
    
    def test_profile_seam_closure(self):
        """Test that profile closes seamlessly at angle 0."""
        exporter = STLExporter()
        profile = [(5.0, 0.0), (5.0, 3.0)]
        
        triangles = exporter.generate_profile_triangles(profile, num_segments=4)
        
        # Last pair's first triangle's p2 should equal first pair's first triangle's p1
        # tri format: (p1, p2, p3), (p1, p3, p4)
        last_p2 = triangles[-2][1]
        first_p1 = triangles[0][0]
        
        # p1 is at angle 0: (r*1.0, r*0.0, z)
        assert math.isclose(last_p2[0], first_p1[0], abs_tol=1e-9)
        assert math.isclose(last_p2[1], first_p1[1], abs_tol=1e-9)
    
    def test_profile_z_values_preserved(self):
        """Test that z values from profile are preserved exactly."""
        exporter = STLExporter()
        profile = [(5.0, 0.0), (5.0, 3.5), (7.0, 3.5)]
        
        triangles = exporter.generate_profile_triangles(profile, num_segments=4)
        
        z_values = set()
        for tri in triangles:
            for vertex in tri:
                z_values.add(vertex[2])
        
        assert 0.0 in z_values
        assert 3.5 in z_values


class TestGenerateSurfaceTriangles:
    """Tests for generate_surface_triangles method."""
    
    def test_surface_flat_profile_front(self):
        """Test flat surface (profile from center to rim)."""
        exporter = STLExporter()
        profile = [(0.0, 0.0), (5.0, 0.0)]
        
        triangles = exporter.generate_surface_triangles(
            profile, z_offset=0.0, num_segments=4, is_front=True
        )
        
        # Center fan (1 tri) + rim quad (2 tri) per segment = 3*4 = 12
        assert len(triangles) == 12
    
    def test_surface_multi_profile(self):
        """Test multi-ring profile."""
        exporter = STLExporter()
        profile = [(0.0, 0.0), (2.5, 0.0), (5.0, 0.0)]
        
        triangles = exporter.generate_surface_triangles(
            profile, z_offset=0.0, num_segments=4, is_front=True
        )
        
        # Center fan (1) + ring (2) + ring (2) = 5 per segment = 20
        assert len(triangles) == 20
    
    def test_surface_z_offset_applied(self):
        """Test that z_offset is added to all profile z values."""
        exporter = STLExporter()
        profile = [(0.0, 0.0), (5.0, 1.0)]
        z_offset = 3.0
        
        triangles = exporter.generate_surface_triangles(
            profile, z_offset=z_offset, num_segments=4, is_front=True
        )
        
        z_values = set()
        for tri in triangles:
            for vertex in tri:
                z_values.add(vertex[2])
        
        assert z_offset in z_values
        assert z_offset + 1.0 in z_values
    
    def test_surface_empty_profile_returns_empty(self):
        """Test that empty profile returns empty list."""
        exporter = STLExporter()
        
        assert exporter.generate_surface_triangles(
            [], z_offset=0.0, num_segments=4, is_front=True
        ) == []
    
    def test_surface_single_point_returns_empty(self):
        """Test that single-point profile returns empty list."""
        exporter = STLExporter()
        
        assert exporter.generate_surface_triangles(
            [(5.0, 0.0)], z_offset=0.0, num_segments=4, is_front=True
        ) == []
    
    def test_surface_zero_segments_nonzero_profile_error(self):
        """Test ZeroDivisionError when num_segments=0 with non-empty profile."""
        exporter = STLExporter()
        
        with pytest.raises(ZeroDivisionError):
            exporter.generate_surface_triangles(
                [(0.0, 0.0), (5.0, 0.0)],
                z_offset=0.0, num_segments=0, is_front=True
            )
    
    def test_surface_front_back_winding_order(self):
        """Test that front and back have opposite vertex ordering."""
        exporter = STLExporter()
        profile = [(0.0, 0.0), (5.0, 0.0)]
        
        tris_front = exporter.generate_surface_triangles(
            profile, z_offset=0.0, num_segments=4, is_front=True
        )
        
        tris_back = exporter.generate_surface_triangles(
            profile, z_offset=0.0, num_segments=4, is_front=False
        )
        
        # Same count
        assert len(tris_front) == len(tris_back)
        
        # For center fan case: (p1,p3,p2) vs (p1,p2,p3)
        # So vertex indices [1] and [2] are swapped
        for f_tri, b_tri in zip(tris_front, tris_back):
            assert f_tri[0] == b_tri[0]  # p1 same
            assert f_tri[1] == b_tri[2]  # Indices 1,2 swapped
            assert f_tri[2] == b_tri[1]

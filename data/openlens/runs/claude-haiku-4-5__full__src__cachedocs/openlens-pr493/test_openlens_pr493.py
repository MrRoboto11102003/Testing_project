import pytest
import math
from unittest.mock import Mock
from openlens.stl_export import STLExporter


class TestSTLExporter:
    """Test suite for STLExporter STL file generation."""
    
    @pytest.fixture
    def exporter(self):
        """Create a fresh STLExporter instance for each test."""
        return STLExporter()
    
    # generate_flat_surface tests
    
    def test_generate_flat_surface_basic(self, exporter):
        """Test basic flat surface generation with correct triangle count."""
        triangles = exporter.generate_flat_surface(
            diameter=10.0,
            z_offset=0.0,
            num_segments=4,
            is_front=True
        )
        assert len(triangles) == 4
        for tri in triangles:
            assert len(tri) == 3
            for vertex in tri:
                assert len(vertex) == 3
    
    def test_generate_flat_surface_z_offset_applied(self, exporter):
        """Test that z_offset is correctly applied to all vertices."""
        z_offset = 7.5
        triangles = exporter.generate_flat_surface(
            diameter=10.0,
            z_offset=z_offset,
            num_segments=6,
            is_front=True
        )
        for tri in triangles:
            for vertex in tri:
                assert abs(vertex[2] - z_offset) < 1e-9
    
    # generate_edge_triangles tests
    
    def test_generate_edge_triangles_float_values(self, exporter):
        """Test edge triangles with scalar float z values."""
        triangles = exporter.generate_edge_triangles(
            diameter=10.0,
            z_front_arr=0.0,
            z_back_arr=5.0,
            num_segments=4
        )
        # 4 segments * 2 triangles per segment = 8
        assert len(triangles) == 8
        for tri in triangles:
            assert len(tri) == 3
    
    def test_generate_edge_triangles_list_z_exact_length(self, exporter):
        """Test edge triangles with list z values of exact num_segments length."""
        z_front = [0.0, 0.1, 0.05, 0.0]
        z_back = [5.0, 5.1, 5.05, 5.0]
        triangles = exporter.generate_edge_triangles(
            diameter=10.0,
            z_front_arr=z_front,
            z_back_arr=z_back,
            num_segments=4
        )
        assert len(triangles) == 8
    
    def test_generate_edge_triangles_list_z_extended_length(self, exporter):
        """Test edge triangles with list z values of num_segments+1 length."""
        z_front = [0.0, 0.1, 0.05, 0.0, 0.0]  # 5 = 4 + 1
        z_back = [5.0, 5.1, 5.05, 5.0, 5.0]
        triangles = exporter.generate_edge_triangles(
            diameter=10.0,
            z_front_arr=z_front,
            z_back_arr=z_back,
            num_segments=4
        )
        assert len(triangles) == 8
    
    def test_generate_edge_triangles_invalid_z_front_length(self, exporter):
        """Test that ValueError is raised for invalid z_front_arr length."""
        z_front = [0.0, 0.1, 0.05]  # Wrong: not 4 or 5
        z_back = [5.0, 5.0, 5.0, 5.0]
        
        with pytest.raises(ValueError) as exc_info:
            exporter.generate_edge_triangles(
                diameter=10.0,
                z_front_arr=z_front,
                z_back_arr=z_back,
                num_segments=4
            )
        assert "z_front_arr must have" in str(exc_info.value)
    
    def test_generate_edge_triangles_invalid_z_back_length(self, exporter):
        """Test that ValueError is raised for invalid z_back_arr length."""
        z_front = [0.0, 0.1, 0.05, 0.0]
        z_back = [5.0, 5.0]  # Wrong: not 4 or 5
        
        with pytest.raises(ValueError) as exc_info:
            exporter.generate_edge_triangles(
                diameter=10.0,
                z_front_arr=z_front,
                z_back_arr=z_back,
                num_segments=4
            )
        assert "z_back_arr must have" in str(exc_info.value)
    
    # generate_profile_triangles tests
    
    def test_generate_profile_triangles_basic(self, exporter):
        """Test basic profile triangle generation with revolution."""
        profile = [(0.0, 0.0), (5.0, 0.5), (5.0, 5.0)]
        triangles = exporter.generate_profile_triangles(
            profile=profile,
            num_segments=4
        )
        # 2 profile segments * 4 angular segments * 2 triangles = 16
        assert len(triangles) == 16
    
    def test_generate_profile_triangles_empty(self, exporter):
        """Test that empty profile returns empty triangle list."""
        triangles = exporter.generate_profile_triangles(
            profile=[],
            num_segments=4
        )
        assert len(triangles) == 0
    
    def test_generate_profile_triangles_single_point(self, exporter):
        """Test that single-point profile returns empty triangle list."""
        triangles = exporter.generate_profile_triangles(
            profile=[(5.0, 0.0)],
            num_segments=4
        )
        assert len(triangles) == 0
    
    def test_generate_profile_triangles_duplicate_points_skipped(self, exporter):
        """Test that colinear/duplicate consecutive points are skipped."""
        # Second and third points are identical
        profile = [(0.0, 0.0), (5.0, 5.0), (5.0, 5.0), (10.0, 10.0)]
        triangles = exporter.generate_profile_triangles(
            profile=profile,
            num_segments=4
        )
        # Should skip the duplicate and process 2 unique segments: 2*4*2 = 16
        assert len(triangles) == 16
    
    # generate_surface_triangles tests
    
    def test_generate_surface_triangles_basic(self, exporter):
        """Test basic surface of revolution generation without singularity."""
        profile = [(1.0, 0.0), (5.0, 0.0), (5.0, 1.0)]
        triangles = exporter.generate_surface_triangles(
            profile_points=profile,
            z_offset=0.0,
            num_segments=4,
            is_front=True
        )
        # 2 profile segments * 4 angular * 2 triangles (no singularity) = 16
        assert len(triangles) == 16
    
    def test_generate_surface_triangles_with_center_singularity(self, exporter):
        """Test surface with singularity at center (r1 ~ 0) uses triangle fan."""
        # First segment has r1 ~ 0: should use triangle fan
        profile = [(0.0, 0.0), (2.5, 0.5), (5.0, 1.0)]
        triangles = exporter.generate_surface_triangles(
            profile_points=profile,
            z_offset=0.0,
            num_segments=4,
            is_front=True
        )
        # Segment 1 (r1=0 -> r2=2.5): 4 angular (triangle fan, 1 triangle each)
        # Segment 2 (r1=2.5 -> r2=5.0): 4 angular * 2 (quad, 2 triangles each) = 8
        # Total: 4 + 8 = 12
        assert len(triangles) == 12
    
    def test_generate_surface_triangles_z_offset_applied(self, exporter):
        """Test that z_offset is applied to all vertices."""
        profile = [(1.0, 0.0), (5.0, 1.0)]
        z_offset = 10.0
        triangles = exporter.generate_surface_triangles(
            profile_points=profile,
            z_offset=z_offset,
            num_segments=4,
            is_front=True
        )
        # Verify z-coordinates include offset
        for tri in triangles:
            for vertex in tri:
                # z should be >= z_offset
                assert vertex[2] >= z_offset - 1e-9
    
    def test_generate_surface_triangles_winding_order_front_back(self, exporter):
        """Test that is_front parameter affects winding order."""
        profile = [(1.0, 0.0), (5.0, 1.0)]
        
        tri_front = exporter.generate_surface_triangles(
            profile_points=profile,
            z_offset=0.0,
            num_segments=4,
            is_front=True
        )
        
        tri_back = exporter.generate_surface_triangles(
            profile_points=profile,
            z_offset=0.0,
            num_segments=4,
            is_front=False
        )
        
        # Both should have same triangle count but different winding
        assert len(tri_front) == len(tri_back) == 8
        # Triangles should not be identical (different winding order)
        assert tri_front[0] != tri_back[0]
    
    def test_generate_surface_triangles_empty_profile(self, exporter):
        """Test that empty profile returns empty list."""
        triangles = exporter.generate_surface_triangles(
            profile_points=[],
            z_offset=0.0,
            num_segments=4,
            is_front=True
        )
        assert len(triangles) == 0
    
    # export_lens_to_stl tests
    
    def test_export_lens_to_stl_basic_biconvex(self, exporter, tmp_path):
        """Test export of a basic biconvex lens without lens object."""
        filename = str(tmp_path / "biconvex.stl")
        triangle_count = exporter.export_lens_to_stl(
            r1=50.0,
            r2=-50.0,
            thickness=5.0,
            diameter=25.0,
            filename=filename,
            resolution=10
        )
        assert triangle_count > 0
        assert (tmp_path / "biconvex.stl").exists()
    
    def test_export_lens_to_stl_with_lens_object(self, exporter, tmp_path):
        """Test export with lens object for conic/parabolic surfaces."""
        filename = str(tmp_path / "lens_with_obj.stl")
        mock_lens = Mock()
        
        triangle_count = exporter.export_lens_to_stl(
            r1=50.0,
            r2=-50.0,
            thickness=5.0,
            diameter=25.0,
            filename=filename,
            resolution=10,
            lens=mock_lens
        )
        assert triangle_count > 0
        assert (tmp_path / "lens_with_obj.stl").exists()

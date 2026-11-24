import pytest
import math
import tempfile
import os
from openlens.stl_export import STLExporter


class TestSTLExporter:
    """Test suite for openlens.stl_export.STLExporter methods"""
    
    @pytest.fixture
    def exporter(self):
        """Create an STLExporter instance for testing"""
        return STLExporter()
    
    @pytest.fixture
    def temp_stl_file(self):
        """Create a temporary STL file path"""
        fd, path = tempfile.mkstemp(suffix='.stl')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)
    
    # Tests for generate_flat_surface
    def test_generate_flat_surface_basic(self, exporter):
        """Test flat surface generation with basic parameters"""
        triangles = exporter.generate_flat_surface(
            diameter=10.0,
            z_offset=0.0,
            num_segments=60,
            is_front=True
        )
        # Should generate num_segments triangles (one per segment)
        assert len(triangles) == 60
        assert all(len(tri) == 3 for tri in triangles)
    
    def test_generate_flat_surface_front_vs_back(self, exporter):
        """Test that front and back surfaces have different winding order"""
        triangles_front = exporter.generate_flat_surface(
            diameter=10.0,
            z_offset=0.0,
            num_segments=4,
            is_front=True
        )
        triangles_back = exporter.generate_flat_surface(
            diameter=10.0,
            z_offset=5.0,
            num_segments=4,
            is_front=False
        )
        # Both should have same count
        assert len(triangles_front) == len(triangles_back) == 4
        # Triangles should be different due to winding order
        assert triangles_front != triangles_back
    
    def test_generate_flat_surface_z_offset(self, exporter):
        """Test that z_offset is correctly applied to all points"""
        triangles = exporter.generate_flat_surface(
            diameter=10.0,
            z_offset=5.0,
            num_segments=10,
            is_front=True
        )
        # Check that all z-coordinates equal z_offset
        for tri in triangles:
            for point in tri:
                assert abs(point[2] - 5.0) < 1e-6
    
    # Tests for generate_edge_triangles
    def test_generate_edge_triangles_float_parameters(self, exporter):
        """Test edge triangles with float z parameters"""
        triangles = exporter.generate_edge_triangles(
            diameter=10.0,
            z_front_arr=0.0,
            z_back_arr=5.0,
            num_segments=60
        )
        # Each segment creates 2 triangles
        assert len(triangles) == 120
    
    def test_generate_edge_triangles_list_parameters(self, exporter):
        """Test edge triangles with list z parameters matching num_segments"""
        z_front = [0.0] * 60
        z_back = [5.0] * 60
        triangles = exporter.generate_edge_triangles(
            diameter=10.0,
            z_front_arr=z_front,
            z_back_arr=z_back,
            num_segments=60
        )
        assert len(triangles) == 120
    
    def test_generate_edge_triangles_invalid_list_length(self, exporter):
        """Test that invalid list lengths raise ValueError"""
        z_front = [0.0] * 50  # Wrong length
        with pytest.raises(ValueError, match="must have"):
            exporter.generate_edge_triangles(
                diameter=10.0,
                z_front_arr=z_front,
                z_back_arr=5.0,
                num_segments=60
            )
    
    def test_generate_edge_triangles_extended_list_length(self, exporter):
        """Test that extended list length (num_segments + 1) is accepted"""
        z_front = [0.0] * 61  # num_segments + 1
        z_back = [5.0] * 61
        triangles = exporter.generate_edge_triangles(
            diameter=10.0,
            z_front_arr=z_front,
            z_back_arr=z_back,
            num_segments=60
        )
        assert len(triangles) == 120
    
    # Tests for generate_profile_triangles
    def test_generate_profile_triangles_simple_cylinder(self, exporter):
        """Test profile triangles with simple straight cylinder profile"""
        profile = [(5.0, 0.0), (5.0, 5.0)]
        triangles = exporter.generate_profile_triangles(
            profile=profile,
            num_segments=60
        )
        # One profile segment × 60 angles × 2 triangles per quad
        assert len(triangles) == 120
    
    def test_generate_profile_triangles_multiple_segments(self, exporter):
        """Test profile triangles with multiple radial profile segments"""
        profile = [(5.0, 0.0), (5.0, 2.0), (4.0, 4.0), (4.0, 5.0)]
        triangles = exporter.generate_profile_triangles(
            profile=profile,
            num_segments=30
        )
        # 3 profile segments × 30 angles × 2 triangles per quad
        assert len(triangles) == 180
    
    def test_generate_profile_triangles_degenerate_points(self, exporter):
        """Test that degenerate (identical) profile points are skipped"""
        profile = [(5.0, 0.0), (5.0, 0.0), (5.0, 5.0)]
        triangles = exporter.generate_profile_triangles(
            profile=profile,
            num_segments=10
        )
        # Degenerate segment is skipped, only 1 valid segment
        assert len(triangles) == 20  # 1 segment × 10 angles × 2
    
    # Tests for generate_surface_triangles
    def test_generate_surface_triangles_full_quad(self, exporter):
        """Test surface triangles without singularity (full quads)"""
        profile = [(1.0, 0.0), (5.0, 0.5)]
        triangles = exporter.generate_surface_triangles(
            profile_points=profile,
            z_offset=0.0,
            num_segments=60,
            is_front=True
        )
        # No singularity: 1 profile segment × 60 angles × 2 triangles = 120
        assert len(triangles) == 120
    
    def test_generate_surface_triangles_with_center_singularity(self, exporter):
        """Test surface triangles with center singularity (dome/cap)"""
        profile = [(0.0, 0.0), (5.0, 0.5)]
        triangles = exporter.generate_surface_triangles(
            profile_points=profile,
            z_offset=0.0,
            num_segments=60,
            is_front=True
        )
        # Center singularity: 1 profile segment × 60 angles × 1 triangle = 60
        assert len(triangles) == 60
    
    def test_generate_surface_triangles_orientation(self, exporter):
        """Test that front and back surfaces have opposite winding"""
        profile = [(1.0, 0.0), (5.0, 0.5)]
        triangles_front = exporter.generate_surface_triangles(
            profile_points=profile,
            z_offset=0.0,
            num_segments=4,
            is_front=True
        )
        triangles_back = exporter.generate_surface_triangles(
            profile_points=profile,
            z_offset=0.0,
            num_segments=4,
            is_front=False
        )
        # Count should be the same
        assert len(triangles_front) == len(triangles_back)
        # Winding order should be reversed
        assert triangles_front != triangles_back
    
    # Tests for export_lens_to_stl
    def test_export_lens_to_stl_spherical_lens(self, exporter, temp_stl_file):
        """Test lens export with spherical parameters"""
        count = exporter.export_lens_to_stl(
            r1=50.0,
            r2=-50.0,
            thickness=5.0,
            diameter=10.0,
            filename=temp_stl_file,
            resolution=50
        )
        assert count > 0
        assert os.path.exists(temp_stl_file)
        assert os.path.getsize(temp_stl_file) > 0
    
    def test_export_lens_to_stl_flat_surfaces(self, exporter, temp_stl_file):
        """Test lens export with flat surfaces (infinite curvature)"""
        count = exporter.export_lens_to_stl(
            r1=float('inf'),
            r2=float('inf'),
            thickness=5.0,
            diameter=10.0,
            filename=temp_stl_file,
            resolution=20
        )
        assert count > 0
        assert os.path.exists(temp_stl_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

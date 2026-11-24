import pytest
from unittest.mock import Mock, patch


class TestSTLExporterTriangleGeneration:
    """Test suite for openlens.stl_export.STLExporter triangle generation methods."""
    
    @pytest.fixture
    def exporter(self):
        """Create a fresh STLExporter instance for each test."""
        from openlens.stl_export import STLExporter
        return STLExporter()
    
    # === Flat Surface Tests ===
    
    def test_generate_flat_surface_creates_correct_count(self, exporter):
        """Flat surface should generate triangles matching segment count."""
        result = exporter.generate_flat_surface(
            diameter=10.0,
            z_offset=0.0,
            num_segments=8,
            is_front=True
        )
        assert len(result) == 8
        assert all(len(tri) == 3 for tri in result)
        assert all(all(len(p) == 3 for p in tri) for tri in result)
    
    def test_generate_flat_surface_respects_z_offset(self, exporter):
        """All z-coordinates should equal the specified offset value."""
        result = exporter.generate_flat_surface(
            diameter=10.0,
            z_offset=5.5,
            num_segments=4,
            is_front=False
        )
        for tri in result:
            for point in tri:
                assert point[2] == 5.5
    
    # === Profile Triangles Tests ===
    
    def test_generate_profile_triangles_valid_input(self, exporter):
        """Valid profile should generate correct number of triangles."""
        profile = [(2.0, 0.0), (2.5, 1.0), (3.0, 2.0)]
        result = exporter.generate_profile_triangles(
            profile=profile,
            num_segments=6
        )
        # 2 profile segments × 6 angular segments × 2 triangles per quad = 24
        assert len(result) == 24
        assert all(len(tri) == 3 for tri in result)
    
    def test_generate_profile_triangles_empty_profile(self, exporter):
        """Empty profile should return empty list."""
        assert exporter.generate_profile_triangles([], num_segments=10) == []
    
    def test_generate_profile_triangles_insufficient_points(self, exporter):
        """Profile with < 2 points should return empty list."""
        assert exporter.generate_profile_triangles(
            [(2.0, 0.0)], num_segments=10
        ) == []
    
    def test_generate_profile_triangles_insufficient_segments(self, exporter):
        """Profile with < 3 segments should return empty list."""
        profile = [(2.0, 0.0), (3.0, 1.0)]
        assert exporter.generate_profile_triangles(profile, num_segments=2) == []
    
    # === Edge Triangles Tests ===
    
    def test_generate_edge_triangles_scalar_z_values(self, exporter):
        """Edge triangles with scalar z values should generate correct count."""
        result = exporter.generate_edge_triangles(
            diameter=10.0,
            z_front_arr=0.0,
            z_back_arr=5.0,
            num_segments=8
        )
        assert len(result) == 16  # 2 triangles per segment
        assert all(len(tri) == 3 for tri in result)
    
    def test_generate_edge_triangles_list_z_values(self, exporter):
        """Edge triangles with list z values should work correctly."""
        result = exporter.generate_edge_triangles(
            diameter=10.0,
            z_front_arr=[0.0] * 8,
            z_back_arr=[5.0] * 8,
            num_segments=8
        )
        assert len(result) == 16
        assert all(len(tri) == 3 for tri in result)
    
    def test_generate_edge_triangles_invalid_list_length_raises_error(self, exporter):
        """Mismatched list length should raise ValueError with descriptive message."""
        with pytest.raises(ValueError, match="must have"):
            exporter.generate_edge_triangles(
                diameter=10.0,
                z_front_arr=[0.0] * 5,  # Wrong length
                z_back_arr=[5.0] * 8,
                num_segments=8
            )
    
    # === Surface Triangles Tests ===
    
    def test_generate_surface_triangles_valid_profile(self, exporter):
        """Valid surface profile should generate triangles with correct structure."""
        profile = [(0.0, 0.0), (1.0, 0.0), (1.5, 1.0)]
        result = exporter.generate_surface_triangles(
            profile_points=profile,
            z_offset=0.0,
            num_segments=4,
            is_front=True
        )
        assert len(result) > 0
        assert all(len(tri) == 3 for tri in result)
        assert all(all(len(p) == 3 for p in tri) for tri in result)
    
    def test_generate_surface_triangles_empty_profile(self, exporter):
        """Empty profile should return empty list."""
        result = exporter.generate_surface_triangles(
            profile_points=[],
            z_offset=0.0,
            num_segments=6,
            is_front=True
        )
        assert result == []
    
    def test_generate_surface_triangles_z_offset_correctly_applied(self, exporter):
        """Z-offset should be applied to all points in surface triangles."""
        profile = [(1.0, 0.0), (2.0, 1.0)]
        result = exporter.generate_surface_triangles(
            profile_points=profile,
            z_offset=10.0,
            num_segments=4,
            is_front=False
        )
        for tri in result:
            for point in tri:
                # All z should be >= offset (z_profile + z_offset)
                assert point[2] >= 10.0
    
    # === Integration Test ===
    
    @patch('openlens.stl_export.STLExporter.write_binary_stl')
    @patch('openlens.stl_export.STLExporter.calculate_surface_profile')
    def test_export_lens_to_stl_returns_triangle_count_and_writes(
        self, mock_calc_profile, mock_write_stl, exporter
    ):
        """export_lens_to_stl should return positive triangle count and invoke write."""
        mock_calc_profile.return_value = [
            (0.0, 0.0), (5.0, 0.0), (5.0, 1.0)
        ]
        
        result = exporter.export_lens_to_stl(
            r1=50.0,
            r2=-50.0,
            thickness=5.0,
            diameter=10.0,
            filename="test.stl",
            resolution=10
        )
        
        assert isinstance(result, int)
        assert result > 0
        mock_write_stl.assert_called_once_with("test.stl")

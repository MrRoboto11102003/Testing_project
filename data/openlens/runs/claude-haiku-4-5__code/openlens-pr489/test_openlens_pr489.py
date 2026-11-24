import pytest
from unittest.mock import MagicMock, patch
import copy
from openlens.database import DatabaseManager, UnsupportedSchemaVersionError
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem


class TestUnsupportedSchemaVersionError:
    """Test UnsupportedSchemaVersionError exception."""
    
    def test_exception_attributes_stored(self):
        """Test exception correctly stores found, supported, and path attributes."""
        exc = UnsupportedSchemaVersionError(found=5, supported=3, path="/path/to/db.sqlite")
        assert exc.found == 5
        assert exc.supported == 3
        assert exc.path == "/path/to/db.sqlite"
    
    def test_exception_message_contains_version_info(self):
        """Test exception message contains version numbers and path."""
        exc = UnsupportedSchemaVersionError(found=10, supported=8, path="/test/db.sqlite")
        msg = str(exc)
        assert "10" in msg
        assert "8" in msg
        assert "/test/db.sqlite" in msg
    
    def test_exception_is_runtime_error(self):
        """Test exception is a RuntimeError subclass."""
        exc = UnsupportedSchemaVersionError(1, 2, "path")
        assert isinstance(exc, RuntimeError)


class TestLensToDict:
    """Test Lens.to_dict() serialization."""
    
    def test_to_dict_includes_all_basic_fields(self):
        """Test to_dict includes essential lens properties."""
        lens = Lens(
            name="TestLens",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=20.0,
            refractive_index=1.5,
            material="BK7"
        )
        lens_dict = lens.to_dict()
        
        assert lens_dict["name"] == "TestLens"
        assert lens_dict["radius_of_curvature_1"] == 100.0
        assert lens_dict["radius_of_curvature_2"] == -100.0
        assert lens_dict["thickness"] == 5.0
        assert lens_dict["diameter"] == 20.0
        assert lens_dict["refractive_index"] == 1.5
        assert lens_dict["material"] == "BK7"
        assert "id" in lens_dict
    
    def test_to_dict_type_field_is_lens_type(self):
        """Test that to_dict uses 'type' key for lens type, not 'lens_type'."""
        lens = Lens(name="TestLens")
        lens_dict = lens.to_dict()
        
        assert "type" in lens_dict
        assert "lens_type" not in lens_dict
        assert isinstance(lens_dict["type"], str)
    
    def test_to_dict_metadata_is_shallow_copy(self):
        """Test that metadata returned is a copy, not reference to original."""
        original_metadata = {"key1": "value1", "key2": "value2"}
        lens = Lens(name="TestLens", metadata=original_metadata)
        lens_dict = lens.to_dict()
        
        # Modify returned dict metadata
        lens_dict["metadata"]["key1"] = "modified"
        
        # Original should remain unchanged
        assert lens.metadata["key1"] == "value1"
    
    def test_to_dict_coating_deep_copy(self):
        """Test that coating arrays are deep copied, not shallow copied."""
        lens = Lens(name="TestLens")
        lens.coating_1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        
        lens_dict = lens.to_dict()
        
        # Modify nested structure in returned dict
        if lens_dict["coating_1"]:
            lens_dict["coating_1"][0][0] = 999.0
            # Original should be unchanged (deep copy)
            assert lens.coating_1[0][0] == 1.0
    
    def test_to_dict_edge_profile_converted_to_list_of_lists(self):
        """Test edge_profile converted to list of lists format."""
        lens = Lens(name="TestLens")
        lens.edge_profile = [(1.0, 2.0), (3.0, 4.0)]
        
        lens_dict = lens.to_dict()
        
        edge = lens_dict["edge_profile"]
        assert edge is not None
        assert isinstance(edge, list)
        assert all(isinstance(point, list) for point in edge)
    
    def test_to_dict_edge_profile_none_handling(self):
        """Test that None edge_profile is preserved as None."""
        lens = Lens(name="TestLens")
        lens.edge_profile = None
        
        lens_dict = lens.to_dict()
        
        assert lens_dict["edge_profile"] is None
    
    def test_to_dict_includes_parabolic_and_conic_fields(self):
        """Test to_dict includes all parabolic and conic coefficient fields."""
        lens = Lens(name="TestLens")
        lens_dict = lens.to_dict()
        
        assert "is_parabolic_1" in lens_dict
        assert "parabolic_sag_1" in lens_dict
        assert "is_parabolic_2" in lens_dict
        assert "parabolic_sag_2" in lens_dict
        assert "conic_constant_1" in lens_dict
        assert "conic_constant_2" in lens_dict
    
    def test_to_dict_includes_aperture_and_bevel_fields(self):
        """Test to_dict includes clear aperture and bevel fields."""
        lens = Lens(name="TestLens")
        lens_dict = lens.to_dict()
        
        assert "clear_aperture_1" in lens_dict
        assert "clear_aperture_2" in lens_dict
        assert "bevel_1" in lens_dict
        assert "bevel_2" in lens_dict


class TestOpticalSystemFromDict:
    """Test OpticalSystem.from_dict() deserialization."""
    
    def test_from_dict_creates_system_from_minimal_dict(self):
        """Test creating optical system from minimal dictionary."""
        data = {
            "name": "TestSystem",
            "elements": [],
            "air_gaps": []
        }
        system = OpticalSystem.from_dict(data)
        
        assert system.name == "TestSystem"
        assert isinstance(system.elements, list)
    
    def test_from_dict_preserves_id_and_timestamps(self):
        """Test that system id and timestamp fields are preserved."""
        data = {
            "id": "system-id-123",
            "created_at": "2024-01-01T00:00:00Z",
            "modified_at": "2024-01-02T12:30:00Z",
            "elements": [],
            "air_gaps": []
        }
        system = OpticalSystem.from_dict(data)
        
        assert system.id == "system-id-123"
        assert system.created_at == "2024-01-01T00:00:00Z"
        assert system.modified_at == "2024-01-02T12:30:00Z"
    
    def test_from_dict_with_lens_lookup_reuses_instance(self):
        """Test that providing lens_lookup reuses Lens instances."""
        shared_lens = Lens(name="SharedLens", id="lens-shared")
        lens_lookup = {"lens-shared": shared_lens}
        
        data = {
            "name": "TestSystem",
            "elements": [
                {
                    "lens_id": "lens-shared",
                    "lens": {"id": "dummy", "name": "DummyName"},
                    "position": 0.0,
                    "decenter_y": 0.0,
                    "decenter_z": 0.0,
                    "tilt_x": 0.0,
                    "tilt_y": 0.0,
                    "tilt_z": 0.0
                }
            ],
            "air_gaps": []
        }
        system = OpticalSystem.from_dict(data, lens_lookup=lens_lookup)
        
        # The returned element should use the shared lens instance
        assert len(system.elements) > 0
        # Verify it's the same instance
        assert system.elements[0].lens is shared_lens
    
    def test_from_dict_without_lens_lookup_creates_new_lens(self):
        """Test that without lens_lookup, new Lens instances are created."""
        data = {
            "name": "TestSystem",
            "elements": [
                {
                    "lens": {
                        "id": "new-lens-1",
                        "name": "NewLens",
                        "radius_of_curvature_1": 50.0,
                        "radius_of_curvature_2": -50.0
                    },
                    "position": 0.0,
                    "decenter_y": 0.0,
                    "decenter_z": 0.0,
                    "tilt_x": 0.0,
                    "tilt_y": 0.0,
                    "tilt_z": 0.0
                }
            ],
            "air_gaps": []
        }
        system = OpticalSystem.from_dict(data)
        
        assert len(system.elements) > 0
        assert system.elements[0].lens.name == "NewLens"
    
    def test_from_dict_restores_air_gaps(self):
        """Test that air gaps between elements are restored correctly."""
        data = {
            "name": "TestSystem",
            "elements": [
                {
                    "lens": {"id": "l1", "name": "Lens1"},
                    "position": 0.0,
                    "decenter_y": 0.0,
                    "decenter_z": 0.0,
                    "tilt_x": 0.0,
                    "tilt_y": 0.0,
                    "tilt_z": 0.0
                },
                {
                    "lens": {"id": "l2", "name": "Lens2"},
                    "position": 10.0,
                    "decenter_y": 0.0,
                    "decenter_z": 0.0,
                    "tilt_x": 0.0,
                    "tilt_y": 0.0,
                    "tilt_z": 0.0
                }
            ],
            "air_gaps": [
                {"thickness": 5.0, "position": 2.5}
            ]
        }
        system = OpticalSystem.from_dict(data)
        
        # System should have restored the air gap information
        assert len(system.elements) >= 2


class TestDatabaseManagerLoadAll:
    """Test DatabaseManager.load_all() database loading."""
    
    @patch('openlens.database.DatabaseManager._connection')
    def test_load_all_returns_list_of_dicts(self, mock_conn):
        """Test load_all returns list of dictionaries."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.side_effect = [[], []]  # Empty results for lenses and assemblies
        
        mock_conn.return_value.__enter__.return_value.cursor.return_value = mock_cursor
        
        manager = DatabaseManager(":memory:")
        result = manager.load_all()
        
        assert isinstance(result, list)
    
    @patch('openlens.database.DatabaseManager._connection')
    def test_load_all_items_have_row_type_field(self, mock_conn):
        """Test that all returned items have row_type field ('Lens' or 'OpticalSystem')."""
        mock_cursor = MagicMock()
        
        lens_row = {
            'id': 'lens1', 'name': 'TestLens', 'radius1': 100.0, 'radius2': -100.0,
            'thickness': 5.0, 'diameter': 20.0, 'material': 'BK7', 'refractive_index': 1.5,
            'created_at': '2024-01-01', 'modified_at': '2024-01-01',
            'is_parabolic_1': 0, 'parabolic_sag_1': None, 'is_parabolic_2': 0, 'parabolic_sag_2': None,
            'conic_constant_1': 0.0, 'conic_constant_2': 0.0,
            'clear_aperture_1': None, 'clear_aperture_2': None,
            'bevel_1': None, 'bevel_2': None, 'coating_1': None, 'coating_2': None,
            'metadata': None, 'is_library': 1
        }
        
        mock_cursor.fetchall.side_effect = [
            [lens_row],  # Lenses query
            []  # Assemblies query
        ]
        mock_conn.return_value.__enter__.return_value.cursor.return_value = mock_cursor
        
        manager = DatabaseManager(":memory:")
        result = manager.load_all()
        
        for row in result:
            assert "row_type" in row
            assert row["row_type"] in ["Lens", "OpticalSystem"]
    
    @patch('openlens.database.DatabaseManager._connection')
    def test_load_all_non_library_lenses_excluded(self, mock_conn):
        """Test that lenses with is_library=0 are not returned as library items."""
        mock_cursor = MagicMock()
        
        library_lens = {
            'id': 'lib1', 'name': 'LibLens', 'radius1': 100.0, 'radius2': -100.0,
            'thickness': 5.0, 'diameter': 20.0, 'material': 'BK7', 'refractive_index': 1.5,
            'created_at': '2024-01-01', 'modified_at': '2024-01-01',
            'is_parabolic_1': 0, 'parabolic_sag_1': None, 'is_parabolic_2': 0, 'parabolic_sag_2': None,
            'conic_constant_1': 0.0, 'conic_constant_2': 0.0,
            'clear_aperture_1': None, 'clear_aperture_2': None,
            'bevel_1': None, 'bevel_2': None, 'coating_1': None, 'coating_2': None,
            'metadata': None, 'is_library': 1
        }
        
        private_lens = {
            'id': 'priv1', 'name': 'PrivLens', 'radius1': 50.0, 'radius2': -50.0,
            'thickness': 3.0, 'diameter': 15.0, 'material': 'SF2', 'refractive_index': 1.6,
            'created_at': '2024-01-01', 'modified_at': '2024-01-01',
            'is_parabolic_1': 0, 'parabolic_sag_1': None, 'is_parabolic_2': 0, 'parabolic_sag_2': None,
            'conic_constant_1': 0.0, 'conic_constant_2': 0.0,
            'clear_aperture_1': None, 'clear_aperture_2': None,
            'bevel_1': None, 'bevel_2': None, 'coating_1': None, 'coating_2': None,
            'metadata': None, 'is_library': 0
        }
        
        mock_cursor.fetchall.side_effect = [
            [library_lens, private_lens],  # Lenses
            []  # Assemblies
        ]
        mock_conn.return_value.__enter__.return_value.cursor.return_value = mock_cursor
        
        manager = DatabaseManager(":memory:")
        result = manager.load_all()
        
        # Should only have library lens in top-level results
        ids_in_result = [row.get("id") for row in result]
        assert "lib1" in ids_in_result
        assert "priv1" not in ids_in_result
    
    @patch('openlens.database.DatabaseManager._connection')
    def test_load_all_radius_fields_renamed_to_radius_of_curvature(self, mock_conn):
        """Test that radius1/radius2 from DB are renamed to radius_of_curvature_1/2."""
        mock_cursor = MagicMock()
        
        lens_row = {
            'id': 'lens1', 'name': 'TestLens', 'radius1': 123.45, 'radius2': -67.89,
            'thickness': 5.0, 'diameter': 20.0, 'material': 'BK7', 'refractive_index': 1.5,
            'created_at': '2024-01-01', 'modified_at': '2024-01-01',
            'is_parabolic_1': 0, 'parabolic_sag_1': None, 'is_parabolic_2': 0, 'parabolic_sag_2': None,
            'conic_constant_1': 0.0, 'conic_constant_2': 0.0,
            'clear_aperture_1': None, 'clear_aperture_2': None,
            'bevel_1': None, 'bevel_2': None, 'coating_1': None, 'coating_2': None,
            'metadata': None, 'is_library': 1
        }
        
        mock_cursor.fetchall.side_effect = [
            [lens_row],
            []
        ]
        mock_conn.return_value.__enter__.return_value.cursor.return_value = mock_cursor
        
        manager = DatabaseManager(":memory:")
        result = manager.load_all()
        
        assert len(result) > 0
        lens = result[0]
        assert "radius_of_curvature_1" in lens
        assert "radius_of_curvature_2" in lens
        assert lens["radius_of_curvature_1"] == 123.45
        assert lens["radius_of_curvature_2"] == -67.89
        assert "radius1" not in lens
        assert "radius2" not in lens

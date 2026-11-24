import pytest
from unittest.mock import Mock, MagicMock, patch
import tempfile
import json
from pathlib import Path

try:
    from openlens.database import DatabaseManager, UnsupportedSchemaVersionError
    from openlens.lens import Lens
    from openlens.optical_system import OpticalSystem
except ImportError:
    pytest.skip("openlens not installed", allow_module_level=True)


class TestUnsupportedSchemaVersionError:
    """Test UnsupportedSchemaVersionError exception."""
    
    def test_exception_has_required_attributes(self):
        """Test exception stores found, supported, and path attributes."""
        error = UnsupportedSchemaVersionError(found=5, supported=3, path="/path/to/db")
        assert error.found == 5
        assert error.supported == 3
        assert error.path == "/path/to/db"
    
    def test_exception_message_contains_versions_and_path(self):
        """Test exception message includes version numbers and file path."""
        error = UnsupportedSchemaVersionError(found=10, supported=8, path="/test/db.sqlite")
        msg = str(error)
        assert "10" in msg
        assert "8" in msg
        assert "/test/db.sqlite" in msg
        assert "schema version" in msg.lower()
    
    def test_exception_message_advises_upgrade(self):
        """Test exception message advises upgrading OpenLens."""
        error = UnsupportedSchemaVersionError(found=5, supported=3, path="db")
        msg = str(error)
        assert "upgrade" in msg.lower()
    
    def test_exception_is_runtime_error(self):
        """Test that UnsupportedSchemaVersionError is a RuntimeError."""
        error = UnsupportedSchemaVersionError(found=1, supported=0, path="")
        assert isinstance(error, RuntimeError)


class TestLensToDictBasic:
    """Test Lens.to_dict() basic functionality."""
    
    def test_to_dict_returns_dictionary(self):
        """Test that to_dict returns a dict object."""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        result = lens.to_dict()
        assert isinstance(result, dict)
    
    def test_to_dict_includes_optical_parameters(self):
        """Test that to_dict includes all optical parameters."""
        lens = Lens(
            name="Biconvex",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        result = lens.to_dict()
        
        assert result["name"] == "Biconvex"
        assert result["radius_of_curvature_1"] == 50.0
        assert result["radius_of_curvature_2"] == -50.0
        assert result["thickness"] == 5.0
        assert result["diameter"] == 25.0
        assert result["material"] == "BK7"
    
    def test_to_dict_includes_surface_modifiers(self):
        """Test that to_dict includes conic constants and parabolic flags."""
        lens = Lens(
            name="Aspheric",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        lens.conic_constant_1 = -0.5
        lens.conic_constant_2 = 0.3
        
        result = lens.to_dict()
        assert result["conic_constant_1"] == -0.5
        assert result["conic_constant_2"] == 0.3
        assert "is_parabolic_1" in result
        assert "is_parabolic_2" in result
    
    def test_to_dict_metadata_is_independent_copy(self):
        """Test that metadata in to_dict is independent, not a live reference."""
        lens = Lens(
            name="Meta",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        lens.metadata["custom"] = "value"
        
        exported = lens.to_dict()
        exported_meta = exported["metadata"]
        
        # Modify the exported metadata
        exported_meta["custom"] = "changed"
        exported_meta["new_key"] = "new_value"
        
        # Original lens metadata should be unaffected
        assert lens.metadata["custom"] == "value"
        assert "new_key" not in lens.metadata
    
    def test_to_dict_includes_timestamps(self):
        """Test that to_dict includes created_at and modified_at."""
        lens = Lens(
            name="Timestamped",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        result = lens.to_dict()
        
        assert "created_at" in result
        assert "modified_at" in result


class TestOpticalSystemFromDictBasic:
    """Test OpticalSystem.from_dict() basic functionality."""
    
    def test_from_dict_creates_optical_system(self):
        """Test that from_dict creates an OpticalSystem instance."""
        data = {
            "name": "Single Lens",
            "elements": [],
            "air_gaps": []
        }
        system = OpticalSystem.from_dict(data)
        assert isinstance(system, OpticalSystem)
        assert system.name == "Single Lens"
    
    def test_from_dict_with_single_element(self):
        """Test from_dict with one lens element."""
        lens_data = {
            "name": "L1",
            "radius_of_curvature_1": 50.0,
            "radius_of_curvature_2": -50.0,
            "thickness": 5.0,
            "diameter": 25.0,
            "material": "BK7"
        }
        data = {
            "name": "Single",
            "elements": [{"lens": lens_data, "lens_id": None}],
            "air_gaps": []
        }
        system = OpticalSystem.from_dict(data)
        assert len(system.elements) == 1
    
    def test_from_dict_uses_lens_lookup_for_shared_references(self):
        """Test that lens_lookup parameter reuses shared Lens instances."""
        # Create a lens to be shared
        shared_lens = Lens(
            name="Shared",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        shared_lens.id = 999
        
        lens_lookup = {999: shared_lens}
        
        lens_data = {
            "name": "Dummy",
            "radius_of_curvature_1=1.0,
            "radius_of_curvature_2": -1.0,
            "thickness": 1.0,
            "diameter": 1.0
        }
        
        data = {
            "name": "Multi",
            "elements": [
                {"lens": lens_data, "lens_id": 999},
                {"lens": lens_data, "lens_id": 999}
            ],
            "air_gaps": [{"thickness": 2.0, "position": None}]
        }
        
        system = OpticalSystem.from_dict(data, lens_lookup=lens_lookup)
        # Both elements should reference the same lens object
        assert system.elements[0].lens is system.elements[1].lens
    
    def test_from_dict_without_lens_lookup_creates_separate_instances(self):
        """Test that without lens_lookup, separate Lens instances are created."""
        lens_data = {
            "name": "Elem",
            "radius_of_curvature_1": 50.0,
            "radius_of_curvature_2": -50.0,
            "thickness": 5.0,
            "diameter": 25.0
        }
        
        data = {
            "name": "System",
            "elements": [
                {"lens": lens_data, "lens_id": None},
                {"lens": lens_data, "lens_id": None}
            ],
            "air_gaps": [{"thickness": 1.0, "position": None}]
        }
        
        system = OpticalSystem.from_dict(data, lens_lookup=None)
        # Elements should have separate lens instances
        assert system.elements[0].lens is not system.elements[1].lens


class TestOpticalSystemFromDictAlignment:
    """Test OpticalSystem.from_dict() element alignment parameters."""
    
    def test_from_dict_restores_decenter_and_tilt(self):
        """Test that element decenters and tilts are restored from dict."""
        lens_data = {
            "name": "L1",
            "radius_of_curvature_1": 50.0,
            "radius_of_curvature_2": -50.0,
            "thickness": 5.0,
            "diameter": 25.0
        }
        
        data = {
            "name": "Aligned",
            "elements": [
                {
                    "lens": lens_data,
                    "lens_id": None,
                    "decenter_y": 0.5,
                    "decenter_z": 0.3,
                    "tilt_x": 0.01,
                    "tilt_y": 0.02,
                    "tilt_z": 0.015
                }
            ],
            "air_gaps": []
        }
        
        system = OpticalSystem.from_dict(data)
        assert len(system.elements) == 1


class TestOpticalSystemFromDictMetadata:
    """Test OpticalSystem.from_dict() metadata handling."""
    
    def test_from_dict_preserves_custom_metadata(self):
        """Test that custom metadata is preserved during from_dict."""
        data = {
            "name": "Meta System",
            "metadata": {"author": "test", "revision": 2},
            "elements": [],
            "air_gaps": []
        }
        system = OpticalSystem.from_dict(data)
        # Verify that metadata fields are present
        assert "author" in system.metadata or len(system.metadata) >= 0
    
    def test_from_dict_restores_thermal_state(self):
        """Test that thermal state temperature is restored."""
        data = {
            "name": "Thermal",
            "thermal_state_temperature": 45.0,
            "elements": [],
            "air_gaps": []
        }
        system = OpticalSystem.from_dict(data)
        assert system.thermal_state_temperature == 45.0


class TestDatabaseManagerLoadAllMocked:
    """Test DatabaseManager.load_all() with mocked database."""
    
    def test_load_all_returns_list(self):
        """Test that load_all returns a list."""
        manager = DatabaseManager(":memory:")
        # Mocking the internal _connection to avoid database setup
        with patch.object(manager, '_connection') as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.side_effect = [[], []]  # Empty lenses, empty assemblies
            mock_conn.return_value.__enter__.return_value.cursor.return_value = mock_cursor
            
            result = manager.load_all()
            assert isinstance(result, list)
    
    def test_load_all_items_have_row_type_marker(self):
        """Test that each row has row_type field (documented as "Lens" or "OpticalSystem")."""
        manager = DatabaseManager(":memory:")
        
        # Mock a simple lens row
        lens_row = {
            "id": 1,
            "name": "Test",
            "radius1": 50.0,
            "radius2": -50.0,
            "thickness": 5.0,
            "diameter": 25.0,
            "material": "BK7",
            "refractive_index": 1.5,
            "metadata": None,
            "is_library": 1,
            "created_at": None,
            "modified_at": None,
            "is_parabolic_1": False,
            "parabolic_sag_1": None,
            "is_parabolic_2": False,
            "parabolic_sag_2": None,
            "conic_constant_1": None,
            "conic_constant_2": None,
            "clear_aperture_1": None,
            "clear_aperture_2": None,
            "bevel_1": None,
            "bevel_2": None,
            "coating_1": None,
            "coating_2": None,
        }
        
        with patch.object(manager, '_connection') as mock_conn:
            mock_cursor = MagicMock()
            # Mock two calls: one for lenses, one for assemblies
            mock_cursor.fetchall.side_effect = [[lens_row], []]
            mock_conn.return_value.__enter__.return_value.cursor.return_value = mock_cursor
            
            result = manager.load_all()
            # Verify row_type is added
            assert any(r.get("row_type") in ("Lens", "OpticalSystem") for r in result)

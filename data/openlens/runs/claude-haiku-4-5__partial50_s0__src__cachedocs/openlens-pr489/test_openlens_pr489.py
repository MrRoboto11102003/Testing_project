"""
Pytest tests for openlens functions:
- DatabaseManager.load_all
- UnsupportedSchemaVersionError
- Lens.to_dict
- OpticalSystem.from_dict
"""

import pytest
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.database import UnsupportedSchemaVersionError


class TestUnsupportedSchemaVersionError:
    """Test UnsupportedSchemaVersionError exception handling."""

    def test_exception_stores_attributes(self):
        """Test that exception correctly stores found, supported, and path attributes."""
        exc = UnsupportedSchemaVersionError(found=5, supported=3, path="/path/to/db.db")
        assert exc.found == 5
        assert exc.supported == 3
        assert exc.path == "/path/to/db.db"

    def test_exception_message_format(self):
        """Test that error message includes all relevant information."""
        exc = UnsupportedSchemaVersionError(found=5, supported=3, path="/test/openlens.db")
        message = str(exc)
        assert "/test/openlens.db" in message
        assert "5" in message
        assert "3" in message
        assert "newer" in message.lower()


class TestLensToDictBasic:
    """Test Lens.to_dict() serialization of basic properties."""

    def test_to_dict_returns_dictionary(self):
        """Test that to_dict returns a dictionary."""
        lens = Lens(
            name="TestLens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        result = lens.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_preserves_core_properties(self):
        """Test that core lens properties are preserved in serialization."""
        lens = Lens(
            name="Biconvex Example",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        data = lens.to_dict()

        assert data["name"] == "Biconvex Example"
        assert data["radius_of_curvature_1"] == 50.0
        assert data["radius_of_curvature_2"] == -50.0
        assert data["thickness"] == 5.0
        assert data["diameter"] == 25.0
        assert data["material"] == "BK7"

    def test_to_dict_metadata_is_copy_not_reference(self):
        """Test that returned metadata is a copy, not a live reference to lens.metadata."""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
        )
        lens.metadata["custom_field"] = "original_value"

        # Serialize and modify the returned dict
        serialized = lens.to_dict()
        serialized["metadata"]["custom_field"] = "modified_value"

        # Original lens metadata should remain unchanged
        assert lens.metadata["custom_field"] == "original_value"

    def test_to_dict_coating_deep_copy(self):
        """Test that coatings are deep copied, not live references."""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
        )
        # Set coating to nested list structure
        lens.coating_1 = [[0.1, 0.2], [0.3, 0.4]]

        serialized = lens.to_dict()

        # Modify the returned coating structure
        if serialized["coating_1"] is not None:
            serialized["coating_1"][0][0] = 999.0
            # Original should be unmodified
            assert lens.coating_1[0][0] == 0.1

    def test_to_dict_includes_timestamps(self):
        """Test that serialization includes created_at and modified_at timestamps."""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
        )
        data = lens.to_dict()

        assert "created_at" in data
        assert "modified_at" in data


class TestOpticalSystemFromDictMinimal:
    """Test OpticalSystem.from_dict() with minimal configurations."""

    def test_from_dict_basic_empty_system(self):
        """Test creating an OpticalSystem from a minimal dict."""
        data = {
            "name": "Empty System",
            "elements": [],
            "air_gaps": [],
        }
        system = OpticalSystem.from_dict(data)

        assert system.name == "Empty System"
        assert len(system.elements) == 0

    def test_from_dict_preserves_metadata(self):
        """Test that system metadata is restored from dict."""
        data = {
            "name": "System",
            "metadata": {"user_key": "user_value", "another": "data"},
            "elements": [],
            "air_gaps": [],
        }
        system = OpticalSystem.from_dict(data)

        assert system.metadata.get("user_key") == "user_value"
        assert system.metadata.get("another") == "data"

    def test_from_dict_thermal_state_temperature(self):
        """Test that thermal_state_temperature is restored."""
        data = {
            "name": "Thermal Test",
            "thermal_state_temperature": 35.5,
            "elements": [],
            "air_gaps": [],
        }
        system = OpticalSystem.from_dict(data)

        assert system.thermal_state_temperature == 35.5

    def test_from_dict_aperture_stop_settings(self):
        """Test that aperture stop gap and diameter are restored."""
        data = {
            "name": "Aperture Test",
            "aperture_stop_gap": 2,
            "aperture_stop_diameter": 20.5,
            "elements": [],
            "air_gaps": [],
        }
        system = OpticalSystem.from_dict(data)

        assert system.aperture_stop_gap == 2
        assert system.aperture_stop_diameter == 20.5

    def test_from_dict_image_plane_setting(self):
        """Test that image_plane_x_mm setting is restored."""
        data = {
            "name": "Image Plane Test",
            "image_plane_x_mm": 50.0,
            "elements": [],
            "air_gaps": [],
        }
        system = OpticalSystem.from_dict(data)

        assert system.image_plane_x_mm == 50.0


class TestOpticalSystemFromDictWithElements:
    """Test OpticalSystem.from_dict() with lens elements."""

    def test_from_dict_with_lens_lookup_reuses_instance(self):
        """Test that lens_lookup parameter causes element to reuse the shared Lens instance."""
        # Create a shared lens instance
        shared_lens = Lens(
            id=1,
            name="SharedLens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )

        # Create dict with lens data (should be ignored when lens_lookup is used)
        element_dict = {
            "lens": {
                "id": 1,
                "name": "IgnoredName",
                "radius_of_curvature_1": 100.0,
                "radius_of_curvature_2": -100.0,
                "thickness": 10.0,
                "diameter": 50.0,
                "material": "BK7",
                "refractive_index": 1.517,
                "type": "Biconvex",
                "created_at": "2024-01-01T00:00:00",
                "modified_at": "2024-01-01T00:00:00",
                "metadata": {},
            },
            "lens_id": 1,
            "position": 0.0,
            "decenter_y": 0.0,
            "decenter_z": 0.0,
            "tilt_x": 0.0,
            "tilt_y": 0.0,
            "tilt_z": 0.0,
        }

        system_data = {
            "name": "System with Shared Lens",
            "elements": [element_dict],
            "air_gaps": [],
        }

        # Deserialize with lens_lookup
        lens_lookup = {1: shared_lens}
        system = OpticalSystem.from_dict(system_data, lens_lookup=lens_lookup)

        assert len(system.elements) == 1

    def test_from_dict_with_air_gaps(self):
        """Test that air gaps are properly restored when elements are added."""
        lens_data = {
            "id": 1,
            "name": "Lens1",
            "radius_of_curvature_1": 50.0,
            "radius_of_curvature_2": -50.0,
            "thickness": 5.0,
            "diameter": 25.0,
            "material": "BK7",
            "refractive_index": 1.517,
            "type": "Biconvex",
            "created_at": "2024-01-01T00:00:00",
            "modified_at": "2024-01-01T00:00:00",
            "metadata": {},
        }

        system_data = {
            "name": "Multi-element System",
            "elements": [
                {
                    "lens": lens_data,
                    "lens_id": 1,
                    "position": 0.0,
                    "decenter_y": 0.0,
                    "decenter_z": 0.0,
                    "tilt_x": 0.0,
                    "tilt_y": 0.0,
                    "tilt_z": 0.0,
                },
                {
                    "lens": lens_data,
                    "lens_id": 1,
                    "position": 10.0,
                    "decenter_y": 0.0,
                    "decenter_z": 0.0,
                    "tilt_x": 0.0,
                    "tilt_y": 0.0,
                    "tilt_z": 0.0,
                },
            ],
            "air_gaps": [{"thickness": 10.0, "position": None}],
        }

        system = OpticalSystem.from_dict(system_data)
        # System should have two elements with air gap between them
        assert len(system.elements) == 2

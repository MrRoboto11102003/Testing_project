import pytest
import tempfile
from pathlib import Path
from openlens.database import DatabaseManager, UnsupportedSchemaVersionError
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem


class TestUnsupportedSchemaVersionError:
    """Tests for UnsupportedSchemaVersionError exception."""

    def test_exception_attributes(self):
        """Exception stores found, supported, and path attributes."""
        e = UnsupportedSchemaVersionError(9, 7, "test.db")
        assert e.found == 9
        assert e.supported == 7
        assert e.path == "test.db"

    def test_exception_is_runtime_error(self):
        """UnsupportedSchemaVersionError is a RuntimeError."""
        e = UnsupportedSchemaVersionError(8, 7, "db.sqlite")
        assert isinstance(e, RuntimeError)

    def test_exception_message_format(self):
        """Error message has the correct format."""
        e = UnsupportedSchemaVersionError(9, 7, "a.db")
        expected = (
            "Database 'a.db' has schema version 9, but this OpenLens only supports up to "
            "version 7. It was written by a newer OpenLens; upgrade OpenLens rather than "
            "opening this file with an older build, which would downgrade the library and "
            "lose data."
        )
        assert str(e) == expected
        assert len(e.args) == 1
        assert e.args[0] == expected


class TestDatabaseManagerLoadAll:
    """Tests for DatabaseManager.load_all."""

    @pytest.fixture
    def db_path(self, tmp_path):
        """Provide a temporary database path."""
        return str(tmp_path / "test.db")

    @pytest.fixture
    def db(self, db_path):
        """Create and return a DatabaseManager instance."""
        return DatabaseManager(db_path)

    def test_empty_database_returns_empty_list(self, db):
        """Empty database returns an empty list."""
        result = db.load_all()
        assert result == []
        assert isinstance(result, list)

    def test_library_lens_appears_in_results(self, db):
        """A library lens (is_library=1) appears in results."""
        lens_dict = {
            "id": "L1",
            "name": "Test Lens",
            "radius_of_curvature_1": 50.0,
            "radius_of_curvature_2": -50.0,
            "thickness": 5.0,
            "material": "BK7",
            "diameter": 25.0,
            "is_library": 1,
        }
        db.save_lens(lens_dict)
        result = db.load_all()
        assert len(result) == 1
        assert result[0]["id"] == "L1"
        assert result[0]["row_type"] == "Lens"

    def test_radius_fields_renamed(self, db):
        """Radius columns are renamed to radius_of_curvature_1/2."""
        lens_dict = {
            "id": "L2",
            "name": "Rename Test",
            "radius_of_curvature_1": 100.0,
            "radius_of_curvature_2": -100.0,
            "thickness": 10.0,
            "material": "BK7",
        }
        db.save_lens(lens_dict)
        result = db.load_all()
        assert "radius1" not in result[0]
        assert "radius2" not in result[0]
        assert result[0]["radius_of_curvature_1"] == 100.0
        assert result[0]["radius_of_curvature_2"] == -100.0

    def test_row_type_marker_set_after_metadata_merge(self, db):
        """row_type is "Lens" and set after metadata merge."""
        lens_dict = {
            "id": "L3",
            "name": "Marker Test",
            "radius_of_curvature_1": 50.0,
            "radius_of_curvature_2": -50.0,
            "thickness": 5.0,
            "material": "BK7",
            "metadata": {"type": "Biconvex"},
        }
        db.save_lens(lens_dict)
        result = db.load_all()
        assert result[0]["row_type"] == "Lens"
        assert isinstance(result[0]["row_type"], str)

    def test_metadata_merged_flat_onto_row(self, db):
        """Metadata blob is merged flat and preserved as metadata key."""
        lens_dict = {
            "id": "L4",
            "name": "Meta",
            "radius_of_curvature_1": 50.0,
            "radius_of_curvature_2": -50.0,
            "thickness": 5.0,
            "material": "BK7",
            "metadata": {"tol": 1, "note": "test"},
        }
        db.save_lens(lens_dict)
        result = db.load_all()
        assert result[0]["tol"] == 1
        assert result[0]["note"] == "test"
        assert result[0]["metadata"] == {"tol": 1, "note": "test"}

    def test_assembly_includes_elements_and_air_gaps(self, db):
        """Assembly dict includes elements and air_gaps lists."""
        lens_dict = {
            "id": "L5",
            "name": "Elem",
            "radius_of_curvature_1": 50.0,
            "radius_of_curvature_2": -50.0,
            "thickness": 5.0,
            "material": "BK7",
        }
        db.save_lens(lens_dict)
        assembly_dict = {
            "id": "A1",
            "name": "Assembly",
            "elements": [{"lens": lens_dict, "position": 0.0}],
            "air_gaps": [{"thickness": 10.0, "position": 5.0}],
        }
        db.save_assembly(assembly_dict)
        result = db.load_all()

        assembly = [r for r in result if r["row_type"] == "OpticalSystem"][0]
        assert assembly["name"] == "Assembly"
        assert len(assembly["elements"]) == 1
        assert len(assembly["air_gaps"]) == 1
        assert assembly["air_gaps"][0]["thickness"] == 10.0

    def test_assembly_element_lens_is_lookup_object(self, db):
        """Assembly element's lens is the same dict object from lookup."""
        lens_dict = {
            "id": "L6",
            "name": "Shared",
            "radius_of_curvature_1": 50.0,
            "radius_of_curvature_2": -50.0,
            "thickness": 5.0,
            "material": "BK7",
            "is_library": 1,
        }
        db.save_lens(lens_dict)
        assembly_dict = {
            "id": "A2",
            "name": "Assembly2",
            "elements": [{"lens": lens_dict, "position": 0.0}],
            "air_gaps": [],
        }
        db.save_assembly(assembly_dict)
        result = db.load_all()

        lens_row = result[0]
        assembly_row = result[1]
        assert assembly_row["elements"][0]["lens"] is lens_row

    def test_lenses_before_assemblies_in_results(self, db):
        """Lenses appear before assemblies in results."""
        lens_dict = {
            "id": "L7",
            "name": "First",
            "radius_of_curvature_1": 50.0,
            "radius_of_curvature_2": -50.0,
            "thickness": 5.0,
            "material": "BK7",
        }
        db.save_lens(lens_dict)
        assembly_dict = {
            "id": "A3",
            "name": "Second",
            "elements": [{"lens": lens_dict, "position": 0.0}],
            "air_gaps": [],
        }
        db.save_assembly(assembly_dict)
        result = db.load_all()

        assert result[0]["row_type"] == "Lens"
        assert result[1]["row_type"] == "OpticalSystem"


class TestLensToDict:
    """Tests for Lens.to_dict."""

    def test_basic_serialization(self):
        """to_dict returns expected field values."""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
            lens_type="Biconvex",
            material="BK7",
        )
        d = lens.to_dict()
        assert d["name"] == "Test"
        assert d["radius_of_curvature_1"] == 50.0
        assert d["refractive_index"] == 1.5
        assert d["type"] == "Biconvex"

    def test_to_dict_has_37_keys(self):
        """to_dict returns exactly 37 keys."""
        lens = Lens(
            name="L",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        d = lens.to_dict()
        assert len(d) == 37

    def test_no_wavelength_nm_in_to_dict(self):
        """to_dict does not include 'wavelength_nm' key."""
        lens = Lens(
            name="L",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        d = lens.to_dict()
        assert "wavelength_nm" not in d

    def test_coating_deep_copied_on_to_dict(self):
        """Coatings are deep copied to prevent mutation of lens."""
        coating = [{"material": "MgF2", "thickness_nm": 100}]
        lens = Lens(
            name="Coated",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
            coating_1=coating,
        )
        d = lens.to_dict()
        d["coating_1"][0]["thickness_nm"] = 200
        assert lens.coating_1[0]["thickness_nm"] == 100

    def test_metadata_shallow_copied_on_to_dict(self):
        """Metadata is shallow copied to prevent direct mutation."""
        metadata = {"tol": 1}
        lens = Lens(
            name="Meta",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
            metadata=metadata,
        )
        d = lens.to_dict()
        d["metadata"]["new_key"] = "new"
        assert "new_key" not in lens.metadata

    def test_edge_profile_to_list_of_lists(self):
        """Edge profile is converted to list of lists."""
        lens = Lens(
            name="Edge",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        lens.edge_profile = ((0.0, 12.5), (5.0, 12.5))
        d = lens.to_dict()
        assert d["edge_profile"] == [[0.0, 12.5], [5.0, 12.5]]

    def test_no_edge_profile_returns_none(self):
        """When edge_profile is not set, to_dict returns None."""
        lens = Lens(
            name="NoEdge",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        d = lens.to_dict()
        assert d["edge_profile"] is None

    def test_to_dict_returns_new_object(self):
        """Each to_dict call returns a new dict object."""
        lens = Lens(
            name="L",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        d1 = lens.to_dict()
        d2 = lens.to_dict()
        assert d1 == d2
        assert d1 is not d2


class TestOpticalSystemFromDict:
    """Tests for OpticalSystem.from_dict."""

    def test_from_dict_empty_creates_default_system(self):
        """Empty dict creates a system with default name."""
        system = OpticalSystem.from_dict({})
        assert system.name == "Optical System"
        assert len(system.elements) == 0
        assert len(system.air_gaps) == 0

    def test_from_dict_preserves_name(self):
        """System name is taken from dict."""
        system = OpticalSystem.from_dict({"name": "My System"})
        assert system.name == "My System"

    def test_from_dict_metadata_filtered(self):
        """Metadata keys in _SYSTEM_SERIALIZED_KEYS are filtered out."""
        data = {
            "metadata": {
                "name": "ignored",
                "tol": 1,
            }
        }
        system = OpticalSystem.from_dict(data)
        assert system.metadata.get("tol") == 1

    def test_from_dict_adds_lens_elements(self):
        """Elements from dict are added to system."""
        lens = Lens(
            name="L1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        data = {
            "elements": [
                {
                    "lens": lens.to_dict(),
                    "position": 0.0,
                    "decenter_y": 0.0,
                    "decenter_z": 0.0,
                    "tilt_x": 0.0,
                    "tilt_y": 0.0,
                    "tilt_z": 0.0,
                }
            ],
            "air_gaps": [],
        }
        system = OpticalSystem.from_dict(data)
        assert len(system.elements) == 1

    def test_from_dict_lens_lookup_by_identity(self):
        """When lens_lookup provided, lenses reused by identity."""
        lens = Lens(
            name="Shared",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        lens_dict = lens.to_dict()
        data = {
            "elements": [
                {
                    "lens": lens_dict,
                    "lens_id": lens.id,
                    "position": 0.0,
                    "decenter_y": 0.0,
                    "decenter_z": 0.0,
                    "tilt_x": 0.0,
                    "tilt_y": 0.0,
                    "tilt_z": 0.0,
                }
            ],
            "air_gaps": [],
        }
        lookup = {lens.id: lens}
        system = OpticalSystem.from_dict(data, lens_lookup=lookup)
        assert system.elements[0].lens is lens

    def test_from_dict_preserves_aperture_stop(self):
        """Aperture stop values are stored from dict."""
        data = {
            "aperture_stop_gap": 2,
            "aperture_stop_diameter": 50.0,
        }
        system = OpticalSystem.from_dict(data)
        assert system.aperture_stop_gap == 2
        assert system.aperture_stop_diameter == 50.0

    def test_from_dict_preserves_timestamps(self):
        """id and timestamps are preserved from dict."""
        test_id = "test-id-123"
        test_created = "2026-01-01T00:00:00"
        test_modified = "2026-01-02T00:00:00"
        data = {
            "id": test_id,
            "created_at": test_created,
            "modified_at": test_modified,
        }
        system = OpticalSystem.from_dict(data)
        assert system.id == test_id
        assert system.created_at == test_created
        assert system.modified_at == test_modified

import math
import pytest
import tempfile
from pathlib import Path

from openlens.database import DatabaseManager, UnsupportedSchemaVersionError
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem


class TestUnsupportedSchemaVersionError:
    """Test UnsupportedSchemaVersionError exception."""

    def test_construction_and_attributes(self):
        """Test that constructor stores attributes and is a RuntimeError."""
        e = UnsupportedSchemaVersionError(9, 7, "test.db")
        assert e.found == 9
        assert e.supported == 7
        assert e.path == "test.db"
        assert isinstance(e, RuntimeError)

    def test_error_message_format(self):
        """Test that error message contains expected parts."""
        e = UnsupportedSchemaVersionError(9, 7, "my.db")
        msg = str(e)
        assert "my.db" in msg
        assert "9" in msg
        assert "7" in msg
        assert "newer OpenLens" in msg


class TestDatabaseManagerLoadAll:
    """Test DatabaseManager.load_all functionality."""

    def test_load_all_empty_database(self):
        """Test load_all on a fresh empty database returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            manager = DatabaseManager(db_path)
            result = manager.load_all()
            assert result == []

    def test_load_all_library_lens_with_radius_renaming(self):
        """Test that library lenses are returned and radius columns are renamed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            manager = DatabaseManager(db_path)
            
            lens_dict = {
                "id": "lens1",
                "name": "Test Lens",
                "radius_of_curvature_1": 50.0,
                "radius_of_curvature_2": -50.0,
                "thickness": 5.0,
                "diameter": 25.0,
                "material": "BK7",
                "refractive_index": 1.5,
            }
            manager.save_lens(lens_dict)
            
            result = manager.load_all()
            assert len(result) == 1
            assert result[0]["row_type"] == "Lens"
            assert result[0]["name"] == "Test Lens"
            # Verify radius renaming
            assert "radius1" not in result[0]
            assert "radius2" not in result[0]
            assert result[0]["radius_of_curvature_1"] == 50.0
            assert result[0]["radius_of_curvature_2"] == -50.0

    def test_load_all_metadata_merge_flat(self):
        """Test that metadata blob is merged flat and row_type set after merge."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            manager = DatabaseManager(db_path)
            
            lens_dict = {
                "id": "lens1",
                "name": "Test",
                "radius_of_curvature_1": 50.0,
                "radius_of_curvature_2": -50.0,
                "thickness": 5.0,
                "diameter": 25.0,
                "material": "BK7",
                "refractive_index": 1.5,
                "type": "Biconvex",
                "metadata": {"custom_key": "custom_value", "row_type": "Spoof"},
            }
            manager.save_lens(lens_dict)
            
            result = manager.load_all()
            lens = result[0]
            
            # Metadata flattened onto row
            assert lens["custom_key"] == "custom_value"
            assert lens["metadata"]["custom_key"] == "custom_value"
            # row_type set after merge (cannot be spoofed)
            assert lens["row_type"] == "Lens"
            # But type (lens type) is preserved
            assert lens["type"] == "Biconvex"

    def test_load_all_assembly_with_elements_and_gaps(self):
        """Test that assemblies are loaded with elements and air gaps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            manager = DatabaseManager(db_path)
            
            lens_dict = {
                "id": "lens1",
                "name": "L1",
                "radius_of_curvature_1": 50.0,
                "radius_of_curvature_2": -50.0,
                "thickness": 5.0,
                "diameter": 25.0,
                "material": "BK7",
                "refractive_index": 1.5,
            }
            manager.save_lens(lens_dict)
            
            assembly_dict = {
                "id": "asm1",
                "name": "Assembly",
                "elements": [{"lens": lens_dict, "position": 0.0}],
                "air_gaps": [{"thickness": 10.0, "position": 5.0}],
            }
            manager.save_assembly(assembly_dict)
            
            result = manager.load_all()
            assemblies = [r for r in result if r["row_type"] == "OpticalSystem"]
            assert len(assemblies) == 1
            
            assembly = assemblies[0]
            assert assembly["name"] == "Assembly"
            assert len(assembly["elements"]) == 1
            assert assembly["elements"][0]["lens"]["name"] == "L1"
            assert len(assembly["air_gaps"]) == 1
            assert assembly["air_gaps"][0]["thickness"] == 10.0


class TestLensToDict:
    """Test Lens.to_dict functionality."""

    def test_to_dict_new_dict_each_call(self):
        """Test that to_dict returns a new dict on each call."""
        lens = Lens(
            name="Test",
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

    def test_to_dict_basic_serialization(self):
        """Test that to_dict includes essential lens properties."""
        lens = Lens(
            name="TestLens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        d = lens.to_dict()
        
        assert d["name"] == "TestLens"
        assert d["radius_of_curvature_1"] == 50.0
        assert d["radius_of_curvature_2"] == -50.0
        assert d["thickness"] == 5.0
        assert d["diameter"] == 25.0
        assert d["refractive_index"] == 1.5
        assert "id" in d
        assert "created_at" in d
        assert "modified_at" in d

    def test_to_dict_coating_deep_copied(self):
        """Test that coatings are deep copied, not referenced."""
        coating = [{"material": "MgF2", "refractive_index": 1.38, "thickness_nm": 100.0}]
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
        
        # Coating is copied
        assert d["coating_1"] is not None
        assert d["coating_1"] is not lens.coating_1
        
        # Mutation of output does not affect lens
        d["coating_1"][0]["thickness_nm"] = 999.0
        assert lens.coating_1[0]["thickness_nm"] == 100.0

    def test_to_dict_metadata_shallow_copy(self):
        """Test that metadata is shallow-copied (dict is new, nested values shared)."""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
            metadata={"tol": [1, 2, 3], "single": "value"},
        )
        d = lens.to_dict()
        
        # Dict itself is new
        assert d["metadata"] is not lens.metadata
        # But nested list is shared
        assert d["metadata"]["tol"] is lens.metadata["tol"]
        # Mutations of keys don't affect lens
        d["metadata"]["new_key"] = "new"
        assert "new_key" not in lens.metadata

    def test_to_dict_zero_radius_to_infinity(self):
        """Test that zero radius is converted to infinity."""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=0.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        d = lens.to_dict()
        
        assert math.isinf(d["radius_of_curvature_1"])


class TestOpticalSystemFromDict:
    """Test OpticalSystem.from_dict functionality."""

    def test_from_dict_empty_creates_default_system(self):
        """Test from_dict with empty dict creates system with defaults."""
        system = OpticalSystem.from_dict({})
        
        assert system.name == "Optical System"
        assert len(system.elements) == 0
        assert len(system.air_gaps) == 0
        assert system.metadata == {}

    def test_from_dict_flat_path_restores_positions(self):
        """Test that saved element positions are restored in flat path."""
        lens1 = Lens(
            name="L1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        lens2 = Lens(
            name="L2",
            radius_of_curvature_1=60.0,
            radius_of_curvature_2=-60.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        
        d = {
            "name": "TestSystem",
            "elements": [
                {"lens": lens1.to_dict(), "position": 10.0},
                {"lens": lens2.to_dict(), "position": 25.0},
            ],
            "air_gaps": [{"thickness": 10.0}],
        }
        
        system = OpticalSystem.from_dict(d)
        
        assert system.name == "TestSystem"
        assert len(system.elements) == 2
        # Positions should be restored from saved values
        assert system.elements[0].position == 10.0
        assert system.elements[1].position == 25.0

    def test_from_dict_lookup_reuse_identity(self):
        """Test that lens_lookup causes lens identity reuse."""
        lens = Lens(
            name="L1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        
        d = {
            "elements": [{"lens": lens.to_dict()}],
            "air_gaps": [],
        }
        
        lookup = {lens.id: lens}
        system = OpticalSystem.from_dict(d, lens_lookup=lookup)
        
        # Lens from lookup should be the same object by identity
        assert system.elements[0].lens is lens

    def test_from_dict_missing_lens_dict_raises_key_error(self):
        """Test that missing lens dict raises KeyError."""
        d = {
            "elements": [{"lens_id": "unknown"}],  # No 'lens' key
            "air_gaps": [],
        }
        
        with pytest.raises(KeyError):
            OpticalSystem.from_dict(d)

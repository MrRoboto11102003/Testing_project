import pytest
from openlens.material_database import MaterialDatabase, MaterialProperties
import tempfile
from pathlib import Path


@pytest.fixture
def temp_db_path():
    """Create a temporary path for a test database."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield str(Path(tmp_dir) / "materials.json")


@pytest.fixture
def db(temp_db_path):
    """Create an isolated test database with a unique path."""
    return MaterialDatabase(db_path=temp_db_path)


class TestClearCache:
    """Test MaterialDatabase.clear_cache() behavior."""

    def test_clear_cache_returns_none(self, db):
        """clear_cache() returns None."""
        assert db.clear_cache() is None

    def test_clear_cache_empties_index_cache(self, db):
        """After clear_cache(), _index_cache is an empty dict."""
        # Populate cache
        db.get_refractive_index("BK7", 587.6)
        assert db._index_cache  # Not empty
        
        # Clear
        db.clear_cache()
        assert db._index_cache == {}

    def test_clear_cache_resets_alias_keys_to_none(self, db):
        """After clear_cache(), _alias_keys is None."""
        # Trigger alias map building
        db.get_material("BK7")
        
        # Clear
        db.clear_cache()
        assert db._alias_keys is None

    def test_clear_cache_safe_on_empty_cache(self, db):
        """clear_cache() doesn't raise when cache is already empty."""
        # Should not raise
        db.clear_cache()
        db.clear_cache()

    def test_clear_cache_invalidates_stale_index(self, db):
        """After material replacement and clear_cache(), indices are recomputed.
        
        Replace BK7 with a material whose n ≈ 1.0 (zero B coefficients).
        Demonstrates that clear_cache() invalidates the stale cached value.
        """
        # Get original index (≈1.5168)
        original_index = db.get_refractive_index("BK7", 587.6)
        assert original_index > 1.5
        
        # Replace with zero-B material
        db.materials["BK7"] = MaterialProperties(
            name="BK7", catalog="X", nd=1.6, vd=40.0
        )
        
        # Stale cache: returns original value
        stale = db.get_refractive_index("BK7", 587.6)
        assert stale == original_index
        
        # After clear: recomputed to ~1.0
        db.clear_cache()
        fresh = db.get_refractive_index("BK7", 587.6)
        assert fresh == pytest.approx(1.0, abs=1e-6)

    def test_clear_cache_per_instance_isolation(self, temp_db_path):
        """clear_cache() on db1 doesn't affect db2's cache.
        
        This tests the documented fix: one instance clearing doesn't wipe
        every other instance's cache.
        """
        db1 = MaterialDatabase(db_path=temp_db_path)
        db2 = MaterialDatabase(db_path=temp_db_path + ".2")
        
        # Warm both caches
        idx1 = db1.get_refractive_index("BK7", 587.6)
        idx2 = db2.get_refractive_index("BK7", 587.6)
        assert idx1 > 1.5
        assert idx2 > 1.5
        
        # Replace material on both
        db1.materials["BK7"] = MaterialProperties(
            name="BK7", catalog="X", nd=1.6, vd=40.0
        )
        db2.materials["BK7"] = MaterialProperties(
            name="BK7", catalog="X", nd=1.6, vd=40.0
        )
        
        # Clear only db1
        db1.clear_cache()
        
        # db1 returns fresh ~1.0
        fresh1 = db1.get_refractive_index("BK7", 587.6)
        assert fresh1 == pytest.approx(1.0, abs=1e-6)
        
        # db2 still cached (old value)
        stale2 = db2.get_refractive_index("BK7", 587.6)
        assert stale2 == pytest.approx(idx2, abs=1e-6)

    def test_clear_cache_resets_alias_lookup(self, db):
        """After clear_cache(), alias map is rebuilt on next lookup miss.
        
        This demonstrates that _alias_keys = None forces the alias map rebuild.
        """
        # Lookup non-existent material (builds alias map if needed)
        assert db.get_material("foo bar") is None
        
        # Add a material with normalized-compatible name
        material = MaterialProperties(
            name="FOO_BAR", catalog="X", nd=1.5, vd=50.0
        )
        db.materials["FOO_BAR"] = material
        
        # Stale alias map: doesn't find it
        assert db.get_material("foo bar") is None
        
        # After clear_cache: alias map rebuilt, now finds it
        db.clear_cache()
        assert db.get_material("foo bar") is material

    def test_clear_cache_does_not_modify_materials_dict(self, db):
        """clear_cache() doesn't add or remove materials."""
        original_keys = set(db.materials.keys())
        db.get_refractive_index("BK7", 587.6)
        db.clear_cache()
        assert set(db.materials.keys()) == original_keys

    def test_clear_cache_preserves_material_identity(self, db):
        """Material objects in materials dict retain their identity."""
        bk7_id_before = id(db.materials["BK7"])
        db.get_refractive_index("BK7", 587.6)
        db.clear_cache()
        bk7_id_after = id(db.materials["BK7"])
        assert bk7_id_before == bk7_id_after

    def test_clear_cache_idempotent(self, db):
        """Calling clear_cache() multiple times is safe."""
        db.get_refractive_index("BK7", 587.6)
        db.clear_cache()
        db.clear_cache()
        db.clear_cache()
        assert db._index_cache == {}
        assert db._alias_keys is None

    def test_clear_cache_allows_rebuild(self, db):
        """After clear_cache(), cache can be rebuilt immediately."""
        db.get_refractive_index("BK7", 587.6)
        db.clear_cache()
        
        # Can rebuild without issues
        idx = db.get_refractive_index("BK7", 587.6)
        assert idx > 1.5
        assert len(db._index_cache) > 0

    def test_clear_cache_with_multiple_entries(self, db):
        """clear_cache() clears all entries in the cache."""
        # Populate cache with multiple entries
        db.get_refractive_index("BK7", 587.6)
        db.get_refractive_index("BK7", 656.3)
        db.get_refractive_index("N-BK7", 587.6)
        
        cache_size = len(db._index_cache)
        assert cache_size >= 3
        
        db.clear_cache()
        assert len(db._index_cache) == 0

    def test_clear_cache_with_added_material(self, db):
        """After clear_cache(), manually added materials are accessible."""
        material = MaterialProperties(
            name="CUSTOM", catalog="TEST", nd=1.5, vd=50.0
        )
        db.materials["CUSTOM"] = material
        db.clear_cache()
        
        # Should be found
        assert db.get_material("CUSTOM") is material

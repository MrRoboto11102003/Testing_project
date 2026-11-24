"""
pytest tests for openlens.material_database.MaterialDatabase.clear_cache()
"""
import pytest
import tempfile
from pathlib import Path
from openlens.material_database import MaterialDatabase, MaterialProperties


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for isolated testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir) / "test_materials.json"


@pytest.fixture
def fresh_db(temp_db_path):
    """Create a fresh MaterialDatabase instance with isolated storage."""
    return MaterialDatabase(db_path=str(temp_db_path))


class TestMaterialDatabaseClearCache:
    """Test suite for MaterialDatabase.clear_cache()"""

    def test_clear_cache_returns_none(self, fresh_db):
        """clear_cache should return None."""
        result = fresh_db.clear_cache()
        assert result is None

    def test_clear_cache_empties_index_cache(self, fresh_db):
        """After clear_cache, the index cache should be empty."""
        # Populate cache with a lookup
        fresh_db.get_refractive_index("BK7", 587.6, 20.0)
        assert len(fresh_db._index_cache) > 0, "Cache should be populated"
        
        fresh_db.clear_cache()
        assert fresh_db._index_cache == {}, "Index cache should be empty after clear"

    def test_clear_cache_resets_alias_keys(self, fresh_db):
        """After clear_cache, the alias keys map should be None."""
        # Build alias map via lookup
        fresh_db.get_material("bk7")
        assert fresh_db._alias_keys is not None, "Alias keys should be built"
        
        fresh_db.clear_cache()
        assert fresh_db._alias_keys is None, "Alias keys should be None after clear"

    def test_clear_cache_safe_on_empty_cache(self, fresh_db):
        """clear_cache should not raise on an empty cache."""
        # Don't populate cache - just call clear
        fresh_db.clear_cache()
        assert fresh_db._index_cache == {}

    def test_clear_cache_safe_repeated_calls(self, fresh_db):
        """clear_cache should be safe to call multiple times."""
        fresh_db.get_refractive_index("BK7", 587.6, 20.0)
        
        # Call multiple times - should not raise
        fresh_db.clear_cache()
        fresh_db.clear_cache()
        fresh_db.clear_cache()
        
        assert fresh_db._index_cache == {}
        assert fresh_db._alias_keys is None

    def test_clear_cache_does_not_modify_materials_dict(self, fresh_db):
        """clear_cache should not modify the materials dict."""
        original_keys = set(fresh_db.materials.keys())
        original_count = len(fresh_db.materials)
        
        fresh_db.clear_cache()
        
        assert set(fresh_db.materials.keys()) == original_keys, "Keys should not change"
        assert len(fresh_db.materials) == original_count, "Count should not change"

    def test_per_instance_isolation(self, temp_db_path):
        """Clearing cache on one instance should not affect another."""
        db1 = MaterialDatabase(db_path=str(temp_db_path))
        db2 = MaterialDatabase(db_path=str(temp_db_path))
        
        # Populate both caches
        db1.get_refractive_index("BK7", 587.6, 20.0)
        db2.get_refractive_index("BK7", 587.6, 20.0)
        
        db1_cache_size = len(db1._index_cache)
        db2_cache_size = len(db2._index_cache)
        assert db1_cache_size > 0 and db2_cache_size > 0, "Both caches populated"
        
        # Clear only db1
        db1.clear_cache()
        
        assert db1._index_cache == {}, "db1 cache should be empty"
        assert len(db2._index_cache) == db2_cache_size, "db2 cache should be unchanged"

    def test_stale_cache_cleared_forces_recomputation(self, fresh_db):
        """Clearing stale cache should force recomputation with updated material."""
        # Get original cached value
        original_value = fresh_db.get_refractive_index("BK7", 587.6, 20.0)
        assert original_value is not None
        
        # Replace BK7 with a material with zero B coefficients
        fresh_db.materials["BK7"] = MaterialProperties(
            name="BK7", catalog="TEST", nd=1.6, vd=40.0
        )
        
        # Without clearing, should return cached value
        stale_value = fresh_db.get_refractive_index("BK7", 587.6, 20.0)
        assert stale_value == original_value, "Should return stale cached value"
        
        # Clear cache and recompute
        fresh_db.clear_cache()
        fresh_value = fresh_db.get_refractive_index("BK7", 587.6, 20.0)
        
        # Fresh value should differ (much lower with zero B coefficients)
        assert fresh_value != original_value, "Should recompute with new material"
        assert fresh_value < original_value, "New material should produce lower index"

    def test_alias_map_rebuilt_after_clear(self, fresh_db):
        """Alias map should be rebuilt after clear_cache."""
        # Build and clear alias map
        fresh_db.get_material("bk7")
        fresh_db.clear_cache()
        
        # Next lookup should rebuild it and still find the material
        result = fresh_db.get_material("bk7")
        assert result is not None, "Should find BK7 by alias after clear"
        assert fresh_db._alias_keys is not None, "Alias map should be rebuilt"

    def test_cache_repopulates_after_clear(self, fresh_db):
        """Cache should repopulate after clear_cache."""
        fresh_db.get_refractive_index("BK7", 587.6, 20.0)
        fresh_db.clear_cache()
        
        assert fresh_db._index_cache == {}, "Cache should be empty after clear"
        
        # Next lookup should repopulate cache
        fresh_db.get_refractive_index("BK7", 587.6, 20.0)
        assert len(fresh_db._index_cache) > 0, "Cache should be repopulated after lookup"

    def test_multiple_cached_items_cleared(self, fresh_db):
        """clear_cache should clear all cached items."""
        # Populate cache with multiple wavelengths
        wavelengths = [400.0, 550.0, 587.6, 700.0]
        for wl in wavelengths:
            fresh_db.get_refractive_index("BK7", wl, 20.0)
        
        cache_size_before = len(fresh_db._index_cache)
        assert cache_size_before >= len(wavelengths), "Should have multiple entries"
        
        fresh_db.clear_cache()
        
        assert len(fresh_db._index_cache) == 0, "All cache entries should be cleared"

    def test_added_material_accessible_after_clear(self, fresh_db):
        """Materials added directly should be accessible after clear_cache."""
        # Add material directly to dict
        new_material = MaterialProperties(
            name="CUSTOM_GLASS", catalog="TEST", nd=1.5, vd=50.0
        )
        fresh_db.materials["CUSTOM_GLASS"] = new_material
        
        # Clear to rebuild alias map
        fresh_db.clear_cache()
        
        # Should be findable by case-insensitive lookup
        found = fresh_db.get_material("custom_glass")
        assert found is not None, "Should find added material after clear_cache"
        assert found.name == "CUSTOM_GLASS"

    def test_material_identity_preserved_after_clear(self, fresh_db):
        """Object identities in materials dict should be preserved."""
        bk7_before = fresh_db.materials.get("BK7")
        fresh_db.clear_cache()
        bk7_after = fresh_db.materials.get("BK7")
        
        assert bk7_before is bk7_after, "Object identity should be preserved"

    def test_cache_state_transitions(self, fresh_db):
        """Verify complete cache state transitions through lifecycle."""
        # Initial state
        assert fresh_db._index_cache == {}, "Cache should start empty"
        assert fresh_db._alias_keys is None, "Alias keys should start as None"
        
        # After lookup
        fresh_db.get_refractive_index("BK7", 587.6, 20.0)
        assert len(fresh_db._index_cache) > 0, "Cache populated after lookup"
        
        # After clear
        fresh_db.clear_cache()
        assert fresh_db._index_cache == {}, "Cache empty after clear"
        assert fresh_db._alias_keys is None, "Alias keys None after clear"
        
        # After second lookup
        fresh_db.get_refractive_index("BK7", 587.6, 20.0)
        assert len(fresh_db._index_cache) > 0, "Cache repopulated after second lookup"

import pytest
from openlens.material_database import MaterialDatabase


class TestMaterialDatabaseClearCache:
    """Tests for MaterialDatabase.clear_cache() method."""

    def test_clear_cache_empties_index_cache(self):
        """Test that clear_cache() empties the _index_cache."""
        db = MaterialDatabase()
        
        # Populate cache by accessing refractive index
        db.get_refractive_index("BK7")
        assert len(db._index_cache) > 0, "Cache should be populated"
        
        # Clear the cache
        db.clear_cache()
        
        # Verify cache is empty
        assert len(db._index_cache) == 0, "Cache should be empty after clear_cache()"

    def test_clear_cache_sets_alias_keys_to_none(self):
        """Test that clear_cache() sets _alias_keys to None."""
        db = MaterialDatabase()
        
        # Access materials to populate alias_keys
        db.list_materials()
        
        # Clear cache
        db.clear_cache()
        
        # Verify _alias_keys is None
        assert db._alias_keys is None, "_alias_keys should be None after clear_cache()"

    def test_clear_cache_does_not_affect_other_instances(self):
        """Test that clearing one instance's cache doesn't affect another instance."""
        db1 = MaterialDatabase()
        db2 = MaterialDatabase()
        
        # Populate both instances' caches
        db1.get_refractive_index("BK7")
        db2.get_refractive_index("BK7")
        
        assert len(db1._index_cache) > 0
        assert len(db2._index_cache) > 0
        
        # Clear only db1's cache
        db1.clear_cache()
        
        # Verify db1 is cleared
        assert len(db1._index_cache) == 0
        
        # Verify db2 is not affected
        assert len(db2._index_cache) > 0

    def test_clear_cache_on_empty_cache(self):
        """Test that clear_cache() handles empty caches gracefully."""
        db = MaterialDatabase()
        
        # Cache should be empty on initialization
        assert len(db._index_cache) == 0
        
        # Calling clear_cache should not raise an error
        db.clear_cache()
        
        # Cache should remain empty
        assert len(db._index_cache) == 0

    def test_clear_cache_idempotent(self):
        """Test that clear_cache() can be called multiple times safely."""
        db = MaterialDatabase()
        
        # Populate cache
        db.get_refractive_index("BK7")
        
        # Clear multiple times without error
        db.clear_cache()
        db.clear_cache()
        db.clear_cache()
        
        # Cache should be empty
        assert len(db._index_cache) == 0

    def test_clear_cache_allows_recaching(self):
        """Test that cache can be repopulated after clearing."""
        db = MaterialDatabase()
        
        # Populate, clear, and repopulate
        db.get_refractive_index("BK7")
        initial_cache_size = len(db._index_cache)
        assert initial_cache_size > 0
        
        db.clear_cache()
        assert len(db._index_cache) == 0
        
        # Repopulate cache
        db.get_refractive_index("BK7")
        assert len(db._index_cache) > 0

    def test_clear_cache_clears_multiple_entries(self):
        """Test that all cache entries are cleared, not just one."""
        db = MaterialDatabase()
        
        # Populate cache with multiple materials/wavelengths
        db.get_refractive_index("BK7")
        db.get_refractive_index("SF2")
        
        cache_size = len(db._index_cache)
        assert cache_size > 0
        
        # Clear and verify all are gone
        db.clear_cache()
        assert len(db._index_cache) == 0

    def test_clear_cache_full_state_reset(self):
        """Test that clear_cache() resets both cache and alias_keys."""
        db = MaterialDatabase()
        
        # Populate both
        db.get_refractive_index("BK7")
        db.list_materials()
        
        # Clear
        db.clear_cache()
        
        # Both should be in cleared state
        assert len(db._index_cache) == 0
        assert db._alias_keys is None

    def test_clear_cache_instance_isolation_multiple_instances(self):
        """Test that three instances maintain independent cache states."""
        db1 = MaterialDatabase()
        db2 = MaterialDatabase()
        db3 = MaterialDatabase()
        
        # Populate all three
        db1.get_refractive_index("BK7")
        db2.get_refractive_index("BK7")
        db3.get_refractive_index("BK7")
        
        # Clear middle instance
        db2.clear_cache()
        
        # Verify isolation
        assert len(db1._index_cache) > 0
        assert len(db2._index_cache) == 0
        assert len(db3._index_cache) > 0

    def test_clear_cache_no_side_effects_on_fresh_instance(self):
        """Test that clear_cache() doesn't cause errors on unused instance."""
        db = MaterialDatabase()
        
        # Don't populate anything, just clear
        db.clear_cache()
        
        # Should not crash and should maintain valid state
        assert len(db._index_cache) == 0
        assert db._alias_keys is None

    def test_clear_cache_behavior_isolation_documented(self):
        """Test that clear_cache() is instance-specific, not class-wide.
        
        This verifies the fix mentioned in the docstring: previously 
        clear_cache() used a class-wide lru_cache that would wipe 
        all instances. This test ensures that no longer happens.
        """
        db1 = MaterialDatabase()
        db2 = MaterialDatabase()
        
        # Populate both
        db1.get_refractive_index("BK7")
        db2.get_refractive_index("BK7")
        
        db1_initial = len(db1._index_cache)
        db2_initial = len(db2._index_cache)
        
        # Clear db1
        db1.clear_cache()
        
        # db1 should be cleared but db2 should not be
        assert len(db1._index_cache) == 0
        assert len(db2._index_cache) == db2_initial

    def test_clear_cache_empty_then_populate(self):
        """Test that after clearing, the instance can be used normally."""
        db = MaterialDatabase()
        
        # Clear empty cache
        db.clear_cache()
        
        # Should be able to use normally after clearing
        db.get_refractive_index("BK7")
        assert len(db._index_cache) > 0

    def test_clear_cache_sequential_clear_populate_cycles(self):
        """Test that multiple clear/populate cycles work correctly."""
        db = MaterialDatabase()
        
        for cycle in range(3):
            # Populate
            db.get_refractive_index("BK7")
            assert len(db._index_cache) > 0, f"Cache not populated in cycle {cycle}"
            
            # Clear
            db.clear_cache()
            assert len(db._index_cache) == 0, f"Cache not cleared in cycle {cycle}"
            assert db._alias_keys is None, f"alias_keys not None in cycle {cycle}"

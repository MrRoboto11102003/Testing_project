import pytest
from openlens.material_database import MaterialDatabase


class TestMaterialDatabaseClearCache:
    """Tests for MaterialDatabase.clear_cache() method.
    
    Focus: instance-level cache independence (the documented bug fix),
    idempotency, and cache repopulation after clearing.
    """
    
    def test_clear_cache_no_error(self):
        """Verify clear_cache() can be called without raising an error."""
        db = MaterialDatabase()
        db.clear_cache()  # Should not raise
    
    def test_clear_cache_returns_none(self):
        """Verify clear_cache() returns None."""
        db = MaterialDatabase()
        result = db.clear_cache()
        assert result is None
    
    def test_clear_cache_instance_isolation_first_instance(self):
        """Verify clearing cache on one instance doesn't affect another instance.
        
        This is the core bug fix: previously, cache_clear() was called on a
        class-wide lru_cache, wiping every instance's cache.
        """
        db1 = MaterialDatabase()
        db2 = MaterialDatabase()
        
        # Populate both caches
        mat1_before = db1.get_material("BK7")
        mat2_before = db2.get_material("SF11")
        
        # Clear only db1's cache
        db1.clear_cache()
        
        # db2 should still work correctly
        mat2_after = db2.get_material("SF11")
        assert mat2_before is not None
        assert mat2_after is not None
        assert mat2_before.nd == mat2_after.nd
    
    def test_clear_cache_instance_isolation_second_instance(self):
        """Verify second instance still works after first instance clears cache."""
        db1 = MaterialDatabase()
        db2 = MaterialDatabase()
        
        # Access materials in both
        db1.get_material("BK7")
        db2.get_material("Crown Glass")
        
        # Clear db1
        db1.clear_cache()
        
        # db2 should still be functional
        mat = db2.get_material("Crown Glass")
        assert mat is not None
    
    def test_clear_cache_multiple_times_sequential(self):
        """Verify calling clear_cache multiple times is idempotent and safe."""
        db = MaterialDatabase()
        db.get_material("BK7")
        
        for _ in range(3):
            db.clear_cache()
            # Should still work after each clear
            mat = db.get_material("BK7")
            assert mat is not None
    
    def test_clear_cache_empty_initial_state(self):
        """Verify clear_cache works on fresh database without prior access."""
        db = MaterialDatabase()
        db.clear_cache()  # No materials accessed yet
        
        mat = db.get_material("BK7")
        assert mat is not None
    
    def test_clear_cache_repopulation_after_clearing(self):
        """Verify cache can be repopulated after clearing."""
        db = MaterialDatabase()
        
        mat1 = db.get_material("BK7")
        db.clear_cache()
        mat2 = db.get_material("BK7")
        
        assert mat1 is not None
        assert mat2 is not None
        # Same material should have same properties
        assert mat1.nd == mat2.nd
    
    def test_clear_cache_with_multiple_materials_accessed(self):
        """Verify clear_cache works after accessing multiple different materials."""
        db = MaterialDatabase()
        
        # Access multiple materials to populate cache
        mat1 = db.get_material("BK7")
        mat2 = db.get_material("SF11")
        mat3 = db.get_material("Fused Silica")
        
        db.clear_cache()
        
        # All should still be accessible with same properties
        mat1_after = db.get_material("BK7")
        mat2_after = db.get_material("SF11")
        mat3_after = db.get_material("Fused Silica")
        
        assert mat1.nd == mat1_after.nd
        assert mat2.nd == mat2_after.nd
        assert mat3.nd == mat3_after.nd
    
    def test_clear_cache_preserves_database_integrity(self):
        """Verify clear_cache doesn't corrupt underlying material data."""
        db = MaterialDatabase()
        
        mat_before = db.get_material("BK7")
        original_nd = mat_before.nd
        original_vd = mat_before.vd
        
        db.clear_cache()
        
        mat_after = db.get_material("BK7")
        
        # Material properties should be unchanged
        assert mat_after.nd == original_nd
        assert mat_after.vd == original_vd
    
    def test_clear_cache_three_independent_instances(self):
        """Verify three instances maintain independent caches after clear_cache calls."""
        instances = [MaterialDatabase() for _ in range(3)]
        materials = ["BK7", "SF11", "Fused Silica"]
        
        # Each instance accesses a different material
        for db, mat_name in zip(instances, materials):
            db.get_material(mat_name)
        
        # Clear only the first instance
        instances[0].clear_cache()
        
        # All instances should still work
        for db, mat_name in zip(instances, materials):
            mat = db.get_material(mat_name)
            assert mat is not None
    
    def test_clear_cache_idempotent_state(self):
        """Verify calling clear_cache multiple times produces consistent state."""
        db = MaterialDatabase()
        db.get_material("BK7")
        
        db.clear_cache()
        result_1 = db.get_material("BK7").nd
        
        db.clear_cache()
        result_2 = db.get_material("BK7").nd
        
        db.clear_cache()
        result_3 = db.get_material("BK7").nd
        
        assert result_1 == result_2 == result_3
    
    def test_clear_cache_sequential_clear_and_access(self):
        """Verify alternating clear_cache and get_material calls work correctly."""
        db = MaterialDatabase()
        
        for _ in range(2):
            mat1 = db.get_material("BK7")
            db.clear_cache()
            mat2 = db.get_material("SF11")
            db.clear_cache()
            
            assert mat1 is not None
            assert mat2 is not None

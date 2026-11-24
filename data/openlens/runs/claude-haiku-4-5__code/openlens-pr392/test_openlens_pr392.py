import pytest
from openlens.material_database import MaterialDatabase


class TestMaterialDatabaseClearCache:
    """Test suite for MaterialDatabase.clear_cache() method."""
    
    def test_clear_cache_exists_and_callable(self):
        """Test that clear_cache is a callable method."""
        db = MaterialDatabase()
        assert hasattr(db, 'clear_cache')
        assert callable(db.clear_cache)
    
    def test_clear_cache_returns_none(self):
        """Test that clear_cache() returns None."""
        db = MaterialDatabase()
        result = db.clear_cache()
        assert result is None
    
    def test_clear_cache_no_arguments(self):
        """Test that clear_cache() takes no arguments."""
        db = MaterialDatabase()
        db.clear_cache()  # Should not raise
        
        with pytest.raises(TypeError):
            db.clear_cache('unexpected_arg')
    
    def test_clear_cache_empties_index_cache(self):
        """Test that clear_cache() clears the _index_cache dictionary."""
        db = MaterialDatabase()
        db._index_cache['BK7'] = 1.517
        db._index_cache['Fused_Silica'] = 1.46
        
        assert len(db._index_cache) == 2
        
        db.clear_cache()
        
        assert len(db._index_cache) == 0
        assert db._index_cache == {}
    
    def test_clear_cache_resets_alias_keys(self):
        """Test that clear_cache() sets _alias_keys to None."""
        db = MaterialDatabase()
        db._alias_keys = {'key1': 'value1', 'key2': 'value2'}
        
        assert db._alias_keys is not None
        
        db.clear_cache()
        
        assert db._alias_keys is None
    
    def test_clear_cache_idempotent(self):
        """Test that multiple clear_cache() calls are safe."""
        db = MaterialDatabase()
        db._index_cache['material'] = 1.5
        db._alias_keys = {'key': 'value'}
        
        db.clear_cache()
        db.clear_cache()
        db.clear_cache()
        
        assert db._index_cache == {}
        assert db._alias_keys is None
    
    def test_clear_cache_instance_isolation_affects_only_one_instance(self):
        """Test that clearing cache on one instance doesn't affect other instances.
        
        This is the key behavior change mentioned in the docstring - previously
        this was class-wide and affected all instances.
        """
        db1 = MaterialDatabase()
        db2 = MaterialDatabase()
        
        db1._index_cache['material1'] = 1.5
        db1._alias_keys = {'alias1': 'data1'}
        
        db2._index_cache['material2'] = 1.33
        db2._alias_keys = {'alias2': 'data2'}
        
        db1.clear_cache()
        
        assert db1._index_cache == {}
        assert db1._alias_keys is None
        
        assert db2._index_cache['material2'] == 1.33
        assert db2._alias_keys is not None
    
    def test_clear_cache_multiple_instances_independent_clears(self):
        """Test that each instance can independently clear its cache."""
        db1 = MaterialDatabase()
        db2 = MaterialDatabase()
        db3 = MaterialDatabase()
        
        for db, material in zip([db1, db2, db3], ['M1', 'M2', 'M3']):
            db._index_cache[material] = 1.5
            db._alias_keys = {material: 'data'}
        
        db2.clear_cache()
        
        assert db1._index_cache['M1'] == 1.5
        assert db1._alias_keys is not None
        
        assert db2._index_cache == {}
        assert db2._alias_keys is None
        
        assert db3._index_cache['M3'] == 1.5
        assert db3._alias_keys is not None
    
    def test_clear_cache_on_empty_cache(self):
        """Test that clear_cache() works on an already empty cache."""
        db = MaterialDatabase()
        db._index_cache = {}
        db._alias_keys = None
        
        db.clear_cache()
        
        assert db._index_cache == {}
        assert db._alias_keys is None
    
    def test_clear_cache_with_large_cache(self):
        """Test clear_cache() with a large number of cached materials."""
        db = MaterialDatabase()
        
        for i in range(100):
            db._index_cache[f'material_{i}'] = 1.5 + i * 0.001
        
        db._alias_keys = {f'alias_{i}': f'value_{i}' for i in range(50)}
        
        assert len(db._index_cache) == 100
        assert len(db._alias_keys) == 50
        
        db.clear_cache()
        
        assert len(db._index_cache) == 0
        assert db._alias_keys is None
    
    def test_clear_cache_attributes_still_exist_after_clear(self):
        """Test that _index_cache and _alias_keys attributes still exist after clearing."""
        db = MaterialDatabase()
        db._index_cache['test'] = 1.5
        db._alias_keys = {'key': 'value'}
        
        db.clear_cache()
        
        assert hasattr(db, '_index_cache')
        assert hasattr(db, '_alias_keys')
        assert isinstance(db._index_cache, dict)
    
    def test_clear_cache_followed_by_repopulation(self):
        """Test that cache can be repopulated after being cleared."""
        db = MaterialDatabase()
        
        db._index_cache['BK7'] = 1.517
        db._alias_keys = {'optical_glass': 'BK7'}
        
        db.clear_cache()
        
        assert db._index_cache == {}
        assert db._alias_keys is None
        
        db._index_cache['Fused_Silica'] = 1.46
        db._alias_keys = {'optical_substrate': 'Fused_Silica'}
        
        assert db._index_cache['Fused_Silica'] == 1.46
        assert db._alias_keys is not None

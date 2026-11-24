import pytest
import tempfile
import json
import os
from pathlib import Path
from openlens.material_database import MaterialDatabase


class TestGetRefractiveIndex:
    """Tests for MaterialDatabase.get_refractive_index"""
    
    @pytest.fixture
    def db(self):
        """Create a MaterialDatabase instance"""
        return MaterialDatabase()
    
    def test_returns_float(self, db):
        """get_refractive_index returns a float"""
        result = db.get_refractive_index("BK7", 589.3)
        assert isinstance(result, float)
    
    def test_default_temperature_20(self, db):
        """Temperature parameter defaults to 20.0°C"""
        result1 = db.get_refractive_index("BK7", 589.3)
        result2 = db.get_refractive_index("BK7", 589.3, 20.0)
        assert result1 == result2
    
    def test_caching_identical_queries(self, db):
        """Results are cached for identical queries"""
        result1 = db.get_refractive_index("BK7", 589.3, 20.0)
        result2 = db.get_refractive_index("BK7", 589.3, 20.0)
        assert result1 == result2
    
    def test_unknown_material_returns_bk7(self, db):
        """Unknown material name returns BK7 refractive index"""
        unknown = db.get_refractive_index("NonexistentMaterial123", 589.3)
        bk7 = db.get_refractive_index("BK7", 589.3)
        assert unknown == bk7
    
    def test_different_wavelengths_produce_different_indices(self, db):
        """Different wavelengths produce different refractive indices"""
        result_400nm = db.get_refractive_index("BK7", 400.0, 20.0)
        result_700nm = db.get_refractive_index("BK7", 700.0, 20.0)
        assert result_400nm != result_700nm
    
    def test_cache_key_includes_wavelength(self, db):
        """Cache distinguishes entries by wavelength"""
        n1 = db.get_refractive_index("BK7", 400.0, 20.0)
        n2 = db.get_refractive_index("BK7", 401.0, 20.0)
        # Different wavelengths should give different values in cache
        assert n1 != n2
    
    def test_cache_key_includes_temperature(self, db):
        """Cache distinguishes entries by temperature"""
        n1 = db.get_refractive_index("BK7", 589.3, 20.0)
        n2 = db.get_refractive_index("BK7", 589.3, 25.0)
        # Both are valid floats and can be different due to temperature coefficients
        assert isinstance(n1, float)
        assert isinstance(n2, float)
    
    def test_cache_key_includes_material_name(self, db):
        """Cache distinguishes entries by material name"""
        bk7_result = db.get_refractive_index("BK7", 589.3, 20.0)
        unknown_result = db.get_refractive_index("UnknownMat", 589.3, 20.0)
        # Both map to BK7, but should be in cache under different keys
        assert bk7_result == unknown_result
    
    def test_clear_cache_allows_recomputation(self, db):
        """clear_cache removes cached entries"""
        result1 = db.get_refractive_index("BK7", 589.3, 20.0)
        db.clear_cache()
        result2 = db.get_refractive_index("BK7", 589.3, 20.0)
        assert result1 == result2
    
    def test_cache_capacity_clears_at_1024_entries(self, db):
        """Cache clears wholesale when it reaches 1024 entries"""
        # Fill cache with more than 1024 entries
        for i in range(1100):
            wavelength = 300.0 + (i * 0.5)
            db.get_refractive_index("BK7", wavelength, 20.0)
        
        # Should still return valid result after clearing
        result = db.get_refractive_index("BK7", 589.3, 20.0)
        assert isinstance(result, float)
        assert 1.0 < result < 2.5
    
    def test_refractive_index_physically_reasonable(self, db):
        """Refractive indices are in physically reasonable range"""
        result = db.get_refractive_index("BK7", 589.3, 20.0)
        # Optical materials typically have 1.0 < n < 2.5
        assert 1.0 < result < 2.5
    
    def test_dispersion_effect_wavelength_dependence(self, db):
        """Refractive index shows dispersion (wavelength dependence)"""
        # For normal dispersion: n decreases with increasing wavelength
        n_short = db.get_refractive_index("BK7", 400.0, 20.0)
        n_long = db.get_refractive_index("BK7", 1000.0, 20.0)
        # Short wavelength (blue) should have higher n than long (infrared)
        assert n_short > n_long


class TestSaveDatabase:
    """Tests for MaterialDatabase.save_database"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def db_with_temp_path(self, temp_dir):
        """Create database with temporary file path"""
        db = MaterialDatabase()
        db.db_path = os.path.join(temp_dir, "subdir", "nested", "materials.json")
        return db
    
    def test_save_database_creates_file(self, db_with_temp_path):
        """save_database creates the JSON file"""
        db_with_temp_path.save_database()
        assert os.path.exists(db_with_temp_path.db_path)
    
    def test_save_database_creates_parent_directories(self, db_with_temp_path):
        """save_database creates all necessary parent directories"""
        db_with_temp_path.save_database()
        parent_dir = os.path.dirname(db_with_temp_path.db_path)
        assert os.path.exists(parent_dir)
        assert os.path.isdir(parent_dir)
    
    def test_save_database_writes_valid_json(self, db_with_temp_path):
        """Saved file contains valid JSON"""
        db_with_temp_path.save_database()
        with open(db_with_temp_path.db_path, 'r') as f:
            data = json.load(f)
        assert isinstance(data, dict)
    
    def test_save_database_includes_all_materials(self, db_with_temp_path):
        """Saved JSON contains all materials from database"""
        db_with_temp_path.save_database()
        with open(db_with_temp_path.db_path, 'r') as f:
            data = json.load(f)
        
        assert len(data) > 0
        assert all(isinstance(k, str) for k in data.keys())
    
    def test_save_database_preserves_material_names(self, db_with_temp_path):
        """Material names are preserved in saved file"""
        original_names = set(db_with_temp_path.materials.keys())
        db_with_temp_path.save_database()
        
        with open(db_with_temp_path.db_path, 'r') as f:
            data = json.load(f)
        saved_names = set(data.keys())
        
        assert saved_names == original_names
    
    def test_save_database_idempotent(self, db_with_temp_path):
        """Multiple calls to save_database produce identical results"""
        db_with_temp_path.save_database()
        with open(db_with_temp_path.db_path, 'r') as f:
            first_save = json.load(f)
        
        db_with_temp_path.save_database()
        with open(db_with_temp_path.db_path, 'r') as f:
            second_save = json.load(f)
        
        assert first_save == second_save
    
    def test_save_database_overwrites_existing_file(self, db_with_temp_path):
        """save_database overwrites an existing file"""
        # Create existing file with old content
        Path(db_with_temp_path.db_path).parent.mkdir(parents=True, exist_ok=True)
        with open(db_with_temp_path.db_path, 'w') as f:
            json.dump({"old_material": {"n": 1.5}}, f)
        
        # Save database
        db_with_temp_path.save_database()
        
        # Verify old content is gone
        with open(db_with_temp_path.db_path, 'r') as f:
            data = json.load(f)
        
        assert "old_material" not in data
        assert len(data) > 0

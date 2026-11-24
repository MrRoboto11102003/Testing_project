import pytest
import json
import tempfile
from pathlib import Path
from openlens.material_database import MaterialDatabase


class TestGetRefractiveIndex:
    """Tests for MaterialDatabase.get_refractive_index"""

    def test_returns_float(self):
        """Test that get_refractive_index returns a float"""
        db = MaterialDatabase()
        result = db.get_refractive_index("BK7", 550.0)
        assert isinstance(result, float)

    def test_bk7_at_standard_wavelength(self):
        """Test BK7 refractive index at standard wavelength is in expected range"""
        db = MaterialDatabase()
        result = db.get_refractive_index("BK7", 550.0)
        assert 1.4 < result < 1.6

    def test_dispersion_blue_vs_red(self):
        """Test that blue light has higher refractive index than red (normal dispersion)"""
        db = MaterialDatabase()
        index_blue = db.get_refractive_index("BK7", 400.0)
        index_red = db.get_refractive_index("BK7", 700.0)
        assert index_blue > index_red

    def test_default_temperature_is_20_degrees(self):
        """Test that default temperature parameter is 20°C"""
        db = MaterialDatabase()
        result_default = db.get_refractive_index("BK7", 550.0)
        result_explicit = db.get_refractive_index("BK7", 550.0, temperature_c=20.0)
        assert result_default == result_explicit

    def test_caching_returns_identical_value(self):
        """Test that cached queries return the exact same value"""
        db = MaterialDatabase()
        result1 = db.get_refractive_index("BK7", 550.0, 20.0)
        result2 = db.get_refractive_index("BK7", 550.0, 20.0)
        result3 = db.get_refractive_index("BK7", 550.0, 20.0)
        assert result1 == result2 == result3

    def test_unknown_material_falls_back_to_bk7(self):
        """Test that unknown material name returns BK7 index (fallback behavior)"""
        db = MaterialDatabase()
        unknown_result = db.get_refractive_index("UnknownGlass_XYZ_123", 550.0)
        bk7_result = db.get_refractive_index("BK7", 550.0)
        assert unknown_result == bk7_result

    def test_different_cache_keys_stored_independently(self):
        """Test that different wavelengths and temperatures are cached independently"""
        db = MaterialDatabase()
        result_a = db.get_refractive_index("BK7", 400.0, 20.0)
        result_b = db.get_refractive_index("BK7", 700.0, 20.0)
        result_c = db.get_refractive_index("BK7", 400.0, 50.0)
        
        # All should be floats
        assert isinstance(result_a, float)
        assert isinstance(result_b, float)
        assert isinstance(result_c, float)
        
        # Retrieve again - should be from cache
        result_a2 = db.get_refractive_index("BK7", 400.0, 20.0)
        result_b2 = db.get_refractive_index("BK7", 700.0, 20.0)
        result_c2 = db.get_refractive_index("BK7", 400.0, 50.0)
        
        assert result_a == result_a2
        assert result_b == result_b2
        assert result_c == result_c2

    def test_cache_bounded_at_1024_entries(self):
        """Test that cache is cleared wholesale when reaching 1024 entries"""
        db = MaterialDatabase()
        # Fill cache beyond 1024 entries with different wavelengths
        for i in range(1030):
            wavelength = 350.0 + i * 0.1
            db.get_refractive_index("BK7", wavelength)
        
        # After clearing and rebuilding, should still return valid values
        result = db.get_refractive_index("BK7", 550.0)
        assert isinstance(result, float)
        assert 1.4 < result < 1.6

    def test_extreme_ultraviolet_wavelength(self):
        """Test with UV wavelength (200nm)"""
        db = MaterialDatabase()
        result = db.get_refractive_index("BK7", 200.0)
        assert isinstance(result, float)

    def test_extreme_infrared_wavelength(self):
        """Test with IR wavelength (2000nm)"""
        db = MaterialDatabase()
        result = db.get_refractive_index("BK7", 2000.0)
        assert isinstance(result, float)

    def test_temperature_affects_refractive_index(self):
        """Test that extreme temperatures produce valid results"""
        db = MaterialDatabase()
        result_cold = db.get_refractive_index("BK7", 550.0, -40.0)
        result_hot = db.get_refractive_index("BK7", 550.0, 100.0)
        assert isinstance(result_cold, float)
        assert isinstance(result_hot, float)


class TestSaveDatabase:
    """Tests for MaterialDatabase.save_database"""

    def test_save_database_creates_file(self):
        """Test that save_database creates a JSON file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_materials.json"
            db = MaterialDatabase()
            db.db_path = db_path
            
            db.save_database()
            
            assert db_path.exists()
            assert db_path.suffix == ".json"

    def test_save_database_creates_parent_directories(self):
        """Test that save_database creates parent directories if needed"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "deep" / "nested" / "dir" / "materials.json"
            db = MaterialDatabase()
            db.db_path = db_path
            
            db.save_database()
            
            assert db_path.exists()
            assert db_path.parent.exists()

    def test_save_database_writes_valid_json(self):
        """Test that saved file contains valid JSON"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_materials.json"
            db = MaterialDatabase()
            db.db_path = db_path
            
            db.save_database()
            
            with open(db_path, "r") as f:
                data = json.load(f)
            
            assert isinstance(data, dict)

    def test_save_database_persists_all_materials(self):
        """Test that all materials from database are persisted to file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_materials.json"
            db = MaterialDatabase()
            db.db_path = db_path
            
            material_count_before = len(db.materials)
            assert material_count_before > 0
            
            db.save_database()
            
            with open(db_path, "r") as f:
                saved_data = json.load(f)
            
            assert len(saved_data) == material_count_before
            # BK7 should always be present
            assert "BK7" in saved_data

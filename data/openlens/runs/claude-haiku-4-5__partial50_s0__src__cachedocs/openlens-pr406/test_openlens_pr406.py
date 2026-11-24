import pytest
import json
import tempfile
import time
from pathlib import Path

from openlens.material_database import MaterialDatabase


class TestMaterialDatabaseGetRefractiveIndex:
    """Tests for MaterialDatabase.get_refractive_index"""

    def test_returns_float_for_known_material(self):
        """Should return float for known material like BK7"""
        db = MaterialDatabase()
        result = db.get_refractive_index("BK7", wavelength_nm=587.6)
        assert isinstance(result, float)
        assert 1.0 < result < 3.0  # Realistic optical glass range

    def test_unknown_material_fallback_to_bk7(self):
        """Unknown material name should return BK7 index as documented"""
        db = MaterialDatabase()
        unknown_idx = db.get_refractive_index("NonexistentGlass", wavelength_nm=587.6)
        bk7_idx = db.get_refractive_index("BK7", wavelength_nm=587.6)
        assert unknown_idx == bk7_idx

    def test_wavelength_dispersion_effect(self):
        """Different wavelengths should produce different indices via Sellmeier"""
        db = MaterialDatabase()
        idx_uv = db.get_refractive_index("BK7", wavelength_nm=380.0)
        idx_ir = db.get_refractive_index("BK7", wavelength_nm=780.0)
        assert idx_uv != idx_ir
        # Normal dispersion: shorter wavelength has higher index
        assert idx_uv > idx_ir

    def test_temperature_coefficient_effect(self):
        """Different temperatures should affect refractive index"""
        db = MaterialDatabase()
        idx_cold = db.get_refractive_index("BK7", wavelength_nm=587.6, temperature_c=0.0)
        idx_warm = db.get_refractive_index("BK7", wavelength_nm=587.6, temperature_c=50.0)
        assert idx_cold != idx_warm

    def test_default_temperature_parameter(self):
        """Default temperature should be 20°C as documented"""
        db = MaterialDatabase()
        default_call = db.get_refractive_index("BK7", wavelength_nm=587.6)
        explicit_20c = db.get_refractive_index("BK7", wavelength_nm=587.6, temperature_c=20.0)
        assert default_call == explicit_20c

    def test_caching_identical_queries(self):
        """Identical parameter queries should return same cached value"""
        db = MaterialDatabase()
        call1 = db.get_refractive_index("BK7", wavelength_nm=550.0, temperature_c=25.0)
        call2 = db.get_refractive_index("BK7", wavelength_nm=550.0, temperature_c=25.0)
        assert call1 == call2
        assert call1 is call2  # Should be identical object

    def test_cache_clear_method_resets_cache(self):
        """clear_cache() method should reset the cache"""
        db = MaterialDatabase()
        # Populate cache
        db.get_refractive_index("BK7", wavelength_nm=587.6)
        assert len(db._index_cache) > 0
        
        # Clear cache
        db.clear_cache()
        assert len(db._index_cache) == 0

    def test_cache_bounded_at_1024_entries(self):
        """Cache should wholesale clear when reaching 1024 entries"""
        db = MaterialDatabase()
        # Generate 1050 unique wavelength queries to exceed limit
        for i in range(1050):
            wavelength = 300.0 + i * 0.5
            db.get_refractive_index("BK7", wavelength_nm=wavelength)
        
        # After overflow, cache size should be much smaller than 1050
        # (clears at 1024, not evict, so ~26 entries left)
        assert len(db._index_cache) < 100


class TestMaterialDatabaseSaveDatabase:
    """Tests for MaterialDatabase.save_database"""

    def test_save_creates_json_file(self):
        """save_database should create JSON file at db_path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "materials.json"
            db = MaterialDatabase()
            db.db_path = str(db_path)
            
            db.save_database()
            
            assert db_path.exists()
            assert db_path.suffix == ".json"

    def test_save_creates_parent_directories(self):
        """save_database should create missing parent directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "a" / "b" / "c" / "materials.json"
            db = MaterialDatabase()
            db.db_path = str(db_path)
            
            db.save_database()
            
            assert db_path.exists()
            assert db_path.parent.exists()

    def test_save_produces_valid_json(self):
        """Saved file should be valid JSON format"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "materials.json"
            db = MaterialDatabase()
            db.db_path = str(db_path)
            
            db.save_database()
            
            data = json.loads(db_path.read_text())
            assert isinstance(data, dict)

    def test_save_preserves_material_names(self):
        """Saved JSON keys should match material names in database"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "materials.json"
            db = MaterialDatabase()
            db.db_path = str(db_path)
            
            db.save_database()
            
            data = json.loads(db_path.read_text())
            for material_name in db.materials.keys():
                assert material_name in data

    def test_save_uses_to_dict_serialization(self):
        """Materials should be saved as dicts (via to_dict method)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "materials.json"
            db = MaterialDatabase()
            db.db_path = str(db_path)
            
            db.save_database()
            
            data = json.loads(db_path.read_text())
            for name, material_dict in data.items():
                assert isinstance(material_dict, dict)

    def test_save_count_matches_database(self):
        """Number of saved materials should match database count"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "materials.json"
            db = MaterialDatabase()
            db.db_path = str(db_path)
            
            original_count = len(db.materials)
            db.save_database()
            
            data = json.loads(db_path.read_text())
            assert len(data) == original_count

    def test_save_overwrites_existing_file(self):
        """save_database should overwrite existing file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "materials.json"
            db = MaterialDatabase()
            db.db_path = str(db_path)
            
            # Save first time
            db.save_database()
            mtime1 = db_path.stat().st_mtime
            
            # Save second time (should overwrite)
            time.sleep(0.01)
            db.save_database()
            mtime2 = db_path.stat().st_mtime
            
            assert mtime2 >= mtime1
            assert db_path.exists()

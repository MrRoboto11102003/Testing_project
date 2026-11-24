import pytest
import json
import math
import tempfile
from pathlib import Path
from openlens.material_database import MaterialDatabase, MaterialProperties


class TestGetRefractiveIndex:
    """Test MaterialDatabase.get_refractive_index"""

    def test_bk7_reference_conditions(self):
        """BK7 at 587.6 nm and 20°C returns nd ≈ 1.5168"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MaterialDatabase(db_path=str(Path(tmpdir) / "test.json"))
            n = db.get_refractive_index("BK7", 587.6, 20.0)
            assert n == pytest.approx(1.5168, abs=1e-3)

    def test_cache_hit_returns_cached_value(self):
        """Repeated calls with same key return cached value"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MaterialDatabase(db_path=str(Path(tmpdir) / "test.json"))
            n1 = db.get_refractive_index("BK7", 550.0, 20.0)
            assert len(db._index_cache) == 1
            n2 = db.get_refractive_index("BK7", 550.0, 20.0)
            assert n1 == n2
            assert len(db._index_cache) == 1

    def test_cache_eviction_at_max_size(self):
        """Cache clears wholesale when reaching 1024 entries"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MaterialDatabase(db_path=str(Path(tmpdir) / "test.json"))
            # Fill cache to 1024 entries with different wavelengths
            for i in range(1024):
                db.get_refractive_index("BK7", 400.0 + i * 0.01, 20.0)
            assert len(db._index_cache) == 1024
            # Add one more triggers wholesale clear
            db.get_refractive_index("BK7", 420.5, 20.0)
            assert len(db._index_cache) == 1

    def test_unknown_material_fallback_to_bk7(self):
        """Unknown material name returns BK7 index as fallback"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MaterialDatabase(db_path=str(Path(tmpdir) / "test.json"))
            n_unknown = db.get_refractive_index("NOTEXIST", 587.6, 20.0)
            n_bk7 = db.get_refractive_index("BK7", 587.6, 20.0)
            assert n_unknown == n_bk7

    def test_cache_key_preserves_name_case(self):
        """Cache keys preserve material name case despite lookup normalization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MaterialDatabase(db_path=str(Path(tmpdir) / "test.json"))
            db.get_refractive_index("BK7", 500.0, 20.0)
            db.get_refractive_index("bk7", 500.0, 20.0)
            # Both entries exist in cache with their original cases
            assert ("BK7", 500.0, 20.0) in db._index_cache
            assert ("bk7", 500.0, 20.0) in db._index_cache

    def test_temperature_effect_bk7(self):
        """Refractive index increases with temperature for BK7"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MaterialDatabase(db_path=str(Path(tmpdir) / "test.json"))
            n_20c = db.get_refractive_index("BK7", 587.6, 20.0)
            n_30c = db.get_refractive_index("BK7", 587.6, 30.0)
            assert n_30c > n_20c

    def test_bk7_dispersion(self):
        """BK7 shows normal dispersion across visible spectrum"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MaterialDatabase(db_path=str(Path(tmpdir) / "test.json"))
            n_blue = db.get_refractive_index("BK7", 486.1, 20.0)
            n_yellow = db.get_refractive_index("BK7", 587.6, 20.0)
            n_red = db.get_refractive_index("BK7", 656.3, 20.0)
            assert n_blue > n_yellow > n_red

    def test_zero_wavelength_returns_one(self):
        """Zero wavelength returns exactly 1.0"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MaterialDatabase(db_path=str(Path(tmpdir) / "test.json"))
            n = db.get_refractive_index("BK7", 0.0, 20.0)
            assert n == 1.0

    def test_custom_glass_sellmeier(self):
        """Custom glass with known Sellmeier coefficients calculates correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MaterialDatabase(db_path=str(Path(tmpdir) / "test.json"))
            # Custom glass with B1=1.0, C1=0.01
            mat = MaterialProperties(
                name="TESTGLASS", catalog="Custom", nd=1.4, vd=50.0,
                B1=1.0, C1=0.01, B2=0.0, C2=0.0, B3=0.0, C3=0.0
            )
            db.add_material(mat, save=False)
            # At 1000 nm: wl_um=1.0, lambda_sq=1.0
            # n_sq = 1 + 1*1/(1-0.01) = 1 + 1/0.99
            n = db.get_refractive_index("TESTGLASS", 1000.0, 20.0)
            expected = math.sqrt(1.0 + 1.0 / 0.99)
            assert n == pytest.approx(expected, rel=1e-12)

    def test_material_without_thermal_data_unchanged_by_temperature(self):
        """Material with zero thermal coefficients unchanged by temperature"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MaterialDatabase(db_path=str(Path(tmpdir) / "test.json"))
            # E-FD60 has no thermal coefficients (D0=D1=D2=E0=E1=0)
            n_20c = db.get_refractive_index("E-FD60", 587.6, 20.0)
            n_80c = db.get_refractive_index("E-FD60", 587.6, 80.0)
            assert n_20c == n_80c


class TestSaveDatabase:
    """Test MaterialDatabase.save_database"""

    def test_save_creates_file(self):
        """save_database creates the JSON file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "materials.json")
            db = MaterialDatabase(db_path=db_path)
            db.save_database()
            assert Path(db_path).exists()

    def test_save_contains_all_builtin_materials(self):
        """Saved file contains all 7 built-in materials"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "materials.json")
            db = MaterialDatabase(db_path=db_path)
            db.save_database()
            
            with open(db_path, encoding="utf-8") as f:
                data = json.load(f)
            
            builtins = ["BK7", "N-BK7", "SF11", "F2", "FUSEDSILICA", "S-LAH79", "E-FD60"]
            for name in builtins:
                assert name in data

    def test_save_preserves_material_values(self):
        """Saved material preserves all property values"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "materials.json")
            db = MaterialDatabase(db_path=db_path)
            db.save_database()
            
            with open(db_path, encoding="utf-8") as f:
                data = json.load(f)
            
            # Verify BK7 properties match known values
            bk7_data = data["BK7"]
            assert bk7_data["nd"] == pytest.approx(1.5168, abs=1e-4)
            assert bk7_data["vd"] == pytest.approx(64.17, abs=0.1)
            assert bk7_data["name"] == "BK7"

    def test_save_custom_material_persists(self):
        """Custom material added to database is saved"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "materials.json")
            db = MaterialDatabase(db_path=db_path)
            
            custom = MaterialProperties(
                name="myglassname", catalog="Custom", nd=1.52, vd=45.0
            )
            db.add_material(custom, save=False)
            db.save_database()
            
            with open(db_path, encoding="utf-8") as f:
                data = json.load(f)
            
            assert "MYGLASSNAME" in data
            assert data["MYGLASSNAME"]["name"] == "myglassname"
            assert data["MYGLASSNAME"]["nd"] == 1.52
            assert data["MYGLASSNAME"]["vd"] == 45.0

    def test_save_overwrites_previous_file(self):
        """Second save overwrites previous file contents"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "materials.json")
            db = MaterialDatabase(db_path=db_path)
            db.save_database()
            
            # Remove a built-in material from memory
            del db.materials["SF11"]
            db.save_database()
            
            with open(db_path, encoding="utf-8") as f:
                data = json.load(f)
            
            assert "SF11" not in data

    def test_save_creates_parent_directories(self):
        """Parent directories are created if they don't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "a" / "b" / "c" / "materials.json")
            db = MaterialDatabase(db_path=db_path)
            db.save_database()
            assert Path(db_path).exists()

    def test_save_returns_none(self):
        """save_database returns None"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "materials.json")
            db = MaterialDatabase(db_path=db_path)
            result = db.save_database()
            assert result is None

    def test_save_and_reload_roundtrip(self):
        """Material properties survive save and reload cycle"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "materials.json")
            
            # Save original
            db1 = MaterialDatabase(db_path=db_path)
            original_bk7_nd = db1.materials["BK7"].nd
            original_bk7_vd = db1.materials["BK7"].vd
            db1.save_database()
            
            # Reload and compare
            db2 = MaterialDatabase(db_path=db_path)
            assert db2.materials["BK7"].nd == original_bk7_nd
            assert db2.materials["BK7"].vd == original_bk7_vd

    def test_save_transmission_data_as_json_arrays(self):
        """Transmission data tuples are converted to JSON arrays"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "materials.json")
            db = MaterialDatabase(db_path=db_path)
            db.save_database()
            
            with open(db_path, encoding="utf-8") as f:
                data = json.load(f)
            
            # Check transmission_data is list of lists (from JSON), not tuples
            if data["BK7"]["transmission_data"]:
                trans = data["BK7"]["transmission_data"]
                assert isinstance(trans, list)
                assert isinstance(trans[0], list)
                assert isinstance(trans[0][0], (int, float))

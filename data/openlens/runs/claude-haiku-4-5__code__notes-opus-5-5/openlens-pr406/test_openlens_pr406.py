import json
import pytest
import math
from pathlib import Path

try:
    from openlens.material_database import MaterialDatabase, MaterialProperties
except ImportError:
    pytest.skip("openlens not installed", allow_module_level=True)


class TestGetRefractiveIndex:
    """Tests for MaterialDatabase.get_refractive_index"""

    def test_builtin_bk7_reference(self, tmp_path):
        """BK7 at 587.6 nm, 20°C returns documented nd ≈ 1.5168"""
        db = MaterialDatabase(db_path=str(tmp_path / "materials.json"))
        n = db.get_refractive_index("BK7", 587.6, 20.0)
        assert n == pytest.approx(1.5168, abs=1e-3)

    def test_unknown_material_fallback(self, tmp_path):
        """Unknown material returns BK7 refractive index"""
        db = MaterialDatabase(db_path=str(tmp_path / "materials.json"))
        n_unknown = db.get_refractive_index("NONEXISTENT", 587.6, 20.0)
        n_bk7 = db.get_refractive_index("BK7", 587.6, 20.0)
        assert n_unknown == n_bk7

    def test_cache_hit(self, tmp_path):
        """Repeated calls return cached value"""
        db = MaterialDatabase(db_path=str(tmp_path / "materials.json"))
        n1 = db.get_refractive_index("BK7", 500.0, 20.0)
        n2 = db.get_refractive_index("BK7", 500.0, 20.0)
        assert n1 == n2
        assert len(db._index_cache) == 1

    def test_dispersion_bk7(self, tmp_path):
        """Dispersion: n(blue) > n(yellow) > n(red)"""
        db = MaterialDatabase(db_path=str(tmp_path / "materials.json"))
        n_blue = db.get_refractive_index("BK7", 486.1, 20.0)
        n_yellow = db.get_refractive_index("BK7", 587.6, 20.0)
        n_red = db.get_refractive_index("BK7", 656.3, 20.0)
        assert n_blue > n_yellow > n_red

    def test_temperature_effect(self, tmp_path):
        """Higher temperature increases refractive index"""
        db = MaterialDatabase(db_path=str(tmp_path / "materials.json"))
        n_20 = db.get_refractive_index("BK7", 587.6, 20.0)
        n_30 = db.get_refractive_index("BK7", 587.6, 30.0)
        assert n_30 > n_20

    def test_cache_full_clears(self, tmp_path):
        """Cache clears wholesale when reaching 1024 entries"""
        db = MaterialDatabase(db_path=str(tmp_path / "materials.json"))
        for i in range(1024):
            db.get_refractive_index("BK7", 400.0 + i * 0.1, 20.0)
        assert len(db._index_cache) == 1024
        db.get_refractive_index("BK7", 700.0, 20.0)
        assert len(db._index_cache) == 1

    def test_zero_wavelength(self, tmp_path):
        """Zero wavelength returns n = 1.0"""
        db = MaterialDatabase(db_path=str(tmp_path / "materials.json"))
        n = db.get_refractive_index("BK7", 0.0, 20.0)
        assert n == 1.0

    def test_custom_sellmeier(self, tmp_path):
        """Custom material with Sellmeier coefficients"""
        db = MaterialDatabase(db_path=str(tmp_path / "materials.json"))
        mat = MaterialProperties(
            name="TESTG", catalog="Custom", nd=1.4, vd=50,
            B1=1.0, C1=0.01, B2=0.0, C2=0.0, B3=0.0, C3=0.0
        )
        db.add_material(mat, save=False)
        # At 1000 nm: n_sq = 1 + 1.0/(1.0 - 0.01) = 1 + 1/0.99
        expected_n = math.sqrt(1.0 + 1.0 / 0.99)
        n = db.get_refractive_index("TESTG", 1000.0, 20.0)
        assert n == pytest.approx(expected_n, rel=1e-12)

    def test_sellmeier_resonance(self, tmp_path):
        """Sellmeier term skipped at resonance"""
        db = MaterialDatabase(db_path=str(tmp_path / "materials.json"))
        mat = MaterialProperties(
            name="RESMAT", catalog="Custom", nd=1.4, vd=50,
            B1=1.0, C1=1.0, B2=0.0, C2=0.0, B3=0.0, C3=0.0
        )
        db.add_material(mat, save=False)
        n = db.get_refractive_index("RESMAT", 1000.0, 20.0)
        assert n == 1.0

    def test_negative_n_squared(self, tmp_path):
        """When n_sq < 0, return 1.0 regardless of temperature"""
        db = MaterialDatabase(db_path=str(tmp_path / "materials.json"))
        mat = MaterialProperties(
            name="NEGMAT", catalog="Custom", nd=1.4, vd=50,
            B1=-2.0, C1=0.01, B2=0.0, C2=0.0, B3=0.0, C3=0.0,
            D0=1e-5
        )
        db.add_material(mat, save=False)
        n = db.get_refractive_index("NEGMAT", 1000.0, 50.0)
        assert n == 1.0


class TestSaveDatabase:
    """Tests for MaterialDatabase.save_database"""

    def test_save_creates_file(self, tmp_path):
        """save_database creates JSON file"""
        db_path = tmp_path / "materials.json"
        db = MaterialDatabase(db_path=str(db_path))
        db.save_database()
        assert db_path.exists()

    def test_save_contains_builtins(self, tmp_path):
        """Saved file contains all 7 built-in materials"""
        db_path = tmp_path / "materials.json"
        db = MaterialDatabase(db_path=str(db_path))
        db.save_database()
        with open(db_path, encoding="utf-8") as f:
            data = json.load(f)
        expected = {"BK7", "N-BK7", "SF11", "F2", "FUSEDSILICA", "S-LAH79", "E-FD60"}
        assert set(data.keys()) == expected

    def test_save_bk7_values(self, tmp_path):
        """BK7 values are correctly saved"""
        db_path = tmp_path / "materials.json"
        db = MaterialDatabase(db_path=str(db_path))
        db.save_database()
        with open(db_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["BK7"]["nd"] == 1.5168
        assert data["BK7"]["catalog"] == "Schott"

    def test_save_round_trip(self, tmp_path):
        """Save and reload preserves material properties"""
        db_path = tmp_path / "materials.json"
        db1 = MaterialDatabase(db_path=str(db_path))
        orig_b1 = db1.materials["BK7"].B1
        db1.save_database()
        
        db2 = MaterialDatabase(db_path=str(db_path))
        assert db2.materials["BK7"].B1 == orig_b1

    def test_save_overwrites(self, tmp_path):
        """Saving overwrites old file, does not merge"""
        db_path = tmp_path / "materials.json"
        db = MaterialDatabase(db_path=str(db_path))
        db.save_database()
        
        del db.materials["SF11"]
        db.save_database()
        
        with open(db_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "SF11" not in data
        assert len(data) == 6

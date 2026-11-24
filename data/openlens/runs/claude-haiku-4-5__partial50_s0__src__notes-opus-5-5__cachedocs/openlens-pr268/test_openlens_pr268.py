import pytest
from pathlib import Path
from openlens.material_database import MaterialDatabase, MaterialProperties


class TestMaterialDatabaseClearCache:
    """Tests for MaterialDatabase.clear_cache()"""

    def test_clear_cache_empties_cache_and_resets_aliases(self, tmp_path):
        """clear_cache() empties _index_cache and sets _alias_keys to None"""
        db = MaterialDatabase(db_path=str(tmp_path / "m.json"))
        db.get_refractive_index("BK7", 587.6, 20.0)
        db.get_material("bk7")
        
        assert len(db._index_cache) > 0
        assert db._alias_keys is not None
        
        db.clear_cache()
        
        assert db._index_cache == {}
        assert db._alias_keys is None

    def test_clear_cache_per_instance_isolation(self, tmp_path):
        """clear_cache() only affects the calling instance"""
        db1 = MaterialDatabase(db_path=str(tmp_path / "m1.json"))
        db2 = MaterialDatabase(db_path=str(tmp_path / "m2.json"))
        
        db1.get_refractive_index("BK7", 587.6, 20.0)
        db2.get_refractive_index("BK7", 587.6, 20.0)
        
        db1.clear_cache()
        
        assert db1._index_cache == {}
        assert len(db2._index_cache) > 0

    def test_clear_cache_observable_via_staleness(self, tmp_path):
        """clear_cache() makes stale cached values fresh"""
        db = MaterialDatabase(db_path=str(tmp_path / "m.json"))
        before = db.get_refractive_index("BK7", 587.6, 20.0)
        
        db.materials["BK7"] = MaterialProperties(
            name="BK7", catalog="X", nd=1.6, vd=40.0,
            B1=0.0, B2=0.0, B3=0.0, C1=0.0, C2=0.0, C3=0.0
        )
        cached = db.get_refractive_index("BK7", 587.6, 20.0)
        assert cached == before
        
        db.clear_cache()
        fresh = db.get_refractive_index("BK7", 587.6, 20.0)
        assert fresh == pytest.approx(1.0, abs=1e-6)


class TestMaterialDatabaseGetTransmission:
    """Tests for MaterialDatabase.get_transmission()"""

    def test_get_transmission_exact_bk7_value(self, tmp_path):
        """get_transmission returns exact catalog value for BK7 at 500 nm"""
        db = MaterialDatabase(db_path=str(tmp_path / "m.json"))
        result = db.get_transmission("BK7", 500.0, thickness_mm=10.0)
        assert result == pytest.approx(0.995, abs=1e-6)

    def test_get_transmission_interpolation(self, tmp_path):
        """get_transmission interpolates between data points"""
        db = MaterialDatabase(db_path=str(tmp_path / "m.json"))
        result = db.get_transmission("BK7", 450.0, thickness_mm=10.0)
        expected = 0.97 + 0.5 * (0.995 - 0.97)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_get_transmission_thickness_scaling(self, tmp_path):
        """get_transmission scales by Beer-Lambert law: T = t_base^(thickness/ref)"""
        db = MaterialDatabase(db_path=str(tmp_path / "m.json"))
        base = db.get_transmission("BK7", 500.0, thickness_mm=10.0)
        double = db.get_transmission("BK7", 500.0, thickness_mm=20.0)
        half = db.get_transmission("BK7", 500.0, thickness_mm=5.0)
        
        assert double == pytest.approx(base**2, abs=1e-6)
        assert half == pytest.approx(base**0.5, abs=1e-6)

    def test_get_transmission_clamping_and_zero_thickness(self, tmp_path):
        """get_transmission clamps to data range and returns 1.0 at thickness 0"""
        db = MaterialDatabase(db_path=str(tmp_path / "m.json"))
        below = db.get_transmission("BK7", 300.0, thickness_mm=10.0)
        above = db.get_transmission("BK7", 3000.0, thickness_mm=10.0)
        zero_thickness = db.get_transmission("BK7", 500.0, thickness_mm=0.0)
        
        assert below == pytest.approx(0.95, abs=1e-6)
        assert above == pytest.approx(0.5, abs=1e-6)
        assert zero_thickness == pytest.approx(1.0, abs=1e-6)

    def test_get_transmission_fallback_for_unknown_material(self, tmp_path):
        """get_transmission uses fallback 0.99 for materials without transmission_data"""
        db = MaterialDatabase(db_path=str(tmp_path / "m.json"))
        base = db.get_transmission("Unobtainium", 500.0, thickness_mm=10.0)
        scaled = db.get_transmission("N-BK7", 500.0, thickness_mm=20.0)
        
        assert base == pytest.approx(0.99, abs=1e-6)
        assert scaled == pytest.approx(0.99**2, abs=1e-6)


class TestMaterialProperties:
    """Tests for MaterialProperties dataclass"""

    def test_construction_required_and_default_fields(self):
        """MaterialProperties sets required fields and proper defaults"""
        m = MaterialProperties("MyGlass", "Custom", 1.5, 60.0)
        
        assert m.name == "MyGlass"
        assert m.catalog == "Custom"
        assert m.nd == 1.5
        assert m.vd == 60.0
        assert m.B1 == 0.0
        assert m.thermal_expansion == pytest.approx(7.1e-6)
        assert m.transmission_thickness == 10.0
        assert m.transmission_data == []

    def test_to_dict_complete_and_ordered(self):
        """to_dict() returns all 23 fields in declaration order"""
        m = MaterialProperties("X", "C", 1.5, 60.0, B1=1.1, thermal_expansion=8e-6)
        d = m.to_dict()
        
        expected_keys = [
            'name', 'catalog', 'nd', 'vd',
            'B1', 'B2', 'B3', 'C1', 'C2', 'C3',
            'D0', 'D1', 'D2', 'E0', 'E1', 'lambda_tk_um',
            'thermal_expansion', 'transmission_data', 'transmission_thickness',
            'density', 'climate_resistance', 'acid_resistance', 'alkali_resistance'
        ]
        assert list(d.keys()) == expected_keys
        assert d["B1"] == 1.1
        assert d["thermal_expansion"] == 8e-6

    def test_to_dict_defensive_copy(self):
        """to_dict() makes defensive copies of mutable fields"""
        m = MaterialProperties("X", "C", 1.5, 60.0, transmission_data=[(500, 0.99)])
        d = m.to_dict()
        
        d["transmission_data"].append((600, 0.98))
        
        assert m.transmission_data == [(500, 0.99)]

    def test_round_trip_to_from_dict(self):
        """Round trip through to_dict/from_dict preserves equality"""
        m1 = MaterialProperties(
            "Glass", "Schott", 1.52, 51.0,
            B1=1.1, thermal_expansion=8e-6,
            transmission_data=[(400, 0.95)], density=2.5
        )
        m2 = MaterialProperties.from_dict(m1.to_dict())
        
        assert m1 == m2

    def test_per_instance_default_transmission_data(self):
        """Each instance gets its own transmission_data list"""
        m1 = MaterialProperties("A", "C", 1.5, 60.0)
        m2 = MaterialProperties("B", "C", 1.5, 60.0)
        
        m1.transmission_data.append((500, 0.99))
        
        assert len(m1.transmission_data) == 1
        assert m2.transmission_data == []

    def test_missing_required_field_raises_typeerror(self):
        """Constructor raises TypeError for missing required fields"""
        with pytest.raises(TypeError):
            MaterialProperties("X", "C", 1.5)

    def test_from_dict_extra_key_raises_typeerror(self):
        """from_dict raises TypeError for unexpected keyword arguments"""
        with pytest.raises(TypeError):
            MaterialProperties.from_dict({
                "name": "X", "catalog": "C", "nd": 1.5, "vd": 60.0,
                "unknown_field": 42
            })

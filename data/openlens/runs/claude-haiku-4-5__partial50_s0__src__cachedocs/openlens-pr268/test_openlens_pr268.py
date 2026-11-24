import pytest
from unittest.mock import Mock, patch
from openlens.material_database import MaterialDatabase, MaterialProperties


class TestMaterialProperties:
    """Test MaterialProperties dataclass"""
    
    def test_material_properties_minimal_creation(self):
        """Create MaterialProperties with only required fields"""
        mat = MaterialProperties(
            name="BK7",
            catalog="Schott",
            nd=1.517,
            vd=64.2
        )
        assert mat.name == "BK7"
        assert mat.catalog == "Schott"
        assert mat.nd == 1.517
        assert mat.vd == 64.2
    
    def test_material_properties_defaults(self):
        """Test that default values are applied correctly"""
        mat = MaterialProperties(
            name="Test",
            catalog="Custom",
            nd=1.5,
            vd=50.0
        )
        assert mat.thermal_expansion == 7.1e-6
        assert mat.transmission_thickness == 10.0
        assert mat.density == 2.51
        assert mat.transmission_data == []
    
    def test_material_properties_full_initialization(self):
        """Create MaterialProperties with all thermal and transmission fields"""
        trans_data = [(400.0, 0.95), (550.0, 0.98), (700.0, 0.96)]
        mat = MaterialProperties(
            name="Custom",
            catalog="Ohara",
            nd=1.6,
            vd=40.0,
            B1=1.03961212,
            B2=0.231792344,
            D0=1.86e-6,
            D1=2.0e-8,
            transmission_data=trans_data,
            transmission_thickness=5.0,
            climate_resistance=3,
            acid_resistance=2
        )
        assert mat.B1 == 1.03961212
        assert mat.D0 == 1.86e-6
        assert len(mat.transmission_data) == 3
        assert mat.transmission_thickness == 5.0
    
    def test_material_properties_to_dict(self):
        """Test to_dict serialization"""
        mat = MaterialProperties(
            name="BK7",
            catalog="Schott",
            nd=1.517,
            vd=64.2,
            B1=1.0
        )
        result = mat.to_dict()
        assert isinstance(result, dict)
        assert result["name"] == "BK7"
        assert result["nd"] == 1.517
        assert result["B1"] == 1.0
    
    def test_material_properties_from_dict(self):
        """Test from_dict deserialization"""
        data = {
            "name": "BK7",
            "catalog": "Schott",
            "nd": 1.517,
            "vd": 64.2,
            "B1": 0.0,
            "B2": 0.0,
            "B3": 0.0,
            "C1": 0.0,
            "C2": 0.0,
            "C3": 0.0,
            "D0": 0.0,
            "D1": 0.0,
            "D2": 0.0,
            "E0": 0.0,
            "E1": 0.0,
            "lambda_tk_um": 0.0,
            "thermal_expansion": 7.1e-6,
            "transmission_data": [],
            "transmission_thickness": 10.0,
            "density": 2.51,
            "climate_resistance": 1,
            "acid_resistance": 1,
            "alkali_resistance": 1
        }
        mat = MaterialProperties.from_dict(data)
        assert mat.name == "BK7"
        assert mat.vd == 64.2
    
    def test_material_properties_roundtrip(self):
        """Test to_dict and from_dict roundtrip"""
        original = MaterialProperties(
            name="TestMaterial",
            catalog="Schott",
            nd=1.52,
            vd=55.0,
            transmission_data=[(550.0, 0.95)],
            transmission_thickness=7.5
        )
        dict_form = original.to_dict()
        restored = MaterialProperties.from_dict(dict_form)
        assert restored.name == original.name
        assert restored.nd == original.nd
        assert restored.transmission_thickness == original.transmission_thickness
        assert restored.transmission_data == original.transmission_data


class TestMaterialDatabaseClearCache:
    """Test MaterialDatabase.clear_cache method"""
    
    def test_clear_cache_clears_internal_state(self):
        """clear_cache should reset _index_cache and _alias_keys"""
        db = MaterialDatabase()
        db._index_cache = {"key": "value"}
        db._alias_keys = {"alias": "real"}
        
        db.clear_cache()
        
        assert len(db._index_cache) == 0
        assert db._alias_keys is None
    
    def test_clear_cache_idempotent(self):
        """clear_cache should be safe to call multiple times"""
        db = MaterialDatabase()
        db.clear_cache()
        db.clear_cache()
        db.clear_cache()
        assert len(db._index_cache) == 0
    
    def test_clear_cache_instance_specific(self):
        """clear_cache on one instance should not affect other instances"""
        db1 = MaterialDatabase()
        db2 = MaterialDatabase()
        
        db1._index_cache["key1"] = "val1"
        db2._index_cache["key2"] = "val2"
        
        db1.clear_cache()
        
        assert len(db1._index_cache) == 0
        assert "key2" in db2._index_cache


class TestMaterialDatabaseGetTransmission:
    """Test MaterialDatabase.get_transmission method"""
    
    def test_get_transmission_fallback_no_material(self):
        """When material not found, use default fallback"""
        db = MaterialDatabase()
        with patch.object(db, 'get_material', return_value=None):
            result = db.get_transmission("NonExistent", 550.0)
        
        expected = 0.99 ** 1.0  # thickness 10mm / reference 10mm
        assert result == pytest.approx(expected)
    
    def test_get_transmission_fallback_half_thickness(self):
        """Fallback scales with thickness correctly (half)"""
        db = MaterialDatabase()
        with patch.object(db, 'get_material', return_value=None):
            result = db.get_transmission("NonExistent", 550.0, thickness_mm=5.0)
        
        expected = 0.99 ** 0.5
        assert result == pytest.approx(expected)
    
    def test_get_transmission_fallback_double_thickness(self):
        """Fallback scales with thickness correctly (double)"""
        db = MaterialDatabase()
        with patch.object(db, 'get_material', return_value=None):
            result = db.get_transmission("NonExistent", 550.0, thickness_mm=20.0)
        
        expected = 0.99 ** 2.0
        assert result == pytest.approx(expected)
    
    def test_get_transmission_no_transmission_data(self):
        """Material with None transmission_data uses fallback"""
        db = MaterialDatabase()
        mat = MaterialProperties(
            name="NoData",
            catalog="Schott",
            nd=1.5,
            vd=50.0,
            transmission_data=None
        )
        with patch.object(db, 'get_material', return_value=mat):
            result = db.get_transmission("NoData", 550.0, thickness_mm=5.0)
        
        expected = 0.99 ** 0.5
        assert result == pytest.approx(expected)
    
    def test_get_transmission_beer_lambert_scaling(self):
        """Test Beer-Lambert law: T2 = T1^(d2/d1)"""
        db = MaterialDatabase()
        mat = MaterialProperties(
            name="BK7",
            catalog="Schott",
            nd=1.517,
            vd=64.2,
            transmission_data=[(550.0, 0.98)],
            transmission_thickness=10.0
        )
        
        with patch.object(db, 'get_material', return_value=mat):
            with patch.object(db, '_interpolate_transmission', return_value=0.98):
                t_ref = db.get_transmission("BK7", 550.0, thickness_mm=10.0)
                t_half = db.get_transmission("BK7", 550.0, thickness_mm=5.0)
                t_double = db.get_transmission("BK7", 550.0, thickness_mm=20.0)
        
        assert t_ref == pytest.approx(0.98)
        assert t_half == pytest.approx(0.98 ** 0.5)
        assert t_double == pytest.approx(0.98 ** 2.0)
        assert t_double < t_ref < t_half
    
    def test_get_transmission_zero_base_returns_zero(self):
        """Zero base transmission should return 0.0"""
        db = MaterialDatabase()
        mat = MaterialProperties(
            name="Black",
            catalog="Schott",
            nd=1.5,
            vd=50.0,
            transmission_data=[(550.0, 0.0)]
        )
        
        with patch.object(db, 'get_material', return_value=mat):
            with patch.object(db, '_interpolate_transmission', return_value=0.0):
                result = db.get_transmission("Black", 550.0, thickness_mm=5.0)
        
        assert result == 0.0
    
    def test_get_transmission_negative_base_returns_zero(self):
        """Negative base transmission should return 0.0"""
        db = MaterialDatabase()
        mat = MaterialProperties(
            name="Invalid",
            catalog="Schott",
            nd=1.5,
            vd=50.0,
            transmission_data=[(550.0, -0.05)]
        )
        
        with patch.object(db, 'get_material', return_value=mat):
            with patch.object(db, '_interpolate_transmission', return_value=-0.05):
                result = db.get_transmission("Invalid", 550.0, thickness_mm=5.0)
        
        assert result == 0.0
    
    def test_get_transmission_physical_bounds(self):
        """Transmission values should stay within [0, 1]"""
        db = MaterialDatabase()
        mat = MaterialProperties(
            name="BK7",
            catalog="Schott",
            nd=1.517,
            vd=64.2,
            transmission_data=[(550.0, 0.95)],
            transmission_thickness=10.0
        )
        
        with patch.object(db, 'get_material', return_value=mat):
            with patch.object(db, '_interpolate_transmission', return_value=0.95):
                thin = db.get_transmission("BK7", 550.0, thickness_mm=0.5)
                thick = db.get_transmission("BK7", 550.0, thickness_mm=100.0)
        
        assert 0.0 <= thin <= 1.0
        assert 0.0 <= thick <= 1.0

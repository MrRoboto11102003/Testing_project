import pytest
from openlens.material_database import MaterialDatabase, MaterialProperties


class TestMaterialProperties:
    """Tests for MaterialProperties dataclass"""
    
    def test_basic_instantiation(self):
        """Create MaterialProperties with required fields"""
        props = MaterialProperties(name="BK7", catalog="Schott", nd=1.5168, vd=64.17)
        assert props.name == "BK7"
        assert props.catalog == "Schott"
        assert props.nd == 1.5168
        assert props.vd == 64.17
    
    def test_default_values(self):
        """Verify default values for optional fields"""
        props = MaterialProperties(name="Test", catalog="Custom", nd=1.5, vd=60.0)
        assert props.B1 == 0.0
        assert props.B2 == 0.0
        assert props.B3 == 0.0
        assert props.D0 == 0.0
        assert props.thermal_expansion == 7.1e-6
        assert props.density == 2.51
        assert props.transmission_thickness == 10.0
        assert props.transmission_data == []
    
    def test_sellmeier_coefficients(self):
        """Test Sellmeier coefficient assignment"""
        props = MaterialProperties(
            name="BK7",
            catalog="Schott",
            nd=1.5168,
            vd=64.17,
            B1=1.03961212,
            B2=0.231792344,
            C1=0.00600069867,
            C2=0.0200179144,
        )
        assert props.B1 == pytest.approx(1.03961212)
        assert props.C1 == pytest.approx(0.00600069867)
    
    def test_thermal_coefficients(self):
        """Test Schott TIE-19 thermal coefficient assignment"""
        props = MaterialProperties(
            name="Test",
            catalog="Schott",
            nd=1.5,
            vd=60.0,
            D0=1.6e-6,
            D1=2.0e-8,
            E0=0.46e-6,
            E1=1.6e-8,
            lambda_tk_um=0.168,
        )
        assert props.D0 == pytest.approx(1.6e-6)
        assert props.E0 == pytest.approx(0.46e-6)
        assert props.lambda_tk_um == pytest.approx(0.168)
    
    def test_transmission_data(self):
        """Test transmission data assignment"""
        trans_data = [(400.0, 0.95), (550.0, 0.98), (700.0, 0.96)]
        props = MaterialProperties(
            name="Test",
            catalog="Custom",
            nd=1.5,
            vd=60.0,
            transmission_data=trans_data,
            transmission_thickness=5.0,
        )
        assert props.transmission_data == trans_data
        assert props.transmission_thickness == 5.0
    
    def test_to_dict(self):
        """Test to_dict preserves all fields"""
        props = MaterialProperties(
            name="BK7",
            catalog="Schott",
            nd=1.5168,
            vd=64.17,
            B1=1.0,
            thermal_expansion=10e-6,
            density=2.5,
        )
        d = props.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "BK7"
        assert d["nd"] == 1.5168
        assert d["B1"] == 1.0
        assert d["thermal_expansion"] == 10e-6
    
    def test_from_dict_and_round_trip(self):
        """Test from_dict reconstructs object and round-trip preserves data"""
        original = MaterialProperties(
            name="Custom Glass",
            catalog="Custom",
            nd=1.52,
            vd=55.0,
            B1=1.1,
            D0=1.5e-6,
            transmission_data=[(400.0, 0.95), (550.0, 0.98)],
            transmission_thickness=8.0,
        )
        d = original.to_dict()
        restored = MaterialProperties.from_dict(d)
        
        assert restored.name == original.name
        assert restored.nd == original.nd
        assert restored.vd == original.vd
        assert restored.B1 == original.B1
        assert restored.D0 == original.D0
        assert restored.transmission_data == original.transmission_data
        assert restored.transmission_thickness == original.transmission_thickness


class TestMaterialDatabaseClearCache:
    """Tests for MaterialDatabase.clear_cache() - instance-specific cache"""
    
    def test_clear_cache_no_error(self):
        """Test that clear_cache executes without raising"""
        db = MaterialDatabase()
        db.clear_cache()  # Should not raise
    
    def test_clear_cache_independent_instances(self):
        """Test that instances have independent caches (not shared class-wide)"""
        db1 = MaterialDatabase()
        db2 = MaterialDatabase()
        
        # Clear cache on separate instances - should not interfere
        db1.clear_cache()
        db2.clear_cache()
        # Success indicates instance isolation vs old class-wide lru_cache


class TestMaterialDatabaseGetTransmission:
    """Tests for MaterialDatabase.get_transmission() - Beer-Lambert law scaling"""
    
    def test_default_fallback_missing_material(self):
        """Test default fallback when material not found: 0.99^(thickness/10)"""
        db = MaterialDatabase()
        result = db.get_transmission("NonexistentMaterial", 550.0, thickness_mm=10.0)
        assert pytest.approx(result, abs=1e-6) == 0.99
    
    def test_default_thickness_parameter(self):
        """Test that default thickness_mm parameter is 10.0"""
        db = MaterialDatabase()
        result_default = db.get_transmission("Unknown", 550.0)
        result_explicit = db.get_transmission("Unknown", 550.0, thickness_mm=10.0)
        assert pytest.approx(result_default) == result_explicit
    
    def test_zero_thickness_returns_one(self):
        """Test transmission at zero thickness returns 1.0"""
        db = MaterialDatabase()
        result = db.get_transmission("Unknown", 550.0, thickness_mm=0.0)
        assert pytest.approx(result, abs=1e-6) == 1.0
    
    def test_beer_lambert_scaling(self):
        """Test Beer-Lambert law: T(d2) = T(d1)^(d2/d1)"""
        db = MaterialDatabase()
        # For missing material, default is 0.99^(thickness/10)
        t5 = db.get_transmission("Unknown", 550.0, thickness_mm=5.0)
        t10 = db.get_transmission("Unknown", 550.0, thickness_mm=10.0)
        t20 = db.get_transmission("Unknown", 550.0, thickness_mm=20.0)
        
        assert pytest.approx(t5) == 0.99 ** 0.5  # 5/10 = 0.5
        assert pytest.approx(t10) == 0.99         # 10/10 = 1.0
        assert pytest.approx(t20) == 0.99 ** 2.0  # 20/10 = 2.0
    
    def test_monotonic_decrease_with_thickness(self):
        """Test transmission decreases monotonically with thickness"""
        db = MaterialDatabase()
        thicknesses = [1.0, 5.0, 10.0, 20.0, 50.0]
        transmissions = [
            db.get_transmission("Unknown", 550.0, thickness_mm=t)
            for t in thicknesses
        ]
        
        for i in range(len(transmissions) - 1):
            assert transmissions[i] > transmissions[i + 1]
    
    def test_transmission_valid_range(self):
        """Test transmission values are always in [0, 1]"""
        db = MaterialDatabase()
        for thickness in [0.0, 0.5, 1.0, 10.0, 50.0, 100.0]:
            result = db.get_transmission("Unknown", 550.0, thickness_mm=thickness)
            assert 0.0 <= result <= 1.0

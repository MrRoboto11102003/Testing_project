import pytest
from openlens.material_database import MaterialDatabase, MaterialProperties


class TestMaterialProperties:
    """Test MaterialProperties dataclass"""
    
    def test_creation_with_required_fields(self):
        """Test creating MaterialProperties with required fields only"""
        mat = MaterialProperties(
            name="BK7",
            catalog="Schott",
            nd=1.517,
            vd=64.17
        )
        assert mat.name == "BK7"
        assert mat.catalog == "Schott"
        assert mat.nd == 1.517
        assert mat.vd == 64.17
    
    def test_default_values_initialized(self):
        """Test that default values are properly initialized"""
        mat = MaterialProperties(
            name="BK7",
            catalog="Schott",
            nd=1.517,
            vd=64.17
        )
        assert mat.B1 == 0.0
        assert mat.B2 == 0.0
        assert mat.B3 == 0.0
        assert mat.C1 == 0.0
        assert mat.transmission_data == []
        assert mat.transmission_thickness == 10.0
        assert mat.thermal_expansion == 7.1e-6
        assert mat.density == 2.51
    
    def test_creation_with_sellmeier_coefficients(self):
        """Test creating with Sellmeier coefficients"""
        mat = MaterialProperties(
            name="BK7",
            catalog="Schott",
            nd=1.517,
            vd=64.17,
            B1=1.03961212,
            B2=0.231792344,
            B3=1.01046945,
            C1=0.00610,
            C2=0.02050,
            C3=103.56
        )
        assert mat.B1 == 1.03961212
        assert mat.C2 == 0.02050
    
    def test_creation_with_transmission_data(self):
        """Test creating with transmission data"""
        transmission = [(400, 0.85), (550, 0.95), (700, 0.90)]
        mat = MaterialProperties(
            name="BK7",
            catalog="Schott",
            nd=1.517,
            vd=64.17,
            transmission_data=transmission,
            transmission_thickness=5.0
        )
        assert mat.transmission_data == transmission
        assert mat.transmission_thickness == 5.0
    
    def test_creation_with_thermal_coefficients(self):
        """Test creating with thermal coefficients"""
        mat = MaterialProperties(
            name="BK7",
            catalog="Schott",
            nd=1.517,
            vd=64.17,
            D0=1.86e-6,
            D1=-1.37e-8,
            D2=2.6e-11,
            E0=5.1e-7,
            E1=-1.7e-9,
            lambda_tk_um=0.17
        )
        assert mat.D0 == 1.86e-6
        assert mat.lambda_tk_um == 0.17
    
    def test_to_dict_conversion(self):
        """Test to_dict() converts to dictionary with all fields"""
        mat = MaterialProperties(
            name="BK7",
            catalog="Schott",
            nd=1.517,
            vd=64.17,
            B1=1.03961212
        )
        result = mat.to_dict()
        assert isinstance(result, dict)
        assert result["name"] == "BK7"
        assert result["nd"] == 1.517
        assert result["B1"] == 1.03961212
    
    def test_from_dict_creation(self):
        """Test from_dict() creates MaterialProperties from dictionary"""
        data = {
            "name": "BK7",
            "catalog": "Schott",
            "nd": 1.517,
            "vd": 64.17,
            "B1": 1.03961212
        }
        mat = MaterialProperties.from_dict(data)
        assert mat.name == "BK7"
        assert mat.nd == 1.517
        assert mat.B1 == 1.03961212
    
    def test_roundtrip_dict_conversion(self):
        """Test round-trip conversion: to_dict -> from_dict"""
        original = MaterialProperties(
            name="BK7",
            catalog="Schott",
            nd=1.517,
            vd=64.17,
            B1=1.03961212,
            transmission_data=[(400, 0.85), (550, 0.95)],
            transmission_thickness=8.0,
            D0=1.86e-6
        )
        restored = MaterialProperties.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.nd == original.nd
        assert restored.B1 == original.B1
        assert restored.transmission_data == original.transmission_data
        assert restored.transmission_thickness == original.transmission_thickness
        assert restored.D0 == original.D0


class TestMaterialDatabaseClearCache:
    """Test MaterialDatabase.clear_cache method"""
    
    def test_clear_cache_callable(self):
        """Test that clear_cache method exists and is callable"""
        db = MaterialDatabase()
        assert hasattr(db, 'clear_cache')
        assert callable(db.clear_cache)
    
    def test_clear_cache_no_exception(self):
        """Test that clear_cache executes without raising exceptions"""
        db = MaterialDatabase()
        db.clear_cache()  # Should not raise
    
    def test_clear_cache_instance_independent(self):
        """Test that clear_cache on one instance doesn't affect others"""
        db1 = MaterialDatabase()
        db2 = MaterialDatabase()
        db1.clear_cache()
        db2.clear_cache()  # Should succeed independently


class TestMaterialDatabaseGetTransmission:
    """Test MaterialDatabase.get_transmission method"""
    
    def test_get_transmission_default_fallback_at_reference_thickness(self):
        """Test default fallback at reference thickness (10mm)"""
        db = MaterialDatabase()
        # For missing material: 0.99 ** (10/10) = 0.99
        result = db.get_transmission("NonexistentMaterial", 550.0)
        assert result == pytest.approx(0.99, rel=1e-6)
    
    def test_get_transmission_fallback_scales_with_thickness(self):
        """Test that default fallback scales correctly with thickness"""
        db = MaterialDatabase()
        # At 20mm: 0.99 ** (20/10) = 0.99^2
        result = db.get_transmission("NonexistentMaterial", 550.0, thickness_mm=20.0)
        expected = 0.99 ** 2.0
        assert result == pytest.approx(expected, rel=1e-6)
    
    def test_get_transmission_fallback_at_half_thickness(self):
        """Test fallback at half reference thickness"""
        db = MaterialDatabase()
        # At 5mm: 0.99 ** (5/10) = 0.99^0.5
        result = db.get_transmission("SomeMaterial", 550.0, thickness_mm=5.0)
        expected = 0.99 ** 0.5
        assert result == pytest.approx(expected, rel=1e-6)
    
    def test_get_transmission_zero_thickness_returns_unity(self):
        """Test that zero thickness returns transmission of 1.0"""
        db = MaterialDatabase()
        result = db.get_transmission("SomeMaterial", 550.0, thickness_mm=0.0)
        assert result == pytest.approx(1.0, rel=1e-6)
    
    def test_get_transmission_returns_float(self):
        """Test that get_transmission always returns a float"""
        db = MaterialDatabase()
        result = db.get_transmission("Material", 400.0, 15.0)
        assert isinstance(result, float)
    
    def test_get_transmission_in_valid_range(self):
        """Test that transmission values are in valid range [0, 1]"""
        db = MaterialDatabase()
        for wavelength in [300, 400, 550, 700, 1000, 2000]:
            for thickness in [1, 10, 50, 100]:
                result = db.get_transmission("Material", float(wavelength), float(thickness))
                assert 0.0 <= result <= 1.0

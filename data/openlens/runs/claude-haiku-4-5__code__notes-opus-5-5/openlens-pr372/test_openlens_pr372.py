import pytest
from openlens.chromatic_analyzer import ChromaticAnalyzer


class FakeDatabase:
    """A fake material database with known, exact values for testing."""
    
    def __init__(self, materials_dict):
        """
        materials_dict: {name: {"d": n_d, "F": n_F, "C": n_C}}
        """
        self.materials = materials_dict
        # Map wavelength values to keys using ChromaticAnalyzer constants
        W = ChromaticAnalyzer.WAVELENGTHS
        self.wavelength_map = {
            W["d"]: "d",
            W["F"]: "F",
            W["C"]: "C",
        }
    
    def get_refractive_index(self, name, wavelength_nm, temperature_c=20.0):
        """Return refractive index for a given material and wavelength."""
        key = self.wavelength_map.get(wavelength_nm)
        if key is None:
            raise ValueError(f"Unknown wavelength: {wavelength_nm}")
        return self.materials[name][key]


@pytest.fixture
def analyzer_with_fake_db():
    """Create a ChromaticAnalyzer with a fake, exact database."""
    # Crown: n_d=1.5, n_F=1.51, n_C=1.50 => V1 ≈ 50
    # Flint: n_d=1.6, n_F=1.62, n_C=1.60 => V2 ≈ 30
    materials = {
        "Crown": {"d": 1.5, "F": 1.51, "C": 1.50},
        "Flint": {"d": 1.6, "F": 1.62, "C": 1.60},
        "NoDis": {"d": 1.5, "F": 1.5, "C": 1.5},  # No dispersion
    }
    db = FakeDatabase(materials)
    return ChromaticAnalyzer(material_db=db)


@pytest.fixture
def analyzer_real_db():
    """Create a ChromaticAnalyzer with the real database."""
    return ChromaticAnalyzer()


class TestDesignAchromaticDoublet:
    """Test suite for design_achromatic_doublet method."""
    
    def test_basic_design_with_fake_db(self, analyzer_with_fake_db):
        """Test a basic doublet design with known values."""
        result = analyzer_with_fake_db.design_achromatic_doublet(
            focal_length=100.0,
            crown_material="Crown",
            flint_material="Flint"
        )
        
        # Check structure
        assert set(result.keys()) == {
            "crown_element", "flint_element",
            "total_focal_length", "total_power"
        }
        assert set(result["crown_element"].keys()) == {
            "focal_length", "power", "material", "n", "abbe_number"
        }
        assert set(result["flint_element"].keys()) == {
            "focal_length", "power", "material", "n", "abbe_number"
        }
        
        # Check values
        assert result["total_focal_length"] == 100.0
        assert result["total_power"] == pytest.approx(0.01)
        assert result["crown_element"]["n"] == 1.5
        assert result["flint_element"]["n"] == 1.6
    
    def test_power_sum_identity(self, analyzer_with_fake_db):
        """Test that phi1 + phi2 = phi_total."""
        result = analyzer_with_fake_db.design_achromatic_doublet(
            focal_length=100.0,
            crown_material="Crown",
            flint_material="Flint"
        )
        
        phi1 = result["crown_element"]["power"]
        phi2 = result["flint_element"]["power"]
        phi_total = result["total_power"]
        
        assert phi1 + phi2 == pytest.approx(phi_total, rel=1e-10)
    
    def test_achromat_condition(self, analyzer_with_fake_db):
        """Test that phi1/V1 + phi2/V2 ≈ 0 (achromat condition)."""
        result = analyzer_with_fake_db.design_achromatic_doublet(
            focal_length=100.0,
            crown_material="Crown",
            flint_material="Flint"
        )
        
        phi1 = result["crown_element"]["power"]
        phi2 = result["flint_element"]["power"]
        V1 = result["crown_element"]["abbe_number"]
        V2 = result["flint_element"]["abbe_number"]
        
        condition = phi1 / V1 + phi2 / V2
        assert condition == pytest.approx(0.0, abs=1e-12)
    
    def test_focal_length_reciprocal_power(self, analyzer_with_fake_db):
        """Test that f = 1/phi for each element."""
        result = analyzer_with_fake_db.design_achromatic_doublet(
            focal_length=50.0,
            crown_material="Crown",
            flint_material="Flint"
        )
        
        # Crown
        phi1 = result["crown_element"]["power"]
        f1 = result["crown_element"]["focal_length"]
        if phi1 != 0:
            assert f1 == pytest.approx(1.0 / phi1)
        
        # Flint
        phi2 = result["flint_element"]["power"]
        f2 = result["flint_element"]["focal_length"]
        if phi2 != 0:
            assert f2 == pytest.approx(1.0 / phi2)
    
    def test_negative_focal_length(self, analyzer_with_fake_db):
        """Test with negative focal length."""
        result = analyzer_with_fake_db.design_achromatic_doublet(
            focal_length=-100.0,
            crown_material="Crown",
            flint_material="Flint"
        )
        
        assert result["total_focal_length"] == -100.0
        assert result["total_power"] == pytest.approx(-0.01)
        # Both powers should flip sign compared to positive focal length
        assert result["crown_element"]["power"] < 0
        assert result["flint_element"]["power"] > 0
    
    def test_default_materials(self, analyzer_real_db):
        """Test with default materials (BK7 and F2)."""
        result = analyzer_real_db.design_achromatic_doublet(focal_length=100.0)
        
        assert result["crown_element"]["material"] == "BK7"
        assert result["flint_element"]["material"] == "F2"
        assert result["total_focal_length"] == 100.0
        
        # Crown should have higher Abbe number than flint
        assert result["crown_element"]["abbe_number"] > result["flint_element"]["abbe_number"]
        # Crown positive, flint negative for positive focal length
        assert result["crown_element"]["power"] > 0
        assert result["flint_element"]["power"] < 0
    
    def test_infinite_focal_length(self, analyzer_with_fake_db):
        """Test with infinite focal length."""
        result = analyzer_with_fake_db.design_achromatic_doublet(
            focal_length=float('inf'),
            crown_material="Crown",
            flint_material="Flint"
        )
        
        assert result["total_focal_length"] == float('inf')
        assert result["total_power"] == 0.0
        assert result["crown_element"]["power"] == 0.0
        assert result["flint_element"]["power"] == 0.0
        assert result["crown_element"]["focal_length"] == float('inf')
        assert result["flint_element"]["focal_length"] == float('inf')
    
    def test_zero_crown_abbe_number(self):
        """Test when crown has zero Abbe number."""
        materials = {
            "ZeroPower": {"d": 1.0, "F": 1.0, "C": 1.0},
            "Normal": {"d": 1.5, "F": 1.51, "C": 1.49},
        }
        db = FakeDatabase(materials)
        analyzer = ChromaticAnalyzer(material_db=db)
        
        result = analyzer.design_achromatic_doublet(
            focal_length=100.0,
            crown_material="ZeroPower",
            flint_material="Normal"
        )
        
        # Crown power should be 0 and focal length infinite
        assert result["crown_element"]["power"] == 0.0
        assert result["crown_element"]["focal_length"] == float('inf')
        # Flint carries all power
        assert result["flint_element"]["power"] == pytest.approx(0.01)
    
    def test_material_names_round_trip(self, analyzer_with_fake_db):
        """Test that material names are preserved in output."""
        result = analyzer_with_fake_db.design_achromatic_doublet(
            focal_length=100.0,
            crown_material="Crown",
            flint_material="Flint"
        )
        
        assert result["crown_element"]["material"] == "Crown"
        assert result["flint_element"]["material"] == "Flint"
    
    def test_abbe_numbers_match_calculated(self, analyzer_with_fake_db):
        """Test that returned Abbe numbers match calculate_abbe_number."""
        result = analyzer_with_fake_db.design_achromatic_doublet(
            focal_length=100.0,
            crown_material="Crown",
            flint_material="Flint"
        )
        
        V1_ret = result["crown_element"]["abbe_number"]
        V1_calc = analyzer_with_fake_db.calculate_abbe_number("Crown")
        assert V1_ret == pytest.approx(V1_calc)
        
        V2_ret = result["flint_element"]["abbe_number"]
        V2_calc = analyzer_with_fake_db.calculate_abbe_number("Flint")
        assert V2_ret == pytest.approx(V2_calc)
    
    def test_focal_length_zero_raises_zero_division_error(self, analyzer_with_fake_db):
        """Test that focal_length=0 raises ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            analyzer_with_fake_db.design_achromatic_doublet(
                focal_length=0.0,
                crown_material="Crown",
                flint_material="Flint"
            )
    
    def test_crown_no_dispersion_raises_value_error(self, analyzer_with_fake_db):
        """Test that crown with no dispersion raises ValueError."""
        with pytest.raises(ValueError, match="Material 'NoDis' has no dispersion"):
            analyzer_with_fake_db.design_achromatic_doublet(
                focal_length=100.0,
                crown_material="NoDis",
                flint_material="Flint"
            )
    
    def test_flint_no_dispersion_raises_value_error(self, analyzer_with_fake_db):
        """Test that flint with no dispersion raises ValueError."""
        with pytest.raises(ValueError, match="Material 'NoDis' has no dispersion"):
            analyzer_with_fake_db.design_achromatic_doublet(
                focal_length=100.0,
                crown_material="Crown",
                flint_material="NoDis"
            )
    
    def test_equal_abbe_numbers_raises_value_error_fake_db(self):
        """Test that equal Abbe numbers raise ValueError with fake database."""
        materials = {
            "Glass1": {"d": 1.5, "F": 1.52, "C": 1.48},
            "Glass2": {"d": 1.5, "F": 1.52, "C": 1.48},
        }
        db = FakeDatabase(materials)
        analyzer = ChromaticAnalyzer(material_db=db)
        
        with pytest.raises(ValueError, match="Cannot design an achromat from 'Glass1' and 'Glass2'"):
            analyzer.design_achromatic_doublet(
                focal_length=100.0,
                crown_material="Glass1",
                flint_material="Glass2"
            )
    
    def test_equal_abbe_numbers_raises_value_error_real_db(self, analyzer_real_db):
        """Test that equal Abbe numbers raise ValueError with real database."""
        with pytest.raises(ValueError, match="Cannot design an achromat from 'BK7' and 'BK7'"):
            analyzer_real_db.design_achromatic_doublet(
                focal_length=100.0,
                crown_material="BK7",
                flint_material="BK7"
            )

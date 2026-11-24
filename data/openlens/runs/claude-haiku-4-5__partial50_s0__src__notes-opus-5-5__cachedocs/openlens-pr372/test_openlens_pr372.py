import pytest
from openlens.chromatic_analyzer import ChromaticAnalyzer


class FakeDB:
    """Fake material database for testing with precise values."""
    def __init__(self, materials):
        self.materials = materials
    
    def get_refractive_index(self, name, wavelength_nm, temperature_c=20.0):
        """Get refractive index from fake database."""
        W = ChromaticAnalyzer.WAVELENGTHS
        wl_key = {
            W["d"]: "d",
            W["F"]: "F",
            W["C"]: "C",
        }[wavelength_nm]
        return self.materials[name][wl_key]


@pytest.fixture
def simple_fake_db():
    """Fake database with normal crown (high V) and flint (low V)."""
    materials = {
        "CROWN": {"d": 1.50, "F": 1.51, "C": 1.50},
        "FLINT": {"d": 1.60, "F": 1.62, "C": 1.60},
    }
    return FakeDB(materials)


@pytest.fixture
def edge_case_fake_db():
    """Fake database for edge cases and error conditions."""
    materials = {
        "NO_DISP": {"d": 1.5, "F": 1.5, "C": 1.5},
        "HIGH_V": {"d": 1.5, "F": 1.51, "C": 1.50},
        "LOW_V": {"d": 1.6, "F": 1.62, "C": 1.60},
        "SAME_V": {"d": 1.55, "F": 1.56, "C": 1.55},
        "ZERO_INDEX": {"d": 1.0, "F": 1.0, "C": 1.0},
    }
    return FakeDB(materials)


class TestDesignAchromaticDoublet:
    """Test suite for ChromaticAnalyzer.design_achromatic_doublet."""

    def test_return_structure_and_keys(self, simple_fake_db):
        """Test that return dict has correct structure and all required keys."""
        analyzer = ChromaticAnalyzer(material_db=simple_fake_db)
        result = analyzer.design_achromatic_doublet(100.0, "CROWN", "FLINT")
        
        # Top-level keys
        assert set(result.keys()) == {
            "crown_element", "flint_element", "total_focal_length", "total_power"
        }
        
        # Element keys
        for elem_key in ["crown_element", "flint_element"]:
            assert set(result[elem_key].keys()) == {
                "focal_length", "power", "material", "n", "abbe_number"
            }

    def test_power_sum_identity(self, simple_fake_db):
        """Test that phi1 + phi2 = phi_total (power conservation)."""
        analyzer = ChromaticAnalyzer(material_db=simple_fake_db)
        result = analyzer.design_achromatic_doublet(100.0, "CROWN", "FLINT")
        
        phi_sum = (result["crown_element"]["power"] + 
                   result["flint_element"]["power"])
        assert phi_sum == pytest.approx(result["total_power"], rel=1e-9)

    def test_achromat_condition_identity(self, simple_fake_db):
        """Test that phi1/V1 + phi2/V2 ≈ 0 (achromatic constraint)."""
        analyzer = ChromaticAnalyzer(material_db=simple_fake_db)
        result = analyzer.design_achromatic_doublet(100.0, "CROWN", "FLINT")
        
        V1 = result["crown_element"]["abbe_number"]
        V2 = result["flint_element"]["abbe_number"]
        phi1 = result["crown_element"]["power"]
        phi2 = result["flint_element"]["power"]
        
        achromat_check = phi1/V1 + phi2/V2
        assert achromat_check == pytest.approx(0.0, abs=1e-11)

    def test_total_focal_length_preserved_as_input(self, simple_fake_db):
        """Test that total_focal_length is the input object itself."""
        analyzer = ChromaticAnalyzer(material_db=simple_fake_db)
        
        # Test with integer
        result_int = analyzer.design_achromatic_doublet(100, "CROWN", "FLINT")
        assert result_int["total_focal_length"] == 100
        assert isinstance(result_int["total_focal_length"], int)
        
        # Test with float
        result_float = analyzer.design_achromatic_doublet(100.5, "CROWN", "FLINT")
        assert result_float["total_focal_length"] == 100.5
        assert isinstance(result_float["total_focal_length"], float)

    def test_material_names_round_trip(self, simple_fake_db):
        """Test that material names are preserved exactly."""
        analyzer = ChromaticAnalyzer(material_db=simple_fake_db)
        result = analyzer.design_achromatic_doublet(100.0, "CROWN", "FLINT")
        
        assert result["crown_element"]["material"] == "CROWN"
        assert result["flint_element"]["material"] == "FLINT"

    def test_focal_length_zero_raises_zero_division_error(self, simple_fake_db):
        """Test that focal_length=0 raises ZeroDivisionError."""
        analyzer = ChromaticAnalyzer(material_db=simple_fake_db)
        
        with pytest.raises(ZeroDivisionError):
            analyzer.design_achromatic_doublet(0.0, "CROWN", "FLINT")

    def test_equal_abbe_numbers_raises_value_error(self, edge_case_fake_db):
        """Test that identical Abbe numbers raise ValueError."""
        analyzer = ChromaticAnalyzer(material_db=edge_case_fake_db)
        
        with pytest.raises(ValueError, match="Cannot design an achromat"):
            analyzer.design_achromatic_doublet(100.0, "SAME_V", "SAME_V")

    def test_crown_no_dispersion_raises_value_error(self, edge_case_fake_db):
        """Test that crown with insufficient dispersion raises ValueError."""
        analyzer = ChromaticAnalyzer(material_db=edge_case_fake_db)
        
        with pytest.raises(ValueError, match="no dispersion"):
            analyzer.design_achromatic_doublet(100.0, "NO_DISP", "LOW_V")

    def test_flint_no_dispersion_raises_value_error(self, edge_case_fake_db):
        """Test that flint with insufficient dispersion raises ValueError."""
        analyzer = ChromaticAnalyzer(material_db=edge_case_fake_db)
        
        with pytest.raises(ValueError, match="no dispersion"):
            analyzer.design_achromatic_doublet(100.0, "HIGH_V", "NO_DISP")

    def test_negative_focal_length_flips_powers(self, simple_fake_db):
        """Test that negative focal length correctly reverses power signs."""
        analyzer = ChromaticAnalyzer(material_db=simple_fake_db)
        
        result_pos = analyzer.design_achromatic_doublet(100.0, "CROWN", "FLINT")
        result_neg = analyzer.design_achromatic_doublet(-100.0, "CROWN", "FLINT")
        
        # Powers should flip signs
        assert result_neg["total_power"] == pytest.approx(
            -result_pos["total_power"]
        )
        assert result_neg["crown_element"]["power"] == pytest.approx(
            -result_pos["crown_element"]["power"]
        )
        assert result_neg["flint_element"]["power"] == pytest.approx(
            -result_pos["flint_element"]["power"]
        )

    def test_power_signs_crown_positive_flint_negative(self, simple_fake_db):
        """Test that for positive f, crown power > 0 and flint power < 0."""
        analyzer = ChromaticAnalyzer(material_db=simple_fake_db)
        result = analyzer.design_achromatic_doublet(100.0, "CROWN", "FLINT")
        
        assert result["crown_element"]["power"] > 0
        assert result["flint_element"]["power"] < 0

    def test_infinite_focal_length_afocal_system(self, simple_fake_db):
        """Test afocal system with infinite focal length."""
        analyzer = ChromaticAnalyzer(material_db=simple_fake_db)
        result = analyzer.design_achromatic_doublet(
            focal_length=float('inf'),
            crown_material="CROWN",
            flint_material="FLINT",
        )
        
        assert result["total_power"] == 0.0
        assert result["crown_element"]["power"] == pytest.approx(0.0)
        assert result["flint_element"]["power"] == pytest.approx(0.0)
        assert result["crown_element"]["focal_length"] == float('inf')
        assert result["flint_element"]["focal_length"] == float('inf')

    def test_refractive_indices_from_database(self, simple_fake_db):
        """Test that refractive indices are retrieved from database."""
        analyzer = ChromaticAnalyzer(material_db=simple_fake_db)
        result = analyzer.design_achromatic_doublet(100.0, "CROWN", "FLINT")
        
        assert result["crown_element"]["n"] == 1.50
        assert result["flint_element"]["n"] == 1.60

    def test_abbe_numbers_calculated_correctly(self, simple_fake_db):
        """Test that Abbe numbers are calculated from V = (n_d - 1)/(n_F - n_C)."""
        analyzer = ChromaticAnalyzer(material_db=simple_fake_db)
        result = analyzer.design_achromatic_doublet(100.0, "CROWN", "FLINT")
        
        # CROWN: V = (1.50 - 1) / (1.51 - 1.50) = 0.50 / 0.01 = 50.0
        assert result["crown_element"]["abbe_number"] == pytest.approx(50.0)
        
        # FLINT: V = (1.60 - 1) / (1.62 - 1.60) = 0.60 / 0.02 = 30.0
        assert result["flint_element"]["abbe_number"] == pytest.approx(30.0)

    def test_very_small_focal_length(self, simple_fake_db):
        """Test with very small focal length produces large powers."""
        analyzer = ChromaticAnalyzer(material_db=simple_fake_db)
        result = analyzer.design_achromatic_doublet(0.1, "CROWN", "FLINT")
        
        assert result["total_power"] == pytest.approx(10.0)
        assert result["crown_element"]["power"] > 0
        assert result["flint_element"]["power"] < 0

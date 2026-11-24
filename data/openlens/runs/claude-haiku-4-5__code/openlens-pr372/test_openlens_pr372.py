import pytest
import numpy as np
from openlens.chromatic_analyzer import ChromaticAnalyzer


class TestDesignAchromaticDoublet:
    """Test suite for ChromaticAnalyzer.design_achromatic_doublet method."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a ChromaticAnalyzer instance for testing."""
        return ChromaticAnalyzer()
    
    def test_basic_design_default_materials(self, analyzer):
        """Test basic achromatic doublet design with default materials."""
        focal_length = 100.0
        result = analyzer.design_achromatic_doublet(focal_length)
        
        assert isinstance(result, dict)
        assert result["total_focal_length"] == focal_length
        assert np.isclose(result["total_power"], 1.0 / focal_length)
    
    def test_return_structure(self, analyzer):
        """Verify returned dictionary contains all required keys and structure."""
        result = analyzer.design_achromatic_doublet(100.0)
        
        # Check top-level keys
        assert "crown_element" in result
        assert "flint_element" in result
        assert "total_focal_length" in result
        assert "total_power" in result
        
        # Check element structure
        for element_key in ["crown_element", "flint_element"]:
            element = result[element_key]
            assert "focal_length" in element
            assert "power" in element
            assert "material" in element
            assert "n" in element
            assert "abbe_number" in element
    
    def test_power_conservation(self, analyzer):
        """Test that individual powers sum to total power (thin lens approximation)."""
        focal_length = 100.0
        result = analyzer.design_achromatic_doublet(focal_length)
        
        phi1 = result["crown_element"]["power"]
        phi2 = result["flint_element"]["power"]
        phi_total = result["total_power"]
        
        assert np.isclose(phi1 + phi2, phi_total, atol=1e-10)
    
    def test_achromatism_condition(self, analyzer):
        """Test that the achromatism condition φ1/V1 + φ2/V2 ≈ 0 is satisfied."""
        focal_length = 100.0
        result = analyzer.design_achromatic_doublet(focal_length)
        
        phi1 = result["crown_element"]["power"]
        phi2 = result["flint_element"]["power"]
        V1 = result["crown_element"]["abbe_number"]
        V2 = result["flint_element"]["abbe_number"]
        
        # Achromatism condition should equal zero
        achromatism = phi1 / V1 + phi2 / V2
        assert np.isclose(achromatism, 0.0, atol=1e-10)
    
    def test_same_abbe_number_error(self, analyzer):
        """Test ValueError raised when crown and flint materials have same Abbe number."""
        # Same material should have same Abbe number
        with pytest.raises(ValueError) as exc_info:
            analyzer.design_achromatic_doublet(100.0, crown_material="BK7", flint_material="BK7")
        
        error_text = str(exc_info.value)
        assert "achromat" in error_text.lower() or "abbe" in error_text.lower()
    
    def test_crown_material_stored(self, analyzer):
        """Test that the specified crown material is correctly stored."""
        result = analyzer.design_achromatic_doublet(100.0, crown_material="BK7")
        assert result["crown_element"]["material"] == "BK7"
    
    def test_flint_material_stored(self, analyzer):
        """Test that the specified flint material is correctly stored."""
        result = analyzer.design_achromatic_doublet(100.0, flint_material="F2")
        assert result["flint_element"]["material"] == "F2"
    
    def test_converging_system(self, analyzer):
        """Test design for positive focal length (converging system)."""
        focal_length = 50.0
        result = analyzer.design_achromatic_doublet(focal_length)
        
        assert result["total_power"] > 0
        # Crown (positive lens) should have positive power
        assert result["crown_element"]["power"] > 0
        # Flint (negative lens) should have negative power
        assert result["flint_element"]["power"] < 0
    
    def test_diverging_system(self, analyzer):
        """Test design for negative focal length (diverging system)."""
        focal_length = -100.0
        result = analyzer.design_achromatic_doublet(focal_length)
        
        assert result["total_focal_length"] == focal_length
        assert result["total_power"] < 0
    
    def test_focal_length_power_reciprocal(self, analyzer):
        """Test that individual focal lengths are reciprocals of their powers."""
        result = analyzer.design_achromatic_doublet(100.0)
        
        phi1 = result["crown_element"]["power"]
        phi2 = result["flint_element"]["power"]
        f1 = result["crown_element"]["focal_length"]
        f2 = result["flint_element"]["focal_length"]
        
        assert np.isclose(f1, 1.0 / phi1)
        assert np.isclose(f2, 1.0 / phi2)
    
    def test_crown_higher_abbe_than_flint(self, analyzer):
        """Test that crown material has higher Abbe number than flint."""
        result = analyzer.design_achromatic_doublet(100.0)
        
        V_crown = result["crown_element"]["abbe_number"]
        V_flint = result["flint_element"]["abbe_number"]
        
        # Crown should have lower dispersion (higher Abbe number)
        assert V_crown > V_flint
    
    def test_refractive_indices_physical_values(self, analyzer):
        """Test that refractive indices are in expected physical range for optical glass."""
        result = analyzer.design_achromatic_doublet(100.0)
        
        n_crown = result["crown_element"]["n"]
        n_flint = result["flint_element"]["n"]
        
        # Standard optical glass typically has n between 1.5 and 1.9
        assert 1.0 < n_crown < 2.0
        assert 1.0 < n_flint < 2.0
    
    def test_large_focal_length(self, analyzer):
        """Test design with very large focal length."""
        focal_length = 10000.0
        result = analyzer.design_achromatic_doublet(focal_length)
        
        assert result["total_focal_length"] == focal_length
        assert np.isclose(result["total_power"], 1.0 / focal_length, atol=1e-15)
    
    def test_small_focal_length(self, analyzer):
        """Test design with small focal length."""
        focal_length = 1.0
        result = analyzer.design_achromatic_doublet(focal_length)
        
        assert result["total_focal_length"] == focal_length
        assert np.isclose(result["total_power"], 1.0 / focal_length)

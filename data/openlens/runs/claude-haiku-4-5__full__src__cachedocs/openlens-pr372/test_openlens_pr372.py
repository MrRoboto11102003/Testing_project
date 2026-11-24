import pytest
from openlens.chromatic_analyzer import ChromaticAnalyzer


class TestDesignAchromaticDoublet:
    """Test suite for ChromaticAnalyzer.design_achromatic_doublet method"""

    @pytest.fixture
    def analyzer(self):
        """Create a ChromaticAnalyzer instance for testing"""
        return ChromaticAnalyzer()

    def test_returns_dictionary(self, analyzer):
        """Test that design_achromatic_doublet returns a dictionary"""
        result = analyzer.design_achromatic_doublet(
            focal_length=100.0,
            crown_material="BK7",
            flint_material="F2"
        )
        assert isinstance(result, dict)

    def test_dictionary_structure(self, analyzer):
        """Test that result dictionary contains all required keys"""
        result = analyzer.design_achromatic_doublet(
            focal_length=100.0,
            crown_material="BK7",
            flint_material="F2"
        )
        assert "crown_element" in result
        assert "flint_element" in result
        assert "total_focal_length" in result
        assert "total_power" in result

    def test_crown_element_fields(self, analyzer):
        """Test that crown_element contains all required fields"""
        result = analyzer.design_achromatic_doublet(
            focal_length=100.0,
            crown_material="BK7",
            flint_material="F2"
        )
        crown = result["crown_element"]
        required_fields = {"focal_length", "power", "material", "n", "abbe_number"}
        assert required_fields.issubset(crown.keys())

    def test_flint_element_fields(self, analyzer):
        """Test that flint_element contains all required fields"""
        result = analyzer.design_achromatic_doublet(
            focal_length=100.0,
            crown_material="BK7",
            flint_material="F2"
        )
        flint = result["flint_element"]
        required_fields = {"focal_length", "power", "material", "n", "abbe_number"}
        assert required_fields.issubset(flint.keys())

    def test_power_sum_condition(self, analyzer):
        """Test that φ1 + φ2 = φ_total"""
        result = analyzer.design_achromatic_doublet(
            focal_length=100.0,
            crown_material="BK7",
            flint_material="F2"
        )
        phi1 = result["crown_element"]["power"]
        phi2 = result["flint_element"]["power"]
        phi_total = result["total_power"]
        
        assert abs((phi1 + phi2) - phi_total) < 1e-10

    def test_achromatism_condition(self, analyzer):
        """Test achromatism condition: φ1/V1 + φ2/V2 ≈ 0"""
        result = analyzer.design_achromatic_doublet(
            focal_length=100.0,
            crown_material="BK7",
            flint_material="F2"
        )
        phi1 = result["crown_element"]["power"]
        phi2 = result["flint_element"]["power"]
        V1 = result["crown_element"]["abbe_number"]
        V2 = result["flint_element"]["abbe_number"]
        
        achromatism_error = abs(phi1 / V1 + phi2 / V2)
        assert achromatism_error < 1e-10

    def test_focal_length_reciprocal(self, analyzer):
        """Test that focal_length = 1 / phi_total"""
        focal_length = 50.0
        result = analyzer.design_achromatic_doublet(
            focal_length=focal_length,
            crown_material="BK7",
            flint_material="F2"
        )
        
        calculated_fl = 1.0 / result["total_power"]
        assert abs(calculated_fl - focal_length) < 1e-10

    def test_total_focal_length_matches_input(self, analyzer):
        """Test that total_focal_length in result matches input"""
        focal_length = 75.0
        result = analyzer.design_achromatic_doublet(
            focal_length=focal_length,
            crown_material="BK7",
            flint_material="F2"
        )
        assert result["total_focal_length"] == focal_length

    def test_same_abbe_number_raises_error(self, analyzer):
        """Test that ValueError is raised for identical materials"""
        with pytest.raises(ValueError, match="Cannot design an achromat"):
            analyzer.design_achromatic_doublet(
                focal_length=100.0,
                crown_material="BK7",
                flint_material="BK7"
            )

    def test_error_message_contains_guidance(self, analyzer):
        """Test that error message provides helpful context"""
        with pytest.raises(ValueError) as exc_info:
            analyzer.design_achromatic_doublet(
                focal_length=100.0,
                crown_material="BK7",
                flint_material="BK7"
            )
        error_msg = str(exc_info.value)
        assert "different glasses are required" in error_msg

    def test_multiple_focal_lengths(self, analyzer):
        """Test design works correctly for various focal lengths"""
        focal_lengths = [5.0, 50.0, 100.0, 500.0, 1000.0]
        for fl in focal_lengths:
            result = analyzer.design_achromatic_doublet(
                focal_length=fl,
                crown_material="BK7",
                flint_material="F2"
            )
            assert abs(result["total_focal_length"] - fl) < 1e-10

    def test_material_names_preserved(self, analyzer):
        """Test that input material names are preserved in output"""
        result = analyzer.design_achromatic_doublet(
            focal_length=100.0,
            crown_material="BK7",
            flint_material="F2"
        )
        assert result["crown_element"]["material"] == "BK7"
        assert result["flint_element"]["material"] == "F2"

    def test_refractive_indices_valid(self, analyzer):
        """Test that refractive indices are physically reasonable"""
        result = analyzer.design_achromatic_doublet(
            focal_length=100.0,
            crown_material="BK7",
            flint_material="F2"
        )
        
        n1 = result["crown_element"]["n"]
        n2 = result["flint_element"]["n"]
        
        assert 1.0 < n1 < 2.5  # Typical range for optical glass
        assert 1.0 < n2 < 2.5

    def test_abbe_numbers_valid(self, analyzer):
        """Test that Abbe numbers are physically reasonable"""
        result = analyzer.design_achromatic_doublet(
            focal_length=100.0,
            crown_material="BK7",
            flint_material="F2"
        )
        
        V1 = result["crown_element"]["abbe_number"]
        V2 = result["flint_element"]["abbe_number"]
        
        # Typical range for glass Abbe numbers
        assert 20 < V1 < 85
        assert 20 < V2 < 85

    def test_powers_opposite_signs(self, analyzer):
        """Test that individual powers have opposite signs for doublet"""
        result = analyzer.design_achromatic_doublet(
            focal_length=100.0,
            crown_material="BK7",
            flint_material="F2"
        )
        
        phi1 = result["crown_element"]["power"]
        phi2 = result["flint_element"]["power"]
        
        # For typical achromatic doublet, powers oppose
        assert phi1 * phi2 < 0

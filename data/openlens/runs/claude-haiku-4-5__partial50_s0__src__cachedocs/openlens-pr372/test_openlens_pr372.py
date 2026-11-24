import pytest
from unittest.mock import Mock
from openlens.chromatic_analyzer import ChromaticAnalyzer


class TestDesignAchromaticDoublet:
    """Unit tests for ChromaticAnalyzer.design_achromatic_doublet method"""

    @pytest.fixture
    def analyzer_with_mocks(self):
        """Create a ChromaticAnalyzer with mocked material database"""
        analyzer = ChromaticAnalyzer()
        
        # Mock the material database
        analyzer.material_db = Mock()
        analyzer.material_db.get_refractive_index = Mock(
            side_effect=self._mock_refractive_index
        )
        
        # Mock the Abbe number calculator
        analyzer.calculate_abbe_number = Mock(side_effect=self._mock_abbe_number)
        
        # Set wavelengths dictionary
        analyzer.WAVELENGTHS = {"d": 587.6, "C": 656.3, "F": 486.1}
        
        return analyzer

    @staticmethod
    def _mock_refractive_index(material, wavelength):
        """Mock refractive index lookup"""
        indices = {
            ("BK7", 587.6): 1.517,
            ("F2", 587.6): 1.620,
            ("BAF10", 587.6): 1.670,
            ("DENSE_FLINT", 587.6): 1.700,
        }
        return indices.get((material, wavelength), 1.5)

    @staticmethod
    def _mock_abbe_number(material):
        """Mock Abbe number lookup"""
        abbe_nums = {
            "BK7": 64.17,
            "F2": 36.37,
            "BAF10": 50.8,
            "DENSE_FLINT": 32.0,
        }
        return abbe_nums.get(material, 50.0)

    def test_default_materials_bk7_f2(self, analyzer_with_mocks):
        """Test design with default BK7 crown and F2 flint"""
        result = analyzer_with_mocks.design_achromatic_doublet(focal_length=100.0)
        
        assert isinstance(result, dict)
        assert result["crown_element"]["material"] == "BK7"
        assert result["flint_element"]["material"] == "F2"
        assert result["total_focal_length"] == 100.0

    def test_custom_materials(self, analyzer_with_mocks):
        """Test design with custom materials"""
        result = analyzer_with_mocks.design_achromatic_doublet(
            focal_length=50.0,
            crown_material="BAF10",
            flint_material="DENSE_FLINT"
        )
        
        assert result["crown_element"]["material"] == "BAF10"
        assert result["flint_element"]["material"] == "DENSE_FLINT"

    def test_power_sum_equals_total_power(self, analyzer_with_mocks):
        """Verify that φ1 + φ2 = 1/f (power conservation)"""
        focal_length = 75.0
        result = analyzer_with_mocks.design_achromatic_doublet(focal_length=focal_length)
        
        phi1 = result["crown_element"]["power"]
        phi2 = result["flint_element"]["power"]
        total_expected = 1.0 / focal_length
        
        assert pytest.approx(phi1 + phi2, rel=1e-10) == total_expected

    def test_achromatism_condition(self, analyzer_with_mocks):
        """Verify achromatism condition: φ1/V1 + φ2/V2 ≈ 0"""
        result = analyzer_with_mocks.design_achromatic_doublet(focal_length=100.0)
        
        phi1 = result["crown_element"]["power"]
        phi2 = result["flint_element"]["power"]
        V1 = result["crown_element"]["abbe_number"]
        V2 = result["flint_element"]["abbe_number"]
        
        achromatism = phi1 / V1 + phi2 / V2
        assert pytest.approx(achromatism, abs=1e-11) == 0.0

    def test_focal_length_power_reciprocal_relationship(self, analyzer_with_mocks):
        """Verify that f = 1/φ for both elements"""
        result = analyzer_with_mocks.design_achromatic_doublet(focal_length=100.0)
        
        f1 = result["crown_element"]["focal_length"]
        phi1 = result["crown_element"]["power"]
        f2 = result["flint_element"]["focal_length"]
        phi2 = result["flint_element"]["power"]
        
        if phi1 != 0:
            assert pytest.approx(f1, rel=1e-10) == 1.0 / phi1
        if phi2 != 0:
            assert pytest.approx(f2, rel=1e-10) == 1.0 / phi2

    def test_crown_converging_flint_diverging(self, analyzer_with_mocks):
        """Verify crown is converging (positive power) and flint is diverging (negative)"""
        result = analyzer_with_mocks.design_achromatic_doublet(focal_length=100.0)
        
        assert result["crown_element"]["power"] > 0, "Crown should be converging"
        assert result["flint_element"]["power"] < 0, "Flint should be diverging"

    def test_various_focal_lengths(self, analyzer_with_mocks):
        """Test with various focal length values"""
        test_focal_lengths = [5.0, 50.0, 100.0, 500.0, 1000.0]
        
        for focal_length in test_focal_lengths:
            result = analyzer_with_mocks.design_achromatic_doublet(focal_length=focal_length)
            
            assert result["total_focal_length"] == focal_length
            assert pytest.approx(result["total_power"], rel=1e-10) == 1.0 / focal_length

    def test_result_structure_completeness(self, analyzer_with_mocks):
        """Verify result contains all required keys and proper structure"""
        result = analyzer_with_mocks.design_achromatic_doublet(focal_length=100.0)
        
        # Top-level keys
        assert set(result.keys()) == {
            "crown_element", "flint_element", "total_focal_length", "total_power"
        }
        
        # Element-level keys
        required_element_keys = {"focal_length", "power", "material", "n", "abbe_number"}
        assert set(result["crown_element"].keys()) == required_element_keys
        assert set(result["flint_element"].keys()) == required_element_keys

    def test_identical_abbe_number_raises_valueerror(self, analyzer_with_mocks):
        """Verify ValueError when crown and flint materials have identical Abbe numbers"""
        analyzer_with_mocks.calculate_abbe_number = Mock(return_value=50.0)
        
        with pytest.raises(ValueError) as exc_info:
            analyzer_with_mocks.design_achromatic_doublet(
                focal_length=100.0,
                crown_material="GLASS1",
                flint_material="GLASS2"
            )
        
        assert "Cannot design an achromat" in str(exc_info.value)

    def test_abbe_numbers_within_tolerance_raises_error(self, analyzer_with_mocks):
        """Verify ValueError when Abbe numbers differ by less than 1e-12"""
        def abbe_side_effect(material):
            return 50.0 if material == "M1" else 50.0 + 1e-13
        
        analyzer_with_mocks.calculate_abbe_number = Mock(side_effect=abbe_side_effect)
        
        with pytest.raises(ValueError):
            analyzer_with_mocks.design_achromatic_doublet(
                focal_length=100.0,
                crown_material="M1",
                flint_material="M2"
            )

    def test_refractive_indices_retrieved_at_d_line(self, analyzer_with_mocks):
        """Verify refractive indices are retrieved at d-line wavelength (587.6 nm)"""
        analyzer_with_mocks.design_achromatic_doublet(focal_length=100.0)
        
        calls = analyzer_with_mocks.material_db.get_refractive_index.call_args_list
        wavelengths_used = [call[0][1] for call in calls]
        
        assert 587.6 in wavelengths_used

    def test_crown_power_dominates_flint(self, analyzer_with_mocks):
        """Verify crown element power magnitude exceeds flint power magnitude"""
        result = analyzer_with_mocks.design_achromatic_doublet(focal_length=100.0)
        
        crown_power_mag = abs(result["crown_element"]["power"])
        flint_power_mag = abs(result["flint_element"]["power"])
        
        assert crown_power_mag > flint_power_mag, \
            "Crown must be stronger to dominate the correction"

    def test_multiple_independent_designs(self, analyzer_with_mocks):
        """Test that multiple independent designs produce expected variations"""
        results = []
        focal_lengths = [25.0, 100.0, 400.0]
        
        for fl in focal_lengths:
            result = analyzer_with_mocks.design_achromatic_doublet(focal_length=fl)
            results.append(result)
            assert result["total_focal_length"] == fl
        
        # Powers should be different for each design
        powers = [r["total_power"] for r in results]
        assert len(set(powers)) == len(focal_lengths)

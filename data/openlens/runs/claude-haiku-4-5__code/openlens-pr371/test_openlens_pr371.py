import pytest
from openlens.optical_system import AchromaticDoubletDesigner, OpticalSystem


class TestAchromaticDoubletDesign:
    """Tests for AchromaticDoubletDesigner.design_cemented_doublet."""

    def test_returns_optical_system(self):
        """Test that design_cemented_doublet returns an OpticalSystem instance."""
        result = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=50.0
        )
        assert isinstance(result, OpticalSystem)

    def test_default_materials_bk7_sf11(self):
        """Test design with default materials (BK7 crown, SF11 flint)."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=50.0
        )
        assert isinstance(system, OpticalSystem)

    def test_explicit_default_materials(self):
        """Test explicitly specifying default materials produces valid system."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=50.0,
            crown_material="BK7",
            flint_material="SF11"
        )
        assert isinstance(system, OpticalSystem)

    def test_alternative_crown_flint_materials(self):
        """Test with alternative valid optical glass combination."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=50.0,
            crown_material="BK7",
            flint_material="SF5"
        )
        assert isinstance(system, OpticalSystem)

    def test_invalid_crown_material_raises_valueerror(self):
        """Test that non-existent crown material raises ValueError."""
        with pytest.raises(ValueError, match="Materials not found"):
            AchromaticDoubletDesigner.design_cemented_doublet(
                focal_length=100.0,
                diameter=50.0,
                crown_material="INVALID_GLASS_XYZ",
                flint_material="SF11"
            )

    def test_invalid_flint_material_raises_valueerror(self):
        """Test that non-existent flint material raises ValueError."""
        with pytest.raises(ValueError, match="Materials not found"):
            AchromaticDoubletDesigner.design_cemented_doublet(
                focal_length=100.0,
                diameter=50.0,
                crown_material="BK7",
                flint_material="INVALID_GLASS_XYZ"
            )

    def test_short_focal_length(self):
        """Test design with short focal length edge case."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=20.0,
            diameter=25.0
        )
        assert isinstance(system, OpticalSystem)

    def test_long_focal_length(self):
        """Test design with long focal length edge case."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=1000.0,
            diameter=50.0
        )
        assert isinstance(system, OpticalSystem)

    def test_small_aperture_diameter(self):
        """Test design with small aperture diameter edge case."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=5.0
        )
        assert isinstance(system, OpticalSystem)

    def test_large_aperture_diameter(self):
        """Test design with large aperture diameter edge case."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=200.0
        )
        assert isinstance(system, OpticalSystem)

    def test_fast_optics_small_fnumber(self):
        """Test design of fast optics with small f-number."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=50.0,
            diameter=100.0  # f/0.5 optics
        )
        assert isinstance(system, OpticalSystem)

    def test_slow_optics_large_fnumber(self):
        """Test design of slow optics with large f-number."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=500.0,
            diameter=50.0  # f/10 optics
        )
        assert isinstance(system, OpticalSystem)

    def test_positional_arguments(self):
        """Test function signature with positional arguments."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(100.0, 50.0)
        assert isinstance(system, OpticalSystem)

    def test_mixed_positional_and_keyword_arguments(self):
        """Test function with mixed positional and keyword arguments."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            100.0, 50.0, crown_material="BK7", flint_material="SF11"
        )
        assert isinstance(system, OpticalSystem)

    def test_all_keyword_arguments(self):
        """Test function with all parameters as keyword arguments."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=50.0,
            crown_material="BK7",
            flint_material="SF11"
        )
        assert isinstance(system, OpticalSystem)

import pytest
from openlens.optical_system import OpticalSystem, AchromaticDoubletDesigner
from openlens.lens import Lens


class TestAchromaticDoubletDesigner:
    """Tests for AchromaticDoubletDesigner.design_cemented_doublet"""

    def test_returns_optical_system(self):
        """Test that design_cemented_doublet returns an OpticalSystem"""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=50.0,
            diameter=25.0
        )
        assert isinstance(system, OpticalSystem)

    def test_system_name_is_achromatic_doublet(self):
        """Test that the system is named 'Achromatic Doublet'"""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=50.0,
            diameter=25.0
        )
        assert system.name == "Achromatic Doublet"

    def test_system_has_two_elements(self):
        """Test that the system contains exactly two lens elements"""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=50.0,
            diameter=25.0
        )
        assert len(system.elements) == 2
        assert all(isinstance(elem, Lens) for elem in system.elements)

    def test_elements_are_cemented_zero_gap(self):
        """Test that the two elements are cemented (zero air gap)"""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=50.0,
            diameter=25.0
        )
        # The second element should have zero air gap before it
        assert system.air_gaps[1] == 0.0

    def test_crown_element_properties(self):
        """Test that crown element has correct material and diameter"""
        crown_material = "BK7"
        diameter = 25.0
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=50.0,
            diameter=diameter,
            crown_material=crown_material
        )
        crown_lens = system.elements[0]
        assert crown_lens.material == crown_material
        assert crown_lens.diameter == diameter
        # Thickness should be approximately 15% of diameter
        assert abs(crown_lens.thickness - diameter * 0.15) < 0.5

    def test_flint_element_properties(self):
        """Test that flint element has correct material and diameter"""
        flint_material = "SF11"
        diameter = 25.0
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=50.0,
            diameter=diameter,
            flint_material=flint_material
        )
        flint_lens = system.elements[1]
        assert flint_lens.material == flint_material
        assert flint_lens.diameter == diameter
        # Thickness should be approximately 8% of diameter
        assert abs(flint_lens.thickness - diameter * 0.08) < 0.3

    def test_system_focal_length_converging(self):
        """Test that system has positive (converging) focal length"""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=50.0,
            diameter=25.0
        )
        efl = system.get_system_focal_length()
        assert efl is not None
        assert efl > 0

    def test_focal_length_in_reasonable_range(self):
        """Test that calculated focal length is in reasonable range of requested"""
        requested_fl = 50.0
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=requested_fl,
            diameter=25.0
        )
        efl = system.get_system_focal_length()
        # Allow 20% tolerance due to thick lens effects and paraxial approximation
        assert 0.8 * requested_fl < efl < 1.2 * requested_fl

    def test_multiple_focal_lengths(self):
        """Test design works for various focal lengths"""
        for focal_length in [25.0, 50.0, 100.0]:
            system = AchromaticDoubletDesigner.design_cemented_doublet(
                focal_length=focal_length,
                diameter=25.0
            )
            assert isinstance(system, OpticalSystem)
            assert system.get_system_focal_length() is not None

    def test_multiple_diameters(self):
        """Test design works for various diameters"""
        for diameter in [10.0, 25.0, 50.0]:
            system = AchromaticDoubletDesigner.design_cemented_doublet(
                focal_length=50.0,
                diameter=diameter
            )
            assert isinstance(system, OpticalSystem)
            assert system.elements[0].diameter == diameter
            assert system.elements[1].diameter == diameter

    def test_invalid_crown_material_raises_value_error(self):
        """Test that invalid crown material raises ValueError"""
        with pytest.raises(ValueError):
            AchromaticDoubletDesigner.design_cemented_doublet(
                focal_length=50.0,
                diameter=25.0,
                crown_material="INVALID_GLASS_NONEXISTENT"
            )

    def test_invalid_flint_material_raises_value_error(self):
        """Test that invalid flint material raises ValueError"""
        with pytest.raises(ValueError):
            AchromaticDoubletDesigner.design_cemented_doublet(
                focal_length=50.0,
                diameter=25.0,
                flint_material="INVALID_GLASS_NONEXISTENT"
            )

    def test_bk7_sf11_standard_combination(self):
        """Test standard BK7-SF11 achromatic combination"""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=50.0,
            diameter=25.0,
            crown_material="BK7",
            flint_material="SF11"
        )
        assert system.elements[0].material == "BK7"
        assert system.elements[1].material == "SF11"
        assert system.get_system_focal_length() is not None

    def test_alternative_material_combination(self):
        """Test alternative achromatic material combination"""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=50.0,
            diameter=25.0,
            crown_material="BK7",
            flint_material="F2"
        )
        assert system.elements[0].material == "BK7"
        assert system.elements[1].material == "F2"
        assert system.get_system_focal_length() is not None

    def test_lens_names_contain_material(self):
        """Test that lens names include material and 'Element' label"""
        crown_material = "BK7"
        flint_material = "SF11"
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=50.0,
            diameter=25.0,
            crown_material=crown_material,
            flint_material=flint_material
        )
        assert crown_material in system.elements[0].name
        assert "Element" in system.elements[0].name
        assert flint_material in system.elements[1].name
        assert "Element" in system.elements[1].name

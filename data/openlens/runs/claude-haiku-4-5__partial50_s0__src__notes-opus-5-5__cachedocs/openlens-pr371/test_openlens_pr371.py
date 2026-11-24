import pytest
from openlens.optical_system import AchromaticDoubletDesigner, OpticalSystem
from openlens.material_database import get_material_database
from openlens.constants import WAVELENGTH_D_LINE


class TestAchromaticDoubletDesignerCementedDoublet:
    """Test suite for AchromaticDoubletDesigner.design_cemented_doublet"""

    def test_basic_design_returns_optical_system(self):
        """Test that design_cemented_doublet returns an OpticalSystem"""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=50.0
        )
        assert isinstance(system, OpticalSystem)
        assert system.name == "Achromatic Doublet"

    def test_system_has_two_elements_cemented(self):
        """Test that system has exactly two elements with zero air gap (cemented)"""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=50.0
        )
        assert len(system.elements) == 2
        assert len(system.air_gaps) == 1
        assert system.air_gaps[0].thickness == 0.0

    def test_crown_lens_equiconvex_shape(self):
        """Test that crown lens is equiconvex (R2 = -R1)"""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=50.0,
            crown_material="BK7"
        )
        crown = system.elements[0].lens
        assert crown.name == "BK7 Element"
        assert crown.radius_of_curvature_2 == pytest.approx(
            -crown.radius_of_curvature_1, rel=1e-9
        )

    def test_flint_lens_negative_power(self):
        """Test that flint lens typically has concave front surface"""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=50.0,
            flint_material="SF11"
        )
        flint = system.elements[1].lens
        assert flint.name == "SF11 Element"
        # Flint R1 equals crown R2, which is negative for typical positive achromat
        assert flint.radius_of_curvature_1 < 0

    def test_lens_diameters_match_input(self):
        """Test that both lenses have the requested diameter"""
        diameter = 50.0
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=diameter
        )
        assert system.elements[0].lens.diameter == diameter
        assert system.elements[1].lens.diameter == diameter

    def test_lens_thicknesses_scaled_by_diameter(self):
        """Test that crown thickness is 0.15*D and flint thickness is 0.08*D"""
        diameter = 50.0
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=diameter
        )
        crown = system.elements[0].lens
        flint = system.elements[1].lens
        
        assert crown.thickness == pytest.approx(diameter * 0.15, rel=1e-9)
        assert flint.thickness == pytest.approx(diameter * 0.08, rel=1e-9)

    def test_cemented_interface_condition(self):
        """Test that flint R1 equals crown R2 (cemented interface)"""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=50.0
        )
        crown = system.elements[0].lens
        flint = system.elements[1].lens
        
        assert flint.radius_of_curvature_1 == pytest.approx(
            crown.radius_of_curvature_2, rel=1e-9
        )

    def test_refractive_indices_from_catalog(self):
        """Test that refractive indices match catalog nd values"""
        db = get_material_database()
        crown_mat = db.get_material("BK7")
        flint_mat = db.get_material("SF11")
        
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=50.0,
            crown_material="BK7",
            flint_material="SF11"
        )
        
        assert system.elements[0].lens.refractive_index == pytest.approx(
            crown_mat.nd, rel=1e-9
        )
        assert system.elements[1].lens.refractive_index == pytest.approx(
            flint_mat.nd, rel=1e-9
        )

    def test_design_wavelength_is_d_line(self):
        """Test that lenses are designed at d-line wavelength"""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=50.0
        )
        assert system.elements[0].lens.wavelength == WAVELENGTH_D_LINE
        assert system.elements[1].lens.wavelength == WAVELENGTH_D_LINE

    def test_error_on_missing_crown_material(self):
        """Test ValueError when crown material not found"""
        with pytest.raises(ValueError) as exc_info:
            AchromaticDoubletDesigner.design_cemented_doublet(
                focal_length=100.0,
                diameter=50.0,
                crown_material="INVALID_MATERIAL",
                flint_material="SF11"
            )
        error_msg = str(exc_info.value)
        assert "Materials not found" in error_msg
        assert "INVALID_MATERIAL" in error_msg
        assert "SF11" in error_msg

    def test_error_on_missing_flint_material(self):
        """Test ValueError when flint material not found"""
        with pytest.raises(ValueError) as exc_info:
            AchromaticDoubletDesigner.design_cemented_doublet(
                focal_length=100.0,
                diameter=50.0,
                crown_material="BK7",
                flint_material="NONEXISTENT"
            )
        error_msg = str(exc_info.value)
        assert "Materials not found" in error_msg
        assert "BK7" in error_msg
        assert "NONEXISTENT" in error_msg

    def test_error_on_zero_focal_length(self):
        """Test that zero focal length raises ZeroDivisionError"""
        with pytest.raises(ZeroDivisionError):
            AchromaticDoubletDesigner.design_cemented_doublet(
                focal_length=0.0,
                diameter=50.0
            )

    def test_negative_focal_length_design(self):
        """Test design with negative focal length"""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=-100.0,
            diameter=50.0
        )
        assert isinstance(system, OpticalSystem)
        # With negative focal length, the crown R1 should be negative
        assert system.elements[0].lens.radius_of_curvature_1 < 0

    def test_different_flint_materials(self):
        """Test design with different flint material (SF5 instead of SF11)"""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=50.0,
            crown_material="BK7",
            flint_material="SF5"
        )
        assert system.elements[0].lens.material == "BK7"
        assert system.elements[1].lens.material == "SF5"
        # Both elements should still be present
        assert len(system.elements) == 2

    def test_equal_abbe_numbers_same_material(self):
        """Test design when both elements use the same material (BK7)"""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=50.0,
            crown_material="BK7",
            flint_material="BK7"
        )
        # Both elements should exist even with identical materials
        assert len(system.elements) == 2
        assert system.elements[0].lens.material == "BK7"
        assert system.elements[1].lens.material == "BK7"

    def test_new_system_id_each_call(self):
        """Test that each call produces a system with unique ID"""
        sys1 = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=50.0
        )
        sys2 = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=50.0
        )
        assert sys1.id != sys2.id
        assert isinstance(sys1.id, str)
        assert isinstance(sys2.id, str)

"""
Test suite for openlens.optical_system.AchromaticDoubletDesigner.design_cemented_doublet.
"""

import pytest
from openlens.optical_system import AchromaticDoubletDesigner
from openlens.material_database import get_material_database


class TestAchromaticDoubletDesignerCementedDoublet:
    """Tests for AchromaticDoubletDesigner.design_cemented_doublet method."""
    
    def test_basic_structure_with_defaults(self):
        """Test basic system structure with default BK7/SF11 materials."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(100.0, 50.0)
        
        assert system.name == "Achromatic Doublet"
        assert len(system.elements) == 2
        assert len(system.air_gaps) == 1
    
    def test_lens_names_default_materials(self):
        """Test lens names reflect default BK7 and SF11 materials."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(100, 50)
        
        assert system.elements[0].lens.name == "BK7 Element"
        assert system.elements[1].lens.name == "SF11 Element"
    
    def test_lens_names_custom_materials(self):
        """Test lens names reflect custom material specifications."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            100, 50,
            crown_material="BK7",
            flint_material="SF2"
        )
        
        assert system.elements[0].lens.name == "BK7 Element"
        assert system.elements[1].lens.name == "SF2 Element"
    
    def test_diameters_propagated_to_both_lenses(self):
        """Test that the input diameter is used for both lenses."""
        diameter = 50.0
        system = AchromaticDoubletDesigner.design_cemented_doublet(100, diameter)
        
        assert system.elements[0].lens.diameter == diameter
        assert system.elements[1].lens.diameter == diameter
    
    def test_thicknesses_calculated_from_diameter(self):
        """Test that crown thickness is 15% of diameter, flint is 8%."""
        diameter = 50.0
        system = AchromaticDoubletDesigner.design_cemented_doublet(100, diameter)
        
        crown_thickness = system.elements[0].lens.thickness
        flint_thickness = system.elements[1].lens.thickness
        
        assert crown_thickness == pytest.approx(diameter * 0.15, rel=1e-9)
        assert flint_thickness == pytest.approx(diameter * 0.08, rel=1e-9)
    
    def test_air_gap_is_zero_for_cemented(self):
        """Test that the air gap between lenses is exactly zero (cemented)."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(100, 50)
        
        assert system.air_gaps[0].thickness == 0.0
    
    def test_crown_has_equiconvex_symmetry(self):
        """Test that crown satisfies R2 = -R1 (equiconvex geometry)."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(100, 50)
        
        R1_crown = system.elements[0].lens.radius_of_curvature_1
        R2_crown = system.elements[0].lens.radius_of_curvature_2
        
        assert R2_crown == pytest.approx(-R1_crown, rel=1e-9)
    
    def test_cemented_interface_has_same_radius(self):
        """Test that cemented interface is continuous: flint R1 = crown R2."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(100, 50)
        
        crown_R2 = system.elements[0].lens.radius_of_curvature_2
        flint_R1 = system.elements[1].lens.radius_of_curvature_1
        
        assert flint_R1 == pytest.approx(crown_R2, rel=1e-9)
    
    def test_refractive_indices_match_catalog_values(self):
        """Test that lens refractive indices match the material catalog nd values."""
        db = get_material_database()
        crown_mat = db.get_material("BK7")
        flint_mat = db.get_material("SF11")
        
        system = AchromaticDoubletDesigner.design_cemented_doublet(100, 50)
        
        assert system.elements[0].lens.refractive_index == pytest.approx(crown_mat.nd, rel=1e-6)
        assert system.elements[1].lens.refractive_index == pytest.approx(flint_mat.nd, rel=1e-6)
    
    def test_invalid_crown_material_raises_error(self):
        """Test that an invalid crown material name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            AchromaticDoubletDesigner.design_cemented_doublet(
                100, 50,
                crown_material="NONEXISTENT"
            )
        
        error_msg = str(exc_info.value)
        assert "Materials not found" in error_msg
        assert "NONEXISTENT" in error_msg
    
    def test_invalid_flint_material_raises_error(self):
        """Test that an invalid flint material name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            AchromaticDoubletDesigner.design_cemented_doublet(
                100, 50,
                flint_material="NONEXISTENT"
            )
        
        error_msg = str(exc_info.value)
        assert "Materials not found" in error_msg
        assert "NONEXISTENT" in error_msg
    
    def test_zero_focal_length_raises_zero_division_error(self):
        """Test that focal_length=0 raises ZeroDivisionError in f2 calculation."""
        with pytest.raises(ZeroDivisionError):
            AchromaticDoubletDesigner.design_cemented_doublet(0, 50)
    
    def test_negative_focal_length_inverts_all_radii(self):
        """Test that negative focal length inverts radius signs."""
        pos_system = AchromaticDoubletDesigner.design_cemented_doublet(100, 50)
        neg_system = AchromaticDoubletDesigner.design_cemented_doublet(-100, 50)
        
        # With opposite focal length, radii should flip
        assert neg_system.elements[0].lens.radius_of_curvature_1 == pytest.approx(
            -pos_system.elements[0].lens.radius_of_curvature_1
        )
    
    def test_equal_abbe_numbers_both_materials_same(self):
        """Test design when both materials have the same Abbe number."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            100, 50,
            crown_material="BK7",
            flint_material="BK7"
        )
        
        assert system is not None
        assert len(system.elements) == 2
        assert system.elements[0].lens.name == "BK7 Element"
        assert system.elements[1].lens.name == "BK7 Element"
    
    def test_each_call_creates_unique_system_instance(self):
        """Test that multiple calls produce systems with different ids."""
        sys1 = AchromaticDoubletDesigner.design_cemented_doublet(100, 50)
        sys2 = AchromaticDoubletDesigner.design_cemented_doublet(100, 50)
        
        # Different object instances
        assert sys1 is not sys2
        # Different system ids
        assert sys1.id != sys2.id

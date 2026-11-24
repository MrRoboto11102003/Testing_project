import pytest
from openlens.optical_system import AchromaticDoubletDesigner, OpticalSystem
from openlens.lens import Lens


class TestAchromaticDoubletDesigner:
    """Test suite for the achromatic doublet designer."""

    def test_design_cemented_doublet_returns_optical_system(self):
        """Test that design_cemented_doublet returns an OpticalSystem instance."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=25.0
        )
        assert isinstance(system, OpticalSystem)

    def test_design_cemented_doublet_system_name(self):
        """Test that the returned system is correctly named."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=25.0
        )
        assert system.name == "Achromatic Doublet"

    def test_design_cemented_doublet_contains_exactly_two_lenses(self):
        """Test that the system contains exactly two lens elements."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=25.0
        )
        assert len(system.lenses) == 2
        assert isinstance(system.lenses[0], Lens)
        assert isinstance(system.lenses[1], Lens)

    def test_design_cemented_doublet_invalid_crown_material(self):
        """Test that invalid crown material raises ValueError."""
        with pytest.raises(ValueError, match="Materials not found"):
            AchromaticDoubletDesigner.design_cemented_doublet(
                focal_length=100.0,
                diameter=25.0,
                crown_material="INVALID_MATERIAL",
                flint_material="SF11"
            )

    def test_design_cemented_doublet_invalid_flint_material(self):
        """Test that invalid flint material raises ValueError."""
        with pytest.raises(ValueError, match="Materials not found"):
            AchromaticDoubletDesigner.design_cemented_doublet(
                focal_length=100.0,
                diameter=25.0,
                crown_material="BK7",
                flint_material="INVALID_MATERIAL"
            )

    def test_design_cemented_doublet_diameters_match_input(self):
        """Test that all lenses have the specified diameter."""
        diameter = 25.0
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=diameter
        )
        assert system.lenses[0].diameter == diameter
        assert system.lenses[1].diameter == diameter

    def test_design_cemented_doublet_all_thicknesses_positive(self):
        """Test that both lenses have positive thickness."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=25.0
        )
        assert system.lenses[0].thickness > 0
        assert system.lenses[1].thickness > 0

    def test_design_cemented_doublet_crown_element_positive(self):
        """Test that crown element has positive power (biconvex)."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=25.0
        )
        crown = system.lenses[0]
        # Biconvex: front radius positive, back radius negative
        assert crown.radius_of_curvature_1 > 0
        assert crown.radius_of_curvature_2 < 0

    def test_design_cemented_doublet_cemented_interface(self):
        """Test that the flint element's front radius matches crown's back radius (cemented)."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=25.0
        )
        crown_back = system.lenses[0].radius_of_curvature_2
        flint_front = system.lenses[1].radius_of_curvature_1
        assert flint_front == crown_back

    def test_design_cemented_doublet_varies_with_focal_length(self):
        """Test that different focal lengths produce different optical designs."""
        sys_50 = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=50.0,
            diameter=25.0
        )
        sys_150 = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=150.0,
            diameter=25.0
        )
        
        # Radii should differ with focal length changes
        assert sys_50.lenses[0].radius_of_curvature_1 != sys_150.lenses[0].radius_of_curvature_1
        assert sys_50.lenses[0].radius_of_curvature_2 != sys_150.lenses[0].radius_of_curvature_2

    def test_design_cemented_doublet_varies_with_diameter(self):
        """Test that different diameters produce different optical designs."""
        sys_small = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=15.0
        )
        sys_large = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=40.0
        )
        
        assert sys_small.lenses[0].diameter == 15.0
        assert sys_large.lenses[0].diameter == 40.0
        assert sys_small.lenses[0].thickness != sys_large.lenses[0].thickness

    def test_design_cemented_doublet_refractive_indices_valid_glass_range(self):
        """Test that refractive indices are within valid optical glass range."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=25.0
        )
        
        # Typical optical glass has refractive index 1.4-2.0
        for lens in system.lenses:
            assert 1.4 < lens.refractive_index < 2.1

    def test_design_cemented_doublet_flint_higher_index_than_crown(self):
        """Test that SF11 (flint) has higher refractive index than BK7 (crown)."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=25.0,
            crown_material="BK7",
            flint_material="SF11"
        )
        
        crown_index = system.lenses[0].refractive_index
        flint_index = system.lenses[1].refractive_index
        
        # SF11 is a high-index flint; BK7 is a crown glass
        assert flint_index > crown_index

    def test_design_cemented_doublet_crown_thicker_than_flint(self):
        """Test that crown element is typically thicker than flint element."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=25.0
        )
        
        # By design: crown thickness ~15% of diameter, flint ~8%
        crown_thickness = system.lenses[0].thickness
        flint_thickness = system.lenses[1].thickness
        
        assert crown_thickness > flint_thickness

    def test_design_cemented_doublet_lens_naming_includes_materials(self):
        """Test that lens names include material information."""
        system = AchromaticDoubletDesigner.design_cemented_doublet(
            focal_length=100.0,
            diameter=25.0,
            crown_material="BK7",
            flint_material="SF11"
        )
        
        crown_name = system.lenses[0].name
        flint_name = system.lenses[1].name
        
        assert "BK7" in crown_name
        assert "SF11" in flint_name

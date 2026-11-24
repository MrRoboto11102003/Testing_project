import pytest
from openlens import PresetLibrary, Lens
from openlens.preset_library import LensPreset


class TestPresetLibraryGetLensCopy:
    """Test suite for PresetLibrary.get_lens_copy()."""
    
    @pytest.fixture
    def lib(self):
        """Fresh PresetLibrary for each test to prevent state leakage."""
        return PresetLibrary()
    
    # ===== Unknown and invalid presets =====
    
    def test_unknown_preset_name_returns_none(self, lib):
        """Unknown preset name returns None."""
        assert lib.get_lens_copy("unknown_preset") is None
    
    def test_empty_string_preset_returns_none(self, lib):
        """Empty string is not a valid preset."""
        assert lib.get_lens_copy("") is None
    
    # ===== Case sensitivity =====
    
    def test_case_sensitive_name_matching(self, lib):
        """Preset name matching is case-sensitive."""
        assert lib.get_lens_copy("50mm Biconvex") is not None
        assert lib.get_lens_copy("50mm biconvex") is None
    
    # ===== Preset name vs lens name distinction =====
    
    def test_preset_name_differs_from_lens_name(self, lib):
        """Must use preset name, not lens.name property."""
        # "25mm Plossl Eyepiece" is preset name; "25mm Plossl" is lens.name
        assert lib.get_lens_copy("25mm Plossl") is None  # Wrong: lens name
        assert lib.get_lens_copy("25mm Plossl Eyepiece") is not None  # Correct
    
    def test_lens_name_property_in_copy(self, lib):
        """Copied lens has correct name property (from lens, not preset key)."""
        copy = lib.get_lens_copy("25mm Plossl Eyepiece")
        # lens.name should be "25mm Plossl", not the preset name
        assert copy.name == "25mm Plossl"
    
    # ===== Basic return type and independence =====
    
    def test_known_preset_returns_lens_instance(self, lib):
        """Known preset name returns a Lens instance."""
        result = lib.get_lens_copy("50mm Biconvex")
        assert isinstance(result, Lens)
        assert result is not None
    
    def test_copy_is_new_distinct_object(self, lib):
        """Returned copy is a different object from preset's lens."""
        original = lib.get_preset("50mm Biconvex").lens
        copy = lib.get_lens_copy("50mm Biconvex")
        assert copy is not original
    
    # ===== ID uniqueness =====
    
    def test_copy_has_new_uuid_different_from_original(self, lib):
        """Copy receives a fresh UUID, different from original."""
        original = lib.get_preset("50mm Biconvex").lens
        copy = lib.get_lens_copy("50mm Biconvex")
        
        assert hasattr(copy, "id")
        assert copy.id != original.id
    
    def test_multiple_calls_produce_unique_ids(self, lib):
        """Each call to get_lens_copy produces a unique ID."""
        copy1 = lib.get_lens_copy("50mm Biconvex")
        copy2 = lib.get_lens_copy("50mm Biconvex")
        
        assert copy1.id != copy2.id
    
    # ===== Optical properties preserved =====
    
    def test_optical_parameters_preserved(self, lib):
        """Radii, thickness, and diameter are preserved in copy."""
        original = lib.get_preset("50mm Biconvex").lens
        copy = lib.get_lens_copy("50mm Biconvex")
        
        assert copy.radius_of_curvature_1 == pytest.approx(original.radius_of_curvature_1)
        assert copy.radius_of_curvature_2 == pytest.approx(original.radius_of_curvature_2)
        assert copy.thickness == pytest.approx(original.thickness)
        assert copy.diameter == pytest.approx(original.diameter)
    
    def test_material_preserved(self, lib):
        """Material is preserved in the copy."""
        original = lib.get_preset("50mm Biconvex").lens
        copy = lib.get_lens_copy("50mm Biconvex")
        assert copy.material == original.material
    
    # ===== Full round-trip fidelity =====
    
    def test_to_dict_from_dict_round_trip_matches(self, lib):
        """Data survives to_dict/from_dict round-trip except for id."""
        original = lib.get_preset("Kellner Eyepiece").lens
        copy = lib.get_lens_copy("Kellner Eyepiece")
        
        # Serialize both
        orig_dict = original.to_dict()
        copy_dict = copy.to_dict()
        
        # Remove ids (should differ by design)
        orig_dict.pop("id", None)
        copy_dict.pop("id", None)
        
        # All other fields should match exactly
        assert orig_dict == copy_dict
    
    # ===== No side effects on original =====
    
    def test_mutating_copy_does_not_affect_original(self, lib):
        """Mutating the copy leaves the original preset lens unchanged."""
        original_thickness = lib.get_preset("50mm Biconvex").lens.thickness
        copy = lib.get_lens_copy("50mm Biconvex")
        
        # Modify the copy
        copy.thickness = 999.0
        
        # Verify original is unchanged
        assert lib.get_preset("50mm Biconvex").lens.thickness == original_thickness
    
    # ===== User-added presets =====
    
    def test_user_added_preset_can_be_copied(self, lib):
        """User-added presets work with get_lens_copy."""
        custom_lens = Lens(
            name="CustomLens",
            radius_of_curvature_1=40.0,
            radius_of_curvature_2=-60.0,
            thickness=5.0,
            diameter=20.0
        )
        lib.add_preset(LensPreset("MyCustom", "CustomCat", "desc", lens=custom_lens))
        
        copy = lib.get_lens_copy("MyCustom")
        assert copy is not None
        assert isinstance(copy, Lens)
        assert copy.name == "CustomLens"
        assert copy.radius_of_curvature_1 == pytest.approx(40.0)
    
    # ===== Preset name uniqueness =====
    
    def test_preset_name_overwrites_previous(self, lib):
        """Adding preset with same name overwrites the previous one."""
        lens1 = Lens(name="V1", radius_of_curvature_1=10.0, radius_of_curvature_2=-20.0,
                     thickness=2.0, diameter=10.0)
        lens2 = Lens(name="V2", radius_of_curvature_1=30.0, radius_of_curvature_2=-40.0,
                     thickness=3.0, diameter=15.0)
        
        lib.add_preset(LensPreset("Duplicate", "Cat", "First", lens=lens1))
        lib.add_preset(LensPreset("Duplicate", "Cat", "Second", lens=lens2))
        
        # Should get the second lens
        copy = lib.get_lens_copy("Duplicate")
        assert copy.radius_of_curvature_1 == pytest.approx(30.0)
        assert copy.name == "V2"
    
    # ===== Special case: flat radius =====
    
    def test_flat_radius_preserved_in_copy(self, lib):
        """Flat surface radius (plano surface) is correctly preserved."""
        # Edmund #45-166 has a plano-convex surface
        original = lib.get_preset("Edmund #45-166 (25mm PCX)").lens
        copy = lib.get_lens_copy("Edmund #45-166 (25mm PCX)")
        
        # The flat radius should match via serialized form
        orig_dict = original.to_dict()
        copy_dict = copy.to_dict()
        assert orig_dict["radius_of_curvature_2"] == copy_dict["radius_of_curvature_2"]

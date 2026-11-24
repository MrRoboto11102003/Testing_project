import pytest
from openlens.preset_library import PresetLibrary, LensPreset
from openlens.lens import Lens


class TestPresetLibraryGetLensCopy:
    """Tests for PresetLibrary.get_lens_copy()"""

    def test_known_preset_returns_lens_instance(self):
        """Known preset name returns a Lens instance."""
        lib = PresetLibrary()
        copy = lib.get_lens_copy("50mm Biconvex")
        assert isinstance(copy, Lens)

    def test_unknown_preset_returns_none(self):
        """Unknown preset name returns None."""
        lib = PresetLibrary()
        assert lib.get_lens_copy("nope") is None
        assert lib.get_lens_copy("") is None

    def test_case_sensitive_name_matching(self):
        """Preset name matching is case-sensitive."""
        lib = PresetLibrary()
        assert lib.get_lens_copy("50mm Biconvex") is not None
        assert lib.get_lens_copy("50mm biconvex") is None

    def test_lens_name_differs_from_preset_name(self):
        """Distinguish between preset name and lens.name for some presets."""
        lib = PresetLibrary()
        copy = lib.get_lens_copy("25mm Plossl Eyepiece")
        assert copy is not None
        assert copy.name == "25mm Plossl"
        # Using just the lens name should return None
        assert lib.get_lens_copy("25mm Plossl") is None

    def test_copy_is_distinct_object(self):
        """Copy is a distinct object, not the same instance as the preset lens."""
        lib = PresetLibrary()
        preset = lib.get_preset("50mm Biconvex")
        orig = preset.lens
        copy = lib.get_lens_copy("50mm Biconvex")
        assert copy is not orig

    def test_copy_has_different_id(self):
        """Copy receives a fresh UUID different from the original."""
        lib = PresetLibrary()
        preset = lib.get_preset("50mm Biconvex")
        orig = preset.lens
        copy = lib.get_lens_copy("50mm Biconvex")
        assert copy.id != orig.id

    def test_each_copy_has_unique_id(self):
        """Each call generates a copy with a different UUID."""
        lib = PresetLibrary()
        copy1 = lib.get_lens_copy("50mm Biconvex")
        copy2 = lib.get_lens_copy("50mm Biconvex")
        assert copy1.id != copy2.id

    def test_copy_dict_full_fidelity(self):
        """Copy's to_dict() matches original after removing id (full fidelity)."""
        lib = PresetLibrary()
        preset = lib.get_preset("50mm Biconvex")
        orig = preset.lens
        copy = lib.get_lens_copy("50mm Biconvex")
        
        orig_dict = orig.to_dict()
        copy_dict = copy.to_dict()
        
        # Remove id fields to compare all other fields
        orig_dict.pop("id", None)
        copy_dict.pop("id", None)
        
        # This verifies coatings, model-glass, parabolic sags, Fresnel grooves, etc.
        assert copy_dict == orig_dict

    def test_copy_independent_mutation(self):
        """Mutating the copy does not affect the original."""
        lib = PresetLibrary()
        preset = lib.get_preset("50mm Biconvex")
        orig = preset.lens
        orig_thickness = orig.thickness
        
        copy = lib.get_lens_copy("50mm Biconvex")
        copy.thickness = 999.0
        
        assert orig.thickness == orig_thickness

    def test_copy_optical_parameters(self):
        """Copy preserves key optical parameters."""
        lib = PresetLibrary()
        copy = lib.get_lens_copy("Beam Expander Element")
        preset = lib.get_preset("Beam Expander Element")
        orig = preset.lens
        
        assert copy.radius_of_curvature_1 == orig.radius_of_curvature_1
        assert copy.radius_of_curvature_2 == orig.radius_of_curvature_2
        assert copy.thickness == orig.thickness
        assert copy.diameter == orig.diameter
        assert copy.material == orig.material

    def test_wavelength_dependent_presets(self):
        """Wavelength-dependent presets preserve wavelength and refractive index."""
        lib = PresetLibrary()
        copy = lib.get_lens_copy("Laser Focusing Lens 532nm")
        if copy is not None:
            preset = lib.get_preset("Laser Focusing Lens 532nm")
            orig = preset.lens
            assert copy.wavelength == orig.wavelength
            assert copy.refractive_index == pytest.approx(orig.refractive_index, rel=1e-6)

    def test_flat_surface_preserved(self):
        """Flat surface radius is correctly preserved through round-trip."""
        lib = PresetLibrary()
        copy = lib.get_lens_copy("Edmund #45-166 (25mm PCX)")
        preset = lib.get_preset("Edmund #45-166 (25mm PCX)")
        if copy is not None and preset is not None:
            orig = preset.lens
            assert copy.radius_of_curvature_2 == orig.radius_of_curvature_2

    def test_user_added_presets(self):
        """Works correctly with user-added presets."""
        lib = PresetLibrary()
        custom = Lens(
            name="CustomLens",
            radius_of_curvature_1=40.0,
            radius_of_curvature_2=-60.0,
            thickness=5.0,
            diameter=20.0,
            material="BK7"
        )
        lib.add_preset(LensPreset("MyCustom", "User", "Description", lens=custom))
        
        copy = lib.get_lens_copy("MyCustom")
        assert copy is not None
        assert copy is not custom
        assert copy.radius_of_curvature_1 == 40.0
        assert copy.radius_of_curvature_2 == -60.0

    def test_id_format_hex_string(self):
        """New id is a 32-character lowercase hex string."""
        lib = PresetLibrary()
        copy = lib.get_lens_copy("50mm Biconvex")
        assert isinstance(copy.id, str)
        assert len(copy.id) == 32
        assert all(c in "0123456789abcdef" for c in copy.id)

    def test_unhashable_key_error(self):
        """Unhashable key raises TypeError."""
        lib = PresetLibrary()
        with pytest.raises(TypeError):
            lib.get_lens_copy(["unhashable"])

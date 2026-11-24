import pytest
import uuid
from openlens.lens import Lens
from openlens.preset_library import PresetLibrary


class TestPresetLibraryGetLensCopy:
    """Tests for PresetLibrary.get_lens_copy method."""

    @pytest.fixture
    def preset_library(self):
        """Create a PresetLibrary instance."""
        return PresetLibrary()

    @pytest.fixture
    def available_preset_name(self, preset_library):
        """Get the name of an available preset for testing."""
        presets = preset_library.list_presets()
        if presets:
            return presets[0].name
        return None

    def test_get_lens_copy_returns_none_for_nonexistent_preset(self, preset_library):
        """Test that get_lens_copy returns None when preset does not exist."""
        result = preset_library.get_lens_copy("NonexistentPreset_XYZ_123456789")
        assert result is None

    def test_get_lens_copy_returns_lens_object(self, preset_library, available_preset_name):
        """Test that get_lens_copy returns a Lens object for a valid preset."""
        if available_preset_name is None:
            pytest.skip("No presets available")
        
        result = preset_library.get_lens_copy(available_preset_name)
        assert result is not None
        assert isinstance(result, Lens)

    def test_get_lens_copy_returns_different_objects(self, preset_library, available_preset_name):
        """Test that successive calls return different object instances."""
        if available_preset_name is None:
            pytest.skip("No presets available")
        
        copy1 = preset_library.get_lens_copy(available_preset_name)
        copy2 = preset_library.get_lens_copy(available_preset_name)
        
        assert copy1 is not copy2

    def test_get_lens_copy_assigns_unique_ids(self, preset_library, available_preset_name):
        """Test that each copy receives a unique ID."""
        if available_preset_name is None:
            pytest.skip("No presets available")
        
        copy1 = preset_library.get_lens_copy(available_preset_name)
        copy2 = preset_library.get_lens_copy(available_preset_name)
        
        assert copy1.id != copy2.id
        
        # IDs should be valid hex UUIDs (32 characters)
        assert isinstance(copy1.id, str)
        assert isinstance(copy2.id, str)
        assert len(copy1.id) == 32
        assert len(copy2.id) == 32

    def test_get_lens_copy_dict_roundtrip_preserves_all_data(self, preset_library, available_preset_name):
        """Test that dict roundtrip preserves all lens data including special properties."""
        if available_preset_name is None:
            pytest.skip("No presets available")
        
        original_preset = preset_library.get_preset(available_preset_name)
        copy = preset_library.get_lens_copy(available_preset_name)
        
        # Docstring mentions this preserves coatings, model-glass, parabolic sags, Fresnel params
        original_dict = original_preset.lens.to_dict()
        copy_dict = copy.to_dict()
        
        # IDs must differ
        original_id = original_dict.pop("id")
        copy_id = copy_dict.pop("id")
        assert original_id != copy_id
        
        # Everything else must match exactly
        assert copy_dict == original_dict

    def test_get_lens_copy_copies_are_independent(self, preset_library, available_preset_name):
        """Test that modifications to one copy don't affect others."""
        if available_preset_name is None:
            pytest.skip("No presets available")
        
        copy1 = preset_library.get_lens_copy(available_preset_name)
        copy2 = preset_library.get_lens_copy(available_preset_name)
        
        original_name = copy1.name
        copy1.name = "Modified Copy 1"
        
        assert copy2.name == original_name
        assert copy1.name != copy2.name

    def test_get_lens_copy_with_empty_string_returns_none(self, preset_library):
        """Test that empty string preset name returns None."""
        result = preset_library.get_lens_copy("")
        assert result is None

    def test_get_lens_copy_preserves_optical_properties(self, preset_library, available_preset_name):
        """Test that core optical properties are preserved through roundtrip."""
        if available_preset_name is None:
            pytest.skip("No presets available")
        
        original_preset = preset_library.get_preset(available_preset_name)
        copy = preset_library.get_lens_copy(available_preset_name)
        
        assert copy.radius_of_curvature_1 == original_preset.lens.radius_of_curvature_1
        assert copy.radius_of_curvature_2 == original_preset.lens.radius_of_curvature_2
        assert copy.thickness == original_preset.lens.thickness
        assert copy.diameter == original_preset.lens.diameter
        assert copy.refractive_index == original_preset.lens.refractive_index

    def test_get_lens_copy_multiple_copies_all_unique_ids(self, preset_library, available_preset_name):
        """Test that multiple copies all have unique IDs."""
        if available_preset_name is None:
            pytest.skip("No presets available")
        
        copies = [preset_library.get_lens_copy(available_preset_name) for _ in range(5)]
        ids = [c.id for c in copies]
        
        # All IDs must be unique
        assert len(ids) == len(set(ids)), "Some copies share the same ID"

    def test_get_lens_copy_preserves_lens_name(self, preset_library, available_preset_name):
        """Test that lens name attribute is preserved."""
        if available_preset_name is None:
            pytest.skip("No presets available")
        
        original_preset = preset_library.get_preset(available_preset_name)
        copy = preset_library.get_lens_copy(available_preset_name)
        
        assert copy.name == original_preset.lens.name

    def test_get_lens_copy_preserves_material(self, preset_library, available_preset_name):
        """Test that material string is preserved."""
        if available_preset_name is None:
            pytest.skip("No presets available")
        
        original_preset = preset_library.get_preset(available_preset_name)
        copy = preset_library.get_lens_copy(available_preset_name)
        
        assert copy.material == original_preset.lens.material

    def test_get_lens_copy_focal_length_calculation_consistent(self, preset_library, available_preset_name):
        """Test that focal length calculations are consistent for original and copy."""
        if available_preset_name is None:
            pytest.skip("No presets available")
        
        original_preset = preset_library.get_preset(available_preset_name)
        copy = preset_library.get_lens_copy(available_preset_name)
        
        original_fl = original_preset.lens.calculate_focal_length()
        copy_fl = copy.calculate_focal_length()
        
        # Both should have the same focal length (or both None)
        if original_fl is not None and copy_fl is not None:
            assert abs(original_fl - copy_fl) < 1e-10
        elif original_fl is None:
            assert copy_fl is None

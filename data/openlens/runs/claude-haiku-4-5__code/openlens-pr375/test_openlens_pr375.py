import pytest
import uuid
from unittest.mock import Mock, patch
from openlens import PresetLibrary, Lens


class TestPresetLibraryGetLensCopy:
    """Tests for PresetLibrary.get_lens_copy method."""
    
    @pytest.fixture
    def preset_library(self):
        """Create a PresetLibrary instance."""
        return PresetLibrary()
    
    def test_get_lens_copy_nonexistent_preset_returns_none(self, preset_library):
        """Test that get_lens_copy returns None for nonexistent presets."""
        result = preset_library.get_lens_copy("nonexistent_preset_xyz_12345")
        assert result is None
    
    def test_get_lens_copy_empty_string_returns_none(self, preset_library):
        """Test that get_lens_copy returns None for empty string preset name."""
        result = preset_library.get_lens_copy("")
        assert result is None
    
    def test_get_lens_copy_special_characters_returns_none(self, preset_library):
        """Test that get_lens_copy returns None for nonexistent special character names."""
        result = preset_library.get_lens_copy("!@#$%^&*()")
        assert result is None
    
    def test_get_lens_copy_returns_different_object(self):
        """Test that the returned lens is a different object (distinct identity)."""
        library = PresetLibrary()
        
        with patch.object(library, 'get_preset') as mock_get_preset:
            original_lens = Mock(spec=Lens)
            original_lens.to_dict.return_value = {
                "id": "original_id_123",
                "name": "test_lens"
            }
            
            mock_preset = Mock()
            mock_preset.lens = original_lens
            mock_get_preset.return_value = mock_preset
            
            with patch.object(Lens, 'from_dict') as mock_from_dict:
                copy_lens = Mock(spec=Lens)
                mock_from_dict.return_value = copy_lens
                
                result = library.get_lens_copy("test_preset")
                
                assert result is copy_lens
                assert result is not original_lens
    
    def test_get_lens_copy_assigns_new_uuid(self):
        """Test that get_lens_copy assigns a new UUID to the copy."""
        library = PresetLibrary()
        original_id = "original_id_abc123"
        
        with patch.object(library, 'get_preset') as mock_get_preset:
            original_lens = Mock(spec=Lens)
            original_lens.to_dict.return_value = {
                "id": original_id,
                "name": "test_lens"
            }
            
            mock_preset = Mock()
            mock_preset.lens = original_lens
            mock_get_preset.return_value = mock_preset
            
            with patch.object(Lens, 'from_dict') as mock_from_dict:
                mock_from_dict.return_value = Mock()
                
                library.get_lens_copy("test_preset")
                
                call_args = mock_from_dict.call_args[0][0]
                new_id = call_args["id"]
                
                assert new_id != original_id
                # Verify it's a valid UUID hex string
                try:
                    uuid.UUID(hex=new_id)
                    is_valid = True
                except (ValueError, TypeError):
                    is_valid = False
                assert is_valid
    
    def test_get_lens_copy_preserves_coatings(self):
        """Test that get_lens_copy preserves coatings (mentioned in docstring)."""
        library = PresetLibrary()
        coatings_data = ["AR_coating", "MR_coating"]
        
        with patch.object(library, 'get_preset') as mock_get_preset:
            original_lens = Mock(spec=Lens)
            original_lens.to_dict.return_value = {
                "id": "original_id",
                "coatings": coatings_data
            }
            
            mock_preset = Mock()
            mock_preset.lens = original_lens
            mock_get_preset.return_value = mock_preset
            
            with patch.object(Lens, 'from_dict') as mock_from_dict:
                mock_from_dict.return_value = Mock()
                
                library.get_lens_copy("test_preset")
                
                call_data = mock_from_dict.call_args[0][0]
                assert call_data["coatings"] == coatings_data
    
    def test_get_lens_copy_preserves_parabolic_sags(self):
        """Test that get_lens_copy preserves parabolic sags (mentioned in docstring)."""
        library = PresetLibrary()
        sags_data = [0.001, 0.002, 0.003]
        
        with patch.object(library, 'get_preset') as mock_get_preset:
            original_lens = Mock(spec=Lens)
            original_lens.to_dict.return_value = {
                "id": "original_id",
                "parabolic_sags": sags_data
            }
            
            mock_preset = Mock()
            mock_preset.lens = original_lens
            mock_get_preset.return_value = mock_preset
            
            with patch.object(Lens, 'from_dict') as mock_from_dict:
                mock_from_dict.return_value = Mock()
                
                library.get_lens_copy("test_preset")
                
                call_data = mock_from_dict.call_args[0][0]
                assert call_data["parabolic_sags"] == sags_data
    
    def test_get_lens_copy_preserves_fresnel_grooves(self):
        """Test that get_lens_copy preserves Fresnel groove parameters."""
        library = PresetLibrary()
        grooves_data = {"pitch": 0.5, "depth": 0.1}
        
        with patch.object(library, 'get_preset') as mock_get_preset:
            original_lens = Mock(spec=Lens)
            original_lens.to_dict.return_value = {
                "id": "original_id",
                "fresnel_grooves": grooves_data
            }
            
            mock_preset = Mock()
            mock_preset.lens = original_lens
            mock_get_preset.return_value = mock_preset
            
            with patch.object(Lens, 'from_dict') as mock_from_dict:
                mock_from_dict.return_value = Mock()
                
                library.get_lens_copy("test_preset")
                
                call_data = mock_from_dict.call_args[0][0]
                assert call_data["fresnel_grooves"] == grooves_data
    
    def test_get_lens_copy_multiple_calls_different_ids(self):
        """Test that multiple get_lens_copy calls produce different UUIDs."""
        library = PresetLibrary()
        
        with patch.object(library, 'get_preset') as mock_get_preset:
            original_lens = Mock(spec=Lens)
            original_lens.to_dict.return_value = {
                "id": "original_id",
                "name": "test"
            }
            
            mock_preset = Mock()
            mock_preset.lens = original_lens
            mock_get_preset.return_value = mock_preset
            
            with patch.object(Lens, 'from_dict') as mock_from_dict:
                mock_from_dict.return_value = Mock()
                
                library.get_lens_copy("test_preset")
                library.get_lens_copy("test_preset")
                
                call1_data = mock_from_dict.call_args_list[0][0][0]
                call2_data = mock_from_dict.call_args_list[1][0][0]
                
                assert call1_data["id"] != call2_data["id"]
    
    def test_get_lens_copy_roundtrips_through_to_dict_from_dict(self):
        """Test that the method implements the documented round-trip behavior."""
        library = PresetLibrary()
        
        with patch.object(library, 'get_preset') as mock_get_preset:
            original_lens = Mock(spec=Lens)
            original_lens.to_dict.return_value = {
                "id": "original_id",
                "some_property": "value"
            }
            
            mock_preset = Mock()
            mock_preset.lens = original_lens
            mock_get_preset.return_value = mock_preset
            
            with patch.object(Lens, 'from_dict') as mock_from_dict:
                mock_from_dict.return_value = Mock()
                
                library.get_lens_copy("test_preset")
                
                # Verify to_dict was called
                original_lens.to_dict.assert_called_once()
                # Verify from_dict was called
                mock_from_dict.assert_called_once()
    
    def test_get_lens_copy_preserves_all_lens_properties(self):
        """Test that all lens properties are preserved in the copy."""
        library = PresetLibrary()
        
        lens_data = {
            "id": "original",
            "focal_length": 50.0,
            "aperture": 2.0,
            "elements": ["elem1", "elem2"],
            "coatings": ["AR"],
            "parabolic_sags": [0.001],
            "fresnel_grooves": {"pitch": 0.5},
            "materials": {"glass": {"n": 1.5}},
            "custom_field": "custom_value"
        }
        
        with patch.object(library, 'get_preset') as mock_get_preset:
            original_lens = Mock(spec=Lens)
            original_lens.to_dict.return_value = lens_data.copy()
            
            mock_preset = Mock()
            mock_preset.lens = original_lens
            mock_get_preset.return_value = mock_preset
            
            with patch.object(Lens, 'from_dict') as mock_from_dict:
                mock_from_dict.return_value = Mock()
                
                library.get_lens_copy("test_preset")
                
                call_data = mock_from_dict.call_args[0][0]
                
                # Verify all original fields are preserved
                for key in lens_data:
                    if key != "id":
                        assert call_data[key] == lens_data[key]

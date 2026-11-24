import pytest
from unittest.mock import Mock, patch
from openlens.preset_library import PresetLibrary
from openlens.lens import Lens


class TestPresetLibraryGetLensCopy:
    """Tests for PresetLibrary.get_lens_copy method."""
    
    @pytest.fixture
    def library(self):
        """Create a PresetLibrary instance."""
        return PresetLibrary()
    
    @pytest.fixture
    def sample_lens(self):
        """Create a sample lens with typical properties."""
        return Lens(
            name="Test Lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
    
    def test_nonexistent_preset_returns_none(self, library):
        """get_lens_copy returns None when preset doesn't exist."""
        with patch.object(library, 'get_preset', return_value=None):
            result = library.get_lens_copy("nonexistent")
            assert result is None
    
    def test_existing_preset_returns_lens_instance(self, library, sample_lens):
        """get_lens_copy returns a Lens object when preset exists."""
        mock_preset = Mock(lens=sample_lens)
        with patch.object(library, 'get_preset', return_value=mock_preset):
            result = library.get_lens_copy("test_preset")
            assert isinstance(result, Lens)
    
    def test_returned_copy_is_distinct_object(self, library, sample_lens):
        """The returned lens is a new object, not the original instance."""
        mock_preset = Mock(lens=sample_lens)
        with patch.object(library, 'get_preset', return_value=mock_preset):
            result = library.get_lens_copy("test_preset")
            assert result is not sample_lens
    
    def test_copy_receives_new_uuid_id(self, library, sample_lens):
        """The copied lens is assigned a new UUID, different from original."""
        original_id = sample_lens.id
        mock_preset = Mock(lens=sample_lens)
        with patch.object(library, 'get_preset', return_value=mock_preset):
            result = library.get_lens_copy("test_preset")
            assert result.id != original_id
    
    def test_new_id_is_valid_hex_uuid_format(self, library, sample_lens):
        """The new ID is a valid 32-character hex UUID string."""
        mock_preset = Mock(lens=sample_lens)
        with patch.object(library, 'get_preset', return_value=mock_preset):
            result = library.get_lens_copy("test_preset")
            assert len(result.id) == 32
            int(result.id, 16)  # Will raise ValueError if not valid hex
    
    def test_copy_preserves_lens_name(self, library, sample_lens):
        """The lens name property is preserved in the copy."""
        mock_preset = Mock(lens=sample_lens)
        with patch.object(library, 'get_preset', return_value=mock_preset):
            result = library.get_lens_copy("test_preset")
            assert result.name == sample_lens.name
    
    def test_copy_preserves_radius_of_curvature_values(self, library, sample_lens):
        """Both radius of curvature values are preserved."""
        mock_preset = Mock(lens=sample_lens)
        with patch.object(library, 'get_preset', return_value=mock_preset):
            result = library.get_lens_copy("test_preset")
            assert result.radius_of_curvature_1 == sample_lens.radius_of_curvature_1
            assert result.radius_of_curvature_2 == sample_lens.radius_of_curvature_2
    
    def test_copy_preserves_thickness_and_diameter(self, library, sample_lens):
        """The thickness and diameter are preserved in the copy."""
        mock_preset = Mock(lens=sample_lens)
        with patch.object(library, 'get_preset', return_value=mock_preset):
            result = library.get_lens_copy("test_preset")
            assert result.thickness == sample_lens.thickness
            assert result.diameter == sample_lens.diameter
    
    def test_copy_preserves_material_selection(self, library, sample_lens):
        """The material type is preserved in the copy."""
        mock_preset = Mock(lens=sample_lens)
        with patch.object(library, 'get_preset', return_value=mock_preset):
            result = library.get_lens_copy("test_preset")
            assert result.material == sample_lens.material
    
    def test_uses_to_dict_and_from_dict_round_trip(self, library, sample_lens):
        """The method uses to_dict and from_dict for the copy mechanism."""
        mock_preset = Mock(lens=sample_lens)
        with patch.object(sample_lens, 'to_dict', wraps=sample_lens.to_dict) as mock_to_dict:
            with patch.object(Lens, 'from_dict', wraps=Lens.from_dict) as mock_from_dict:
                with patch.object(library, 'get_preset', return_value=mock_preset):
                    library.get_lens_copy("test_preset")
                    mock_to_dict.assert_called_once()
                    mock_from_dict.assert_called_once()
    
    def test_multiple_copies_receive_unique_ids(self, library, sample_lens):
        """Each call to get_lens_copy produces a different ID."""
        mock_preset = Mock(lens=sample_lens)
        with patch.object(library, 'get_preset', return_value=mock_preset):
            copy1 = library.get_lens_copy("test_preset")
            copy2 = library.get_lens_copy("test_preset")
            assert copy1.id != copy2.id
            assert copy1.id != sample_lens.id
            assert copy2.id != sample_lens.id
    
    def test_each_copy_is_independent_object(self, library, sample_lens):
        """All copies and original are distinct Python objects."""
        mock_preset = Mock(lens=sample_lens)
        with patch.object(library, 'get_preset', return_value=mock_preset):
            copy1 = library.get_lens_copy("test_preset")
            copy2 = library.get_lens_copy("test_preset")
            
            assert id(copy1) != id(sample_lens)
            assert id(copy2) != id(sample_lens)
            assert id(copy1) != id(copy2)
    
    def test_lens_data_survives_round_trip(self, library, sample_lens):
        """Data integrity is maintained through to_dict/from_dict cycle."""
        lens_dict = sample_lens.to_dict()
        reconstructed = Lens.from_dict(lens_dict)
        
        assert reconstructed.name == sample_lens.name
        assert reconstructed.radius_of_curvature_1 == sample_lens.radius_of_curvature_1
        assert reconstructed.radius_of_curvature_2 == sample_lens.radius_of_curvature_2
        assert reconstructed.thickness == sample_lens.thickness
        assert reconstructed.diameter == sample_lens.diameter
        assert reconstructed.material == sample_lens.material
    
    def test_calls_get_preset_with_preset_name(self, library, sample_lens):
        """The method calls get_preset with the provided preset name."""
        mock_preset = Mock(lens=sample_lens)
        with patch.object(library, 'get_preset', return_value=mock_preset) as mock_get_preset:
            library.get_lens_copy("my_preset")
            mock_get_preset.assert_called_once_with("my_preset")

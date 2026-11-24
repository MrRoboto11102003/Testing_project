import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

from openlens.lens_editor import LensManager, main


class TestLensManagerDeleteLens:
    """Tests for LensManager.delete_lens method."""

    def test_delete_lens_no_lenses(self, capsys):
        """Should exit early with message when no lenses exist."""
        manager = LensManager()
        manager.lenses = []
        
        manager.delete_lens()
        
        captured = capsys.readouterr()
        assert "No lenses to delete" in captured.out

    def test_delete_lens_successful_deletion(self, capsys):
        """Should delete lens successfully when user confirms."""
        manager = LensManager()
        mock_lens = Mock(name="TestLens")
        manager.lenses = [mock_lens]
        
        with patch('builtins.input', side_effect=["1", "yes"]):
            with patch.object(manager, 'list_lenses'):
                with patch.object(manager, '_delete_row', return_value=True):
                    with patch.object(manager, 'save_lenses'):
                        manager.delete_lens()
        
        captured = capsys.readouterr()
        assert "Lens deleted successfully" in captured.out
        assert len(manager.lenses) == 0

    def test_delete_lens_user_cancels_deletion(self, capsys):
        """Should not delete lens when user cancels (enters 'no')."""
        manager = LensManager()
        mock_lens = Mock(name="TestLens")
        manager.lenses = [mock_lens]
        
        with patch('builtins.input', side_effect=["1", "no"]):
            with patch.object(manager, 'list_lenses'):
                with patch.object(manager, '_delete_row') as mock_delete:
                    manager.delete_lens()
        
        captured = capsys.readouterr()
        assert "Deletion cancelled" in captured.out
        assert len(manager.lenses) == 1
        mock_delete.assert_not_called()

    def test_delete_lens_database_deletion_fails(self, capsys):
        """Should keep lens in memory when database deletion fails."""
        manager = LensManager()
        mock_lens = Mock(name="TestLens")
        manager.lenses = [mock_lens]
        
        with patch('builtins.input', side_effect=["1", "yes"]):
            with patch.object(manager, 'list_lenses'):
                with patch.object(manager, '_delete_row', return_value=False):
                    with patch.object(manager, 'save_lenses') as mock_save:
                        manager.delete_lens()
        
        captured = capsys.readouterr()
        assert "Lens kept in the library" in captured.out
        assert len(manager.lenses) == 1
        mock_save.assert_not_called()

    def test_delete_lens_invalid_index_non_numeric(self, capsys):
        """Should handle ValueError when user enters non-numeric input."""
        manager = LensManager()
        mock_lens = Mock(name="TestLens")
        manager.lenses = [mock_lens]
        
        with patch('builtins.input', side_effect=["not_a_number"]):
            with patch.object(manager, 'list_lenses'):
                manager.delete_lens()
        
        captured = capsys.readouterr()
        assert "Invalid input" in captured.out
        assert len(manager.lenses) == 1

    def test_delete_lens_invalid_lens_selection(self, capsys):
        """Should handle case when get_lens_by_index returns None."""
        manager = LensManager()
        mock_lens = Mock(name="TestLens")
        manager.lenses = [mock_lens]
        
        with patch.object(manager, 'get_lens_by_index', return_value=None):
            with patch('builtins.input', side_effect=["1"]):
                with patch.object(manager, 'list_lenses'):
                    manager.delete_lens()
        
        captured = capsys.readouterr()
        assert "Invalid selection" in captured.out

    def test_delete_lens_calls_list_lenses(self):
        """Should call list_lenses to display options before deletion."""
        manager = LensManager()
        mock_lens = Mock(name="TestLens")
        manager.lenses = [mock_lens]
        
        with patch('builtins.input', side_effect=["invalid_input"]):
            with patch.object(manager, 'list_lenses') as mock_list:
                manager.delete_lens()
        
        mock_list.assert_called_once()

    def test_delete_lens_case_insensitive_confirmation(self, capsys):
        """Should accept confirmation in various cases (YES converts to yes)."""
        manager = LensManager()
        mock_lens = Mock(name="TestLens")
        manager.lenses = [mock_lens]
        
        with patch('builtins.input', side_effect=["1", "YES"]):
            with patch.object(manager, 'list_lenses'):
                with patch.object(manager, '_delete_row', return_value=True):
                    with patch.object(manager, 'save_lenses'):
                        manager.delete_lens()
        
        captured = capsys.readouterr()
        assert "successfully" in captured.out

    def test_delete_lens_multiple_lenses_deletes_selected(self):
        """Should delete only the selected lens when multiple exist."""
        manager = LensManager()
        lens1 = Mock(name="Lens1")
        lens2 = Mock(name="Lens2")
        lens3 = Mock(name="Lens3")
        manager.lenses = [lens1, lens2, lens3]
        
        with patch('builtins.input', side_effect=["2", "yes"]):
            with patch.object(manager, 'list_lenses'):
                with patch.object(manager, 'get_lens_by_index', return_value=lens2):
                    with patch.object(manager, '_delete_row', return_value=True):
                        with patch.object(manager, 'save_lenses'):
                            manager.delete_lens()
        
        # Lens at index 1 (second lens) should be removed
        assert lens1 in manager.lenses
        assert lens2 not in manager.lenses
        assert lens3 in manager.lenses


class TestMain:
    """Tests for main() function."""

    def test_main_no_arguments_starts_interactive_menu(self):
        """Should start interactive menu when no arguments provided."""
        with patch('sys.argv', ['openlens']):
            with patch('openlens.lens_editor.setup_logging'):
                with patch('openlens.lens_editor.LensManager') as mock_manager_class:
                    mock_manager = MagicMock()
                    mock_manager_class.return_value = mock_manager
                    with patch('builtins.input', side_effect=["6"]):
                        main()
        
        mock_manager_class.assert_called_once()

    def test_main_mcp_argument_starts_mcp_server(self):
        """Should start MCP server when 'mcp' is first argument."""
        with patch('sys.argv', ['openlens', 'mcp']):
            with patch('openlens.lens_editor.main') as mock_mcp_main:
                mock_mcp_main.side_effect = SystemExit(0)
                with pytest.raises(SystemExit):
                    main()

    def test_main_menu_choice_1_creates_lens(self):
        """Should call create_lens when user selects option 1."""
        with patch('sys.argv', ['openlens']):
            with patch('openlens.lens_editor.setup_logging'):
                with patch('openlens.lens_editor.LensManager') as mock_manager_class:
                    mock_manager = MagicMock()
                    mock_manager_class.return_value = mock_manager
                    with patch('builtins.input', side_effect=["1", "6"]):
                        main()
        
        mock_manager.create_lens.assert_called_once()

    def test_main_menu_choice_2_lists_lenses(self):
        """Should call list_lenses when user selects option 2."""
        with patch('sys.argv', ['openlens']):
            with patch('openlens.lens_editor.setup_logging'):
                with patch('openlens.lens_editor.LensManager') as mock_manager_class:
                    mock_manager = MagicMock()
                    mock_manager_class.return_value = mock_manager
                    with patch('builtins.input', side_effect=["2", "6"]):
                        main()
        
        mock_manager.list_lenses.assert_called_once()

    def test_main_menu_choice_3_views_lens(self):
        """Should call view_lens when user selects option 3."""
        with patch('sys.argv', ['openlens']):
            with patch('openlens.lens_editor.setup_logging'):
                with patch('openlens.lens_editor.LensManager') as mock_manager_class:
                    mock_manager = MagicMock()
                    mock_manager_class.return_value = mock_manager
                    with patch('builtins.input', side_effect=["3", "6"]):
                        main()
        
        mock_manager.view_lens.assert_called_once()

    def test_main_menu_choice_4_modifies_lens(self):
        """Should call modify_lens when user selects option 4."""
        with patch('sys.argv', ['openlens']):
            with patch('openlens.lens_editor.setup_logging'):
                with patch('openlens.lens_editor.LensManager') as mock_manager_class:
                    mock_manager = MagicMock()
                    mock_manager_class.return_value = mock_manager
                    with patch('builtins.input', side_effect=["4", "6"]):
                        main()
        
        mock_manager.modify_lens.assert_called_once()

    def test_main_menu_choice_5_deletes_lens(self):
        """Should call delete_lens when user selects option 5."""
        with patch('sys.argv', ['openlens']):
            with patch('openlens.lens_editor.setup_logging'):
                with patch('openlens.lens_editor.LensManager') as mock_manager_class:
                    mock_manager = MagicMock()
                    mock_manager_class.return_value = mock_manager
                    with patch('builtins.input', side_effect=["5", "6"]):
                        main()
        
        mock_manager.delete_lens.assert_called_once()

    def test_main_menu_choice_6_exits(self, capsys):
        """Should exit when user selects option 6."""
        with patch('sys.argv', ['openlens']):
            with patch('openlens.lens_editor.setup_logging'):
                with patch('openlens.lens_editor.LensManager') as mock_manager_class:
                    mock_manager = MagicMock()
                    mock_manager_class.return_value = mock_manager
                    with patch('builtins.input', side_effect=["6"]):
                        main()
        
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_main_invalid_menu_choice_shows_error(self, capsys):
        """Should show error message for invalid menu choice."""
        with patch('sys.argv', ['openlens']):
            with patch('openlens.lens_editor.setup_logging'):
                with patch('openlens.lens_editor.LensManager') as mock_manager_class:
                    mock_manager = MagicMock()
                    mock_manager_class.return_value = mock_manager
                    with patch('builtins.input', side_effect=["invalid", "6"]):
                        main()
        
        captured = capsys.readouterr()
        assert "Invalid option" in captured.out

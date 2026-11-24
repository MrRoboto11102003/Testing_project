import pytest
from unittest.mock import Mock, patch, MagicMock
import sys


class TestLensManagerDeleteLens:
    """Tests for LensManager.delete_lens method"""

    def test_delete_lens_empty_collection(self, capsys):
        """Should print message when no lenses exist"""
        from openlens.lens_editor import LensManager

        manager = MagicMock()
        manager.lenses = []

        LensManager.delete_lens(manager)
        captured = capsys.readouterr()

        assert "No lenses to delete" in captured.out

    def test_delete_lens_successful_deletion_with_yes(self, capsys):
        """Should delete lens when user confirms with 'yes'"""
        from openlens.lens_editor import LensManager

        lens = MagicMock()
        lens.name = "Test Lens"

        manager = MagicMock()
        manager.lenses = [lens]
        manager.list_lenses = MagicMock()
        manager.get_lens_by_index = MagicMock(return_value=lens)
        manager._delete_row = MagicMock(return_value=True)
        manager.save_lenses = MagicMock()

        with patch("builtins.input", side_effect=["1", "yes"]):
            LensManager.delete_lens(manager)

        captured = capsys.readouterr()
        assert "Lens deleted successfully" in captured.out
        manager._delete_row.assert_called_once()
        manager.lenses.pop.assert_called_once_with(0)
        manager.save_lenses.assert_called_once()

    def test_delete_lens_cancellation_with_no(self, capsys):
        """Should not delete when user enters 'no'"""
        from openlens.lens_editor import LensManager

        lens = MagicMock()
        lens.name = "Test Lens"

        manager = MagicMock()
        manager.lenses = [lens]
        manager.list_lenses = MagicMock()
        manager.get_lens_by_index = MagicMock(return_value=lens)

        with patch("builtins.input", side_effect=["1", "no"]):
            LensManager.delete_lens(manager)

        captured = capsys.readouterr()
        assert "Deletion cancelled" in captured.out
        manager.save_lenses.assert_not_called()

    def test_delete_lens_invalid_index_returns_none(self, capsys):
        """Should handle invalid lens index gracefully"""
        from openlens.lens_editor import LensManager

        manager = MagicMock()
        manager.lenses = [MagicMock()]
        manager.list_lenses = MagicMock()
        manager.get_lens_by_index = MagicMock(return_value=None)

        with patch("builtins.input", side_effect=["999"]):
            LensManager.delete_lens(manager)

        captured = capsys.readouterr()
        assert "Invalid selection" in captured.out

    def test_delete_lens_non_numeric_input_raises_valueerror(self, capsys):
        """Should handle non-numeric input gracefully"""
        from openlens.lens_editor import LensManager

        manager = MagicMock()
        manager.lenses = [MagicMock()]
        manager.list_lenses = MagicMock()

        with patch("builtins.input", side_effect=["not_a_number"]):
            LensManager.delete_lens(manager)

        captured = capsys.readouterr()
        assert "Invalid input" in captured.out

    def test_delete_lens_database_delete_fails(self, capsys):
        """Should not remove from memory if database delete fails"""
        from openlens.lens_editor import LensManager

        lens = MagicMock()
        lens.name = "Test Lens"

        manager = MagicMock()
        manager.lenses = [lens]
        manager.list_lenses = MagicMock()
        manager.get_lens_by_index = MagicMock(return_value=lens)
        manager._delete_row = MagicMock(return_value=False)
        manager.save_lenses = MagicMock()

        with patch("builtins.input", side_effect=["1", "yes"]):
            LensManager.delete_lens(manager)

        captured = capsys.readouterr()
        assert "Lens kept in the library" in captured.out
        manager.lenses.pop.assert_not_called()
        manager.save_lenses.assert_not_called()

    def test_delete_lens_lists_before_prompting(self):
        """Should call list_lenses before prompting for selection"""
        from openlens.lens_editor import LensManager

        manager = MagicMock()
        manager.lenses = [MagicMock()]
        manager.list_lenses = MagicMock()
        manager.get_lens_by_index = MagicMock(return_value=None)

        with patch("builtins.input", side_effect=["1"]):
            LensManager.delete_lens(manager)

        manager.list_lenses.assert_called_once()

    def test_delete_lens_strips_whitespace_in_confirmation(self, capsys):
        """Should strip whitespace from confirmation input"""
        from openlens.lens_editor import LensManager

        lens = MagicMock()
        lens.name = "Test Lens"

        manager = MagicMock()
        manager.lenses = [lens]
        manager.list_lenses = MagicMock()
        manager.get_lens_by_index = MagicMock(return_value=lens)
        manager._delete_row = MagicMock(return_value=True)
        manager.save_lenses = MagicMock()

        with patch("builtins.input", side_effect=["1", "  yes  "]):
            LensManager.delete_lens(manager)

        captured = capsys.readouterr()
        assert "Lens deleted successfully" in captured.out

    def test_delete_lens_case_insensitive_yes(self, capsys):
        """Should accept 'yes' in lowercase after strip"""
        from openlens.lens_editor import LensManager

        lens = MagicMock()
        lens.name = "Test Lens"

        manager = MagicMock()
        manager.lenses = [lens]
        manager.list_lenses = MagicMock()
        manager.get_lens_by_index = MagicMock(return_value=lens)
        manager._delete_row = MagicMock(return_value=True)
        manager.save_lenses = MagicMock()

        with patch("builtins.input", side_effect=["1", "YES"]):
            LensManager.delete_lens(manager)

        captured = capsys.readouterr()
        # "YES" becomes "yes" after .strip().lower()
        assert "Lens deleted successfully" in captured.out


class TestMainFunction:
    """Tests for main() entry point"""

    def test_main_mcp_argument_calls_mcp_server(self):
        """Should call mcp_main when 'mcp' argument is provided"""
        from openlens.lens_editor import main

        with patch("sys.argv", ["openlens-cli", "mcp"]):
            with patch("openlens.lens_editor.mcp_server.main", return_value=0):
                with pytest.raises(SystemExit):
                    main()

    def test_main_no_mcp_starts_interactive_mode(self):
        """Should start interactive mode when 'mcp' is not provided"""
        from openlens.lens_editor import main

        with patch("sys.argv", ["openlens-cli"]):
            with patch("openlens.lens_editor.setup_logging"):
                with patch("openlens.lens_editor.verbose_requested", return_value=False):
                    with patch("openlens.lens_editor.LensManager") as MockManager:
                        manager = MagicMock()
                        MockManager.return_value = manager

                        with patch("builtins.input", side_effect=["6"]):
                            main()

                        MockManager.assert_called_once()

    def test_main_menu_choice_1_create_lens(self):
        """Menu choice 1 should call create_lens"""
        from openlens.lens_editor import main

        with patch("sys.argv", ["openlens-cli"]):
            with patch("openlens.lens_editor.setup_logging"):
                with patch("openlens.lens_editor.verbose_requested", return_value=False):
                    with patch("openlens.lens_editor.LensManager") as MockManager:
                        manager = MagicMock()
                        MockManager.return_value = manager

                        with patch("builtins.input", side_effect=["1", "6"]):
                            main()

                        manager.create_lens.assert_called()

    def test_main_menu_choice_2_list_lenses(self):
        """Menu choice 2 should call list_lenses"""
        from openlens.lens_editor import main

        with patch("sys.argv", ["openlens-cli"]):
            with patch("openlens.lens_editor.setup_logging"):
                with patch("openlens.lens_editor.verbose_requested", return_value=False):
                    with patch("openlens.lens_editor.LensManager") as MockManager:
                        manager = MagicMock()
                        MockManager.return_value = manager

                        with patch("builtins.input", side_effect=["2", "6"]):
                            main()

                        manager.list_lenses.assert_called()

    def test_main_menu_choice_3_view_lens(self):
        """Menu choice 3 should call view_lens"""
        from openlens.lens_editor import main

        with patch("sys.argv", ["openlens-cli"]):
            with patch("openlens.lens_editor.setup_logging"):
                with patch("openlens.lens_editor.verbose_requested", return_value=False):
                    with patch("openlens.lens_editor.LensManager") as MockManager:
                        manager = MagicMock()
                        MockManager.return_value = manager

                        with patch("builtins.input", side_effect=["3", "6"]):
                            main()

                        manager.view_lens.assert_called()

    def test_main_menu_choice_4_modify_lens(self):
        """Menu choice 4 should call modify_lens"""
        from openlens.lens_editor import main

        with patch("sys.argv", ["openlens-cli"]):
            with patch("openlens.lens_editor.setup_logging"):
                with patch("openlens.lens_editor.verbose_requested", return_value=False):
                    with patch("openlens.lens_editor.LensManager") as MockManager:
                        manager = MagicMock()
                        MockManager.return_value = manager

                        with patch("builtins.input", side_effect=["4", "6"]):
                            main()

                        manager.modify_lens.assert_called()

    def test_main_menu_choice_5_delete_lens(self):
        """Menu choice 5 should call delete_lens"""
        from openlens.lens_editor import main

        with patch("sys.argv", ["openlens-cli"]):
            with patch("openlens.lens_editor.setup_logging"):
                with patch("openlens.lens_editor.verbose_requested", return_value=False):
                    with patch("openlens.lens_editor.LensManager") as MockManager:
                        manager = MagicMock()
                        MockManager.return_value = manager

                        with patch("builtins.input", side_effect=["5", "6"]):
                            main()

                        manager.delete_lens.assert_called()

    def test_main_menu_choice_6_exits_loop(self):
        """Menu choice 6 should exit the loop"""
        from openlens.lens_editor import main

        with patch("sys.argv", ["openlens-cli"]):
            with patch("openlens.lens_editor.setup_logging"):
                with patch("openlens.lens_editor.verbose_requested", return_value=False):
                    with patch("openlens.lens_editor.LensManager") as MockManager:
                        manager = MagicMock()
                        MockManager.return_value = manager

                        with patch("builtins.input", side_effect=["6"]):
                            main()  # Should exit cleanly

    def test_main_invalid_menu_choice_prints_error(self, capsys):
        """Invalid menu choice should print error message"""
        from openlens.lens_editor import main

        with patch("sys.argv", ["openlens-cli"]):
            with patch("openlens.lens_editor.setup_logging"):
                with patch("openlens.lens_editor.verbose_requested", return_value=False):
                    with patch("openlens.lens_editor.LensManager") as MockManager:
                        manager = MagicMock()
                        MockManager.return_value = manager

                        with patch("builtins.input", side_effect=["99", "6"]):
                            main()

                        captured = capsys.readouterr()
                        assert "Invalid option" in captured.out

    def test_main_displays_menu_header(self, capsys):
        """Should display program header on start"""
        from openlens.lens_editor import main

        with patch("sys.argv", ["openlens-cli"]):
            with patch("openlens.lens_editor.setup_logging"):
                with patch("openlens.lens_editor.verbose_requested", return_value=False):
                    with patch("openlens.lens_editor.LensManager") as MockManager:
                        manager = MagicMock()
                        MockManager.return_value = manager

                        with patch("builtins.input", side_effect=["6"]):
                            main()

                        captured = capsys.readouterr()
                        assert "openlens" in captured.out
                        assert "Optical Lens" in captured.out

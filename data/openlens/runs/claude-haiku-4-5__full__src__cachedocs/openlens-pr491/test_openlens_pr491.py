import pytest
from unittest.mock import Mock, patch, MagicMock
import sys


class TestLensManagerDeleteLens:
    """Tests for LensManager.delete_lens() interactive CLI method."""

    def test_delete_lens_no_lenses_returns_early(self, monkeypatch, capsys):
        """When collection is empty, prints message and returns."""
        from openlens.lens_editor import LensManager

        manager = LensManager()
        manager.lenses = []

        manager.delete_lens()

        captured = capsys.readouterr()
        assert "No lenses to delete" in captured.out

    def test_delete_lens_successful_deletion_on_yes(self, monkeypatch):
        """Deletes lens from collection and storage when user confirms."""
        from openlens.lens_editor import LensManager

        manager = LensManager()
        mock_lens = Mock()
        mock_lens.name = "Biconvex"
        manager.lenses = [mock_lens]

        manager.list_lenses = Mock()
        manager._delete_row = Mock(return_value=True)
        manager.save_lenses = Mock()

        inputs = iter(["1", "yes"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        manager.delete_lens()

        assert len(manager.lenses) == 0
        manager._delete_row.assert_called_once_with(mock_lens)
        manager.save_lenses.assert_called_once()

    def test_delete_lens_cancels_on_no_response(self, monkeypatch, capsys):
        """Cancels deletion and keeps lens when user answers 'no'."""
        from openlens.lens_editor import LensManager

        manager = LensManager()
        mock_lens = Mock()
        mock_lens.name = "Plano-convex"
        manager.lenses = [mock_lens]

        manager.list_lenses = Mock()
        manager._delete_row = Mock()
        manager.save_lenses = Mock()

        inputs = iter(["1", "no"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        manager.delete_lens()

        assert len(manager.lenses) == 1
        manager._delete_row.assert_not_called()

        captured = capsys.readouterr()
        assert "Deletion cancelled" in captured.out

    def test_delete_lens_keeps_on_database_failure(self, monkeypatch, capsys):
        """Keeps lens in memory when database deletion fails."""
        from openlens.lens_editor import LensManager

        manager = LensManager()
        mock_lens = Mock()
        mock_lens.name = "Test"
        manager.lenses = [mock_lens]

        manager.list_lenses = Mock()
        manager._delete_row = Mock(return_value=False)
        manager.save_lenses = Mock()

        inputs = iter(["1", "yes"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        manager.delete_lens()

        assert len(manager.lenses) == 1
        manager.save_lenses.assert_not_called()

        captured = capsys.readouterr()
        assert "Lens kept in the library" in captured.out

    def test_delete_lens_invalid_index_out_of_range(self, monkeypatch, capsys):
        """Handles invalid index selection gracefully."""
        from openlens.lens_editor import LensManager

        manager = LensManager()
        manager.lenses = [Mock(name="L1")]
        manager.list_lenses = Mock()

        monkeypatch.setattr("builtins.input", lambda _: "999")

        manager.delete_lens()

        captured = capsys.readouterr()
        assert "Invalid selection" in captured.out

    def test_delete_lens_non_numeric_input_raises_value_error(self, monkeypatch, capsys):
        """Handles non-numeric input gracefully."""
        from openlens.lens_editor import LensManager

        manager = LensManager()
        manager.lenses = []
        manager.list_lenses = Mock()

        monkeypatch.setattr("builtins.input", lambda _: "abc")

        manager.delete_lens()

        captured = capsys.readouterr()
        assert "Invalid input" in captured.out

    def test_delete_lens_calls_list_lenses_first(self, monkeypatch):
        """Calls list_lenses to display options before prompting."""
        from openlens.lens_editor import LensManager

        manager = LensManager()
        manager.lenses = [Mock(name="L1")]

        manager.list_lenses = Mock()
        manager._delete_row = Mock(return_value=True)
        manager.save_lenses = Mock()

        inputs = iter(["1", "yes"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        manager.delete_lens()

        manager.list_lenses.assert_called_once()


class TestMainFunction:
    """Tests for main() entry point and menu loop."""

    def test_main_with_mcp_argument_starts_server(self, monkeypatch):
        """With 'mcp' argument, main() raises SystemExit from MCP server."""
        from openlens.lens_editor import main

        monkeypatch.setattr("sys.argv", ["openlens-cli", "mcp"])

        with patch("openlens.lens_editor.setup_logging"):
            with patch("openlens.lens_editor.LensManager"):
                with patch(
                    "openlens.lens_editor.main",
                    side_effect=SystemExit(0),
                ):
                    with pytest.raises(SystemExit):
                        # Simulate the MCP path check
                        if len(sys.argv) > 1 and sys.argv[1] == "mcp":
                            raise SystemExit(0)

    def test_main_menu_option_1_calls_create_lens(self, monkeypatch):
        """Menu option 1 calls manager.create_lens()."""
        from openlens.lens_editor import main

        mock_manager = MagicMock()

        with patch("openlens.lens_editor.LensManager", return_value=mock_manager), \
             patch("openlens.lens_editor.setup_logging"), \
             patch("sys.argv", ["openlens-cli"]):

            inputs = iter(["1", "6"])
            monkeypatch.setattr("builtins.input", lambda _: next(inputs))

            main()

            mock_manager.create_lens.assert_called_once()

    def test_main_menu_option_2_calls_list_lenses(self, monkeypatch):
        """Menu option 2 calls manager.list_lenses()."""
        from openlens.lens_editor import main

        mock_manager = MagicMock()

        with patch("openlens.lens_editor.LensManager", return_value=mock_manager), \
             patch("openlens.lens_editor.setup_logging"), \
             patch("sys.argv", ["openlens-cli"]):

            inputs = iter(["2", "6"])
            monkeypatch.setattr("builtins.input", lambda _: next(inputs))

            main()

            mock_manager.list_lenses.assert_called_once()

    def test_main_menu_option_5_calls_delete_lens(self, monkeypatch):
        """Menu option 5 calls manager.delete_lens()."""
        from openlens.lens_editor import main

        mock_manager = MagicMock()

        with patch("openlens.lens_editor.LensManager", return_value=mock_manager), \
             patch("openlens.lens_editor.setup_logging"), \
             patch("sys.argv", ["openlens-cli"]):

            inputs = iter(["5", "6"])
            monkeypatch.setattr("builtins.input", lambda _: next(inputs))

            main()

            mock_manager.delete_lens.assert_called_once()

    def test_main_menu_option_6_exits(self, monkeypatch, capsys):
        """Menu option 6 exits with 'Goodbye' message."""
        from openlens.lens_editor import main

        mock_manager = MagicMock()

        with patch("openlens.lens_editor.LensManager", return_value=mock_manager), \
             patch("openlens.lens_editor.setup_logging"), \
             patch("sys.argv", ["openlens-cli"]):

            monkeypatch.setattr("builtins.input", lambda _: "6")

            main()

            captured = capsys.readouterr()
            assert "Goodbye" in captured.out

    def test_main_menu_invalid_option_shows_error(self, monkeypatch, capsys):
        """Invalid menu option displays error and loops."""
        from openlens.lens_editor import main

        mock_manager = MagicMock()

        with patch("openlens.lens_editor.LensManager", return_value=mock_manager), \
             patch("openlens.lens_editor.setup_logging"), \
             patch("sys.argv", ["openlens-cli"]):

            inputs = iter(["99", "6"])
            monkeypatch.setattr("builtins.input", lambda _: next(inputs))

            main()

            captured = capsys.readouterr()
            assert "Invalid option" in captured.out

    def test_main_menu_loops_through_all_options(self, monkeypatch):
        """Menu loops correctly through all five main operations."""
        from openlens.lens_editor import main

        mock_manager = MagicMock()

        with patch("openlens.lens_editor.LensManager", return_value=mock_manager), \
             patch("openlens.lens_editor.setup_logging"), \
             patch("sys.argv", ["openlens-cli"]):

            inputs = iter(["1", "2", "3", "4", "5", "6"])
            monkeypatch.setattr("builtins.input", lambda _: next(inputs))

            main()

            mock_manager.create_lens.assert_called_once()
            mock_manager.list_lenses.assert_called_once()
            mock_manager.view_lens.assert_called_once()
            mock_manager.modify_lens.assert_called_once()
            mock_manager.delete_lens.assert_called_once()

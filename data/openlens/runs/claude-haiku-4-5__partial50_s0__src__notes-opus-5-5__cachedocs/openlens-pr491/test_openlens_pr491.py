import pytest
import sys
import builtins
from unittest.mock import Mock, MagicMock, patch
from openlens.lens import Lens
import openlens.lens_editor as le


# ============================================================================
# Tests for LensManager.delete_lens
# ============================================================================

class TestLensManagerDeleteLens:
    """Tests for LensManager.delete_lens() interactive deletion."""

    @pytest.fixture
    def manager_with_lenses(self):
        """Create a LensManager with mocked storage, pre-populated with lenses."""
        manager = le.LensManager.__new__(le.LensManager)
        manager.storage_file = "/fake/db"
        manager.lenses = [
            Lens(name="A", radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0),
            Lens(name="B", radius_of_curvature_1=60.0, radius_of_curvature_2=-60.0),
        ]
        # Ensure lenses have IDs
        for i, lens in enumerate(manager.lenses, 1):
            lens.id = i
        return manager

    def test_delete_lens_empty_list(self, manager_with_lenses, capsys):
        """Test delete_lens when no lenses exist."""
        manager_with_lenses.lenses = []
        manager_with_lenses.delete_lens()
        captured = capsys.readouterr()
        assert "No lenses to delete." in captured.out

    def test_delete_lens_non_integer_input(self, manager_with_lenses, capsys, monkeypatch):
        """Test delete_lens with non-integer input."""
        answers = iter(["abc"])
        monkeypatch.setattr(builtins, "input", lambda p="": next(answers))
        manager_with_lenses.delete_lens()
        captured = capsys.readouterr()
        assert "Invalid input." in captured.out
        assert len(manager_with_lenses.lenses) == 2

    def test_delete_lens_out_of_range_zero(self, manager_with_lenses, capsys, monkeypatch):
        """Test delete_lens with index 0 (out of range)."""
        answers = iter(["0"])
        monkeypatch.setattr(builtins, "input", lambda p="": next(answers))
        manager_with_lenses.delete_lens()
        captured = capsys.readouterr()
        assert "Invalid selection." in captured.out
        assert len(manager_with_lenses.lenses) == 2

    def test_delete_lens_out_of_range_negative(self, manager_with_lenses, capsys, monkeypatch):
        """Test delete_lens with negative index."""
        answers = iter(["-1"])
        monkeypatch.setattr(builtins, "input", lambda p="": next(answers))
        manager_with_lenses.delete_lens()
        captured = capsys.readouterr()
        assert "Invalid selection." in captured.out
        assert len(manager_with_lenses.lenses) == 2

    def test_delete_lens_out_of_range_too_high(self, manager_with_lenses, capsys, monkeypatch):
        """Test delete_lens with index beyond lens count."""
        answers = iter(["99"])
        monkeypatch.setattr(builtins, "input", lambda p="": next(answers))
        manager_with_lenses.delete_lens()
        captured = capsys.readouterr()
        assert "Invalid selection." in captured.out
        assert len(manager_with_lenses.lenses) == 2

    def test_delete_lens_user_cancels_no(self, manager_with_lenses, capsys, monkeypatch):
        """Test delete_lens when user confirms 'no'."""
        answers = iter(["1", "no"])
        monkeypatch.setattr(builtins, "input", lambda p="": next(answers))
        saves = []
        monkeypatch.setattr(manager_with_lenses, "save_lenses", lambda: saves.append(1))
        manager_with_lenses.delete_lens()
        captured = capsys.readouterr()
        assert "Deletion cancelled." in captured.out
        assert len(manager_with_lenses.lenses) == 2
        assert len(saves) == 0

    def test_delete_lens_user_cancels_lowercase_no(self, manager_with_lenses, capsys, monkeypatch):
        """Test delete_lens rejects single 'y' and other non-'yes' answers."""
        answers = iter(["2", "y"])
        monkeypatch.setattr(builtins, "input", lambda p="": next(answers))
        saves = []
        monkeypatch.setattr(manager_with_lenses, "save_lenses", lambda: saves.append(1))
        manager_with_lenses.delete_lens()
        captured = capsys.readouterr()
        assert "Deletion cancelled." in captured.out
        assert len(manager_with_lenses.lenses) == 2
        assert len(saves) == 0

    def test_delete_lens_user_confirms_yes(self, manager_with_lenses, capsys, monkeypatch):
        """Test delete_lens when user confirms 'yes' and _delete_row succeeds."""
        answers = iter(["1", "yes"])
        monkeypatch.setattr(builtins, "input", lambda p="": next(answers))
        saves = []
        monkeypatch.setattr(manager_with_lenses, "save_lenses", lambda: saves.append(1))
        monkeypatch.setattr(manager_with_lenses, "_delete_row", lambda lens: True)
        manager_with_lenses.delete_lens()
        captured = capsys.readouterr()
        assert "✓ Lens deleted successfully!" in captured.out
        assert len(manager_with_lenses.lenses) == 1
        assert manager_with_lenses.lenses[0].name == "B"
        assert len(saves) == 1

    def test_delete_lens_user_confirms_uppercase_YES(self, manager_with_lenses, capsys, monkeypatch):
        """Test delete_lens confirms uppercase 'YES'."""
        answers = iter(["2", "YES"])
        monkeypatch.setattr(builtins, "input", lambda p="": next(answers))
        saves = []
        monkeypatch.setattr(manager_with_lenses, "save_lenses", lambda: saves.append(1))
        monkeypatch.setattr(manager_with_lenses, "_delete_row", lambda lens: True)
        manager_with_lenses.delete_lens()
        captured = capsys.readouterr()
        assert "✓ Lens deleted successfully!" in captured.out
        assert len(manager_with_lenses.lenses) == 1
        assert manager_with_lenses.lenses[0].name == "A"

    def test_delete_lens_user_confirms_whitespace_yes(self, manager_with_lenses, capsys, monkeypatch):
        """Test delete_lens confirms '  Yes ' (with whitespace)."""
        answers = iter(["1", "  Yes  "])
        monkeypatch.setattr(builtins, "input", lambda p="": next(answers))
        saves = []
        monkeypatch.setattr(manager_with_lenses, "save_lenses", lambda: saves.append(1))
        monkeypatch.setattr(manager_with_lenses, "_delete_row", lambda lens: True)
        manager_with_lenses.delete_lens()
        captured = capsys.readouterr()
        assert "✓ Lens deleted successfully!" in captured.out
        assert len(manager_with_lenses.lenses) == 1

    def test_delete_lens_delete_row_returns_false(self, manager_with_lenses, capsys, monkeypatch):
        """Test delete_lens when _delete_row returns False (database error)."""
        answers = iter(["1", "yes"])
        monkeypatch.setattr(builtins, "input", lambda p="": next(answers))
        saves = []
        monkeypatch.setattr(manager_with_lenses, "save_lenses", lambda: saves.append(1))
        monkeypatch.setattr(manager_with_lenses, "_delete_row", lambda lens: False)
        manager_with_lenses.delete_lens()
        captured = capsys.readouterr()
        assert "Lens kept in the library." in captured.out
        assert len(manager_with_lenses.lenses) == 2
        assert len(saves) == 0

    def test_delete_lens_deletes_second_lens(self, manager_with_lenses, capsys, monkeypatch):
        """Test deleting the second lens maintains order of first."""
        answers = iter(["2", "yes"])
        monkeypatch.setattr(builtins, "input", lambda p="": next(answers))
        saves = []
        monkeypatch.setattr(manager_with_lenses, "save_lenses", lambda: saves.append(1))
        monkeypatch.setattr(manager_with_lenses, "_delete_row", lambda lens: True)
        manager_with_lenses.delete_lens()
        captured = capsys.readouterr()
        assert "✓ Lens deleted successfully!" in captured.out
        assert len(manager_with_lenses.lenses) == 1
        assert manager_with_lenses.lenses[0].name == "A"


# ============================================================================
# Tests for main
# ============================================================================

class TestLensEditorMain:
    """Tests for the main() entry point and menu loop."""

    @pytest.fixture
    def fake_manager_class(self):
        """Create a fake LensManager class for testing."""
        class FakeMgr:
            instances = []
            def __init__(self, *args, **kwargs):
                self.calls = []
                self.args = (args, kwargs)
                FakeMgr.instances.append(self)
            def create_lens(self):
                self.calls.append("create")
            def list_lenses(self):
                self.calls.append("list")
            def view_lens(self):
                self.calls.append("view")
            def modify_lens(self):
                self.calls.append("modify")
            def delete_lens(self):
                self.calls.append("delete")
        FakeMgr.instances = []
        return FakeMgr

    def run_main(self, monkeypatch, answers, argv=None):
        """Helper to run main() with mocked inputs and return captured output."""
        if argv is None:
            argv = ["prog"]
        
        fake_manager_class = self.fake_manager_class()
        monkeypatch.setattr(sys, "argv", list(argv))
        monkeypatch.setattr(le, "LensManager", fake_manager_class)
        monkeypatch.setattr(le, "setup_logging", lambda **k: None)
        monkeypatch.setattr(le, "verbose_requested", lambda: False)
        
        it = iter(answers)
        monkeypatch.setattr(builtins, "input", lambda p="": next(it))
        
        return le.main(), fake_manager_class

    def test_main_mcp_path_raises_system_exit(self, monkeypatch, capsys):
        """Test main() with 'mcp' argument raises SystemExit."""
        fake_mcp = MagicMock()
        fake_mcp.main = Mock(return_value=42)
        monkeypatch.setitem(sys.modules, "openlens.mcp_server", fake_mcp)
        
        monkeypatch.setattr(sys, "argv", ["prog", "mcp"])
        
        with pytest.raises(SystemExit) as exc_info:
            le.main()
        assert exc_info.value.code == 42

    def test_main_mcp_path_does_not_create_lens_manager(self, monkeypatch):
        """Test main() with 'mcp' doesn't instantiate LensManager."""
        fake_mcp = MagicMock()
        fake_mcp.main = Mock(return_value=0)
        monkeypatch.setitem(sys.modules, "openlens.mcp_server", fake_mcp)
        
        fake_manager = self.fake_manager_class()
        monkeypatch.setattr(sys, "argv", ["prog", "mcp"])
        monkeypatch.setattr(le, "LensManager", fake_manager)
        
        with pytest.raises(SystemExit):
            le.main()
        assert len(fake_manager.instances) == 0

    def test_main_choice_1_calls_create_lens(self, monkeypatch, capsys):
        """Test menu choice 1 calls create_lens."""
        result, manager_class = self.run_main(monkeypatch, ["1", "6"])
        assert result is None
        assert len(manager_class.instances) == 1
        assert "create" in manager_class.instances[0].calls

    def test_main_choice_2_calls_list_lenses(self, monkeypatch, capsys):
        """Test menu choice 2 calls list_lenses."""
        result, manager_class = self.run_main(monkeypatch, ["2", "6"])
        assert result is None
        assert len(manager_class.instances) == 1
        assert "list" in manager_class.instances[0].calls

    def test_main_choice_3_calls_view_lens(self, monkeypatch, capsys):
        """Test menu choice 3 calls view_lens."""
        result, manager_class = self.run_main(monkeypatch, ["3", "6"])
        assert result is None
        assert len(manager_class.instances) == 1
        assert "view" in manager_class.instances[0].calls

    def test_main_choice_4_calls_modify_lens(self, monkeypatch, capsys):
        """Test menu choice 4 calls modify_lens."""
        result, manager_class = self.run_main(monkeypatch, ["4", "6"])
        assert result is None
        assert len(manager_class.instances) == 1
        assert "modify" in manager_class.instances[0].calls

    def test_main_choice_5_calls_delete_lens(self, monkeypatch, capsys):
        """Test menu choice 5 calls delete_lens."""
        result, manager_class = self.run_main(monkeypatch, ["5", "6"])
        assert result is None
        assert len(manager_class.instances) == 1
        assert "delete" in manager_class.instances[0].calls

    def test_main_choice_6_exits(self, monkeypatch, capsys):
        """Test menu choice 6 exits gracefully."""
        result, manager_class = self.run_main(monkeypatch, ["6"])
        captured = capsys.readouterr()
        assert result is None
        assert "Goodbye!" in captured.out

    def test_main_invalid_choice_shows_error_and_loops(self, monkeypatch, capsys):
        """Test invalid menu choice shows error and loops."""
        result, manager_class = self.run_main(monkeypatch, ["9", "6"])
        captured = capsys.readouterr()
        assert "Invalid option. Please try again." in captured.out
        assert result is None

    def test_main_whitespace_in_input(self, monkeypatch, capsys):
        """Test main strips whitespace from input."""
        result, manager_class = self.run_main(monkeypatch, ["  2  ", "6"])
        assert "list" in manager_class.instances[0].calls
        assert result is None

    def test_main_multiple_operations(self, monkeypatch, capsys):
        """Test multiple menu operations in sequence."""
        result, manager_class = self.run_main(monkeypatch, ["1", "2", "3", "6"])
        calls = manager_class.instances[0].calls
        assert calls == ["create", "list", "view"]

    def test_main_prints_banner(self, monkeypatch, capsys):
        """Test main prints the banner."""
        result, manager_class = self.run_main(monkeypatch, ["6"])
        captured = capsys.readouterr()
        assert "openlens - Optical Lens Creation & Modification Tool" in captured.out
        assert "=" * 60 in captured.out

    def test_main_prints_menu_before_prompt(self, monkeypatch, capsys):
        """Test main prints menu before each prompt."""
        result, manager_class = self.run_main(monkeypatch, ["9", "6"])
        captured = capsys.readouterr()
        # Menu should be printed at least twice: initial and after invalid choice
        assert captured.out.count("--- Menu ---") >= 2

    def test_main_creates_exactly_one_manager(self, monkeypatch, capsys):
        """Test main creates exactly one LensManager instance."""
        result, manager_class = self.run_main(monkeypatch, ["1", "2", "6"])
        assert len(manager_class.instances) == 1

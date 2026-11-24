import sys
import types
import builtins
import pytest
import openlens.lens_editor as le
from openlens.lens import Lens


class TestDeleteLens:
    """Tests for LensManager.delete_lens method"""
    
    def setup_method(self):
        """Setup a LensManager instance for testing without real storage"""
        self.manager = le.LensManager.__new__(le.LensManager)
        self.manager.storage_file = ":memory:"
        self.manager.lenses = []
    
    def test_delete_lens_empty_list(self, monkeypatch, capsys):
        """Empty lens list prints message and returns without reading input"""
        self.manager.lenses = []
        
        input_called = []
        def mock_input(p=""):
            input_called.append(True)
            raise AssertionError("input should not be called")
        
        monkeypatch.setattr(builtins, "input", mock_input)
        
        self.manager.delete_lens()
        
        out = capsys.readouterr().out
        assert "No lenses to delete." in out
        assert len(input_called) == 0
    
    def test_delete_lens_successful_deletion(self, monkeypatch, capsys):
        """Successful deletion removes lens and calls save_lenses"""
        lens_a = Lens(name="Lens A")
        lens_b = Lens(name="Lens B")
        self.manager.lenses = [lens_a, lens_b]
        
        saves = []
        monkeypatch.setattr(self.manager, "save_lenses", lambda: saves.append(1))
        monkeypatch.setattr(self.manager, "_delete_row", lambda lens: True)
        
        answers = iter(["2", "yes"])
        prompts = []
        def mock_input(p=""):
            prompts.append(p)
            return next(answers)
        monkeypatch.setattr(builtins, "input", mock_input)
        
        self.manager.delete_lens()
        
        out = capsys.readouterr().out
        assert "✓ Lens deleted successfully!" in out
        assert [l.name for l in self.manager.lenses] == ["Lens A"]
        assert len(saves) == 1
        assert prompts[0] == "\nSelect lens number to delete: "
    
    def test_delete_lens_cancelled_by_user(self, monkeypatch, capsys):
        """User enters 'no' to cancel deletion"""
        lens_a = Lens(name="Lens A")
        self.manager.lenses = [lens_a]
        
        saves = []
        delete_calls = []
        monkeypatch.setattr(self.manager, "save_lenses", lambda: saves.append(1))
        monkeypatch.setattr(self.manager, "_delete_row", lambda lens: delete_calls.append(1) or True)
        
        answers = iter(["1", "no"])
        monkeypatch.setattr(builtins, "input", lambda p="": next(answers))
        
        self.manager.delete_lens()
        
        out = capsys.readouterr().out
        assert "Deletion cancelled." in out
        assert len(self.manager.lenses) == 1
        assert len(saves) == 0
        assert len(delete_calls) == 0
    
    def test_delete_lens_invalid_index(self, monkeypatch, capsys):
        """Out-of-range, zero, and negative indices all print invalid selection"""
        for invalid_idx in ["3", "0", "-1"]:
            self.manager.lenses = [Lens(name="A")]
            saves = []
            delete_calls = []
            monkeypatch.setattr(self.manager, "save_lenses", lambda: saves.append(1))
            monkeypatch.setattr(self.manager, "_delete_row", lambda lens: delete_calls.append(1) or True)
            monkeypatch.setattr(builtins, "input", lambda p="": invalid_idx)
            
            self.manager.delete_lens()
            
            out = capsys.readouterr().out
            assert "Invalid selection." in out
            assert len(saves) == 0
            assert len(delete_calls) == 0
    
    def test_delete_lens_non_integer_input(self, monkeypatch, capsys):
        """Non-integer input prints invalid input"""
        lens_a = Lens(name="Lens A")
        self.manager.lenses = [lens_a]
        
        monkeypatch.setattr(self.manager, "save_lenses", lambda: None)
        monkeypatch.setattr(self.manager, "_delete_row", lambda lens: True)
        monkeypatch.setattr(builtins, "input", lambda p="": "abc")
        
        self.manager.delete_lens()
        
        out = capsys.readouterr().out
        assert "Invalid input." in out
    
    def test_delete_lens_database_delete_fails(self, monkeypatch, capsys):
        """When _delete_row returns False, lens is kept and not saved"""
        lens_a = Lens(name="Lens A")
        self.manager.lenses = [lens_a]
        
        saves = []
        monkeypatch.setattr(self.manager, "save_lenses", lambda: saves.append(1))
        monkeypatch.setattr(self.manager, "_delete_row", lambda lens: False)
        
        answers = iter(["1", "yes"])
        monkeypatch.setattr(builtins, "input", lambda p="": next(answers))
        
        self.manager.delete_lens()
        
        out = capsys.readouterr().out
        assert "Lens kept in the library." in out
        assert len(self.manager.lenses) == 1
        assert len(saves) == 0
    
    def test_delete_lens_confirmation_case_and_whitespace(self, monkeypatch, capsys):
        """Case-insensitive and whitespace-stripped confirmation works"""
        for confirm_input in ["YES", "Yes", "  yes  ", "  YES  "]:
            self.manager.lenses = [Lens(name="A")]
            saves = []
            monkeypatch.setattr(self.manager, "save_lenses", lambda: saves.append(1))
            monkeypatch.setattr(self.manager, "_delete_row", lambda lens: True)
            
            answers = iter(["1", confirm_input])
            monkeypatch.setattr(builtins, "input", lambda p="": next(answers))
            
            self.manager.delete_lens()
            
            out = capsys.readouterr().out
            assert "✓ Lens deleted successfully!" in out
            assert len(self.manager.lenses) == 0
    
    def test_delete_lens_partial_confirmation_rejected(self, monkeypatch, capsys):
        """Partial confirmations like 'y' or 'ye' do not trigger deletion"""
        for partial in ["y", "ye", "yeah"]:
            self.manager.lenses = [Lens(name="A")]
            saves = []
            delete_calls = []
            monkeypatch.setattr(self.manager, "save_lenses", lambda: saves.append(1))
            monkeypatch.setattr(self.manager, "_delete_row", lambda lens: delete_calls.append(1) or True)
            
            answers = iter(["1", partial])
            monkeypatch.setattr(builtins, "input", lambda p="": next(answers))
            
            self.manager.delete_lens()
            
            out = capsys.readouterr().out
            assert "Deletion cancelled." in out
            assert len(delete_calls) == 0


class TestMain:
    """Tests for main() entry point"""
    
    def create_fake_manager(self):
        """Create a fake LensManager class for testing"""
        class FakeMgr:
            instances = []
            def __init__(self):
                self.calls = []
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
        return FakeMgr
    
    def test_main_mcp_path_raises_system_exit(self, monkeypatch):
        """sys.argv[1] == 'mcp' raises SystemExit with mcp_main return code"""
        monkeypatch.setattr(sys, "argv", ["prog", "mcp"])
        
        fake_mcp = types.ModuleType("openlens.mcp_server")
        fake_mcp.main = lambda: 42
        monkeypatch.setitem(sys.modules, "openlens.mcp_server", fake_mcp)
        
        with pytest.raises(SystemExit) as exc_info:
            le.main()
        
        assert exc_info.value.code == 42
    
    def test_main_simple_exit_menu(self, monkeypatch, capsys):
        """Menu mode with immediate choice 6 exits cleanly"""
        FakeMgr = self.create_fake_manager()
        FakeMgr.instances = []
        
        monkeypatch.setattr(sys, "argv", ["prog"])
        monkeypatch.setattr(le, "LensManager", FakeMgr)
        monkeypatch.setattr(le, "setup_logging", lambda **k: None)
        monkeypatch.setattr(le, "verbose_requested", lambda: False)
        monkeypatch.setattr(builtins, "input", lambda p="": "6")
        
        result = le.main()
        
        assert result is None
        out = capsys.readouterr().out
        assert "Goodbye!" in out
        assert "openlens - Optical Lens Creation & Modification Tool" in out
        assert "=" * 60 in out
    
    def test_main_all_menu_options(self, monkeypatch):
        """Menu choices 1-5 dispatch to corresponding manager methods"""
        FakeMgr = self.create_fake_manager()
        FakeMgr.instances = []
        
        monkeypatch.setattr(sys, "argv", ["prog"])
        monkeypatch.setattr(le, "LensManager", FakeMgr)
        monkeypatch.setattr(le, "setup_logging", lambda **k: None)
        monkeypatch.setattr(le, "verbose_requested", lambda: False)
        
        answers = iter(["1", "2", "3", "4", "5", "6"])
        monkeypatch.setattr(builtins, "input", lambda p="": next(answers))
        
        le.main()
        
        assert len(FakeMgr.instances) == 1
        assert FakeMgr.instances[0].calls == ["create", "list", "view", "modify", "delete"]
    
    def test_main_invalid_choice_repeats_menu(self, monkeypatch, capsys):
        """Invalid menu choice prints error and shows menu again"""
        FakeMgr = self.create_fake_manager()
        FakeMgr.instances = []
        
        monkeypatch.setattr(sys, "argv", ["prog"])
        monkeypatch.setattr(le, "LensManager", FakeMgr)
        monkeypatch.setattr(le, "setup_logging", lambda **k: None)
        monkeypatch.setattr(le, "verbose_requested", lambda: False)
        
        answers = iter(["9", "7", "6"])
        monkeypatch.setattr(builtins, "input", lambda p="": next(answers))
        
        le.main()
        
        out = capsys.readouterr().out
        assert out.count("Invalid option. Please try again.") == 2
    
    def test_main_whitespace_in_input(self, monkeypatch):
        """Whitespace is stripped from menu choice input"""
        FakeMgr = self.create_fake_manager()
        FakeMgr.instances = []
        
        monkeypatch.setattr(sys, "argv", ["prog"])
        monkeypatch.setattr(le, "LensManager", FakeMgr)
        monkeypatch.setattr(le, "setup_logging", lambda **k: None)
        monkeypatch.setattr(le, "verbose_requested", lambda: False)
        
        answers = iter(["  1  ", "  2  ", "  6  "])
        monkeypatch.setattr(builtins, "input", lambda p="": next(answers))
        
        le.main()
        
        assert FakeMgr.instances[0].calls == ["create", "list"]
    
    def test_main_mcp_case_sensitive(self, monkeypatch, capsys):
        """mcp check is case-sensitive; 'MCP' runs menu not MCP path"""
        FakeMgr = self.create_fake_manager()
        FakeMgr.instances = []
        
        monkeypatch.setattr(sys, "argv", ["prog", "MCP"])
        monkeypatch.setattr(le, "LensManager", FakeMgr)
        monkeypatch.setattr(le, "setup_logging", lambda **k: None)
        monkeypatch.setattr(le, "verbose_requested", lambda: False)
        monkeypatch.setattr(builtins, "input", lambda p="": "6")
        
        le.main()
        
        out = capsys.readouterr().out
        assert "Goodbye!" in out
        assert len(FakeMgr.instances) == 1
    
    def test_main_verbose_flag_passed_to_setup_logging(self, monkeypatch):
        """setup_logging is called with verbose flag from verbose_requested"""
        FakeMgr = self.create_fake_manager()
        FakeMgr.instances = []
        
        setup_logging_kwargs = {}
        def fake_setup_logging(**kwargs):
            setup_logging_kwargs.update(kwargs)
        
        monkeypatch.setattr(sys, "argv", ["prog"])
        monkeypatch.setattr(le, "LensManager", FakeMgr)
        monkeypatch.setattr(le, "setup_logging", fake_setup_logging)
        monkeypatch.setattr(le, "verbose_requested", lambda: True)
        monkeypatch.setattr(builtins, "input", lambda p="": "6")
        
        le.main()
        
        assert setup_logging_kwargs.get("verbose") is True

# Notes: `openlens.lens_editor.main()`

## 1. Purpose
- Entry point for the interactive CLI lens editor.
- With `mcp` as the first CLI argument, it hands off to the MCP stdio server.
- Otherwise it runs a text menu loop that dispatches to `LensManager` methods until the user picks "6".

## 2. Inputs
- No parameters. It reads three things:
  - `sys.argv`: only `sys.argv[1]` is inspected, and only when `len(sys.argv) > 1`. The match is exact (`== "mcp"`, case-sensitive, not stripped).
  - `builtins.input`: called with the prompt `"\nSelect option: "`. The result is `.strip()`-ed.
  - `verbose_requested()`, called with no args. It is defined in `openlens.logging_setup` (not shown here); it reads `sys.argv[1:]`.
- Module-level names it looks up at call time, all patchable on `openlens.lens_editor`:
  - `setup_logging`
  - `verbose_requested`
  - `LensManager`
  - `sys`

## 3. Raises
- **`SystemExit(mcp_main())`** when `sys.argv[1] == "mcp"`.
  - `.code` is whatever `openlens.mcp_server.main()` returns.
  - `mcp_server` is imported lazily, inside this branch only.
  - Example: `sys.argv = ["openlens-cli", "mcp"]` with a fake `mcp_server.main` returning `3` gives `SystemExit` with `.code == 3`.
- **No try/except in `main`.** Any exception from these propagates unchanged:
  - `input()`: e.g. `EOFError`, or `StopIteration` from a fake input iterator that runs out.
  - `LensManager()`.
  - Any manager method.
  - `setup_logging`.

## 4. Returns and guarantees
- Returns `None`, only after choice `"6"` (it `break`s the loop).
- Creates exactly one `LensManager()` with no arguments, so it uses the default DB path. The real class may seed example data into the user's app DB, so always patch it.
- Stdout for an immediate `"6"` (with `input` patched, so no prompt text is printed):

```
============================================================
   openlens - Optical Lens Creation & Modification Tool
============================================================

--- Menu ---
1. Create new lens
2. List all lenses
3. View lens details
4. Modify lens
5. Delete lens
6. Exit

Goodbye!
```

- Formatting details:
  - Banner lines are `"=" * 60`, i.e. 60 `=` characters.
  - The title line starts with 3 spaces.
  - `"\n--- Menu ---"` and `"\nGoodbye!"` each produce a blank line first.

## 5. Side effects
- Calls `setup_logging(verbose=verbose_requested())` once. This happens only on the menu path, never on the mcp path.
- Prints to stdout.
- Any persistence happens inside `LensManager` methods, not in `main` itself.

## 6. Branches and edge cases
- **mcp path:** `argv[1] == "mcp"`. Extra args are allowed (`["x", "mcp", "--foo"]` still takes this path).
  - Raises `SystemExit` before `setup_logging` and before `LensManager` is constructed.
  - `"MCP"`, `" mcp"` and `"mcp2"` do NOT match; they run the menu.
- **`argv` length ≤ 1** (e.g. `["prog"]`): menu path.
- **Menu choices** (after strip):

  | Choice | Action |
  |---|---|
  | `"1"` | `manager.create_lens()` |
  | `"2"` | `manager.list_lenses()` |
  | `"3"` | `manager.view_lens()` |
  | `"4"` | `manager.modify_lens()` |
  | `"5"` | `manager.delete_lens()` |
  | `"6"` | prints `"\nGoodbye!"`, then returns |
  | anything else | prints `Invalid option. Please try again.` and loops |

  - Each method is called with no args, and its return value is ignored.
  - "Anything else" includes `""`, `"7"`, `"01"`, `"1.0"`, `"exit"`.
- **Whitespace is stripped:** `"  2  "` calls `list_lenses`, and `" 6\n"` exits.
- **The menu reprints before every prompt.** Inputs `["9", "6"]` print the menu block twice, `Invalid option. Please try again.` once, then `Goodbye!`.
- **One call per choice:** inputs `["1", "1", "6"]` call `create_lens` twice and nothing else.

## 7. Rules stated in comments or docstrings
- The docstring states that `mcp` as the first argument starts the MCP stdio server instead of the menu (`openlens-cli mcp`).

## 8. Testing hints

Patch everything; never use the real `LensManager`, logging setup, or stdin:

```python
import sys, types, builtins, pytest
import openlens.lens_editor as le

class FakeMgr:
    instances = []
    def __init__(self, *a, **k):
        self.calls = []; self.args = (a, k); FakeMgr.instances.append(self)
    def create_lens(self): self.calls.append("create")
    def list_lenses(self): self.calls.append("list")
    def view_lens(self): self.calls.append("view")
    def modify_lens(self): self.calls.append("modify")
    def delete_lens(self): self.calls.append("delete")

def run(monkeypatch, answers, argv=("prog",)):
    FakeMgr.instances = []
    monkeypatch.setattr(sys, "argv", list(argv))
    monkeypatch.setattr(le, "LensManager", FakeMgr)
    monkeypatch.setattr(le, "setup_logging", lambda **k: None)
    monkeypatch.setattr(le, "verbose_requested", lambda *a: False)
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(it))
    return le.main()
```

What to assert:
- **Dispatch order:** `["1","2","3","4","5","6"]` gives `calls == ["create","list","view","modify","delete"]`.
- **Constructor:** exactly one instance, with `args == ((), {})`.
- **Return value:** `main()` returns `None`.
- **Output:** use `capsys`. Assert on exact lines (e.g. `"Goodbye!"`, `"Invalid option. Please try again."`, the banner title) or on counts, such as `out.count("--- Menu ---") == number of prompts`.
- **Prompt text:** record the `prompt` argument in the fake input and assert it equals `"\nSelect option: "`.
- **Logging:** capture `setup_logging` kwargs and assert `{"verbose": <value from fake verbose_requested>}`. Note that `verbose_requested` is called with no args.
- **mcp path:**
  - Install a fake module with `monkeypatch.setitem(sys.modules, "openlens.mcp_server", m)`, where `m = types.ModuleType("openlens.mcp_server"); m.main = lambda: 7`.
  - Assert `pytest.raises(SystemExit)` with `.code == 7`.
  - Assert `FakeMgr.instances == []` and that `setup_logging` was never called.
  - Do not rely on the real `mcp_server` (it may need the optional `mcp` package; that module is not shown).

Pitfalls:
- **Always end the input sequence with `"6"`.** Otherwise `next(it)` raises `StopIteration` out of `main`; you can test that deliberately with `pytest.raises(StopIteration)`.
- **Banner line length:** it is exactly 60 `=` characters, so don't miscount.
- **Patch target:** patch `le.LensManager`, not `openlens.lens_editor.LensManager` via a fresh import alias. It is looked up as a module global at call time.

# Notes: `openlens.lens_editor.LensManager.delete_lens(self) -> None`

## 1. Purpose
- Interactive CLI step. It lists the lenses, asks for a 1-based number, and asks for confirmation.
- It deletes the database row first, through `_delete_row`. Only if that succeeds does it remove the lens from `self.lenses` and call `save_lenses()`.
- It returns `None` on every path. All outcomes show up only as printed text and state changes.

## 2. Inputs
- No parameters apart from `self`. It uses `self.lenses` (a list of `Lens`) and `self.storage_file`, which `_delete_row` uses.
- It reads `builtins.input` up to twice:
  1. `input("\nSelect lens number to delete: ")` is passed to `int(...)`.
     - `int` itself accepts surrounding whitespace, so `" 2 "` works.
     - `"2.0"`, `"abc"` and `""` raise `ValueError`.
  2. `input(f"Delete '{lens.name}'? (yes/no): ")` is passed through `.strip().lower()`.
     - Only the exact result `"yes"` confirms. `"YES"` and `"  Yes "` confirm.
     - `"y"`, `"no"` and `""` cancel.
- Index check (`get_lens_by_index`): valid only when `1 <= idx <= len(self.lenses)`. Otherwise it returns `None`. Values `0` and negatives are invalid, so there is no Python negative indexing.

## 3. Raises
- In practice nothing escapes for normal string input:
  - `ValueError` and `IndexError` inside the `try` are caught and print `Invalid input.`.
  - `_delete_row` catches every `Exception`.
  - `save_lenses` catches every `Exception`.
- `EOFError` and `KeyboardInterrupt` from `input` are **not** caught and propagate. For example, a mocked input that raises `StopIteration` will propagate.
- `list_lenses()` runs before the `try`. An error from `lens.calculate_focal_length()` would propagate.

## 4. Returns / state
- Always returns `None`.
- On success:
  - `self.lenses.pop(idx - 1)` removes that exact object; the remaining order is preserved.
  - `self.save_lenses()` is called once, after the pop.
- On every other path, `self.lenses` is unchanged and `save_lenses` is **not** called.

## 5. Side effects
- Prints to stdout.
- Database deletion: `_delete_row` imports `LensStorage` from `openlens.gui.storage` at call time and calls `LensStorage(self.storage_file).delete_item(lens.id)`. That code is in another module and its behaviour is not visible here.
- Unexpected errors are logged with `logger.error` (logger `openlens.lens_editor`).

## 6. Branches (stdout lines, in order; prompts are not printed when `input` is mocked)
| Case | Inputs | Output after the listing | State |
|---|---|---|---|
| Empty list | (no input called) | `"\nNo lenses to delete."` only; no listing at all | unchanged |
| Non-integer | `"abc"` | `Invalid input.` | unchanged |
| Out of range | `"0"`, `"-1"`, `"3"` with 2 lenses | `Invalid selection.` (confirmation prompt never asked) | unchanged |
| Cancelled | `"1"`, `"no"` | `Deletion cancelled.` (`_delete_row` not called) | unchanged |
| `_delete_row` returns False | `"1"`, `"yes"` | (any message from `_delete_row`), then `Lens kept in the library.` | unchanged, no save |
| Success | `"1"`, `"yes"` | `✓ Lens deleted successfully!` | lens removed, saved once |

How `_delete_row` returns False:
- **`STORAGE_AVAILABLE` is False:** it prints nothing extra.
- **`LensInUseError` is raised:** it prints `f"Cannot delete '{lens.name}': {e}"`. The text of `e` comes from `openlens.database` and is not shown here.
- **Any other exception `e`:** it prints `f"Could not delete '{lens.name}' from the database: {e}"`. Example: `RuntimeError("boom")` prints `Could not delete 'A' from the database: boom`.

Listing printed first (from `list_lenses`, for a non-empty list):
- Header: `f"\n=== All Optical Lenses ({n}) ==="`
- One line per lens: `f"{i}. {name} - {material} ({lens_type}) - f={focal:.2f}mm or Undefined"`

## 7. Rules from comments
- Popping the list and re-saving does NOT delete anything. `save_lenses` defaults to `reconcile=False` and only inserts or replaces rows. That is why the explicit row delete happens first.
- The lens is dropped from memory only once the row is really gone.
- Success is never printed when the database refused the deletion.

## 8. Testing hints
Build the manager without touching any real database or the storage layer:
```python
import openlens.lens_editor as le
from openlens.lens import Lens
m = le.LensManager.__new__(le.LensManager)
m.storage_file = str(tmp_path / "t.db"); m.lenses = [Lens(name="A"), Lens(name="B")]
saves = []; monkeypatch.setattr(m, "save_lenses", lambda: saves.append(1) or True)
monkeypatch.setattr(m, "_delete_row", lambda lens: True)   # or False
answers = iter(["2", "yes"]); prompts = []
monkeypatch.setattr("builtins.input", lambda p="": (prompts.append(p), next(answers))[1])
m.delete_lens(); out = capsys.readouterr().out
```

What to assert:
- `prompts == ["\nSelect lens number to delete: ", "Delete 'B'? (yes/no): "]`
- `[l.name for l in m.lenses] == ["A"]`
- `saves == [1]`
- `"✓ Lens deleted successfully!" in out`

To exercise the real `_delete_row`:
- Do not mock `_delete_row`. Instead `monkeypatch.setattr(le, "STORAGE_AVAILABLE", False)`. The global is read at call time, and the result is `Lens kept in the library.`.
- Alternatively, patch `openlens.gui.storage.LensStorage` with a fake class whose `delete_item` records the `lens.id` argument or raises. The import happens inside the function, so patching that module attribute works.
- `lens.id` is assumed to be set by `Lens`; it is not visible here.

Spy on `_delete_row` to check it is **not** called when the user cancels or picks an invalid index.

Pitfalls:
- Use substring checks (`in out`), not exact equality on all of stdout. The listing line depends on the `lens_type` string and the focal length, which are defined elsewhere.
- `"No lenses to delete."` is preceded by a blank line, because the printed string starts with `\n`.
- If the mocked `input` runs out of answers, `StopIteration` propagates and the test errors. Supply exactly the answers that path consumes: zero, one or two.
- Do not call the real `LensManager(...)` with the default path; it may seed or read the user's application database.

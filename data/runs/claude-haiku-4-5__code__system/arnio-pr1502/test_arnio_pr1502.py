import pytest
import pandas as pd
import numpy as np
import sys
import subprocess
from pathlib import Path
import arnio as ar


class TestAutoCleanBasic:
    """Test basic auto_clean functionality with safe mode."""

    @pytest.fixture
    def messy_frame(self, tmp_path):
        """Create a messy frame with whitespace and type issues."""
        csv_file = tmp_path / "messy.csv"
        data = {
            "name": ["  Alice  ", "Bob", "  Charlie  "],
            "age": ["25", "30", "35"],
            "value": [100.5, 200.0, 100.5],
        }
        pd.DataFrame(data).to_csv(csv_file, index=False)
        return ar.read_csv(csv_file)

    def test_safe_mode_default(self, messy_frame):
        """Test that auto_clean with default mode returns cleaned frame."""
        cleaned = ar.auto_clean(messy_frame)
        assert isinstance(cleaned, ar.ArFrame)
        assert len(cleaned) == len(messy_frame)

    def test_safe_mode_explicit(self, messy_frame):
        """Test safe mode cleaning."""
        cleaned = ar.auto_clean(messy_frame, mode="safe")
        assert isinstance(cleaned, ar.ArFrame)
        # Whitespace should be stripped
        assert cleaned["name"].iloc[0] == "Alice"
        assert cleaned["name"].iloc[2] == "Charlie"

    def test_safe_mode_preserves_rows(self, messy_frame):
        """Test that safe mode doesn't remove rows."""
        original_len = len(messy_frame)
        cleaned = ar.auto_clean(messy_frame, mode="safe")
        assert len(cleaned) == original_len


class TestAutoCleanReports:
    """Test auto_clean with reporting and dry_run modes."""

    @pytest.fixture
    def messy_frame(self, tmp_path):
        """Create a messy frame for testing."""
        csv_file = tmp_path / "messy.csv"
        pd.DataFrame({
            "name": ["  Alice  ", "Bob", "  Charlie  "],
            "age": ["25", "30", "35"],
        }).to_csv(csv_file, index=False)
        return ar.read_csv(csv_file)

    def test_dry_run_returns_report(self, messy_frame):
        """Test that dry_run=True returns report without modification."""
        result = ar.auto_clean(messy_frame, mode="safe", dry_run=True)
        assert hasattr(result, 'suggestions')
        # Original frame unchanged
        assert "  Alice  " in messy_frame["name"].values

    def test_return_report_with_frame(self, messy_frame):
        """Test return_report=True returns both frame and report."""
        cleaned, report = ar.auto_clean(messy_frame, return_report=True)
        assert isinstance(cleaned, ar.ArFrame)
        assert hasattr(report, 'suggestions')
        assert len(cleaned) == len(messy_frame)

    def test_dry_run_no_frame_modification(self, messy_frame):
        """Test that dry_run doesn't modify the frame."""
        original_name = messy_frame["name"].iloc[0]
        ar.auto_clean(messy_frame, dry_run=True)
        assert messy_frame["name"].iloc[0] == original_name


class TestAutoCleanExplain:
    """Test auto_clean with explain parameter."""

    @pytest.fixture
    def messy_frame(self, tmp_path):
        """Create a messy frame for testing."""
        csv_file = tmp_path / "messy.csv"
        pd.DataFrame({
            "name": ["  Alice  ", "Bob"],
            "age": ["25", "30"],
        }).to_csv(csv_file, index=False)
        return ar.read_csv(csv_file)

    def test_explain_returns_explanation(self, messy_frame):
        """Test that explain=True returns CleanExplanation."""
        cleaned, explanation = ar.auto_clean(messy_frame, explain=True)
        assert isinstance(cleaned, ar.ArFrame)
        assert hasattr(explanation, 'steps')
        assert hasattr(explanation, 'rows_before')
        assert hasattr(explanation, 'rows_after')

    def test_explain_with_report(self, messy_frame):
        """Test explain=True with return_report=True."""
        cleaned, report, explanation = ar.auto_clean(
            messy_frame, 
            explain=True, 
            return_report=True
        )
        assert isinstance(cleaned, ar.ArFrame)
        assert hasattr(report, 'suggestions')
        assert hasattr(explanation, 'steps')

    def test_explain_tracks_rows(self, messy_frame):
        """Test that explanation tracks row counts."""
        cleaned, explanation = ar.auto_clean(messy_frame, explain=True)
        assert explanation.rows_before >= explanation.rows_after
        assert explanation.rows_removed == (
            explanation.rows_before - explanation.rows_after
        )


class TestAutoCleanStrictMode:
    """Test strict mode and type casting."""

    @pytest.fixture
    def messy_frame(self, tmp_path):
        """Create a frame with type issues."""
        csv_file = tmp_path / "messy.csv"
        pd.DataFrame({
            "name": ["Alice", "Bob"],
            "age": ["25", "30"],
            "score": ["95.5", "87.3"],
        }).to_csv(csv_file, index=False)
        return ar.read_csv(csv_file)

    def test_strict_requires_confirmed_casts(self, messy_frame):
        """Test that strict mode requires confirmed_casts for type casting."""
        with pytest.raises(ValueError, match="allow_lossy_casts"):
            ar.auto_clean(messy_frame, mode="strict")

    def test_strict_dry_run_shows_suggestions(self, messy_frame):
        """Test that strict mode dry_run shows cast suggestions."""
        report = ar.auto_clean(messy_frame, mode="strict", dry_run=True)
        assert hasattr(report, 'suggestions')

    def test_confirmed_casts_workflow(self, messy_frame):
        """Test workflow: dry_run -> inspect -> apply with confirmed_casts."""
        # Step 1: dry run to inspect
        report = ar.auto_clean(messy_frame, mode="strict", dry_run=True)
        
        # Step 2: extract cast suggestions
        casts = None
        for suggestion in report.suggestions:
            if hasattr(suggestion, 'step') and suggestion.step == 'cast_types':
                casts = suggestion.kwargs
                break
        
        # Step 3: apply with confirmed casts if found
        if casts:
            cleaned = ar.auto_clean(
                messy_frame,
                mode="strict",
                allow_lossy_casts=True,
                confirmed_casts=casts
            )
            assert isinstance(cleaned, ar.ArFrame)

    def test_mismatched_confirmed_casts_fails(self, messy_frame):
        """Test that mismatched confirmed_casts raises ValueError."""
        wrong_casts = {"age": "object", "score": "object"}
        with pytest.raises(ValueError, match="confirmed_casts must match"):
            ar.auto_clean(
                messy_frame,
                mode="strict",
                allow_lossy_casts=True,
                confirmed_casts=wrong_casts
            )


class TestAutoCleanErrors:
    """Test error handling and parameter validation."""

    @pytest.fixture
    def messy_frame(self, tmp_path):
        """Create a messy frame for testing."""
        csv_file = tmp_path / "messy.csv"
        pd.DataFrame({"col": ["a", "b"]}).to_csv(csv_file, index=False)
        return ar.read_csv(csv_file)

    def test_invalid_mode_raises_valueerror(self, messy_frame):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            ar.auto_clean(messy_frame, mode="invalid")

    def test_non_bool_return_report_raises_typeerror(self, messy_frame):
        """Test that non-bool return_report raises TypeError."""
        with pytest.raises(TypeError, match="return_report must be a bool"):
            ar.auto_clean(messy_frame, return_report="true")

    def test_non_bool_dry_run_raises_typeerror(self, messy_frame):
        """Test that non-bool dry_run raises TypeError."""
        with pytest.raises(TypeError, match="dry_run must be a bool"):
            ar.auto_clean(messy_frame, dry_run=1)

    def test_non_bool_explain_raises_typeerror(self, messy_frame):
        """Test that non-bool explain raises TypeError."""
        with pytest.raises(TypeError, match="explain must be a bool"):
            ar.auto_clean(messy_frame, explain="yes")

    def test_dry_run_with_explain_raises_valueerror(self, messy_frame):
        """Test that dry_run=True with explain=True raises ValueError."""
        with pytest.raises(ValueError, match="explain=True cannot be used with dry_run=True"):
            ar.auto_clean(messy_frame, dry_run=True, explain=True)

    def test_dry_run_with_return_report_raises_valueerror(self, messy_frame):
        """Test that dry_run=True with return_report=True raises ValueError."""
        with pytest.raises(ValueError, match="return_report=True cannot be used with dry_run=True"):
            ar.auto_clean(messy_frame, dry_run=True, return_report=True)


class TestAutoCleanEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_frame(self, tmp_path):
        """Test auto_clean on empty frame."""
        csv_file = tmp_path / "empty.csv"
        pd.DataFrame({"col": []}).to_csv(csv_file, index=False)
        frame = ar.read_csv(csv_file)
        
        cleaned = ar.auto_clean(frame)
        assert len(cleaned) == 0
        assert isinstance(cleaned, ar.ArFrame)

    def test_single_row_frame(self, tmp_path):
        """Test auto_clean on single-row frame."""
        csv_file = tmp_path / "single.csv"
        pd.DataFrame({"name": ["  Alice  "]}).to_csv(csv_file, index=False)
        frame = ar.read_csv(csv_file)
        
        cleaned = ar.auto_clean(frame)
        assert len(cleaned) == 1
        assert cleaned["name"].iloc[0] == "Alice"

    def test_already_clean_data(self, tmp_path):
        """Test auto_clean on already clean data."""
        csv_file = tmp_path / "clean.csv"
        pd.DataFrame({
            "name": ["Alice", "Bob"],
            "age": [25, 30],
        }).to_csv(csv_file, index=False)
        frame = ar.read_csv(csv_file)
        original_len = len(frame)
        
        cleaned = ar.auto_clean(frame, mode="safe")
        assert len(cleaned) == original_len


class TestAutoCleanPandasAccessor:
    """Test auto_clean via pandas accessor."""

    def test_accessor_basic(self, tmp_path):
        """Test using .arnio accessor."""
        csv_file = tmp_path / "data.csv"
        pd.DataFrame({"name": ["  Alice  "], "age": ["25"]}).to_csv(csv_file, index=False)
        df = pd.read_csv(csv_file)
        
        cleaned = df.arnio.clean()
        
        assert isinstance(cleaned, pd.DataFrame)
        assert len(cleaned) == len(df)
        assert cleaned["name"].iloc[0] == "Alice"

    def test_accessor_with_mode(self, tmp_path):
        """Test accessor with mode parameter."""
        csv_file = tmp_path / "data.csv"
        pd.DataFrame({"col": ["  value  "]}).to_csv(csv_file, index=False)
        df = pd.read_csv(csv_file)
        
        cleaned = df.arnio.clean(mode="safe")
        
        assert isinstance(cleaned, pd.DataFrame)
        assert cleaned["col"].iloc[0] == "value"


class TestAutoCleanCLI:
    """Test auto_clean via command-line interface."""

    def test_cli_cleaning(self, tmp_path):
        """Test CLI invocation for cleaning."""
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"
        
        pd.DataFrame({
            "name": ["  Alice  ", "  Bob  "],
            "age": ["25", "30"],
        }).to_csv(input_file, index=False)
        
        result = subprocess.run(
            [
                sys.executable, "-m", "arnio.cli",
                "clean", str(input_file), "-o", str(output_file)
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert output_file.exists()
        
        # Verify cleaned output
        cleaned = pd.read_csv(output_file)
        assert cleaned["name"].iloc[0] == "Alice"
        assert cleaned["name"].iloc[1] == "Bob"

    def test_cli_dry_run(self, tmp_path):
        """Test CLI with dry-run mode."""
        input_file = tmp_path / "input.csv"
        pd.DataFrame({"name": ["  Alice  "]}).to_csv(input_file, index=False)
        
        result = subprocess.run(
            [
                sys.executable, "-m", "arnio.cli",
                "clean", str(input_file), "--dry-run"
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0

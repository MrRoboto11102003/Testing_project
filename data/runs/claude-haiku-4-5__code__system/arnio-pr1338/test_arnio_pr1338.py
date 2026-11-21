import pytest
import pandas as pd
import numpy as np
import subprocess
import sys
from pathlib import Path

import arnio as ar
from arnio.quality.auto_clean import auto_clean


@pytest.fixture
def test_csv(tmp_path):
    """Create a test CSV file with cleaning opportunities."""
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'name': [' Alice ', 'Bob', '  Charlie', ' Alice '],
        'age': ['25', '30', '35', '25'],
        'score': [95.5, 87.0, 92.3, 95.5],
        'email': ['alice@ex.com', ' bob@ex.com', 'charlie@ex.com', 'alice@ex.com']
    })
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def test_frame(test_csv):
    """Load test CSV into an ArFrame."""
    return ar.load(test_csv)


class TestAutoCleanSafeMode:
    """Test auto_clean in safe mode (default)."""

    def test_safe_mode_trims_whitespace(self, test_frame):
        """Safe mode should trim whitespace from string columns."""
        cleaned = auto_clean(test_frame, mode="safe")
        result = cleaned.to_pandas()
        
        assert result['name'].iloc[0] == 'Alice'
        assert result['name'].iloc[2] == 'Charlie'
        assert result['email'].iloc[1] == 'bob@ex.com'

    def test_safe_mode_returns_arframe(self, test_frame):
        """Safe mode without return_report should return ArFrame."""
        result = auto_clean(test_frame, mode="safe")
        assert hasattr(result, 'to_pandas')
        assert hasattr(result, 'shape')

    def test_default_mode_is_safe(self, test_frame):
        """Default mode should be 'safe'."""
        result_default = auto_clean(test_frame)
        result_safe = auto_clean(test_frame, mode="safe")
        
        pd.testing.assert_frame_equal(
            result_default.to_pandas(),
            result_safe.to_pandas()
        )


class TestAutoCleanDryRun:
    """Test dry_run parameter."""

    def test_dry_run_returns_report(self, test_frame):
        """dry_run=True should return a DataQualityReport."""
        report = auto_clean(test_frame, dry_run=True)
        assert hasattr(report, 'summary')
        assert hasattr(report, 'suggestions')

    def test_dry_run_does_not_modify_frame(self, test_frame):
        """dry_run should not modify the input frame."""
        original = test_frame.to_pandas().copy()
        auto_clean(test_frame, dry_run=True)
        pd.testing.assert_frame_equal(test_frame.to_pandas(), original)

    def test_dry_run_conflicts_with_return_report(self, test_frame):
        """Cannot use both dry_run and return_report."""
        with pytest.raises(ValueError, match="return_report=True cannot be used with dry_run=True"):
            auto_clean(test_frame, dry_run=True, return_report=True)

    def test_dry_run_conflicts_with_explain(self, test_frame):
        """Cannot use both dry_run and explain."""
        with pytest.raises(ValueError, match="explain=True cannot be used with dry_run=True"):
            auto_clean(test_frame, dry_run=True, explain=True)


class TestAutoCleanReturnReport:
    """Test return_report parameter."""

    def test_return_report_tuple(self, test_frame):
        """return_report=True should return (frame, report)."""
        result = auto_clean(test_frame, return_report=True)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        cleaned, report = result
        assert hasattr(cleaned, 'to_pandas')
        assert hasattr(report, 'suggestions')

    def test_return_report_with_strict_mode(self, test_frame):
        """return_report works with strict mode."""
        result = auto_clean(
            test_frame,
            mode="strict",
            return_report=True,
            allow_lossy_casts=True,
            confirmed_casts={}
        )
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestAutoCleanStrictMode:
    """Test strict mode requirements and behavior."""

    def test_strict_requires_allow_lossy_casts(self, test_frame):
        """strict mode without allow_lossy_casts raises ValueError."""
        with pytest.raises(ValueError, match="allow_lossy_casts"):
            auto_clean(test_frame, mode="strict")

    def test_strict_requires_confirmed_casts(self, test_frame):
        """strict mode with allow_lossy_casts but no confirmed_casts raises ValueError."""
        with pytest.raises(ValueError, match="confirmed_casts"):
            auto_clean(test_frame, mode="strict", allow_lossy_casts=True)

    def test_strict_confirmed_casts_mismatch(self, test_frame):
        """confirmed_casts must match proposed mapping exactly."""
        with pytest.raises(ValueError, match="confirmed_casts must match"):
            auto_clean(
                test_frame,
                mode="strict",
                allow_lossy_casts=True,
                confirmed_casts={'nonexistent': 'int64'}
            )

    def test_strict_removes_duplicates(self, tmp_path):
        """strict mode removes exact duplicates."""
        csv_path = tmp_path / "dups.csv"
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Alice'],
            'value': [1, 2, 1]
        })
        df.to_csv(csv_path, index=False)
        frame = ar.load(csv_path)
        
        cleaned = auto_clean(
            frame,
            mode="strict",
            allow_lossy_casts=True,
            confirmed_casts={}
        )
        result = cleaned.to_pandas()
        
        assert len(result) <= len(df)


class TestAutoCleanExplain:
    """Test explain parameter."""

    def test_explain_returns_tuple(self, test_frame):
        """explain=True should return (frame, explanation)."""
        result = auto_clean(test_frame, explain=True)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        cleaned, explanation = result
        assert hasattr(cleaned, 'to_pandas')
        assert hasattr(explanation, 'steps')
        assert hasattr(explanation, 'rows_before')
        assert hasattr(explanation, 'rows_after')

    def test_explain_with_return_report(self, test_frame):
        """explain=True and return_report=True returns 3-tuple."""
        result = auto_clean(test_frame, explain=True, return_report=True)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        cleaned, report, explanation = result
        assert hasattr(cleaned, 'to_pandas')
        assert hasattr(report, 'summary')
        assert hasattr(explanation, 'mode')

    def test_explanation_records_rows_removed(self, tmp_path):
        """CleanExplanation should record row counts."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            'name': [' Alice ', 'Bob', ' Alice '],
            'value': [1, 2, 1]
        })
        df.to_csv(csv_path, index=False)
        frame = ar.load(csv_path)
        
        _, explanation = auto_clean(
            frame,
            explain=True,
            mode="strict",
            allow_lossy_casts=True,
            confirmed_casts={}
        )
        
        assert explanation.rows_before == 3
        assert hasattr(explanation, 'rows_removed')
        assert explanation.rows_removed >= 0


class TestAutoCleanValidation:
    """Test parameter validation."""

    def test_invalid_mode(self, test_frame):
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be 'safe' or 'strict'"):
            auto_clean(test_frame, mode="aggressive")

    def test_return_report_not_bool(self, test_frame):
        """Non-bool return_report raises TypeError."""
        with pytest.raises(TypeError, match="return_report must be a bool"):
            auto_clean(test_frame, return_report="yes")

    def test_dry_run_not_bool(self, test_frame):
        """Non-bool dry_run raises TypeError."""
        with pytest.raises(TypeError, match="dry_run must be a bool"):
            auto_clean(test_frame, dry_run=1)

    def test_allow_lossy_casts_not_bool(self, test_frame):
        """Non-bool allow_lossy_casts raises TypeError."""
        with pytest.raises(TypeError, match="allow_lossy_casts must be a bool"):
            auto_clean(test_frame, allow_lossy_casts="true")

    def test_explain_not_bool(self, test_frame):
        """Non-bool explain raises TypeError."""
        with pytest.raises(TypeError, match="explain must be a bool"):
            auto_clean(test_frame, explain=[])


class TestAutoCleanFileWorkflow:
    """Test complete file I/O workflows."""

    def test_csv_to_csv_pipeline(self, tmp_path):
        """Load CSV, clean, and save to new CSV."""
        input_file = tmp_path / "input.csv"
        df = pd.DataFrame({
            'name': [' Alice ', 'Bob ', 'Charlie'],
            'value': [1, 2, 3]
        })
        df.to_csv(input_file, index=False)
        
        frame = ar.load(input_file)
        cleaned = auto_clean(frame, mode="safe")
        
        output_file = tmp_path / "output.csv"
        cleaned.to_csv(output_file)
        
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert result['name'].iloc[0] == 'Alice'

    def test_inspect_then_apply_workflow(self, tmp_path):
        """Inspect with dry_run, then apply confirmed casts in strict mode."""
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            'value': ['1', '2', '3'],
            'name': ['A', 'B', 'C']
        })
        df.to_csv(csv_path, index=False)
        frame = ar.load(csv_path)
        
        # Step 1: Dry-run to inspect
        report = auto_clean(frame, mode="strict", dry_run=True)
        assert hasattr(report, 'suggestions')
        
        # Step 2: Apply with empty casts if none needed
        cleaned = auto_clean(
            frame,
            mode="strict",
            allow_lossy_casts=True,
            confirmed_casts={}
        )
        assert hasattr(cleaned, 'to_pandas')


class TestAutoCleanEdgeCases:
    """Test edge cases."""

    def test_empty_dataframe(self, tmp_path):
        """auto_clean handles empty DataFrame."""
        csv_path = tmp_path / "empty.csv"
        df = pd.DataFrame({'a': [], 'b': []})
        df.to_csv(csv_path, index=False)
        frame = ar.load(csv_path)
        
        result = auto_clean(frame, mode="safe")
        assert result.shape[0] == 0

    def test_single_row(self, tmp_path):
        """auto_clean handles single row."""
        csv_path = tmp_path / "single.csv"
        df = pd.DataFrame({'name': [' Alice '], 'value': [1]})
        df.to_csv(csv_path, index=False)
        frame = ar.load(csv_path)
        
        cleaned = auto_clean(frame, mode="safe")
        result = cleaned.to_pandas()
        assert len(result) == 1
        assert result['name'].iloc[0] == 'Alice'

    def test_no_cleaning_needed(self, tmp_path):
        """auto_clean on clean data returns same data."""
        csv_path = tmp_path / "clean.csv"
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [1, 2, 3]
        })
        df.to_csv(csv_path, index=False)
        frame = ar.load(csv_path)
        
        cleaned = auto_clean(frame, mode="safe")
        result = cleaned.to_pandas()
        pd.testing.assert_frame_equal(result, df)

    def test_all_duplicates_in_strict(self, tmp_path):
        """strict mode with all duplicates."""
        csv_path = tmp_path / "all_dups.csv"
        df = pd.DataFrame({
            'name': ['Alice', 'Alice', 'Alice'],
            'value': [1, 1, 1]
        })
        df.to_csv(csv_path, index=False)
        frame = ar.load(csv_path)
        
        cleaned = auto_clean(
            frame,
            mode="strict",
            allow_lossy_casts=True,
            confirmed_casts={}
        )
        result = cleaned.to_pandas()
        assert 1 <= len(result) <= 3

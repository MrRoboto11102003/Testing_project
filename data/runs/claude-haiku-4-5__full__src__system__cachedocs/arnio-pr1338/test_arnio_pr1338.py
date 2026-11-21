"""
System (end-to-end) tests for arnio.quality.auto_clean.

These tests drive complete user workflows through the public API:
- Reading messy CSV files from disk
- Cleaning and validating data
- Writing results and converting to pandas
- Using the pandas accessor
- Error handling and edge cases
"""

import pytest
import pandas as pd
import numpy as np
import arnio as ar
from pathlib import Path


class TestAutoCleanBasicUsage:
    """Test basic auto_clean workflows matching typical user patterns."""

    def test_safe_mode_strips_whitespace(self, tmp_path):
        """Safe mode (default) trims whitespace from string columns."""
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text(
            "customer_name,city,revenue\n"
            " Alice , Paris ,1000\n"
            " Bob  , London ,2000\n"
        )

        frame = ar.read_csv(csv_file)
        clean = ar.auto_clean(frame)

        df = ar.to_pandas(clean)
        assert df["customer_name"].tolist() == ["Alice", "Bob"]
        assert df["city"].tolist() == ["Paris", "London"]
        assert len(df) == 2  # Safe mode preserves all rows

    def test_safe_mode_no_data_loss(self, tmp_path):
        """Safe mode never drops rows."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,name\n"
            "1, Alice\n"
            "1, Alice\n"  # duplicate
            "2, Bob\n"
        )

        frame = ar.read_csv(csv_file)
        rows_before = frame.shape[0]
        clean = ar.auto_clean(frame, mode="safe")
        rows_after = clean.shape[0]

        assert rows_before == rows_after == 3

    def test_strict_mode_deduplicates_rows(self, tmp_path):
        """Strict mode removes exact duplicate rows."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,name\n"
            "1, Alice\n"
            "2, Bob\n"
            "2, Bob\n"
        )

        frame = ar.read_csv(csv_file)
        clean = ar.auto_clean(
            frame,
            mode="strict",
            allow_lossy_casts=True,
            confirmed_casts={}
        )

        df = ar.to_pandas(clean)
        assert len(df) == 2  # One duplicate removed


class TestAutoCleanDryRun:
    """Test dry_run mode for previewing changes without modifying data."""

    def test_dry_run_returns_report_only(self, tmp_path):
        """dry_run=True returns DataQualityReport without modifying frame."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "name,value\n"
            " Alice ,100\n"
            " Bob  ,200\n"
        )

        frame = ar.read_csv(csv_file)
        report = ar.auto_clean(frame, dry_run=True)

        assert isinstance(report, ar.DataQualityReport)
        assert report.row_count == 2

    def test_dry_run_strict_previews_changes(self, tmp_path):
        """dry_run with strict mode shows proposed cleaning steps."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "name,value\n"
            " Alice ,100\n"
            " Bob  ,200\n"
            " Bob  ,200\n"
        )

        frame = ar.read_csv(csv_file)
        report = ar.auto_clean(frame, mode="strict", dry_run=True)

        assert isinstance(report, ar.DataQualityReport)
        assert hasattr(report, 'suggestions')
        assert len(report.suggestions) > 0

    def test_dry_run_does_not_mutate_frame(self, tmp_path):
        """dry_run should not modify the input frame."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "name\n"
            " Alice\n"
            " Bob \n"
        )

        frame = ar.read_csv(csv_file)
        df_before = ar.to_pandas(frame)

        ar.auto_clean(frame, dry_run=True, mode="strict")

        # Original unchanged
        df_after = ar.to_pandas(frame)
        pd.testing.assert_frame_equal(df_before, df_after)


class TestAutoCleanReturnTypes:
    """Test different return type combinations based on flags."""

    def test_default_returns_frame(self, tmp_path):
        """Default call returns just ArFrame."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name\nAlice\n")

        frame = ar.read_csv(csv_file)
        result = ar.auto_clean(frame)

        assert isinstance(result, ar.ArFrame)

    def test_return_report_flag_returns_tuple(self, tmp_path):
        """return_report=True returns (ArFrame, DataQualityReport)."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,value\n Alice ,100\n")

        frame = ar.read_csv(csv_file)
        result = ar.auto_clean(frame, return_report=True)

        assert isinstance(result, tuple)
        assert len(result) == 2
        clean_frame, report = result
        assert isinstance(clean_frame, ar.ArFrame)
        assert isinstance(report, ar.DataQualityReport)

    def test_explain_flag_returns_explanation(self, tmp_path):
        """explain=True returns (ArFrame, CleanExplanation)."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "name,city\n"
            " Alice , Paris\n"
            " Bob  , London\n"
        )

        frame = ar.read_csv(csv_file)
        result = ar.auto_clean(frame, explain=True)

        assert isinstance(result, tuple)
        assert len(result) == 2
        clean_frame, explanation = result
        assert isinstance(clean_frame, ar.ArFrame)
        assert hasattr(explanation, 'mode')
        assert hasattr(explanation, 'steps')

    def test_both_return_report_and_explain(self, tmp_path):
        """Both flags return 3-tuple (frame, report, explanation)."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name\nAlice\n")

        frame = ar.read_csv(csv_file)
        result = ar.auto_clean(frame, return_report=True, explain=True)

        assert isinstance(result, tuple)
        assert len(result) == 3
        clean_frame, report, explanation = result
        assert isinstance(clean_frame, ar.ArFrame)
        assert isinstance(report, ar.DataQualityReport)
        assert hasattr(explanation, 'mode')


class TestAutoCleanStrictModeValidation:
    """Test strict mode's cast confirmation requirements."""

    def test_strict_requires_allow_lossy_casts(self, tmp_path):
        """Strict mode without allow_lossy_casts=True should raise."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1, Alice\n")

        frame = ar.read_csv(csv_file)

        with pytest.raises(ValueError, match="allow_lossy_casts"):
            ar.auto_clean(frame, mode="strict")

    def test_strict_requires_confirmed_casts_match(self, tmp_path):
        """confirmed_casts must match proposed mapping exactly."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1, Alice\n")

        frame = ar.read_csv(csv_file)

        with pytest.raises(ValueError, match="confirmed_casts must match"):
            ar.auto_clean(
                frame,
                mode="strict",
                allow_lossy_casts=True,
                confirmed_casts={"id": "int64", "extra": "string"}
            )

    def test_strict_accepts_matching_confirmed_casts(self, tmp_path):
        """Strict mode works when confirmed_casts matches proposal."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,name\n"
            "1, Alice\n"
            "2, Bob\n"
        )

        frame = ar.read_csv(csv_file)
        clean = ar.auto_clean(
            frame,
            mode="strict",
            allow_lossy_casts=True,
            confirmed_casts={}
        )

        df = ar.to_pandas(clean)
        assert len(df) == 2


class TestAutoCleanErrorHandling:
    """Test error cases and input validation."""

    def test_invalid_mode_raises_value_error(self, tmp_path):
        """Invalid mode string should raise ValueError."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name\nAlice\n")

        frame = ar.read_csv(csv_file)

        with pytest.raises(ValueError, match="mode must be"):
            ar.auto_clean(frame, mode="invalid_mode")

    def test_dry_run_with_explain_raises(self, tmp_path):
        """Combining dry_run=True and explain=True should raise."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name\nAlice\n")

        frame = ar.read_csv(csv_file)

        with pytest.raises(ValueError, match="explain=True cannot be used with dry_run=True"):
            ar.auto_clean(frame, dry_run=True, explain=True)

    def test_dry_run_with_return_report_raises(self, tmp_path):
        """Combining dry_run=True and return_report=True should raise."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name\nAlice\n")

        frame = ar.read_csv(csv_file)

        with pytest.raises(ValueError, match="return_report=True cannot be used with dry_run=True"):
            ar.auto_clean(frame, dry_run=True, return_report=True)


class TestAutoCleanFileWorkflows:
    """Test complete end-to-end file I/O workflows."""

    def test_read_clean_write_roundtrip(self, tmp_path):
        """Full user workflow: read messy CSV -> clean -> write clean CSV."""
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"

        input_csv.write_text(
            "customer_id,customer_name,city,revenue\n"
            "101, Alice Smith , New York ,50000\n"
            "102, Bob Jones  , London ,75000\n"
        )

        # Clean pipeline
        frame = ar.read_csv(input_csv)
        clean = ar.auto_clean(frame)
        ar.write_csv(clean, output_csv)

        # Verify output file
        result_frame = ar.read_csv(output_csv)
        df = ar.to_pandas(result_frame)

        assert len(df) == 2
        assert df["customer_name"].tolist() == ["Alice Smith", "Bob Jones"]
        assert df["city"].tolist() == ["New York", "London"]
        assert output_csv.exists()

    def test_clean_and_profile_together(self, tmp_path):
        """Combine auto_clean with profiling for quality assessment."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "name,score,grade\n"
            " Alice , 95,A\n"
            " Bob  , 87,B\n"
        )

        frame = ar.read_csv(csv_file)
        clean, report = ar.auto_clean(frame, return_report=True)

        assert isinstance(report, ar.DataQualityReport)
        assert hasattr(report, 'quality_score')
        assert isinstance(clean, ar.ArFrame)
        assert clean.shape[0] == 2


class TestAutoCleanPandasIntegration:
    """Test integration with pandas accessor and pd.DataFrame."""

    def test_pandas_accessor_clean_method(self):
        """Test pandas .arnio.clean() accessor."""
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  "],
            "age": [25, 30]
        })

        clean_df = df.arnio.clean(strip_whitespace=True)

        assert isinstance(clean_df, pd.DataFrame)
        assert clean_df["name"].tolist() == ["Alice", "Bob"]
        assert len(clean_df) == 2

    def test_from_pandas_to_auto_clean_to_pandas(self):
        """Full roundtrip: pandas -> arnio -> clean -> pandas."""
        df_input = pd.DataFrame({
            "id": [1, 2, 2],
            "name": [" Alice ", " Bob ", " Bob "],
            "city": [" NYC ", " London ", " London "]
        })

        frame = ar.from_pandas(df_input)
        clean = ar.auto_clean(frame, mode="strict", allow_lossy_casts=True, confirmed_casts={})
        df_output = ar.to_pandas(clean)

        assert isinstance(df_output, pd.DataFrame)
        assert len(df_output) == 2  # Deduped
        assert df_output["name"].tolist() == ["Alice", "Bob"]


class TestAutoCleanEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_csv_with_headers(self, tmp_path):
        """Handle CSV with only headers, no data rows."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("name,value,city\n")

        frame = ar.read_csv(csv_file)
        clean = ar.auto_clean(frame)

        assert clean.shape[0] == 0

    def test_single_row_csv(self, tmp_path):
        """Handle CSV with single data row."""
        csv_file = tmp_path / "single.csv"
        csv_file.write_text("name,value\n Alice ,100\n")

        frame = ar.read_csv(csv_file)
        clean = ar.auto_clean(frame)

        assert clean.shape[0] == 1
        df = ar.to_pandas(clean)
        assert df["name"].iloc[0] == "Alice"

    def test_all_duplicate_rows_strict(self, tmp_path):
        """Strict mode with all duplicate rows."""
        csv_file = tmp_path / "dupes.csv"
        csv_file.write_text(
            "id,name\n"
            "1, Alice\n"
            "1, Alice\n"
            "1, Alice\n"
        )

        frame = ar.read_csv(csv_file)
        clean = ar.auto_clean(frame, mode="strict", allow_lossy_casts=True, confirmed_casts={})

        df = ar.to_pandas(clean)
        assert len(df) == 1

    def test_no_cleaning_needed(self, tmp_path):
        """Clean data requiring no cleaning."""
        csv_file = tmp_path / "clean.csv"
        csv_file.write_text(
            "name,value\n"
            "Alice,100\n"
            "Bob,200\n"
        )

        frame = ar.read_csv(csv_file)
        clean = ar.auto_clean(frame)
        df = ar.to_pandas(clean)

        assert df["name"].tolist() == ["Alice", "Bob"]
        assert len(df) == 2

    def test_all_nulls_column(self, tmp_path):
        """Handle column with all null values."""
        csv_file = tmp_path / "nulls.csv"
        csv_file.write_text(
            "id,name,notes\n"
            "1, Alice,\n"
            "2, Bob,\n"
        )

        frame = ar.read_csv(csv_file)
        clean = ar.auto_clean(frame)
        df = ar.to_pandas(clean)

        assert df["id"].tolist() == [1, 2]
        assert pd.isna(df["notes"]).all()


class TestAutoCleanExplanationOutput:
    """Test CleanExplanation metadata output."""

    def test_explanation_contains_step_details(self, tmp_path):
        """Explanation records before/after row counts for each step."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "name,value\n"
            " Alice ,100\n"
            " Bob  ,200\n"
            " Bob  ,200\n"
        )

        frame = ar.read_csv(csv_file)
        clean, explanation = ar.auto_clean(
            frame,
            mode="strict",
            explain=True,
            allow_lossy_casts=True,
            confirmed_casts={}
        )

        assert explanation.mode == "strict"
        assert explanation.rows_before >= explanation.rows_after
        assert len(explanation.steps) > 0

        # Each step should have metadata
        for step in explanation.steps:
            assert hasattr(step, 'step')
            assert hasattr(step, 'rows_before')
            assert hasattr(step, 'rows_after')
            assert step.rows_before >= step.rows_after

    def test_explanation_in_combination_with_report(self, tmp_path):
        """Get both report and explanation together."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "name,score\n"
            " Alice , 95\n"
            " Bob  , 87\n"
        )

        frame = ar.read_csv(csv_file)
        clean, report, explanation = ar.auto_clean(
            frame,
            return_report=True,
            explain=True
        )

        assert isinstance(report, ar.DataQualityReport)
        assert isinstance(clean, ar.ArFrame)
        assert hasattr(explanation, 'mode')
        assert hasattr(explanation, 'steps')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

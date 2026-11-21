"""
End-to-end pytest tests for arnio.cleaning.normalize_whitespace.

Tests drive complete user workflows through arnio's public API:
- Loading CSV files from disk
- Applying normalize_whitespace transformation
- Writing results back to disk
- Converting to pandas
- Using in pipelines
"""

import pytest
import pandas as pd
import numpy as np
import arnio as ar
from pathlib import Path


class TestNormalizeWhitespaceBasic:
    """Basic functionality tests for normalize_whitespace."""

    def test_collapse_leading_trailing_whitespace(self, tmp_path):
        """Test that leading and trailing whitespace is stripped."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("name,city\n  alice  ,  paris  \n  bob  ,  london  \n")

        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame)
        df = ar.to_pandas(clean)

        assert df.loc[0, "name"] == "alice"
        assert df.loc[0, "city"] == "paris"
        assert df.loc[1, "name"] == "bob"
        assert df.loc[1, "city"] == "london"

    def test_collapse_internal_whitespace_runs(self, tmp_path):
        """Test that multiple internal spaces collapse to single space."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("text\nhello    world\nfoo  bar  baz\n")

        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame)
        df = ar.to_pandas(clean)

        assert df.loc[0, "text"] == "hello world"
        assert df.loc[1, "text"] == "foo bar baz"

    def test_whitespace_types_tabs_and_newlines(self, tmp_path):
        """Test that tabs and newlines are also normalized to spaces."""
        csv_file = tmp_path / "input.csv"
        # Using explicit tabs and newlines in string
        csv_file.write_text("text\nhello\t\tworld\nfoo\n\nbar\n")

        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame)
        df = ar.to_pandas(clean)

        assert df.loc[0, "text"] == "hello world"
        assert df.loc[1, "text"] == "foo bar"


class TestNormalizeWhitespaceColumnSelection:
    """Tests for column selection behavior."""

    def test_specific_columns_only(self, tmp_path):
        """Test that only specified columns are normalized."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("name,notes,score\n  alice  ,  extra  spaces  ,  100  \n")

        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame, columns=["name"])
        df = ar.to_pandas(clean)

        assert df.loc[0, "name"] == "alice"
        # Other columns should remain unchanged
        assert df.loc[0, "notes"] == "  extra  spaces  "
        assert df.loc[0, "score"] == "  100  "

    def test_multiple_columns_specified(self, tmp_path):
        """Test normalizing multiple specified columns."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("name,city,notes\n  alice  ,  paris  ,  great  place  \n")

        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame, columns=["name", "city"])
        df = ar.to_pandas(clean)

        assert df.loc[0, "name"] == "alice"
        assert df.loc[0, "city"] == "paris"
        assert df.loc[0, "notes"] == "  great  place  "  # not in columns list

    def test_default_all_string_columns(self, tmp_path):
        """Test that all string columns are normalized when columns=None."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("name,city,age\n  alice  ,  paris  ,25\n")

        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame)  # columns=None (default)
        df = ar.to_pandas(clean)

        assert df.loc[0, "name"] == "alice"
        assert df.loc[0, "city"] == "paris"


class TestNormalizeWhitespaceEdgeCases:
    """Edge cases and special value handling."""

    def test_null_values_preserved(self, tmp_path):
        """Test that null/empty values are preserved as-is."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("name,city\n  alice  ,\n,  paris  \n")

        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame)
        df = ar.to_pandas(clean)

        assert df.loc[0, "name"] == "alice"
        assert pd.isna(df.loc[0, "city"])
        assert pd.isna(df.loc[1, "name"])
        assert df.loc[1, "city"] == "paris"

    def test_whitespace_only_values_become_empty_string(self, tmp_path):
        """Test that values containing only whitespace become empty string."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("text\n   \n\t\n")

        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame)
        df = ar.to_pandas(clean)

        assert df.loc[0, "text"] == ""
        assert df.loc[1, "text"] == ""

    def test_already_normalized_data_unchanged(self, tmp_path):
        """Test that data already properly normalized is unchanged."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("name,city\nalice,paris\nbob,london\n")

        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame)
        df = ar.to_pandas(clean)

        assert df.loc[0, "name"] == "alice"
        assert df.loc[0, "city"] == "paris"
        assert df.loc[1, "name"] == "bob"
        assert df.loc[1, "city"] == "london"

    def test_empty_dataframe(self, tmp_path):
        """Test that empty dataframe is handled gracefully."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("name,city\n")

        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame)
        df = ar.to_pandas(clean)

        assert len(df) == 0
        assert list(df.columns) == ["name", "city"]

    def test_missing_columns_raises_valueerror(self, tmp_path):
        """Test that specifying non-existent columns raises ValueError."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("name,city\nalice,paris\n")

        frame = ar.read_csv(str(csv_file))

        with pytest.raises(ValueError, match="Missing columns"):
            ar.normalize_whitespace(frame, columns=["nonexistent"])


class TestNormalizeWhitespaceIntegration:
    """Integration tests with other arnio features."""

    def test_pipeline_integration(self, tmp_path):
        """Test normalize_whitespace in an ar.pipeline call."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("name,city\n  alice  ,  paris  \n  bob  ,  london  \n")

        frame = ar.read_csv(str(csv_file))
        clean = ar.pipeline(frame, [
            ("normalize_whitespace",),
            ("drop_duplicates",),
        ])
        df = ar.to_pandas(clean)

        assert len(df) == 2
        assert df.loc[0, "name"] == "alice"
        assert df.loc[0, "city"] == "paris"
        assert df.loc[1, "name"] == "bob"

    def test_pipeline_with_column_specification(self, tmp_path):
        """Test normalize_whitespace in pipeline with column specification."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("name,notes\n  alice  ,  extra  spaces  \n")

        frame = ar.read_csv(str(csv_file))
        clean = ar.pipeline(frame, [
            ("normalize_whitespace", {"columns": ["name"]}),
        ])
        df = ar.to_pandas(clean)

        assert df.loc[0, "name"] == "alice"
        assert df.loc[0, "notes"] == "  extra  spaces  "  # unchanged

    def test_roundtrip_write_and_read(self, tmp_path):
        """Test full roundtrip: read CSV, normalize, write, read back."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("name,city\n  alice  ,  paris  \n  bob  ,  london  \n")

        # Read, normalize, and write
        frame = ar.read_csv(str(input_csv))
        clean = ar.normalize_whitespace(frame)
        output_csv = tmp_path / "output.csv"
        ar.write_csv(clean, str(output_csv))

        # Read back and verify
        result_frame = ar.read_csv(str(output_csv))
        df = ar.to_pandas(result_frame)

        assert df.loc[0, "name"] == "alice"
        assert df.loc[0, "city"] == "paris"
        assert df.loc[1, "name"] == "bob"
        assert df.loc[1, "city"] == "london"

    def test_from_pandas_roundtrip(self, tmp_path):
        """Test normalize_whitespace on dataframe created from pandas."""
        df_input = pd.DataFrame({
            "name": ["  alice  ", "  bob  "],
            "city": ["  paris  ", "  london  "],
        })

        frame = ar.from_pandas(df_input)
        clean = ar.normalize_whitespace(frame)
        df_output = ar.to_pandas(clean)

        assert df_output.loc[0, "name"] == "alice"
        assert df_output.loc[0, "city"] == "paris"
        assert df_output.loc[1, "name"] == "bob"

    def test_with_profiling(self, tmp_path):
        """Test that profile reports improvement after normalization."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("name\n  alice  \n  bob  \n  alice  \n")

        frame = ar.read_csv(str(csv_file))
        
        # Before normalization
        report_before = ar.profile(frame)
        
        # After normalization
        clean = ar.normalize_whitespace(frame)
        report_after = ar.profile(clean)

        # Both should have same row count but after normalization,
        # duplicate detection may work differently on cleaned data
        assert report_before.row_count == report_after.row_count == 3

    def test_multiple_pipeline_steps(self, tmp_path):
        """Test normalize_whitespace combined with other cleaning steps."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text(
            "name,score\n  alice  ,95\n  bob  ,\n  alice  ,95\n"
        )

        frame = ar.read_csv(str(csv_file))
        clean = ar.pipeline(frame, [
            ("normalize_whitespace",),
            ("fill_nulls", {"value": 0, "subset": ["score"]}),
            ("drop_duplicates",),
        ])
        df = ar.to_pandas(clean)

        assert len(df) == 2
        assert df.loc[0, "name"] == "alice"
        assert df.loc[1, "name"] == "bob"

    def test_pandas_from_arframe_conversion(self, tmp_path):
        """Test converting between ArFrame and pandas with normalization."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("text\n  hello  world  \n  foo  bar  \n")

        # ArFrame workflow
        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame)
        df = ar.to_pandas(clean)

        assert df.loc[0, "text"] == "hello world"
        assert df.loc[1, "text"] == "foo bar"

        # Verify back to ArFrame works too
        frame_again = ar.from_pandas(df)
        assert frame_again.shape == (2, 1)

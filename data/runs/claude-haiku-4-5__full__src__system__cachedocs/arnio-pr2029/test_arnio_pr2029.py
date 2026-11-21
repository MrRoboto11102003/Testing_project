"""
End-to-end tests for arnio.cleaning.clean_column_names.

Tests drive complete user workflows through arnio's public interfaces:
- Real CSV files on disk
- Loading and cleaning
- Validation/profiling after cleaning
- Round-trip write/read
- Error cases and edge cases
"""

import pytest
import pandas as pd
import numpy as np
import sys
import subprocess
from pathlib import Path
import arnio as ar


class TestCleanColumnNamesEndToEnd:
    """End-to-end tests for arnio.cleaning.clean_column_names."""

    def test_basic_cleanup_default_lowercase(self, tmp_path):
        """Load CSV with messy names, clean to lowercase, verify output."""
        # User workflow: read CSV with mixed-case, special-char column names
        df = pd.DataFrame({
            "First Name": [1, 2, 3],
            "Last-Name": ["alice", "bob", "charlie"],
            "E_mail@Address": [True, False, True],
        })
        
        csv_path = tmp_path / "messy.csv"
        df.to_csv(csv_path, index=False)
        
        # Load via arnio
        frame = ar.read_csv(csv_path)
        assert len(frame.columns) == 3
        
        # Clean column names (default: lowercase)
        cleaned = ar.clean_column_names(frame)
        
        # Verify cleaned names are lowercase and alphanumeric
        assert "first_name" in cleaned.columns
        assert "last_name" in cleaned.columns
        assert "email_address" in cleaned.columns
        
        # Round-trip to pandas
        result_df = ar.to_pandas(cleaned)
        assert len(result_df) == 3
        assert result_df["first_name"].tolist() == [1, 2, 3]

    def test_cleanup_uppercase_case_type(self, tmp_path):
        """Clean with uppercase normalization."""
        df = pd.DataFrame({
            "first_name": [1, 2],
            "last-name": ["x", "y"],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)
        
        frame = ar.read_csv(csv_path)
        cleaned = ar.clean_column_names(frame, case_type="upper")
        
        assert cleaned.columns == ["FIRST_NAME", "LAST_NAME"]

    def test_cleanup_title_case_type(self, tmp_path):
        """Clean with title case normalization."""
        df = pd.DataFrame({
            "first_name": [1, 2],
            "last_name": ["x", "y"],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)
        
        frame = ar.read_csv(csv_path)
        cleaned = ar.clean_column_names(frame, case_type="title")
        
        assert cleaned.columns == ["First_Name", "Last_Name"]

    def test_cleanup_camel_case_type(self, tmp_path):
        """Clean with camelCase normalization."""
        df = pd.DataFrame({
            "first_name": [1, 2],
            "last_name": ["x", "y"],
            "user_id": [100, 101],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)
        
        frame = ar.read_csv(csv_path)
        cleaned = ar.clean_column_names(frame, case_type="camel")
        
        assert "firstName" in cleaned.columns
        assert "lastName" in cleaned.columns
        assert "userId" in cleaned.columns

    def test_cleanup_special_chars_replaced_with_underscores(self, tmp_path):
        """Special characters are replaced with underscores."""
        df = pd.DataFrame({
            "First@Name": [1, 2],
            "Last#Name!": ["x", "y"],
            "E$mail%Address": [True, False],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)
        
        frame = ar.read_csv(csv_path)
        cleaned = ar.clean_column_names(frame)
        
        # @ # ! $ % replaced with _
        assert "first_name" in cleaned.columns
        assert "last_name" in cleaned.columns
        assert "email_address" in cleaned.columns

    def test_cleanup_collapse_consecutive_underscores(self, tmp_path):
        """Consecutive underscores are collapsed to single."""
        df = pd.DataFrame({
            "First___Name": [1, 2],
            "Last__Name": ["x", "y"],
            "___middle___": ["a", "b"],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)
        
        frame = ar.read_csv(csv_path)
        cleaned = ar.clean_column_names(frame)
        
        # Multiple consecutive underscores collapsed
        assert "first_name" in cleaned.columns
        assert "last_name" in cleaned.columns
        assert "middle" in cleaned.columns

    def test_cleanup_trim_boundary_underscores(self, tmp_path):
        """Leading/trailing underscores are trimmed."""
        df = pd.DataFrame({
            "_First_Name_": [1, 2],
            "__Last_Name__": ["x", "y"],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)
        
        frame = ar.read_csv(csv_path)
        cleaned = ar.clean_column_names(frame)
        
        # Boundary underscores removed
        assert "first_name" in cleaned.columns
        assert "last_name" in cleaned.columns

    def test_cleanup_round_trip_write_read(self, tmp_path):
        """Clean, write to CSV, read back — verify names persist."""
        df = pd.DataFrame({
            "First Name": [1, 2, 3],
            "Last-Name": ["alice", "bob", "charlie"],
            "Score": [100, 200, 300],
        })
        
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        df.to_csv(input_path, index=False)
        
        # Load, clean, write
        frame = ar.read_csv(input_path)
        cleaned = ar.clean_column_names(frame, case_type="lower")
        ar.write_csv(cleaned, output_path)
        
        # Read back and verify
        frame2 = ar.read_csv(output_path)
        result_df = ar.to_pandas(frame2)
        
        assert "first_name" in result_df.columns
        assert "last_name" in result_df.columns
        assert "score" in result_df.columns
        assert len(result_df) == 3

    def test_cleanup_pipeline_integration(self, tmp_path):
        """Use cleaned frame with other pipeline steps."""
        df = pd.DataFrame({
            "First Name": [1, 2, None],
            "Last-Name": ["alice", "bob", None],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)
        
        frame = ar.read_csv(csv_path)
        cleaned = ar.clean_column_names(frame)
        
        # Apply pipeline steps on cleaned frame
        result = ar.pipeline(cleaned, [
            ("strip_whitespace",),
            ("drop_nulls",),
        ])
        
        # Verify pipeline worked and column names are still clean
        assert "first_name" in result.columns
        assert "last_name" in result.columns
        assert len(result) == 2  # One row dropped due to nulls

    def test_cleanup_with_profiling(self, tmp_path):
        """Profile frame after cleaning column names."""
        df = pd.DataFrame({
            "First Name": [1, 2, 3],
            "Last-Name": ["x", "y", "z"],
            "Score": [100, 200, 300],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)
        
        frame = ar.read_csv(csv_path)
        cleaned = ar.clean_column_names(frame)
        
        # Profile shows cleaned names
        report = ar.profile(cleaned)
        
        assert report.row_count == 3
        # Column names in report should be cleaned
        for col_name in report.columns.keys():
            assert col_name.islower() or "_" in col_name or col_name.isdigit()

    def test_error_invalid_case_type_string(self, tmp_path):
        """Raise ValueError for unrecognized case_type string."""
        df = pd.DataFrame({"First Name": [1, 2]})
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)
        
        frame = ar.read_csv(csv_path)
        
        with pytest.raises(ValueError, match="case_type must be one of"):
            ar.clean_column_names(frame, case_type="invalid_case")

    def test_error_case_type_not_string_type(self, tmp_path):
        """Raise TypeError if case_type is not a string."""
        df = pd.DataFrame({"First Name": [1, 2]})
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)
        
        frame = ar.read_csv(csv_path)
        
        with pytest.raises(TypeError, match="case_type must be a string"):
            ar.clean_column_names(frame, case_type=123)
        
        with pytest.raises(TypeError, match="case_type must be a string"):
            ar.clean_column_names(frame, case_type=None)

    def test_error_duplicate_names_after_cleaning(self, tmp_path):
        """Raise ValueError if cleaning would create duplicate column names."""
        # Create columns that become identical after cleaning
        df = pd.DataFrame({
            "First Name": [1, 2],
            "First-Name": ["x", "y"],  # Both become "first_name"
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)
        
        frame = ar.read_csv(csv_path)
        
        with pytest.raises(ValueError, match="duplicate"):
            ar.clean_column_names(frame)

    def test_cleanup_all_valid_case_types(self, tmp_path):
        """Verify all documented case_type options work."""
        df = pd.DataFrame({
            "user_id": [1, 2],
            "first_name": ["a", "b"],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)
        
        frame = ar.read_csv(csv_path)
        
        # Test each valid case_type
        for case_type in ["lower", "upper", "title", "camel", "none"]:
            cleaned = ar.clean_column_names(frame, case_type=case_type)
            assert len(cleaned.columns) == 2
            assert all(isinstance(col, str) for col in cleaned.columns)

    def test_cleanup_preserves_data_integrity(self, tmp_path):
        """Cleaning column names does not change data values."""
        df = pd.DataFrame({
            "First Name": [1, 2, 3],
            "Last-Name": ["alice", "bob", "charlie"],
            "Score": [100.5, 200.5, 300.5],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)
        
        frame = ar.read_csv(csv_path)
        original_values = ar.to_pandas(frame)
        
        cleaned = ar.clean_column_names(frame)
        cleaned_values = ar.to_pandas(cleaned)
        
        # Data unchanged, only column names changed
        assert original_values["First Name"].tolist() == cleaned_values["first_name"].tolist()
        assert original_values["Last-Name"].tolist() == cleaned_values["last_name"].tolist()
        assert original_values["Score"].tolist() == cleaned_values["score"].tolist()

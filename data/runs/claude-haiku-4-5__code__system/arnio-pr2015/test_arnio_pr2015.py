import pytest
import pandas as pd
import numpy as np
import sys
import subprocess
import tempfile
from pathlib import Path
import arnio as ar


class TestParseBoolStrings:
    """System tests for arnio.cleaning.parse_bool_strings"""

    def test_basic_bool_string_conversion_from_csv(self, tmp_path):
        """Test basic conversion of common boolean strings to booleans from CSV file."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,is_active,flag\n1,true,yes\n2,false,no\n3,True,Y\n4,FALSE,N")
        
        df = pd.read_csv(csv_file)
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["is_active"].iloc[0] is True
        assert result_df["is_active"].iloc[1] is False
        assert result_df["is_active"].iloc[2] is True
        assert result_df["is_active"].iloc[3] is False
        assert result_df["flag"].iloc[0] is True
        assert result_df["flag"].iloc[1] is False

    def test_numeric_string_values(self, tmp_path):
        """Test that numeric string representations (1/0) are converted correctly."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,status\n1,1\n2,0\n3,yes\n4,no")
        
        df = pd.read_csv(csv_file)
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["status"].iloc[0] is True
        assert result_df["status"].iloc[1] is False
        assert result_df["status"].iloc[2] is True
        assert result_df["status"].iloc[3] is False

    def test_custom_true_false_values(self, tmp_path):
        """Test conversion with custom true/false value sets."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,response\n1,agree\n2,disagree\n3,AGREE\n4,DISAGREE")
        
        df = pd.read_csv(csv_file)
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(
            frame,
            true_values={"agree"},
            false_values={"disagree"}
        )
        result_df = ar.to_pandas(result)
        
        assert result_df["response"].iloc[0] is True
        assert result_df["response"].iloc[1] is False
        assert result_df["response"].iloc[2] is True
        assert result_df["response"].iloc[3] is False

    def test_subset_parameter_targets_specific_columns(self, tmp_path):
        """Test that subset parameter applies conversion to only specified columns."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,col1,col2\n1,true,yes\n2,false,no")
        
        df = pd.read_csv(csv_file)
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame, subset=["col1"])
        result_df = ar.to_pandas(result)
        
        assert result_df["col1"].iloc[0] is True
        assert result_df["col1"].iloc[1] is False
        assert isinstance(result_df["col2"].iloc[0], str)
        assert result_df["col2"].iloc[0] == "yes"

    def test_whitespace_stripping_normalization(self, tmp_path):
        """Test that whitespace is stripped before matching."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,value\n1, true \n2,  false  \n3, YES \n4, no ")
        
        df = pd.read_csv(csv_file)
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["value"].iloc[0] is True
        assert result_df["value"].iloc[1] is False
        assert result_df["value"].iloc[2] is True
        assert result_df["value"].iloc[3] is False

    def test_unmatched_string_values_preserved(self, tmp_path):
        """Test that string values not in true/false sets are preserved unchanged."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,status\n1,true\n2,maybe\n3,false\n4,unknown")
        
        df = pd.read_csv(csv_file)
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["status"].iloc[0] is True
        assert result_df["status"].iloc[1] == "maybe"
        assert result_df["status"].iloc[2] is False
        assert result_df["status"].iloc[3] == "unknown"

    def test_error_bare_string_true_values(self):
        """Test TypeError when true_values is a bare string instead of set/list/tuple."""
        frame = ar.from_pandas(pd.DataFrame({"col": ["true"]}))
        
        with pytest.raises(TypeError, match="true_values must be a set/list/tuple of strings"):
            ar.parse_bool_strings(frame, true_values="true")

    def test_error_bare_string_false_values(self):
        """Test TypeError when false_values is a bare string instead of set/list/tuple."""
        frame = ar.from_pandas(pd.DataFrame({"col": ["false"]}))
        
        with pytest.raises(TypeError, match="false_values must be a set/list/tuple of strings"):
            ar.parse_bool_strings(frame, false_values="false")

    def test_error_non_string_in_true_values(self):
        """Test TypeError when true_values contains non-string values."""
        frame = ar.from_pandas(pd.DataFrame({"col": ["true"]}))
        
        with pytest.raises(TypeError, match="true_values must contain only strings"):
            ar.parse_bool_strings(frame, true_values=["true", 1])

    def test_error_non_string_in_false_values(self):
        """Test TypeError when false_values contains non-string values."""
        frame = ar.from_pandas(pd.DataFrame({"col": ["false"]}))
        
        with pytest.raises(TypeError, match="false_values must contain only strings"):
            ar.parse_bool_strings(frame, false_values=["false", 0])

    def test_error_overlapping_true_false_values(self):
        """Test ValueError when true_values and false_values overlap after normalization."""
        frame = ar.from_pandas(pd.DataFrame({"col": ["yes"]}))
        
        with pytest.raises(ValueError, match="true_values and false_values overlap"):
            ar.parse_bool_strings(
                frame,
                true_values={"yes", "true"},
                false_values={"no", "yes"}
            )

    def test_error_missing_subset_column(self):
        """Test ValueError when subset references non-existent columns."""
        frame = ar.from_pandas(pd.DataFrame({"col1": ["true"]}))
        
        with pytest.raises(ValueError, match="Columns not found in frame"):
            ar.parse_bool_strings(frame, subset=["nonexistent"])

    def test_already_boolean_columns_skipped_in_subset(self, tmp_path):
        """Test that already-boolean columns are not reprocessed even when in subset."""
        df = pd.DataFrame({
            "bool_col": [True, False],
            "str_col": ["true", "false"]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame, subset=["bool_col", "str_col"])
        result_df = ar.to_pandas(result)
        
        assert result_df["bool_col"].iloc[0] is True
        assert result_df["bool_col"].iloc[1] is False
        assert result_df["str_col"].iloc[0] is True
        assert result_df["str_col"].iloc[1] is False

    def test_numeric_columns_ignored_by_default(self, tmp_path):
        """Test that numeric columns are not processed when subset is None."""
        df = pd.DataFrame({
            "num_col": [1, 0, 1],
            "str_col": ["true", "false", "yes"]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["num_col"].iloc[0] == 1
        assert result_df["num_col"].iloc[1] == 0
        assert result_df["str_col"].iloc[0] is True
        assert result_df["str_col"].iloc[1] is False
        assert result_df["str_col"].iloc[2] is True

    def test_round_trip_csv_io_preserves_booleans(self, tmp_path):
        """Test that parsed data can be written to CSV and read with correct types."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("id,active,approved\n1,true,yes\n2,false,no\n3,yes,true")
        
        df = pd.read_csv(input_file)
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        output_file = tmp_path / "output.csv"
        result_df.to_csv(output_file, index=False)
        
        reloaded = pd.read_csv(output_file)
        assert len(reloaded) == 3
        assert "active" in reloaded.columns
        assert "approved" in reloaded.columns

    def test_case_insensitive_all_variations(self, tmp_path):
        """Test various case combinations are normalized correctly."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,val\n1,TrUe\n2,FaLsE\n3,YeS\n4,NO\n5,y\n6,N")
        
        df = pd.read_csv(csv_file)
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        for idx in [0, 2, 4]:
            assert result_df["val"].iloc[idx] is True
        for idx in [1, 3, 5]:
            assert result_df["val"].iloc[idx] is False

    def test_multiple_subset_columns(self, tmp_path):
        """Test subset with multiple column names selects correct columns."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,a,b,c\n1,true,yes,maybe\n2,false,no,unclear")
        
        df = pd.read_csv(csv_file)
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame, subset=["a", "b"])
        result_df = ar.to_pandas(result)
        
        assert result_df["a"].iloc[0] is True
        assert result_df["b"].iloc[0] is True
        assert result_df["c"].iloc[0] == "maybe"
        assert result_df["c"].iloc[1] == "unclear"

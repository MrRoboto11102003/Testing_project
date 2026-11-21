import pytest
import pandas as pd
import arnio as ar
from pathlib import Path


class TestParseBoolStrings:
    """End-to-end tests for parse_bool_strings functionality."""

    def test_basic_yes_no_conversion(self, tmp_path):
        """Convert basic yes/no strings to boolean."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,active\nAlice,yes\nBob,no\nCharlie,maybe")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.parse_bool_strings(frame)
        df = ar.to_pandas(result)
        
        assert df["active"][0] == True
        assert df["active"][1] == False
        assert df["active"][2] == "maybe"

    def test_default_true_values(self, tmp_path):
        """Test default true values: true, yes, y, 1."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val\ntrue\nyes\ny\n1")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.parse_bool_strings(frame)
        df = ar.to_pandas(result)
        
        for val in df["val"]:
            assert val == True

    def test_default_false_values(self, tmp_path):
        """Test default false values: false, no, n, 0."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val\nfalse\nno\nn\n0")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.parse_bool_strings(frame)
        df = ar.to_pandas(result)
        
        for val in df["val"]:
            assert val == False

    def test_case_insensitive_matching(self, tmp_path):
        """Test that matching is case-insensitive."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val\nTRUE\nYES\nTrue\nFALSE\nNO\nFalse")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.parse_bool_strings(frame)
        df = ar.to_pandas(result)
        
        assert df["val"][0] == True
        assert df["val"][1] == True
        assert df["val"][2] == True
        assert df["val"][3] == False
        assert df["val"][4] == False
        assert df["val"][5] == False

    def test_whitespace_trimming(self, tmp_path):
        """Test that whitespace is trimmed before matching."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val\n yes \n no \n  true  ")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.parse_bool_strings(frame)
        df = ar.to_pandas(result)
        
        assert df["val"][0] == True
        assert df["val"][1] == False
        assert df["val"][2] == True

    def test_custom_true_values(self, tmp_path):
        """Test custom true values."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("status\nactive\ninactive\npending")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.parse_bool_strings(frame, true_values={"active"})
        df = ar.to_pandas(result)
        
        assert df["status"][0] == True
        assert df["status"][1] == "inactive"
        assert df["status"][2] == "pending"

    def test_subset_parameter(self, tmp_path):
        """Test that subset parameter limits which columns are processed."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2,col3\nyes,yes,yes\nno,no,no")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.parse_bool_strings(frame, subset=["col1"])
        df = ar.to_pandas(result)
        
        assert df["col1"][0] == True
        assert df["col1"][1] == False
        assert df["col2"][0] == "yes"  # Not converted
        assert df["col2"][1] == "no"   # Not converted
        assert df["col3"][0] == "yes"  # Not converted

    def test_multiple_columns_no_subset(self, tmp_path):
        """Test that all string columns are converted when subset is None."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("flag1,flag2,flag3\ntrue,yes,1\nfalse,no,0")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.parse_bool_strings(frame)
        df = ar.to_pandas(result)
        
        for i in range(1, 4):
            assert df[f"flag{i}"][0] == True
            assert df[f"flag{i}"][1] == False

    def test_error_bare_string_true_values(self):
        """Test error when true_values is a bare string."""
        frame = ar.from_dict({"val": ["yes", "no"]})
        
        with pytest.raises(TypeError, match="true_values must be a set/list/tuple"):
            ar.parse_bool_strings(frame, true_values="yes")

    def test_error_bare_string_false_values(self):
        """Test error when false_values is a bare string."""
        frame = ar.from_dict({"val": ["yes", "no"]})
        
        with pytest.raises(TypeError, match="false_values must be a set/list/tuple"):
            ar.parse_bool_strings(frame, false_values="no")

    def test_error_overlapping_values(self):
        """Test error when true_values and false_values overlap after normalization."""
        frame = ar.from_dict({"val": ["yes", "no"]})
        
        with pytest.raises(ValueError, match="overlap"):
            ar.parse_bool_strings(
                frame,
                true_values={"yes", "maybe"},
                false_values={"no", "maybe"}
            )

    def test_error_non_string_in_true_values(self):
        """Test error when true_values contains non-strings."""
        frame = ar.from_dict({"val": ["yes", "no"]})
        
        with pytest.raises(TypeError, match="true_values must contain only strings"):
            ar.parse_bool_strings(frame, true_values={"yes", 1})

    def test_error_nonexistent_column_in_subset(self, tmp_path):
        """Test error when subset contains non-existent column."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1\nyes")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError, match="Columns not found"):
            ar.parse_bool_strings(frame, subset=["nonexistent"])

    def test_mixed_matched_and_unmatched(self, tmp_path):
        """Test column with both matched and unmatched values preserves unmatched."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val\nyes\nno\nmaybe\ntrue\nfalse\nunknown")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.parse_bool_strings(frame)
        df = ar.to_pandas(result)
        
        assert df["val"][0] == True
        assert df["val"][1] == False
        assert df["val"][2] == "maybe"
        assert df["val"][3] == True
        assert df["val"][4] == False
        assert df["val"][5] == "unknown"

    def test_round_trip_csv_write_read(self, tmp_path):
        """Test full round-trip: read CSV, parse bools, write CSV, read back."""
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"
        input_file.write_text("id,is_active,verified\n1,yes,true\n2,no,false")
        
        frame = ar.read_csv(str(input_file))
        result = ar.parse_bool_strings(frame)
        df = ar.to_pandas(result)
        
        # Verify conversion happened
        assert df["is_active"][0] == True
        assert df["is_active"][1] == False
        assert df["verified"][0] == True
        assert df["verified"][1] == False
        
        # Write back and verify file contains expected data
        df.to_csv(str(output_file), index=False)
        output_content = output_file.read_text()
        assert "1,True,True" in output_content or "1,1,1" in output_content
        assert "2,False,False" in output_content or "2,0,0" in output_content

    def test_null_values_preserved(self, tmp_path):
        """Test that null/empty values are preserved as nulls."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val\nyes\n\nno")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.parse_bool_strings(frame)
        df = ar.to_pandas(result)
        
        assert df["val"][0] == True
        assert pd.isna(df["val"][1])
        assert df["val"][2] == False

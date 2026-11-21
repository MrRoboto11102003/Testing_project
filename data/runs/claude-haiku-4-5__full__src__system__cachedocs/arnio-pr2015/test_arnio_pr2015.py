import pytest
import pandas as pd
import numpy as np
import arnio as ar
from pathlib import Path


class TestParseBoolStringsBasic:
    """Test basic parse_bool_strings functionality with realistic workflows"""

    def test_parse_bool_strings_with_defaults(self, tmp_path):
        """Test parse_bool_strings converts common boolean strings with defaults"""
        csv_path = tmp_path / "users.csv"
        csv_path.write_text("user_id,is_active,is_verified\n1,true,yes\n2,false,no\n3,yes,1\n4,no,0\n")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.parse_bool_strings(frame)
        df = ar.to_pandas(result)
        
        # Verify boolean conversions
        assert df["is_active"].iloc[0] in (True, 1) or df["is_active"].iloc[0] is True
        assert df["is_active"].iloc[1] in (False, 0) or df["is_active"].iloc[1] is False
        assert df["is_verified"].iloc[0] in (True, 1) or df["is_verified"].iloc[0] is True
        assert df["is_verified"].iloc[1] in (False, 0) or df["is_verified"].iloc[1] is False

    def test_parse_bool_strings_custom_values(self, tmp_path):
        """Test parse_bool_strings with custom true/false values"""
        csv_path = tmp_path / "devices.csv"
        csv_path.write_text("device_id,status\n100,ON\n101,OFF\n102,ON\n")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.parse_bool_strings(
            frame,
            true_values={"ON"},
            false_values={"OFF"}
        )
        df = ar.to_pandas(result)
        
        assert df["status"].iloc[0] is True or df["status"].iloc[0] == True
        assert df["status"].iloc[1] is False or df["status"].iloc[1] == False
        assert df["status"].iloc[2] is True or df["status"].iloc[2] == True

    def test_parse_bool_strings_subset_parameter(self, tmp_path):
        """Test parse_bool_strings subset parameter applies conversion only to specified columns"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("convert_this,leave_as_string\ntrue,yes\nfalse,no\n")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.parse_bool_strings(frame, subset=["convert_this"])
        df = ar.to_pandas(result)
        
        # convert_this should be boolean
        assert isinstance(df["convert_this"].iloc[0], (bool, np.bool_)) or df["convert_this"].iloc[0] is True
        # leave_as_string should remain string
        assert isinstance(df["leave_as_string"].iloc[0], str)
        assert df["leave_as_string"].iloc[0] == "yes"

    def test_parse_bool_strings_case_insensitive(self, tmp_path):
        """Test parse_bool_strings is case-insensitive"""
        csv_path = tmp_path / "flags.csv"
        csv_path.write_text("flag\nTRUE\nFalse\nYES\nNo\n1\n0\n")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.parse_bool_strings(frame)
        df = ar.to_pandas(result)
        
        # All should be parsed as boolean regardless of case
        for i in range(6):
            assert df["flag"].iloc[i] in (True, False, 0, 1) or isinstance(df["flag"].iloc[i], (bool, np.bool_))

    def test_parse_bool_strings_whitespace_normalization(self, tmp_path):
        """Test that whitespace is handled correctly"""
        csv_path = tmp_path / "messy.csv"
        csv_path.write_text("value\n  true  \n false \n  yes  \n  no\n")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.parse_bool_strings(frame)
        df = ar.to_pandas(result)
        
        # All should be parsed despite whitespace
        assert df["value"].iloc[0] is True or df["value"].iloc[0] == True
        assert df["value"].iloc[1] is False or df["value"].iloc[1] == False
        assert df["value"].iloc[2] is True or df["value"].iloc[2] == True
        assert df["value"].iloc[3] is False or df["value"].iloc[3] == False

    def test_parse_bool_strings_unsupported_values_preserved(self, tmp_path):
        """Test that unsupported string values are preserved unchanged"""
        csv_path = tmp_path / "mixed.csv"
        csv_path.write_text("decision\ntrue\nunknown\nfalse\npending\n")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.parse_bool_strings(frame)
        df = ar.to_pandas(result)
        
        assert df["decision"].iloc[0] is True or df["decision"].iloc[0] == True
        assert df["decision"].iloc[1] == "unknown"
        assert df["decision"].iloc[2] is False or df["decision"].iloc[2] == False
        assert df["decision"].iloc[3] == "pending"

    def test_parse_bool_strings_numeric_strings(self, tmp_path):
        """Test parsing numeric boolean strings (1 for true, 0 for false)"""
        csv_path = tmp_path / "binary.csv"
        csv_path.write_text("binary_flag\n1\n0\n1\n1\n0\n")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.parse_bool_strings(frame)
        df = ar.to_pandas(result)
        
        assert df["binary_flag"].iloc[0] is True or df["binary_flag"].iloc[0] == True
        assert df["binary_flag"].iloc[1] is False or df["binary_flag"].iloc[1] == False
        assert df["binary_flag"].iloc[4] is False or df["binary_flag"].iloc[4] == False


class TestParseBoolStringsErrors:
    """Test error handling and validation"""

    def test_error_true_values_bare_string(self, tmp_path):
        """Test that passing true_values as bare string raises TypeError"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("flag\ntrue\nfalse\n")
        
        frame = ar.read_csv(str(csv_path))
        with pytest.raises(TypeError, match="true_values must be a set"):
            ar.parse_bool_strings(frame, true_values="true")

    def test_error_false_values_bare_string(self, tmp_path):
        """Test that passing false_values as bare string raises TypeError"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("flag\ntrue\nfalse\n")
        
        frame = ar.read_csv(str(csv_path))
        with pytest.raises(TypeError, match="false_values must be a set"):
            ar.parse_bool_strings(frame, false_values="false")

    def test_error_overlapping_values(self, tmp_path):
        """Test that overlapping true_values and false_values raise ValueError"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("flag\ntrue\nfalse\n")
        
        frame = ar.read_csv(str(csv_path))
        with pytest.raises(ValueError, match="overlap"):
            ar.parse_bool_strings(
                frame,
                true_values={"yes", "true"},
                false_values={"yes", "false"}  # "yes" overlaps
            )

    def test_error_non_string_in_true_values(self, tmp_path):
        """Test that non-string values in true_values raise TypeError"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("flag\ntrue\nfalse\n")
        
        frame = ar.read_csv(str(csv_path))
        with pytest.raises(TypeError, match="must contain only strings"):
            ar.parse_bool_strings(frame, true_values={"true", 1})

    def test_error_non_string_in_false_values(self, tmp_path):
        """Test that non-string values in false_values raise TypeError"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("flag\ntrue\nfalse\n")
        
        frame = ar.read_csv(str(csv_path))
        with pytest.raises(TypeError, match="must contain only strings"):
            ar.parse_bool_strings(frame, false_values={"false", None})

    def test_error_subset_nonexistent_column(self, tmp_path):
        """Test that subset with nonexistent column raises ValueError"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("flag\ntrue\nfalse\n")
        
        frame = ar.read_csv(str(csv_path))
        with pytest.raises(ValueError, match="not found"):
            ar.parse_bool_strings(frame, subset=["does_not_exist"])


class TestParseBoolStringsIntegration:
    """Test parse_bool_strings in realistic end-to-end workflows"""

    def test_parse_bool_pipeline_integration(self, tmp_path):
        """Test parse_bool_strings as part of a cleaning pipeline"""
        csv_path = tmp_path / "customers.csv"
        csv_path.write_text(
            "customer_id,has_account,is_premium,status\n"
            "1,yes,true,ACTIVE\n"
            "2,no,false,INACTIVE\n"
            "3,yes,yes,ACTIVE\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        # Apply boolean parsing as part of a pipeline
        clean = ar.pipeline(
            frame,
            [
                ("parse_bool_strings", {}),
                ("drop_nulls",),
            ]
        )
        df = ar.to_pandas(clean)
        
        assert df["has_account"].iloc[0] is True or df["has_account"].iloc[0] == True
        assert df["is_premium"].iloc[2] is True or df["is_premium"].iloc[2] == True

    def test_parse_bool_write_csv_roundtrip(self, tmp_path):
        """Test parse_bool_strings result can be written to CSV and read back"""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("id,is_deleted\n1,no\n2,yes\n3,false\n")
        
        frame = ar.read_csv(str(input_csv))
        parsed = ar.parse_bool_strings(frame)
        
        # Write to CSV
        output_csv = tmp_path / "output.csv"
        ar.write_csv(parsed, str(output_csv))
        
        # Verify file created
        assert output_csv.exists()
        content = output_csv.read_text()
        assert "id" in content
        assert "is_deleted" in content

    def test_parse_bool_from_pandas_roundtrip(self):
        """Test parse_bool_strings works with from_pandas conversion"""
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "enabled": ["TRUE", "false", "YES"],
            "admin": ["yes", "NO", "1"]
        })
        
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        # Verify conversion happened
        assert result_df["enabled"].iloc[0] is True or result_df["enabled"].iloc[0] == True
        assert result_df["enabled"].iloc[1] is False or result_df["enabled"].iloc[1] == False
        assert result_df["admin"].iloc[2] is True or result_df["admin"].iloc[2] == True

    def test_parse_bool_all_nulls(self, tmp_path):
        """Test parse_bool_strings with column of all null values"""
        csv_path = tmp_path / "nulls.csv"
        csv_path.write_text("id,flag\n1,\n2,\n3,\n")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.parse_bool_strings(frame)
        df = ar.to_pandas(result)
        
        # All nulls should be preserved
        assert df["flag"].isna().all()

    def test_parse_bool_empty_frame(self, tmp_path):
        """Test parse_bool_strings with empty frame"""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("flag\n")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.parse_bool_strings(frame)
        df = ar.to_pandas(result)
        
        assert len(df) == 0
        assert "flag" in df.columns

    def test_parse_bool_multiple_columns(self, tmp_path):
        """Test parse_bool_strings on multiple columns simultaneously"""
        csv_path = tmp_path / "multi.csv"
        csv_path.write_text(
            "col1,col2,col3\n"
            "yes,1,true\n"
            "no,0,false\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        result = ar.parse_bool_strings(frame)
        df = ar.to_pandas(result)
        
        for col in ["col1", "col2", "col3"]:
            assert df[col].iloc[0] is True or df[col].iloc[0] == True
            assert df[col].iloc[1] is False or df[col].iloc[1] == False

    def test_parse_bool_normalization_overlap_case_insensitive(self, tmp_path):
        """Test that overlap detection works after case/whitespace normalization"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("flag\ntrue\nfalse\n")
        
        frame = ar.read_csv(str(csv_path))
        with pytest.raises(ValueError, match="overlap"):
            ar.parse_bool_strings(
                frame,
                true_values={"YES", " yes "},  # Will normalize to "yes"
                false_values={"No", " YES "}   # Will normalize to "yes"
            )

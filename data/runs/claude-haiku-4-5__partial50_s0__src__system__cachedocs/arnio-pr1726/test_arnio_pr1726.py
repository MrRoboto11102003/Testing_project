import pytest
import pandas as pd
import numpy as np
import arnio as ar
import tempfile
from pathlib import Path
import csv


class TestStandardizeMissingTokens:
    """End-to-end system tests for arnio.standardize_missing_tokens"""

    def test_default_missing_tokens_csv_workflow(self, tmp_path):
        """Read messy CSV, standardize default missing tokens, verify output"""
        input_csv = tmp_path / "messy_sales.csv"
        input_csv.write_text(
            "order_id,customer,revenue,notes\n"
            "1001,Alice,500.0,processed\n"
            "1002,N/A,750.0,pending\n"
            "1003,Bob,null,no notes\n"
            "1004,Charlie,N/A,?\n"
        )
        
        frame = ar.read_csv(str(input_csv))
        result = ar.standardize_missing_tokens(frame)
        df = ar.to_pandas(result)
        
        # Verify default tokens converted to NaN
        assert pd.isna(df.iloc[1, 1])  # N/A -> NaN
        assert pd.isna(df.iloc[2, 2])  # null -> NaN
        assert pd.isna(df.iloc[3, 2])  # N/A -> NaN
        assert pd.isna(df.iloc[3, 3])  # ? -> NaN
        assert df.iloc[0, 1] == "Alice"  # Non-tokens unchanged
        assert df.iloc[2, 1] == "Bob"

    def test_custom_tokens_replace_missing_values(self):
        """Test custom token list replaces specified missing values"""
        df = pd.DataFrame({
            "metric": ["100", "MISSING", "200", "ERROR", "300"],
            "status": ["OK", "FAIL", "OK", "FAIL", "OK"]
        })
        
        result = ar.standardize_missing_tokens(
            df, 
            tokens=["MISSING", "ERROR"]
        )
        
        assert pd.isna(result.iloc[1, 0])
        assert pd.isna(result.iloc[3, 0])
        assert result.iloc[0, 0] == "100"
        assert result.iloc[2, 0] == "200"
        # Default tokens not applied when custom list provided
        assert result.iloc[1, 1] == "FAIL"

    def test_subset_parameter_selective_columns(self):
        """Apply standardization only to specified columns via subset"""
        df = pd.DataFrame({
            "col_a": ["NA", "value", "null"],
            "col_b": ["NA", "value", "null"],
            "col_c": ["NA", "value", "null"]
        })
        
        result = ar.standardize_missing_tokens(df, subset=["col_a", "col_c"])
        
        # col_a and col_c standardized
        assert pd.isna(result.iloc[0, 0])
        assert pd.isna(result.iloc[2, 0])
        assert pd.isna(result.iloc[0, 2])
        # col_b untouched
        assert result.iloc[0, 1] == "NA"
        assert result.iloc[2, 1] == "null"

    def test_case_insensitive_whitespace_trimmed(self):
        """Verify case-insensitive matching with whitespace trimming"""
        df = pd.DataFrame({
            "value": [" NA ", "\tN/A\t", "  None  ", "NaMe"]
        })
        
        result = ar.standardize_missing_tokens(
            df,
            tokens=["NA", "N/A", "NONE"]
        )
        
        assert pd.isna(result.iloc[0, 0])
        assert pd.isna(result.iloc[1, 0])
        assert pd.isna(result.iloc[2, 0])
        assert result.iloc[3, 0] == "NaMe"  # Not in token list

    def test_pipeline_integration_multiple_steps(self, tmp_path):
        """Standardize in pipeline with other cleaning steps"""
        input_csv = tmp_path / "data.csv"
        input_csv.write_text(
            "name,score,status\n"
            "  Alice  ,95,active\n"
            " Bob ,null,inactive\n"
            "Charlie,N/A,pending\n"
        )
        
        frame = ar.read_csv(str(input_csv))
        clean = ar.pipeline(frame, [
            ("standardize_missing_tokens",),
            ("strip_whitespace",),
            ("drop_nulls",)
        ])
        df = ar.to_pandas(clean)
        
        # After standardize, strip, and drop_nulls
        assert len(df) == 1
        assert df.iloc[0, 0] == "Alice"
        assert df.iloc[0, 1] == "95"

    def test_write_csv_after_standardizing(self, tmp_path):
        """Read, standardize missing tokens, write to new CSV"""
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"
        
        input_csv.write_text(
            "id,value\n1,100\n2,N/A\n3,null\n4,200"
        )
        
        frame = ar.read_csv(str(input_csv))
        clean = ar.standardize_missing_tokens(frame)
        ar.write_csv(clean, str(output_csv))
        
        # Read back output and verify structure
        output_df = pd.read_csv(output_csv)
        assert len(output_df) == 4
        assert output_df.columns.tolist() == ["id", "value"]
        # CSV NaN representation: empty or the string "nan"
        assert pd.isna(output_df.iloc[1, 1])
        assert pd.isna(output_df.iloc[2, 1])

    def test_profile_before_after_standardization(self, tmp_path):
        """Profile dataset before and after standardizing shows improvement"""
        input_csv = tmp_path / "messy.csv"
        input_csv.write_text(
            "value\n"
            "100\n"
            "N/A\n"
            "200\n"
            "NA\n"
            "300\n"
            "null\n"
        )
        
        frame = ar.read_csv(str(input_csv))
        
        # Before standardization, default tokens appear as strings
        report_before = ar.profile(frame)
        null_count_before = report_before.columns["value"].null_count
        
        # Standardize
        clean = ar.standardize_missing_tokens(frame)
        
        # After standardization, more nulls recognized
        report_after = ar.profile(clean)
        null_count_after = report_after.columns["value"].null_count
        
        assert null_count_after > null_count_before
        assert null_count_after >= 3  # At least N/A, NA, null

    def test_validate_schema_after_standardizing(self):
        """Validate data against schema after standardizing missing tokens"""
        df = pd.DataFrame({
            "user_id": ["1", "2", "3"],
            "age": ["25", "NA", "35"],
            "active": ["true", "false", "null"]
        })
        
        clean = ar.standardize_missing_tokens(df)
        frame = ar.from_pandas(clean)
        
        schema = ar.Schema({
            "user_id": ar.Int64(),
            "age": ar.Int64(nullable=True),
            "active": ar.String(nullable=True)
        })
        
        result = ar.validate(frame, schema)
        assert result.passed

    def test_error_tokens_wrong_type_string_not_list(self):
        """TypeError raised when tokens is a string instead of list"""
        df = pd.DataFrame({"col": ["NA", "value"]})
        
        with pytest.raises(
            TypeError, 
            match="tokens must be a list of strings, not a bare string"
        ):
            ar.standardize_missing_tokens(df, tokens="NA")

    def test_error_subset_wrong_type_string_not_list(self):
        """TypeError raised when subset is a string instead of list"""
        df = pd.DataFrame({"col": ["NA", "value"]})
        
        with pytest.raises(
            TypeError,
            match="subset must be a list of column names, not a string"
        ):
            ar.standardize_missing_tokens(df, subset="col")

    def test_error_nonexistent_column_in_subset(self):
        """ValueError raised for nonexistent column in subset"""
        df = pd.DataFrame({"col_a": ["NA", "value"]})
        
        with pytest.raises(ValueError, match="Unknown columns in subset"):
            ar.standardize_missing_tokens(df, subset=["col_b"])

    def test_numeric_and_mixed_type_columns_unaffected(self):
        """Numeric columns and non-string values in mixed columns preserved"""
        df = pd.DataFrame({
            "integers": [1, 2, 3],
            "floats": [1.5, 2.5, 3.5],
            "mixed": [100, "NA", None, "value", 42]
        })
        
        result = ar.standardize_missing_tokens(df)
        
        # Numeric columns unchanged
        assert result["integers"].tolist() == [1, 2, 3]
        assert result["floats"].tolist() == [1.5, 2.5, 3.5]
        # Mixed: numeric values preserved, NA becomes NaN
        assert result.iloc[0, 2] == 100
        assert pd.isna(result.iloc[1, 2])
        assert pd.isna(result.iloc[2, 2])
        assert result.iloc[3, 2] == "value"
        assert result.iloc[4, 2] == 42

    def test_empty_and_all_missing_values(self):
        """Handle empty DataFrame and all-missing columns"""
        # Empty DataFrame
        df_empty = pd.DataFrame({"col": pd.Series([], dtype=str)})
        result_empty = ar.standardize_missing_tokens(df_empty)
        assert len(result_empty) == 0
        
        # All values are missing tokens
        df_all_missing = pd.DataFrame({
            "col": ["NA", "null", "N/A", "none"]
        })
        result_all_missing = ar.standardize_missing_tokens(df_all_missing)
        assert result_all_missing["col"].isna().all()

    def test_no_modifications_to_input_frame(self):
        """Standardize does not mutate the input DataFrame"""
        df_original = pd.DataFrame({
            "col": ["NA", "value", "null"]
        })
        df_copy = df_original.copy()
        
        _ = ar.standardize_missing_tokens(df_original)
        
        pd.testing.assert_frame_equal(df_original, df_copy)

    def test_arframe_to_pandas_conversion_workflow(self, tmp_path):
        """Full workflow: ArFrame -> standardize -> to_pandas"""
        input_csv = tmp_path / "data.csv"
        input_csv.write_text(
            "product,quantity,price\n"
            "Widget,10,19.99\n"
            "Gadget,N/A,29.99\n"
            "Doohickey,5,null\n"
            "Thingamabob,?,39.99\n"
        )
        
        # Read as ArFrame
        frame = ar.read_csv(str(input_csv))
        
        # Standardize missing tokens
        clean_frame = ar.standardize_missing_tokens(frame)
        
        # Convert to pandas
        df = ar.to_pandas(clean_frame)
        
        # Verify results
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert pd.isna(df.iloc[1, 1])  # N/A
        assert pd.isna(df.iloc[2, 2])  # null
        assert pd.isna(df.iloc[3, 1])  # ?

    def test_tokens_non_string_item_raises_error(self):
        """TypeError raised when tokens list contains non-string items"""
        df = pd.DataFrame({"col": ["NA", "value"]})
        
        with pytest.raises(TypeError, match="tokens must be a list of strings"):
            ar.standardize_missing_tokens(df, tokens=["NA", 42, None])

"""
Pytest end-to-end system tests for arnio.cleaning functions.

Tests drive complete user workflows through arnio's public API:
- Real CSV files on disk using pytest's tmp_path
- Loading, cleaning, writing, and converting to pandas
- Error cases and edge cases with documented behavior
- Observable outputs: files, frames, error types/messages
"""

import sys
import csv
import tempfile
from pathlib import Path

import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCoalesceColumnsWorkflow:
    """Test coalesce_columns: select first non-null value from multiple columns."""

    def test_coalesce_basic_and_roundtrip(self, tmp_path):
        """Load CSV with partial data, coalesce to first non-null, write and verify."""
        csv_data = """primary,secondary,tertiary
,backup1,fallback1
value1,,fallback1
,backup2,
value3,backup3,fallback3"""
        
        csv_file = tmp_path / "contacts.csv"
        csv_file.write_text(csv_data)
        
        # Load and coalesce
        frame = ar.read_csv(csv_file)
        result = ar.coalesce_columns(
            frame,
            subset=["primary", "secondary", "tertiary"],
            output_column="best_contact"
        )
        
        # Verify results
        df = ar.to_pandas(result)
        assert "best_contact" in df.columns
        expected = ["backup1", "value1", "backup2", "value3"]
        assert df["best_contact"].tolist() == expected
        
        # Write to file and roundtrip
        output_file = tmp_path / "coalesced.csv"
        ar.write_csv(result, output_file)
        reloaded = ar.read_csv(output_file)
        reloaded_df = ar.to_pandas(reloaded)
        assert reloaded_df["best_contact"].tolist() == expected
        
        # Test error: missing column
        with pytest.raises(KeyError):
            ar.coalesce_columns(frame, subset=["primary", "nonexistent"])


class TestCombineColumnsWorkflow:
    """Test combine_columns: merge multiple columns with separator."""

    def test_combine_names_and_data_loss_on_nulls(self, tmp_path):
        """Combine first/last names; all-null rows produce null output."""
        csv_data = """first_name,last_name,age
John,Doe,30
Jane,Smith,25
,,,
Bob,,40"""
        
        csv_file = tmp_path / "names.csv"
        csv_file.write_text(csv_data)
        
        frame = ar.read_csv(csv_file)
        result = ar.combine_columns(
            frame,
            subset=["first_name", "last_name"],
            separator=" ",
            output_column="full_name"
        )
        
        df = ar.to_pandas(result)
        assert df["full_name"].iloc[0] == "John Doe"
        assert df["full_name"].iloc[1] == "Jane Smith"
        # Row 2: both nulls -> null output
        assert pd.isna(df["full_name"].iloc[2])
        assert df["full_name"].iloc[3] == "Bob "  # One value present
        
        # Verify pipeline integration
        output_file = tmp_path / "combined.csv"
        ar.write_csv(result, output_file)
        assert output_file.stat().st_size > 0


class TestDropColumnsMatchingWorkflow:
    """Test drop_columns_matching: remove columns by regex pattern."""

    def test_drop_temp_and_debug_columns(self, tmp_path):
        """Drop temporary/debug columns matching patterns."""
        csv_data = """id,name,temp_x,temp_y,debug_flag,value
1,Alice,0.1,0.2,yes,100
2,Bob,0.3,0.4,no,200"""
        
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(csv_data)
        
        frame = ar.read_csv(csv_file)
        result = ar.drop_columns_matching(frame, pattern="^(temp_|debug_)")
        
        df = ar.to_pandas(result)
        assert set(df.columns) == {"id", "name", "value"}
        
        # Test error: invalid regex
        with pytest.raises(Exception):
            ar.drop_columns_matching(frame, pattern="[invalid(")
        
        # Test error: pattern matches all columns
        with pytest.raises(ValueError, match="matches all columns"):
            ar.drop_columns_matching(frame, pattern=".*")


class TestDropConstantColumnsWorkflow:
    """Test drop_constant_columns: remove columns with single unique value."""

    def test_drop_constant_and_preserve_nonconstant(self, tmp_path):
        """Remove constant columns; preserve rows and varying columns."""
        csv_data = """id,status,country,value
1,active,USA,100
2,active,USA,200
3,active,USA,300"""
        
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(csv_data)
        
        frame = ar.read_csv(csv_file)
        result = ar.drop_constant_columns(frame)
        
        df = ar.to_pandas(result)
        assert "status" not in df.columns
        assert "country" not in df.columns
        assert set(df.columns) == {"id", "value"}
        assert len(df) == 3  # Rows preserved
        
        # Verify empty frame edge case
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("a,b\n")
        empty_frame = ar.read_csv(empty_csv)
        empty_result = ar.drop_constant_columns(empty_frame)
        empty_df = ar.to_pandas(empty_result)
        assert len(empty_df) == 0


class TestDropNullsWorkflow:
    """Test drop_nulls: remove rows with nulls in subset columns."""

    def test_drop_nulls_subset_and_full(self, tmp_path):
        """Drop null rows in specific columns; verify counts decrease."""
        csv_data = """id,email,phone,notes
1,alice@example.com,555-1234,complete
2,,555-5678,no email
3,bob@example.com,,no phone
4,charlie@example.com,555-9999,
5,diana@example.com,555-0000,complete"""
        
        csv_file = tmp_path / "contacts.csv"
        csv_file.write_text(csv_data)
        
        frame = ar.read_csv(csv_file)
        
        # Drop rows with nulls in email or phone
        result = ar.drop_nulls(frame, subset=["email", "phone"])
        df = ar.to_pandas(result)
        assert len(df) == 3  # Rows 1, 4, 5
        assert df["id"].tolist() == [1, 4, 5]
        
        # Drop all rows with any null
        result_all = ar.drop_nulls(frame)
        df_all = ar.to_pandas(result_all)
        assert len(df_all) == 2  # Only rows 1, 5 have no nulls
        
        # Write output
        output_file = tmp_path / "complete.csv"
        ar.write_csv(result, output_file)
        final_df = pd.read_csv(output_file)
        assert len(final_df) == 3


class TestFilterRowsWorkflow:
    """Test filter_rows: keep rows matching column conditions."""

    def test_filter_operators_gt_lt_eq_ne(self, tmp_path):
        """Filter with >, <, >=, <=, ==, != operators."""
        csv_data = """id,score,status
1,85,pass
2,42,fail
3,91,pass
4,65,pass
5,30,fail"""
        
        csv_file = tmp_path / "scores.csv"
        csv_file.write_text(csv_data)
        
        frame = ar.read_csv(csv_file)
        
        # Test >
        result_gt = ar.filter_rows(frame, column="score", op=">", value=80)
        assert len(ar.to_pandas(result_gt)) == 2
        
        # Test >=
        result_gte = ar.filter_rows(frame, column="score", op=">=", value=65)
        assert len(ar.to_pandas(result_gte)) == 3
        
        # Test ==
        result_eq = ar.filter_rows(frame, column="status", op="==", value="pass")
        assert len(ar.to_pandas(result_eq)) == 3
        
        # Test !=
        result_ne = ar.filter_rows(frame, column="status", op="!=", value="fail")
        assert len(ar.to_pandas(result_ne)) == 3
        
        # Verify results are ArFrame-compatible
        output_file = tmp_path / "filtered.csv"
        ar.write_csv(result_gte, output_file)
        reloaded = ar.read_csv(output_file)
        assert len(reloaded.columns) == 3


class TestKeepRowsWithNullsWorkflow:
    """Test keep_rows_with_nulls: audit incomplete rows."""

    def test_keep_incomplete_rows_global_and_subset(self, tmp_path):
        """Keep only rows with at least one null; test subset parameter."""
        csv_data = """id,name,email,phone
1,Alice,alice@example.com,555-1234
2,Bob,,555-5678
3,Charlie,charlie@example.com,
4,Diana,diana@example.com,555-9999
5,Eve,,"""
        
        csv_file = tmp_path / "incomplete.csv"
        csv_file.write_text(csv_data)
        
        frame = ar.read_csv(csv_file)
        
        # Keep rows with any null
        result_any = ar.keep_rows_with_nulls(frame)
        df_any = ar.to_pandas(result_any)
        assert len(df_any) == 3
        assert df_any["id"].tolist() == [2, 3, 5]
        
        # Keep rows with null in specific column
        result_email = ar.keep_rows_with_nulls(frame, subset=["email"])
        df_email = ar.to_pandas(result_email)
        assert len(df_email) == 2  # Rows 2, 5
        assert all(pd.isna(df_email["email"]))
        
        # Test error: subset as string instead of list
        with pytest.raises(TypeError, match="must be a list"):
            ar.keep_rows_with_nulls(frame, subset="email")


class TestReplaceValuesWorkflow:
    """Test replace_values: substitute values using mapping."""

    def test_replace_in_column_and_null_handling(self, tmp_path):
        """Replace values in specific column; handle null keys/values."""
        csv_data = """id,status,priority
1,active,high
2,inactive,low
3,active,medium
4,inactive,high"""
        
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(csv_data)
        
        frame = ar.read_csv(csv_file)
        
        # Replace in column
        result = ar.replace_values(
            frame,
            mapping={"active": "on", "inactive": "off"},
            column="status"
        )
        df = ar.to_pandas(result)
        assert df["status"].tolist() == ["on", "off", "on", "off"]
        
        # Replace null values
        frame_null = ar.from_pandas(pd.DataFrame({
            "value": ["a", None, "b", None]
        }))
        result_null = ar.replace_values(
            frame_null,
            mapping={None: "MISSING"}
        )
        df_null = ar.to_pandas(result_null)
        assert df_null["value"].tolist() == ["a", "MISSING", "b", "MISSING"]


class TestSafeDivideColumnsWorkflow:
    """Test safe_divide_columns: divide with zero/null protection."""

    def test_safe_divide_handles_zero_and_nulls(self, tmp_path):
        """Divide safely; zero/null denominators use fill_value."""
        csv_data = """id,revenue,cost
1,1000,100
2,500,0
3,0,50
4,2000,200"""
        
        csv_file = tmp_path / "finance.csv"
        csv_file.write_text(csv_data)
        
        frame = ar.read_csv(csv_file)
        
        # Safe divide
        result = ar.safe_divide_columns(
            frame,
            numerator="revenue",
            denominator="cost",
            output_column="ratio",
            fill_value=0.0
        )
        
        df = ar.to_pandas(result)
        assert "ratio" in df.columns
        assert df["ratio"].iloc[0] == 10.0  # 1000/100
        assert df["ratio"].iloc[1] == 0.0   # 500/0 -> fill_value
        assert df["ratio"].iloc[3] == 10.0  # 2000/200
        
        # Verify file output
        output_file = tmp_path / "with_ratio.csv"
        ar.write_csv(result, output_file)
        reloaded_df = pd.read_csv(output_file)
        assert "ratio" in reloaded_df.columns


class TestStandardizeMissingTokensWorkflow:
    """Test standardize_missing_tokens: normalize null-like strings."""

    def test_standardize_common_null_formats(self, tmp_path):
        """Convert N/A, NA, none, null, - etc. to NaN."""
        csv_data = """id,name,status,notes
1,Alice,active,
2,Bob,N/A,no data
3,Charlie,active,none
4,Diana,NA,-
5,Eve,active,NULL"""
        
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text(csv_data)
        
        frame = ar.read_csv(csv_file)
        result = ar.standardize_missing_tokens(frame)
        
        df = ar.to_pandas(result)
        
        # Verify null tokens are now NaN
        assert pd.isna(df["status"].iloc[1])  # N/A
        assert pd.isna(df["notes"].iloc[2])   # none
        assert pd.isna(df["status"].iloc[3])  # NA
        assert pd.isna(df["notes"].iloc[3])   # -
        assert pd.isna(df["notes"].iloc[4])   # NULL
        
        # Active and Alice remain unchanged
        assert df["status"].iloc[0] == "active"
        assert df["name"].iloc[0] == "Alice"
        
        # Test custom tokens
        frame2 = ar.from_pandas(pd.DataFrame({
            "value": ["keep", "IGNORE", "keep"]
        }))
        result2 = ar.standardize_missing_tokens(
            frame2,
            tokens=["IGNORE"]
        )
        df2 = ar.to_pandas(result2)
        assert pd.isna(df2["value"].iloc[1])


class TestPipelineIntegration:
    """Test combining multiple cleaning steps."""

    def test_pipeline_multiple_steps_csv_to_csv(self, tmp_path):
        """Chain multiple cleaning operations in one pipeline."""
        csv_data = """id,name,status,score,temp_col
1, Alice ,active,85,x
2, Bob ,inactive,42,y
3, Charlie ,active,91,z"""
        
        csv_file = tmp_path / "raw.csv"
        csv_file.write_text(csv_data)
        
        frame = ar.read_csv(csv_file)
        
        # Multi-step pipeline
        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_columns_matching", {"pattern": "^temp_"}),
            ("filter_rows", {"column": "score", "op": ">", "value": 50}),
        ])
        
        df = ar.to_pandas(result)
        assert "temp_col" not in df.columns
        assert len(df) == 2  # Only rows with score > 50
        assert df["name"].iloc[0] == "Alice"
        
        # Write output
        output_file = tmp_path / "processed.csv"
        ar.write_csv(result, output_file)
        reloaded = ar.read_csv(output_file)
        assert len(reloaded.columns) == 4  # id, name, status, score


class TestPandasAccessorIntegration:
    """Test pandas .arnio accessor methods."""

    def test_accessor_clean_method(self):
        """Use .arnio.clean() accessor on pandas DataFrame."""
        df = pd.DataFrame({
            "name": [" Alice ", " Bob ", " Alice "],
            "constant": [1, 1, 1],
        })
        
        # Use accessor to clean
        result = df.arnio.clean(
            strip_whitespace=True,
            drop_duplicates=True
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Duplicates removed
        assert result["name"].iloc[0] == "Alice"


class TestFullRoundtripAndEdgeCases:
    """Test complete end-to-end workflows and edge cases."""

    def test_complete_data_cleaning_roundtrip(self, tmp_path):
        """Full workflow: CSV → clean → analysis → write → verify."""
        csv_data = """order_id,customer,city,status,amount
1, John Doe , Paris ,active,100.50
2, Jane Smith , London ,N/A,200.00
3, John Doe , Paris ,active,100.50
4,Bob Brown,,inactive,0
5, Diana Lee , Tokyo ,active,"""
        
        csv_file = tmp_path / "orders.csv"
        csv_file.write_text(csv_data)
        
        # Load
        frame = ar.read_csv(csv_file)
        
        # Clean pipeline
        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("standardize_missing_tokens",),
            ("drop_duplicates",),
        ])
        
        # Analyze with pandas
        df = ar.to_pandas(result)
        assert len(df) == 4  # Duplicate row removed
        assert df["customer"].iloc[0] == "John Doe"
        assert pd.isna(df["status"].iloc[1])  # N/A standardized
        
        # Combine names for final output
        final = ar.combine_columns(
            result,
            subset=["customer"],
            separator="|",
            output_column="contact"
        )
        
        # Write final output
        output_file = tmp_path / "cleaned_orders.csv"
        ar.write_csv(final, output_file)
        
        # Verify file was created and can be re-read
        reloaded = pd.read_csv(output_file)
        assert len(reloaded) == 4
        assert "contact" in reloaded.columns

"""
System (end-to-end) tests for arnio column selection/dropping functions.

Tests drive real workflows: load CSV → select/drop columns → write to CSV or convert to pandas.
"""

import pytest
import pandas as pd
import arnio as ar
from pathlib import Path


class TestDropColumns:
    """Tests for ArFrame.drop_columns"""

    def test_drop_columns_basic_workflow(self, tmp_path):
        """Drop columns and write result to CSV (end-to-end workflow)"""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("id,name,score,debug_flag\n1,Alice,95,true\n2,Bob,87,false\n")
        
        frame = ar.read_csv(input_csv)
        trimmed = frame.drop_columns(["debug_flag"])
        
        # Write to CSV
        output_csv = tmp_path / "output.csv"
        ar.write_csv(trimmed, output_csv)
        
        # Verify file contents
        output_text = output_csv.read_text()
        assert "debug_flag" not in output_text
        assert "id,name,score" in output_text
        
        # Verify column order preserved
        assert trimmed.columns == ["id", "name", "score"]

    def test_drop_columns_tuple_input(self, tmp_path):
        """Drop columns accepts tuple and returns new frame"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c,d\n1,2,3,4\n")
        
        frame = ar.read_csv(csv_file)
        result = frame.drop_columns(("b", "d"))
        df = ar.to_pandas(result)
        
        assert list(df.columns) == ["a", "c"]
        assert df.shape[1] == 2

    def test_drop_columns_empty_list_returns_copy(self, tmp_path):
        """Empty drop list returns unchanged copy of frame"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x,y,z\n1,2,3\n")
        
        frame = ar.read_csv(csv_file)
        result = frame.drop_columns([])
        
        assert result.columns == ["x", "y", "z"]
        assert result._frame != frame._frame  # Different object

    def test_drop_columns_error_missing_column(self, tmp_path):
        """Missing column raises ValueError with helpful message"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(csv_file)
        with pytest.raises(ValueError, match="Unknown column"):
            frame.drop_columns(["missing"])

    def test_drop_columns_error_wrong_type(self, tmp_path):
        """String input instead of list raises TypeError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(csv_file)
        with pytest.raises(TypeError, match="must be a list or tuple"):
            frame.drop_columns("a")

    def test_drop_columns_error_remove_all_columns(self, tmp_path):
        """Cannot drop all columns from frame"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(csv_file)
        with pytest.raises(ValueError, match="cannot remove all columns"):
            frame.drop_columns(["a", "b"])


class TestSelectColumns:
    """Tests for ArFrame.select_columns"""

    def test_select_columns_basic_workflow(self, tmp_path):
        """Select columns and validate pipeline (end-to-end workflow)"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("user_id,email,age,city,revenue\n1,alice@test.com,30,NYC,5000\n2,bob@test.com,25,LA,3000\n")
        
        frame = ar.read_csv(csv_file)
        
        # Select only email and revenue
        selected = frame.select_columns(["email", "revenue"])
        
        # Validate schema on selected columns
        schema = ar.Schema({
            "email": ar.Email(nullable=False),
            "revenue": ar.Float64(nullable=False, min=0),
        })
        result = ar.validate(selected, schema)
        
        assert result.passed
        assert selected.columns == ["email", "revenue"]

    def test_select_columns_reorders_columns(self, tmp_path):
        """Selection order overrides original column order"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c,d\n1,2,3,4\n")
        
        frame = ar.read_csv(csv_file)
        result = frame.select_columns(["d", "b", "a"])
        df = ar.to_pandas(result)
        
        assert list(df.columns) == ["d", "b", "a"]

    def test_select_columns_error_empty_selection(self, tmp_path):
        """Empty column list raises ValueError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(csv_file)
        with pytest.raises(ValueError, match="cannot be empty"):
            frame.select_columns([])

    def test_select_columns_error_duplicates(self, tmp_path):
        """Duplicate column names in selection raises ValueError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3\n")
        
        frame = ar.read_csv(csv_file)
        with pytest.raises(ValueError, match="Duplicate"):
            frame.select_columns(["a", "b", "a"])

    def test_select_columns_error_unknown_column(self, tmp_path):
        """Unknown column raises ValueError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(csv_file)
        with pytest.raises(ValueError, match="Unknown columns"):
            frame.select_columns(["a", "missing"])

    def test_select_columns_error_string_not_list(self, tmp_path):
        """String input (not list) raises TypeError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(csv_file)
        with pytest.raises(TypeError, match="not a string"):
            frame.select_columns("a")


class TestSelectDtypes:
    """Tests for ArFrame.select_dtypes"""

    def test_select_dtypes_include_workflow(self, tmp_path):
        """Select numeric columns and profile (end-to-end workflow)"""
        csv_file = tmp_path / "data.csv"
        # Columns will be inferred as: id=int64, name=string, score=float64
        csv_file.write_text("id,name,score\n1,Alice,95.5\n2,Bob,87.0\n")
        
        frame = ar.read_csv(csv_file)
        
        # Select only numeric columns
        numeric = frame.select_dtypes(include=["int64", "float64"])
        
        # Profile the numeric subset
        report = ar.profile(numeric)
        
        assert "name" not in numeric.columns
        assert report.row_count == 2
        assert report.column_count == 2

    def test_select_dtypes_exclude_workflow(self, tmp_path):
        """Exclude string columns (end-to-end workflow)"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,active\n1,Alice,true\n2,Bob,false\n")
        
        frame = ar.read_csv(csv_file)
        
        # Exclude string columns
        result = frame.select_dtypes(exclude="string")
        df = ar.to_pandas(result)
        
        # name is string, so should be excluded
        assert "name" not in df.columns
        assert df.shape[1] >= 1  # At least some columns remain

    def test_select_dtypes_error_no_arguments(self, tmp_path):
        """Missing both include and exclude raises ValueError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(csv_file)
        with pytest.raises(ValueError, match="at least one"):
            frame.select_dtypes()

    def test_select_dtypes_error_overlap(self, tmp_path):
        """Include and exclude overlap raises ValueError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(csv_file)
        with pytest.raises(ValueError, match="overlap"):
            frame.select_dtypes(include="int64", exclude="int64")

    def test_select_dtypes_error_unknown_dtype(self, tmp_path):
        """Unknown dtype string raises ValueError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(csv_file)
        with pytest.raises(ValueError, match="Unrecognised dtype"):
            frame.select_dtypes(include="unknown_dtype")


class TestIntegration:
    """Integration tests combining column operations with pipelines"""

    def test_drop_select_pipeline_full_workflow(self, tmp_path):
        """Drop columns → select columns → clean → write (full workflow)"""
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"
        
        input_csv.write_text(
            "id, name , email , score , debug_col\n"
            "1, Alice , alice@test.com , 95 , X\n"
            "2, Bob , bob@test.com , 87 , Y\n"
        )
        
        # Load and process
        frame = ar.read_csv(input_csv)
        
        # Remove debug column
        without_debug = frame.drop_columns(["debug_col"])
        
        # Select only name and email
        final = without_debug.select_columns(["name", "email"])
        
        # Clean and write
        clean = ar.pipeline(final, [("strip_whitespace",)])
        ar.write_csv(clean, output_csv)
        
        # Verify output
        output_text = output_csv.read_text()
        assert "debug_col" not in output_text
        assert "score" not in output_text
        assert "name" in output_text
        assert "email" in output_text
        # Check whitespace was trimmed
        assert "Alice" in output_text  # Not " Alice "

    def test_select_dtypes_then_validate(self, tmp_path):
        """Select dtypes → apply schema validation (workflow)"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "user_id,score,notes\n"
            "1,95,good\n"
            "2,87,excellent\n"
        )
        
        frame = ar.read_csv(csv_file)
        
        # Select only numeric columns
        numeric = frame.select_dtypes(include=["int64", "float64"])
        
        # Create schema for numeric columns only
        schema = ar.Schema({
            "user_id": ar.Int64(nullable=False, min=1),
            "score": ar.Float64(nullable=False, min=0, max=100),
        })
        
        result = ar.validate(numeric, schema)
        assert result.passed
        assert "notes" not in numeric.columns

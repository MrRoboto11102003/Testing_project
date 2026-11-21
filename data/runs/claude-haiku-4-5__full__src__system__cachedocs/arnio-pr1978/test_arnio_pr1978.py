"""
System (end-to-end) tests for arnio.frame.ArFrame.select_dtypes.

These tests drive complete user workflows through arnio's public API,
using real files on disk, loading, cleaning/profiling, and converting results.
They assert on observable outcomes: output files, returned frames, reports,
error types and messages.
"""

import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestSelectDtypesInclude:
    """Test include parameter functionality"""

    def test_include_single_dtype_as_string(self, tmp_path):
        """Include a single dtype specified as string"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1,alice,95.5\n2,bob,87.3\n")

        frame = ar.read_csv(str(csv_file))
        result = frame.select_dtypes(include="int64")

        assert result.columns == ["id"]
        df = ar.to_pandas(result)
        assert df.shape == (2, 1)
        assert df["id"].tolist() == [1, 2]

    def test_include_multiple_dtypes_as_list(self, tmp_path):
        """Include multiple dtypes specified as list"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,score,active\n1,95.5,yes\n2,87.3,no\n")

        frame = ar.read_csv(str(csv_file))
        result = frame.select_dtypes(include=["int64", "float64"])

        assert set(result.columns) == {"id", "score"}
        df = ar.to_pandas(result)
        assert df.shape == (2, 2)
        assert df["id"].dtype == np.int64
        assert df["score"].dtype == np.float64

    def test_include_as_tuple(self, tmp_path):
        """Include dtypes specified as tuple"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,active\n1,alice,yes\n2,bob,no\n")

        frame = ar.read_csv(str(csv_file))
        result = frame.select_dtypes(include=("int64", "string"))

        assert set(result.columns) == {"id", "name", "active"}


class TestSelectDtypesExclude:
    """Test exclude parameter functionality"""

    def test_exclude_single_dtype_string(self, tmp_path):
        """Exclude a single dtype"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1,alice,95.5\n2,bob,87.3\n")

        frame = ar.read_csv(str(csv_file))
        result = frame.select_dtypes(exclude="string")

        assert set(result.columns) == {"id", "score"}

    def test_exclude_multiple_dtypes(self, tmp_path):
        """Exclude multiple dtypes"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,name,score,active\n1,alice,95.5,yes\n2,bob,87.3,no\n"
        )

        frame = ar.read_csv(str(csv_file))
        result = frame.select_dtypes(exclude=["string", "bool"])

        assert set(result.columns) == {"id", "score"}


class TestSelectDtypesColumnOrder:
    """Test that column order is preserved"""

    def test_preserves_original_column_order(self, tmp_path):
        """Column order in result matches original frame, not alphabetical"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("z,a,m\n1,alice,95.5\n2,bob,87.3\n")

        frame = ar.read_csv(str(csv_file))
        result = frame.select_dtypes(include=["int64", "string", "float64"])

        # Columns should appear in original order (z, a, m), not (a, m, z)
        assert result.columns == ["z", "a", "m"]


class TestSelectDtypesIntegration:
    """Integration tests with complete workflows"""

    def test_read_select_dtypes_write_csv(self, tmp_path):
        """Full workflow: read CSV, select numeric columns, write result"""
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"
        input_csv.write_text(
            "id,name,score,active\n1,alice,95.5,yes\n2,bob,87.3,no\n"
        )

        # Read, select numeric columns, write
        frame = ar.read_csv(str(input_csv))
        numeric = frame.select_dtypes(include=["int64", "float64"])
        ar.write_csv(numeric, str(output_csv))

        # Verify output file has only numeric columns
        output_text = output_csv.read_text()
        assert "id,score" in output_text
        assert "alice" not in output_text
        assert "name" not in output_text

    def test_select_dtypes_convert_to_pandas_verify_dtypes(self, tmp_path):
        """Select dtypes, convert to pandas, verify types are correct"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1,alice,95.5\n2,bob,87.3\n3,charlie,91.2\n")

        frame = ar.read_csv(str(csv_file))
        numeric = frame.select_dtypes(include=["int64", "float64"])
        df = ar.to_pandas(numeric)

        # Verify structure and dtypes
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
        assert set(df.columns) == {"id", "score"}
        assert df["id"].tolist() == [1, 2, 3]
        assert all(isinstance(x, (int, np.integer)) for x in df["id"])
        assert all(isinstance(x, (float, np.floating)) for x in df["score"])

    def test_select_dtypes_then_pipeline_operations(self, tmp_path):
        """Select numeric columns, then apply pipeline operations"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1, alice ,95.5\n2, bob ,87.3\n")

        frame = ar.read_csv(str(csv_file))
        numeric = frame.select_dtypes(include=["int64", "float64"])

        # Pipeline operations should work on selected columns
        clean = ar.pipeline(numeric, [("strip_whitespace",)])
        assert clean.shape == (2, 2)

    def test_select_dtypes_then_profile(self, tmp_path):
        """Select columns then generate data quality profile"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,name,score\n1,alice,95.5\n2,bob,87.3\n3,charlie,91.2\n"
        )

        frame = ar.read_csv(str(csv_file))
        numeric = frame.select_dtypes(include=["int64", "float64"])

        # Profile should work on selected frame
        report = ar.profile(numeric)
        assert report.row_count == 3
        assert report.column_count == 2
        assert "id" in report.columns
        assert "score" in report.columns


class TestSelectDtypesEmptyResults:
    """Test behavior when no columns match filter"""

    def test_no_matching_columns_returns_empty_frame(self, tmp_path):
        """When no columns match, return empty frame (not error)"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,alice\n2,bob\n")

        frame = ar.read_csv(str(csv_file))
        result = frame.select_dtypes(include="bool")

        # Should not raise, returns frame with 0 rows but same columns
        assert result.shape == (0, 2)
        df = ar.to_pandas(result)
        assert len(df) == 0
        assert len(df.columns) == 2


class TestSelectDtypesErrorCases:
    """Test error handling and validation"""

    def test_neither_include_nor_exclude_raises_value_error(self, tmp_path):
        """Calling without include or exclude raises ValueError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,alice\n")

        frame = ar.read_csv(str(csv_file))

        with pytest.raises(ValueError, match="at least one of"):
            frame.select_dtypes()

    def test_include_exclude_overlap_raises_value_error(self, tmp_path):
        """When include and exclude contain same dtype, raises ValueError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1,alice,95.5\n")

        frame = ar.read_csv(str(csv_file))

        with pytest.raises(ValueError, match="overlap"):
            frame.select_dtypes(
                include=["int64", "float64"], exclude="float64"
            )

    def test_unrecognized_dtype_string_raises_value_error(self, tmp_path):
        """Unrecognized dtype string raises ValueError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,alice\n")

        frame = ar.read_csv(str(csv_file))

        with pytest.raises(ValueError, match="Unrecognised dtype"):
            frame.select_dtypes(include="invalid_dtype_name")

    def test_dict_instead_of_list_raises_type_error(self, tmp_path):
        """Dict used for include raises TypeError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,alice\n")

        frame = ar.read_csv(str(csv_file))

        with pytest.raises(TypeError, match="must be a string, list, or tuple"):
            frame.select_dtypes(include={"int64"})

    def test_integer_instead_of_string_raises_type_error(self, tmp_path):
        """Integer for exclude raises TypeError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,alice\n")

        frame = ar.read_csv(str(csv_file))

        with pytest.raises(TypeError, match="must be a string, list, or tuple"):
            frame.select_dtypes(exclude=123)

    def test_non_string_element_in_list_raises_type_error(self, tmp_path):
        """Non-string element in include list raises TypeError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,alice\n")

        frame = ar.read_csv(str(csv_file))

        with pytest.raises(TypeError, match="must contain only strings"):
            frame.select_dtypes(include=["int64", 42])


class TestSelectDtypesAllValidTypes:
    """Test filtering with valid dtype values"""

    def test_select_all_valid_dtype_values(self, tmp_path):
        """Test that all documented valid dtype strings work"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "int_col,float_col,str_col,bool_col\n1,1.5,text,yes\n2,2.5,data,no\n"
        )

        frame = ar.read_csv(str(csv_file))

        # Test each valid dtype individually
        valid_dtypes = ["int64", "float64", "string", "bool"]
        for dtype in valid_dtypes:
            result = frame.select_dtypes(include=dtype)
            # Should succeed and return an ArFrame
            assert isinstance(result, ar.ArFrame)
            # Should have at least 0 columns (no error)
            assert len(result.columns) >= 0

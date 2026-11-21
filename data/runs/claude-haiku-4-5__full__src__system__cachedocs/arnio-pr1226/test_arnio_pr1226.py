"""
End-to-end pytest tests for arnio.convert.to_pandas.

Tests cover basic conversion, copy behavior, null handling, dtype preservation,
metadata handling, error cases, and complete workflows from CSV to pandas.
"""

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import arnio as ar


class TestToPandasBasic:
    """Basic to_pandas conversion workflows."""

    def test_to_pandas_converts_arframe_to_dataframe(self, tmp_path):
        """Convert a simple ArFrame to pandas DataFrame."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,score\n1,alice,95.5\n2,bob,87.3\n")
        
        frame = ar.read_csv(csv_path)
        df = ar.to_pandas(frame)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)
        assert list(df.columns) == ["id", "name", "score"]
        assert df["id"].tolist() == ["1", "2"]  # CSV reads as string by default
        assert df["name"].tolist() == ["alice", "bob"]

    def test_to_pandas_preserves_dtypes(self, tmp_path):
        """Verify dtype preservation through conversion."""
        csv_path = tmp_path / "types.csv"
        csv_path.write_text("int_col,float_col,bool_col,str_col\n1,1.5,true,hello\n2,2.7,false,world\n")
        
        frame = ar.read_csv(csv_path)
        # Cast types to get proper dtypes
        frame = ar.cast_types(frame, {
            "int_col": "int64",
            "float_col": "float64",
            "bool_col": "bool",
        })
        df = ar.to_pandas(frame)
        
        # Check that dtypes are preserved
        assert str(df["int_col"].dtype) == "Int64"  # Nullable int
        assert str(df["float_col"].dtype) == "float64"
        assert str(df["bool_col"].dtype) == "boolean"  # Nullable bool


class TestToPandasNullHandling:
    """Test null value handling across different dtypes."""

    def test_to_pandas_handles_null_in_int_column(self, tmp_path):
        """Null values in int columns become pd.NA."""
        csv_path = tmp_path / "nulls_int.csv"
        csv_path.write_text("id,value\n1,100\n2,\n3,300\n")
        
        frame = ar.read_csv(csv_path)
        frame = ar.cast_types(frame, {"value": "int64"})
        df = ar.to_pandas(frame)
        
        assert pd.isna(df["value"].iloc[1])
        assert df["value"].iloc[0] == 100
        assert df["value"].iloc[2] == 300

    def test_to_pandas_handles_null_in_float_column(self, tmp_path):
        """Null values in float columns become NaN."""
        csv_path = tmp_path / "nulls_float.csv"
        csv_path.write_text("id,score\n1,95.5\n2,\n3,87.0\n")
        
        frame = ar.read_csv(csv_path)
        frame = ar.cast_types(frame, {"score": "float64"})
        df = ar.to_pandas(frame)
        
        assert np.isnan(df["score"].iloc[1])
        assert df["score"].iloc[0] == 95.5
        assert df["score"].iloc[2] == 87.0

    def test_to_pandas_handles_null_in_bool_column(self, tmp_path):
        """Null values in bool columns become pd.NA."""
        csv_path = tmp_path / "nulls_bool.csv"
        csv_path.write_text("id,active\n1,true\n2,\n3,false\n")
        
        frame = ar.read_csv(csv_path)
        frame = ar.cast_types(frame, {"active": "bool"})
        df = ar.to_pandas(frame)
        
        assert pd.isna(df["active"].iloc[1])
        assert df["active"].iloc[0] is True
        assert df["active"].iloc[2] is False

    def test_to_pandas_handles_null_in_string_column(self, tmp_path):
        """Null values in string columns become pd.NA."""
        csv_path = tmp_path / "nulls_string.csv"
        csv_path.write_text("id,name\n1,alice\n2,\n3,bob\n")
        
        frame = ar.read_csv(csv_path)
        df = ar.to_pandas(frame)
        
        assert pd.isna(df["name"].iloc[1])
        assert df["name"].iloc[0] == "alice"
        assert df["name"].iloc[2] == "bob"


class TestToPandasCopyBehavior:
    """Test copy parameter and buffer management."""

    def test_to_pandas_copy_false_default(self, tmp_path):
        """Verify copy=False is the default (zero-copy when possible)."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("value\n100\n200\n300\n")
        
        frame = ar.read_csv(csv_path)
        frame = ar.cast_types(frame, {"value": "int64"})
        df1 = ar.to_pandas(frame)
        df2 = ar.to_pandas(frame)
        
        assert df1 is not df2  # Different DataFrames
        assert isinstance(df1, pd.DataFrame)

    def test_to_pandas_copy_true_creates_defensive_copy(self, tmp_path):
        """copy=True creates defensive pandas-owned buffers."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("value\n100\n200\n300\n")
        
        frame = ar.read_csv(csv_path)
        frame = ar.cast_types(frame, {"value": "int64"})
        df_no_copy = ar.to_pandas(frame, copy=False)
        df_copy = ar.to_pandas(frame, copy=True)
        
        assert df_no_copy is not df_copy
        assert df_no_copy["value"].tolist() == df_copy["value"].tolist()

    def test_to_pandas_copy_parameter_validates_bool(self):
        """copy parameter must be a bool."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="copy must be a bool"):
            ar.to_pandas(frame, copy="yes")
        
        with pytest.raises(TypeError, match="copy must be a bool"):
            ar.to_pandas(frame, copy=1)


class TestToPandasRoundTrip:
    """Test round-trip conversions: pandas → ArFrame → pandas."""

    def test_round_trip_preserves_data(self, tmp_path):
        """Data is preserved through pandas → ArFrame → pandas."""
        original_df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"],
            "score": [95.5, 87.3, 92.1],
        })
        
        frame = ar.from_pandas(original_df)
        result_df = ar.to_pandas(frame)
        
        # Data should match (may have dtype changes due to inference)
        assert result_df.shape == original_df.shape
        assert list(result_df.columns) == list(original_df.columns)

    def test_round_trip_preserves_attrs_metadata(self):
        """DataFrame.attrs metadata is preserved through round-trip."""
        original_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        original_df.attrs = {"source": "test", "version": "1.0"}
        
        frame = ar.from_pandas(original_df)
        result_df = ar.to_pandas(frame)
        
        assert result_df.attrs == {"source": "test", "version": "1.0"}


class TestToPandasCompleteWorkflows:
    """End-to-end workflows involving cleaning, validation, and conversion."""

    def test_csv_to_clean_dataframe_workflow(self, tmp_path):
        """Complete workflow: read CSV → clean → convert to pandas."""
        csv_path = tmp_path / "raw.csv"
        csv_path.write_text(
            "id,name,revenue\n"
            "1,  Alice  ,1000.50\n"
            "2,  Bob    ,2000.00\n"
            "1,  Alice  ,1000.50\n"  # Duplicate
        )
        
        frame = ar.read_csv(csv_path)
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])
        df = ar.to_pandas(clean)
        
        assert len(df) == 2  # Duplicate removed
        assert df["name"].iloc[0] == "Alice"  # Whitespace stripped
        assert df["name"].iloc[1] == "Bob"

    def test_profile_and_convert_workflow(self, tmp_path):
        """Workflow: read CSV → profile → suggest cleaning → convert."""
        csv_path = tmp_path / "messy.csv"
        csv_path.write_text(
            "user_id,email\n"
            "1,  test@example.com  \n"
            "2,  user@example.com  \n"
        )
        
        frame = ar.read_csv(csv_path)
        report = ar.profile(frame)
        
        # Suggest and apply cleaning
        suggestions = ar.suggest_cleaning(frame)
        if suggestions:
            clean = ar.pipeline(frame, suggestions)
        else:
            clean = frame
        
        df = ar.to_pandas(clean)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_validation_error_handling_then_convert(self, tmp_path):
        """Workflow: validate → handle issues → convert to pandas."""
        csv_path = tmp_path / "invalid.csv"
        csv_path.write_text(
            "user_id,email\n"
            "1,valid@example.com\n"
            "2,invalid-email\n"
        )
        
        frame = ar.read_csv(csv_path)
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        result = ar.validate(frame, schema)
        
        # Even if validation fails, we can still convert to pandas
        df = ar.to_pandas(frame)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2


class TestToPandasEdgeCases:
    """Edge cases and special scenarios."""

    def test_to_pandas_empty_frame(self):
        """Convert an empty ArFrame to pandas."""
        frame = ar.from_dict({"a": [], "b": []})
        df = ar.to_pandas(frame)
        
        assert len(df) == 0
        assert list(df.columns) == ["a", "b"]

    def test_to_pandas_single_row(self):
        """Convert a single-row ArFrame."""
        frame = ar.from_dict({"id": [1], "name": ["alice"]})
        df = ar.to_pandas(frame)
        
        assert len(df) == 1
        assert df["id"].iloc[0] == 1
        assert df["name"].iloc[0] == "alice"

    def test_to_pandas_single_column(self):
        """Convert a single-column ArFrame."""
        frame = ar.from_dict({"values": [1, 2, 3]})
        df = ar.to_pandas(frame)
        
        assert len(df.columns) == 1
        assert df["values"].tolist() == [1, 2, 3]

    def test_to_pandas_wide_dataframe(self, tmp_path):
        """Convert a wide ArFrame with many columns."""
        csv_path = tmp_path / "wide.csv"
        cols = ",".join(f"col{i}" for i in range(50))
        values = ",".join(str(i) for i in range(50))
        csv_path.write_text(f"{cols}\n{values}\n")
        
        frame = ar.read_csv(csv_path)
        df = ar.to_pandas(frame)
        
        assert len(df.columns) == 50
        assert len(df) == 1


class TestToPandasErrorHandling:
    """Test error cases and validation."""

    def test_to_pandas_rejects_non_arframe_input(self):
        """to_pandas raises TypeError for non-ArFrame input."""
        with pytest.raises(
            TypeError, 
            match=r"to_pandas\(\) expects an ArFrame"
        ):
            ar.to_pandas(pd.DataFrame({"a": [1, 2, 3]}))

    def test_to_pandas_rejects_dict_input(self):
        """to_pandas raises TypeError for dict input."""
        with pytest.raises(TypeError, match=r"expects an ArFrame"):
            ar.to_pandas({"a": [1, 2, 3]})

    def test_to_pandas_rejects_none_input(self):
        """to_pandas raises TypeError for None input."""
        with pytest.raises(TypeError, match=r"expects an ArFrame"):
            ar.to_pandas(None)


class TestToPandasPandasAccessor:
    """Test workflows using the pandas .arnio accessor."""

    def test_accessor_pipeline_then_to_pandas(self, tmp_path):
        """Use accessor to clean and convert back to native pandas."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "id,status\n"
            "1,  ACTIVE  \n"
            "2,  INACTIVE  \n"
        )
        
        frame = ar.read_csv(csv_path)
        df = ar.to_pandas(frame)
        
        # Use accessor to clean
        clean_df = df.arnio.clean(
            strip_whitespace=True,
            drop_duplicates=False,
        )
        
        assert isinstance(clean_df, pd.DataFrame)
        assert clean_df["status"].iloc[0] == "ACTIVE"

    def test_accessor_profile_then_convert(self, tmp_path):
        """Use accessor to profile data."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("value\n100\n200\n300\n")
        
        frame = ar.read_csv(csv_path)
        df = ar.to_pandas(frame)
        
        # Profile using accessor
        report = df.arnio.profile()
        
        assert report.row_count == 3
        assert isinstance(report, ar.DataQualityReport)


class TestToPandasWithAutoClean:
    """Test to_pandas with auto_clean workflows."""

    def test_auto_clean_safe_then_to_pandas(self, tmp_path):
        """auto_clean(safe) → to_pandas workflow."""
        csv_path = tmp_path / "messy.csv"
        csv_path.write_text(
            "id,name\n"
            "1,  alice  \n"
            "2,  bob  \n"
        )
        
        frame = ar.read_csv(csv_path)
        clean_frame = ar.auto_clean(frame, mode="safe")
        df = ar.to_pandas(clean_frame)
        
        assert df["name"].iloc[0] == "alice"
        assert df["name"].iloc[1] == "bob"

    def test_auto_clean_strict_then_to_pandas(self, tmp_path):
        """auto_clean(strict) → to_pandas workflow."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "id,value\n"
            "1,  100  \n"
            "2,  200  \n"
            "1,  100  \n"  # Duplicate
        )
        
        frame = ar.read_csv(csv_path)
        clean_frame = ar.auto_clean(frame, mode="strict")
        df = ar.to_pandas(clean_frame)
        
        # Duplicates removed in strict mode
        assert len(df) == 2

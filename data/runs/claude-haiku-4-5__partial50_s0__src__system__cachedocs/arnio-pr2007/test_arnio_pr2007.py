import pytest
import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from unittest.mock import patch

import arnio as ar


class TestWriteParquetBasic:
    """Basic write_parquet functionality."""

    def test_write_parquet_simple(self, tmp_path):
        """Write a simple ArFrame to .parquet file."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"],
            "score": [85.5, 92.0, 78.5]
        })
        frame = ar.from_pandas(df)
        output_path = tmp_path / "test.parquet"
        
        ar.write_parquet(frame, output_path)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        result_df = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(result_df, df)

    def test_write_parquet_pq_extension(self, tmp_path):
        """Write to .pq file extension."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        frame = ar.from_pandas(df)
        output_path = tmp_path / "data.pq"
        
        ar.write_parquet(frame, output_path)
        
        assert output_path.exists()
        result_df = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(result_df, df)

    def test_write_parquet_pathlike(self, tmp_path):
        """Accept os.PathLike objects."""
        df = pd.DataFrame({"value": [10, 20, 30]})
        frame = ar.from_pandas(df)
        output_path = Path(tmp_path) / "result.parquet"
        
        ar.write_parquet(frame, output_path)
        
        assert output_path.exists()


class TestWriteParquetCompression:
    """Test different compression codecs."""

    @pytest.mark.parametrize("compression", ["snappy", "gzip", "zstd", "brotli", "none"])
    def test_write_parquet_compression_codecs(self, tmp_path, compression):
        """Write with different compression codecs."""
        df = pd.DataFrame({
            "id": range(50),
            "value": np.random.randn(50),
            "text": ["item"] * 50
        })
        frame = ar.from_pandas(df)
        output_path = tmp_path / f"test_{compression}.parquet"
        
        ar.write_parquet(frame, output_path, compression=compression)
        
        assert output_path.exists()
        result_df = pd.read_parquet(output_path)
        assert len(result_df) == 50
        assert list(result_df.columns) == ["id", "value", "text"]

    def test_write_parquet_invalid_compression(self, tmp_path):
        """Reject invalid compression codec."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        output_path = tmp_path / "test.parquet"
        
        with pytest.raises(ValueError, match="Unknown compression codec"):
            ar.write_parquet(frame, output_path, compression="invalid_codec")


class TestWriteParquetRowGroupSize:
    """Test row_group_size parameter."""

    def test_write_parquet_row_group_size_valid(self, tmp_path):
        """Write with custom row group size."""
        df = pd.DataFrame({
            "id": range(1000),
            "value": np.random.randn(1000)
        })
        frame = ar.from_pandas(df)
        output_path = tmp_path / "test.parquet"
        
        ar.write_parquet(frame, output_path, row_group_size=250)
        
        assert output_path.exists()
        result_df = pd.read_parquet(output_path)
        assert len(result_df) == 1000

    def test_write_parquet_row_group_size_zero(self, tmp_path):
        """Reject zero row_group_size."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        output_path = tmp_path / "test.parquet"
        
        with pytest.raises(ValueError, match="positive integer"):
            ar.write_parquet(frame, output_path, row_group_size=0)

    def test_write_parquet_row_group_size_negative(self, tmp_path):
        """Reject negative row_group_size."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        output_path = tmp_path / "test.parquet"
        
        with pytest.raises(ValueError, match="positive integer"):
            ar.write_parquet(frame, output_path, row_group_size=-100)

    def test_write_parquet_row_group_size_bool(self, tmp_path):
        """Reject boolean for row_group_size."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        output_path = tmp_path / "test.parquet"
        
        with pytest.raises(TypeError):
            ar.write_parquet(frame, output_path, row_group_size=True)


class TestWriteParquetAttributes:
    """Test DataFrame.attrs handling."""

    def test_write_parquet_preserve_attrs_serializable(self, tmp_path):
        """Preserve JSON-serializable attributes."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        df.attrs = {
            "source": "test_data",
            "version": 1,
            "tags": ["important", "clean"],
            "metadata": {"nested": "value"}
        }
        frame = ar.from_pandas(df)
        output_path = tmp_path / "test.parquet"
        
        ar.write_parquet(frame, output_path, preserve_attrs=True)
        
        result_df = pd.read_parquet(output_path)
        assert result_df.attrs == df.attrs

    def test_write_parquet_preserve_attrs_false(self, tmp_path):
        """Drop attributes when preserve_attrs=False."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        df.attrs = {"source": "test_data", "version": 1}
        frame = ar.from_pandas(df)
        output_path = tmp_path / "test.parquet"
        
        ar.write_parquet(frame, output_path, preserve_attrs=False)
        
        result_df = pd.read_parquet(output_path)
        assert result_df.attrs == {}

    def test_write_parquet_non_serializable_attrs_error(self, tmp_path):
        """Reject non-JSON-serializable attributes when preserve_attrs=True."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        df.attrs = {
            "valid": "string",
            "invalid": set([1, 2, 3])  # Sets not JSON-serializable
        }
        frame = ar.from_pandas(df)
        output_path = tmp_path / "test.parquet"
        
        with pytest.raises(TypeError, match="JSON-serializable"):
            ar.write_parquet(frame, output_path, preserve_attrs=True)


class TestWriteParquetValidation:
    """Test input validation and error cases."""

    def test_write_parquet_invalid_extension_csv(self, tmp_path):
        """Reject .csv extension."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        output_path = tmp_path / "test.csv"
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(frame, output_path)

    def test_write_parquet_invalid_extension_json(self, tmp_path):
        """Reject .json extension."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        output_path = tmp_path / "test.json"
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(frame, output_path)

    def test_write_parquet_not_arframe(self, tmp_path):
        """Reject non-ArFrame input."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        output_path = tmp_path / "test.parquet"
        
        with pytest.raises(TypeError, match="must be an ArFrame"):
            ar.write_parquet(df, output_path)

    def test_write_parquet_invalid_path_type(self, tmp_path):
        """Reject invalid path types."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError):
            ar.write_parquet(frame, 12345)

    def test_write_parquet_invalid_compression_type(self, tmp_path):
        """compression parameter must be a string."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        output_path = tmp_path / "test.parquet"
        
        with pytest.raises(TypeError):
            ar.write_parquet(frame, output_path, compression=123)

    def test_write_parquet_invalid_preserve_attrs_type(self, tmp_path):
        """preserve_attrs must be boolean."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        output_path = tmp_path / "test.parquet"
        
        with pytest.raises(TypeError):
            ar.write_parquet(frame, output_path, preserve_attrs="yes")


class TestWriteParquetWorkflows:
    """End-to-end workflows with real data operations."""

    def test_read_clean_write_parquet_workflow(self, tmp_path):
        """Read CSV, clean with pipeline, write to Parquet."""
        # Create messy CSV
        csv_path = tmp_path / "messy.csv"
        csv_path.write_text("id, name , score\n1, Alice , 85\n2, Bob , 92\n3, Charlie , 78\n")
        
        # Read and clean
        frame = ar.read_csv(csv_path)
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
        ])
        
        # Write to Parquet
        parquet_path = tmp_path / "clean.parquet"
        ar.write_parquet(clean, parquet_path, compression="gzip")
        
        assert parquet_path.exists()
        result_df = pd.read_parquet(parquet_path)
        assert len(result_df) == 3
        assert "alice" in result_df["name"].values

    def test_pandas_accessor_workflow(self, tmp_path):
        """Use pandas accessor for cleaning, then write to Parquet."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 3],
            "name": [" Alice ", " Bob ", " Charlie ", " Charlie "],
            "score": [85, 92, 78, 78]
        })
        
        # Clean using accessor
        clean_df = df.arnio.clean(
            strip_whitespace=True,
            drop_duplicates=True,
        )
        
        # Convert to ArFrame and write
        frame = ar.from_pandas(clean_df)
        output_path = tmp_path / "result.parquet"
        ar.write_parquet(frame, output_path)
        
        assert output_path.exists()
        result_df = pd.read_parquet(output_path)
        assert len(result_df) == 3

    def test_profile_validate_write_workflow(self, tmp_path):
        """Profile data, validate, then write to Parquet."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
            "score": [85.5, 92.0, 78.5]
        })
        frame = ar.from_pandas(df)
        
        # Profile
        report = ar.profile(frame)
        assert report.row_count == 3
        
        # Validate
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
            "score": ar.Float64(nullable=False),
        })
        result = ar.validate(frame, schema)
        assert result.passed
        
        # Write
        output_path = tmp_path / "validated.parquet"
        ar.write_parquet(frame, output_path, compression="snappy")
        
        assert output_path.exists()


class TestWriteParquetEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_write_parquet_empty_dataframe(self, tmp_path):
        """Write empty DataFrame (0 rows, with columns)."""
        df = pd.DataFrame({
            "id": pd.Series([], dtype="int64"),
            "name": pd.Series([], dtype="object"),
            "value": pd.Series([], dtype="float64")
        })
        frame = ar.from_pandas(df)
        output_path = tmp_path / "empty.parquet"
        
        ar.write_parquet(frame, output_path)
        
        assert output_path.exists()
        result_df = pd.read_parquet(output_path)
        assert len(result_df) == 0
        assert list(result_df.columns) == ["id", "name", "value"]

    def test_write_parquet_unicode_characters(self, tmp_path):
        """Write DataFrame with unicode characters."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Carlos", "Élena", "李明", "日本"],
            "city": ["NYC", "London", "Madrid", "Paris", "Beijing", "Tokyo"]
        })
        frame = ar.from_pandas(df)
        output_path = tmp_path / "unicode.parquet"
        
        ar.write_parquet(frame, output_path)
        
        result_df = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(result_df, df)

    def test_write_parquet_mixed_types_with_nulls(self, tmp_path):
        """Write DataFrame with mixed types and null values."""
        df = pd.DataFrame({
            "int_col": [1, 2, None],
            "float_col": [1.1, None, 3.3],
            "str_col": ["a", None, "c"],
            "bool_col": [True, False, None],
        })
        frame = ar.from_pandas(df)
        output_path = tmp_path / "mixed.parquet"
        
        ar.write_parquet(frame, output_path)
        
        result_df = pd.read_parquet(output_path)
        assert list(result_df.columns) == list(df.columns)
        assert len(result_df) == 3

    def test_write_parquet_case_insensitive_extension(self, tmp_path):
        """Accept uppercase .PARQUET and .PQ extensions."""
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)
        
        output_path1 = tmp_path / "data.PARQUET"
        ar.write_parquet(frame, output_path1)
        assert output_path1.exists()
        
        output_path2 = tmp_path / "data.PQ"
        ar.write_parquet(frame, output_path2)
        assert output_path2.exists()

    def test_write_parquet_overwrite_existing_file(self, tmp_path):
        """Overwrite existing Parquet file."""
        df1 = pd.DataFrame({"x": [1, 2, 3]})
        df2 = pd.DataFrame({"x": [10, 20, 30, 40, 50]})
        output_path = tmp_path / "test.parquet"
        
        # Write first time
        ar.write_parquet(ar.from_pandas(df1), output_path)
        result1 = pd.read_parquet(output_path)
        assert len(result1) == 3
        
        # Overwrite
        ar.write_parquet(ar.from_pandas(df2), output_path)
        result2 = pd.read_parquet(output_path)
        assert len(result2) == 5


class TestWriteParquetLargeData:
    """Test with larger datasets."""

    def test_write_parquet_large_dataset(self, tmp_path):
        """Write a larger dataset to verify correctness at scale."""
        n_rows = 10000
        df = pd.DataFrame({
            "id": range(n_rows),
            "value": np.random.randn(n_rows),
            "category": np.random.choice(["A", "B", "C", "D"], n_rows),
            "date": pd.date_range("2020-01-01", periods=n_rows, freq="D").astype(str)
        })
        frame = ar.from_pandas(df)
        output_path = tmp_path / "large.parquet"
        
        ar.write_parquet(frame, output_path, compression="snappy", row_group_size=1000)
        
        assert output_path.exists()
        result_df = pd.read_parquet(output_path)
        assert len(result_df) == n_rows
        assert list(result_df.columns) == list(df.columns)
        assert result_df["id"].min() == 0
        assert result_df["id"].max() == n_rows - 1

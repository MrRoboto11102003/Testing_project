"""
System (end-to-end) tests for arnio.io.write_parquet.

These tests verify the complete workflow: reading/creating data, cleaning/validating,
and writing to Parquet files using the public arnio API.
"""

import pytest
import os
import json
from pathlib import Path

import pandas as pd
import numpy as np

import arnio as ar


class TestWriteParquetBasic:
    """Basic write_parquet functionality and file I/O."""

    def test_write_parquet_simple_frame(self, tmp_path):
        """Write a simple ArFrame to Parquet and verify content."""
        data = {"id": [1, 2, 3], "name": ["alice", "bob", "charlie"]}
        df = pd.DataFrame(data)
        frame = ar.from_pandas(df)

        output = tmp_path / "test.parquet"
        ar.write_parquet(frame, output)

        assert output.exists()
        result = pd.read_parquet(output)
        pd.testing.assert_frame_equal(df, result)

    def test_write_parquet_pq_extension(self, tmp_path):
        """Write to .pq file extension (alternative Parquet format)."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        frame = ar.from_pandas(df)

        output = tmp_path / "data.pq"
        ar.write_parquet(frame, output)

        assert output.exists()
        result = pd.read_parquet(output)
        assert len(result) == 3
        assert list(result.columns) == ["x", "y"]

    def test_write_parquet_with_nulls(self, tmp_path):
        """Write ArFrame containing null values."""
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", None]})
        frame = ar.from_pandas(df)

        output = tmp_path / "nulls.parquet"
        ar.write_parquet(frame, output)

        result = pd.read_parquet(output)
        assert result["a"].isna().sum() == 1
        assert result["b"].isna().sum() == 1

    def test_write_parquet_mixed_dtypes(self, tmp_path):
        """Write ArFrame with multiple data types."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )
        frame = ar.from_pandas(df)

        output = tmp_path / "mixed.parquet"
        ar.write_parquet(frame, output)

        result = pd.read_parquet(output)
        assert list(result.columns) == [
            "int_col",
            "float_col",
            "str_col",
            "bool_col",
        ]
        assert len(result) == 3

    def test_write_parquet_empty_frame(self, tmp_path):
        """Write an empty ArFrame (0 rows, columns present)."""
        df = pd.DataFrame(
            {
                "a": pd.Series([], dtype="int64"),
                "b": pd.Series([], dtype="string"),
            }
        )
        frame = ar.from_pandas(df)

        output = tmp_path / "empty.parquet"
        ar.write_parquet(frame, output)

        result = pd.read_parquet(output)
        assert len(result) == 0
        assert list(result.columns) == ["a", "b"]


class TestWriteParquetCompression:
    """Test various compression codec options."""

    @pytest.mark.parametrize(
        "codec", ["snappy", "gzip", "zstd", "brotli", "none"]
    )
    def test_write_parquet_compression_codecs(self, tmp_path, codec):
        """Verify all supported compression codecs work correctly."""
        df = pd.DataFrame({"x": range(100), "y": range(100, 200)})
        frame = ar.from_pandas(df)

        output = tmp_path / f"{codec}.parquet"
        ar.write_parquet(frame, output, compression=codec)

        result = pd.read_parquet(output)
        pd.testing.assert_frame_equal(df, result)


class TestWriteParquetRowGroupSize:
    """Test row_group_size parameter."""

    def test_write_parquet_custom_row_group_size(self, tmp_path):
        """Write with custom row group size."""
        df = pd.DataFrame({"x": range(1000)})
        frame = ar.from_pandas(df)

        output = tmp_path / "rg.parquet"
        ar.write_parquet(frame, output, row_group_size=100)

        result = pd.read_parquet(output)
        assert len(result) == 1000
        pd.testing.assert_frame_equal(df, result)

    def test_write_parquet_row_group_size_larger_than_data(self, tmp_path):
        """Row group size larger than dataset size."""
        df = pd.DataFrame({"x": range(50)})
        frame = ar.from_pandas(df)

        output = tmp_path / "rg_large.parquet"
        ar.write_parquet(frame, output, row_group_size=10000)

        result = pd.read_parquet(output)
        assert len(result) == 50


class TestWriteParquetPreserveAttrs:
    """Test preserve_attrs parameter."""

    def test_write_parquet_preserve_attrs_true_simple(self, tmp_path):
        """Write with preserve_attrs=True for JSON-serializable attrs."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        df.attrs = {"source": "test_data", "version": 1, "tags": ["important"]}
        frame = ar.from_pandas(df)

        output = tmp_path / "attrs.parquet"
        ar.write_parquet(frame, output, preserve_attrs=True)

        # Verify file is created and readable
        assert output.exists()
        result = pd.read_parquet(output)
        assert len(result) == 3

    def test_write_parquet_preserve_attrs_false_drops_attrs(self, tmp_path):
        """Write with preserve_attrs=False silently drops attrs."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        df.attrs = {"source": "test_data"}
        frame = ar.from_pandas(df)

        output = tmp_path / "no_attrs.parquet"
        ar.write_parquet(frame, output, preserve_attrs=False)

        # File should still be created successfully
        assert output.exists()


class TestWriteParquetErrorHandling:
    """Test error cases and validation."""

    def test_write_parquet_invalid_extension(self, tmp_path):
        """ValueError for unsupported file extension."""
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)

        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(frame, tmp_path / "file.csv")

    def test_write_parquet_invalid_extension_txt(self, tmp_path):
        """ValueError for .txt extension."""
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)

        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(frame, tmp_path / "file.txt")

    def test_write_parquet_invalid_compression_codec(self, tmp_path):
        """ValueError for unknown compression codec."""
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)

        with pytest.raises(ValueError, match="Unknown compression codec"):
            ar.write_parquet(frame, tmp_path / "out.parquet", compression="badcodec")

    def test_write_parquet_negative_row_group_size(self, tmp_path):
        """ValueError for negative row_group_size."""
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)

        with pytest.raises(ValueError, match="positive integer"):
            ar.write_parquet(frame, tmp_path / "out.parquet", row_group_size=-1)

    def test_write_parquet_zero_row_group_size(self, tmp_path):
        """ValueError for zero row_group_size."""
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)

        with pytest.raises(ValueError, match="positive integer"):
            ar.write_parquet(frame, tmp_path / "out.parquet", row_group_size=0)

    def test_write_parquet_non_integer_row_group_size(self, tmp_path):
        """TypeError for non-integer row_group_size."""
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)

        with pytest.raises(TypeError, match="must be an integer"):
            ar.write_parquet(
                frame, tmp_path / "out.parquet", row_group_size="100"
            )

    def test_write_parquet_boolean_row_group_size_rejected(self, tmp_path):
        """TypeError for boolean row_group_size (even though bool is int subclass)."""
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)

        with pytest.raises(TypeError, match="must be an integer"):
            ar.write_parquet(frame, tmp_path / "out.parquet", row_group_size=True)

    def test_write_parquet_not_arframe(self, tmp_path):
        """TypeError when input is not an ArFrame."""
        df = pd.DataFrame({"x": [1, 2]})

        with pytest.raises(TypeError, match="ArFrame"):
            ar.write_parquet(df, tmp_path / "out.parquet")

    def test_write_parquet_preserve_attrs_not_bool(self, tmp_path):
        """TypeError if preserve_attrs is not a boolean."""
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)

        with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
            ar.write_parquet(frame, tmp_path / "out.parquet", preserve_attrs="yes")

    def test_write_parquet_non_serializable_attrs_error(self, tmp_path):
        """TypeError for non-JSON-serializable attrs with preserve_attrs=True."""
        df = pd.DataFrame({"x": [1, 2]})
        # Add an object that is not JSON serializable
        df.attrs = {"bad_obj": set([1, 2, 3])}
        frame = ar.from_pandas(df)

        with pytest.raises(TypeError, match="JSON-serializable"):
            ar.write_parquet(frame, tmp_path / "out.parquet", preserve_attrs=True)


class TestWriteParquetIntegration:
    """Integration tests with other arnio operations."""

    def test_write_parquet_csv_to_parquet_pipeline(self, tmp_path):
        """Read CSV, clean, write to Parquet."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("id,name,score\n1,alice,95.5\n2,bob,87.3\n3,charlie,92.1\n")

        # Read and clean
        frame = ar.read_csv(csv_file)
        clean = ar.pipeline(frame, [("strip_whitespace",)])

        output = tmp_path / "output.parquet"
        ar.write_parquet(clean, output)

        result = pd.read_parquet(output)
        assert len(result) == 3
        assert list(result.columns) == ["id", "name", "score"]

    def test_write_parquet_after_pipeline_with_drop_duplicates(self, tmp_path):
        """Apply cleaning pipeline including drop_duplicates, then write."""
        df = pd.DataFrame(
            {"id": [1, 2, 2, 3], "value": ["a", "b", "b", "c"]}
        )
        frame = ar.from_pandas(df)

        clean = ar.pipeline(frame, [("drop_duplicates",)])

        output = tmp_path / "dedup.parquet"
        ar.write_parquet(clean, output)

        result = pd.read_parquet(output)
        assert len(result) == 3

    def test_write_parquet_pathlib_path_support(self, tmp_path):
        """Accept pathlib.Path objects as destination."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        frame = ar.from_pandas(df)

        output = Path(tmp_path) / "data.parquet"
        ar.write_parquet(frame, output)

        assert output.exists()
        result = pd.read_parquet(output)
        assert len(result) == 3

    def test_write_parquet_string_path_support(self, tmp_path):
        """Accept string paths as destination."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        frame = ar.from_pandas(df)

        output_str = str(tmp_path / "data.parquet")
        ar.write_parquet(frame, output_str)

        assert os.path.exists(output_str)
        result = pd.read_parquet(output_str)
        assert len(result) == 3

    def test_write_parquet_roundtrip_preserves_data(self, tmp_path):
        """Full roundtrip: create → write → read back preserves all data."""
        original = pd.DataFrame(
            {
                "int": [1, 2, 3],
                "float": [1.5, 2.5, 3.5],
                "str": ["x", "y", "z"],
                "bool": [True, False, True],
            }
        )
        frame = ar.from_pandas(original)

        output = tmp_path / "roundtrip.parquet"
        ar.write_parquet(frame, output, compression="zstd", row_group_size=10)

        result = pd.read_parquet(output)
        pd.testing.assert_frame_equal(original, result)

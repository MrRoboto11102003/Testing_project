"""
System (end-to-end) tests for arnio.io.write_parquet.

Tests drive complete user workflows: load data, clean/validate,
write to Parquet, and verify roundtrips.
"""

import pytest
import sys
import os
import json
from pathlib import Path
import tempfile

import pandas as pd
import numpy as np

import arnio as ar


class TestWriteParquetBasicWorkflow:
    """Basic write_parquet functionality and file I/O."""

    def test_write_parquet_simple_roundtrip(self, tmp_path):
        """Write a simple ArFrame to Parquet and read it back."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"],
            "score": [95.5, 87.3, 92.1],
        })
        frame = ar.from_pandas(df)
        
        output = tmp_path / "data.parquet"
        ar.write_parquet(frame, output)
        
        assert output.exists()
        result = pd.read_parquet(output)
        pd.testing.assert_frame_equal(result, df)

    def test_write_parquet_both_extensions(self, tmp_path):
        """Accept both .parquet and .pq file extensions."""
        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]}))
        
        # .parquet extension
        p1 = tmp_path / "data.parquet"
        ar.write_parquet(frame, p1)
        assert p1.exists()
        
        # .pq extension
        p2 = tmp_path / "data.pq"
        ar.write_parquet(frame, p2)
        assert p2.exists()
        
        # Both should be readable
        df1 = pd.read_parquet(p1)
        df2 = pd.read_parquet(p2)
        pd.testing.assert_frame_equal(df1, df2)

    @pytest.mark.parametrize("compression", ["snappy", "gzip", "zstd", "brotli", "none"])
    def test_write_parquet_compression_codecs(self, tmp_path, compression):
        """All supported compression codecs work correctly."""
        frame = ar.from_pandas(pd.DataFrame({"data": list(range(500))}))
        
        output = tmp_path / f"data_{compression}.parquet"
        ar.write_parquet(frame, output, compression=compression)
        
        assert output.exists()
        result = pd.read_parquet(output)
        assert len(result) == 500

    def test_write_parquet_row_group_size(self, tmp_path):
        """row_group_size parameter is accepted and produces valid output."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": range(1000),
            "value": np.random.randn(1000),
        }))
        
        output = tmp_path / "data.parquet"
        ar.write_parquet(frame, output, row_group_size=100)
        
        result = pd.read_parquet(output)
        assert len(result) == 1000

    def test_write_parquet_preserve_attrs(self, tmp_path):
        """DataFrame attrs are preserved when preserve_attrs=True."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        df.attrs = {"source": "test_data", "version": 1, "valid": True}
        frame = ar.from_pandas(df)
        
        output = tmp_path / "data.parquet"
        ar.write_parquet(frame, output, preserve_attrs=True)
        
        result = pd.read_parquet(output)
        assert result.attrs.get("source") == "test_data"
        assert result.attrs.get("version") == 1

    def test_write_parquet_discard_attrs(self, tmp_path):
        """DataFrame attrs are dropped when preserve_attrs=False."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        df.attrs = {"metadata": "value"}
        frame = ar.from_pandas(df)
        
        output = tmp_path / "data.parquet"
        ar.write_parquet(frame, output, preserve_attrs=False)
        
        result = pd.read_parquet(output)
        assert len(result.attrs) == 0


class TestWriteParquetErrorCases:
    """Error handling and validation."""

    def test_write_parquet_invalid_extension(self, tmp_path):
        """Reject files with unsupported extensions."""
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(frame, tmp_path / "data.csv")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(frame, tmp_path / "data.txt")

    def test_write_parquet_invalid_compression(self, tmp_path):
        """Reject unknown compression codec."""
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))
        
        with pytest.raises(ValueError, match="Unknown compression codec"):
            ar.write_parquet(frame, tmp_path / "data.parquet", compression="invalid")

    def test_write_parquet_invalid_parameters(self, tmp_path):
        """Reject invalid parameter types and values."""
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))
        
        # Invalid row_group_size (negative)
        with pytest.raises(ValueError, match="positive integer"):
            ar.write_parquet(frame, tmp_path / "d1.parquet", row_group_size=-100)
        
        # Invalid row_group_size (zero)
        with pytest.raises(ValueError, match="positive integer"):
            ar.write_parquet(frame, tmp_path / "d2.parquet", row_group_size=0)
        
        # Invalid row_group_size (string)
        with pytest.raises(TypeError):
            ar.write_parquet(frame, tmp_path / "d3.parquet", row_group_size="100")
        
        # Invalid row_group_size (bool is not int)
        with pytest.raises(TypeError):
            ar.write_parquet(frame, tmp_path / "d4.parquet", row_group_size=True)
        
        # Invalid preserve_attrs (non-bool)
        with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
            ar.write_parquet(frame, tmp_path / "d5.parquet", preserve_attrs="true")

    def test_write_parquet_invalid_input_types(self, tmp_path):
        """Reject non-ArFrame inputs and invalid path types."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        
        # Non-ArFrame input
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_parquet(df, tmp_path / "data.parquet")
        
        # Invalid path type
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError):
            ar.write_parquet(frame, 12345)
        
        with pytest.raises(TypeError):
            ar.write_parquet(frame, ["path", "list"])

    def test_write_parquet_non_serializable_attrs(self, tmp_path):
        """Reject non-JSON-serializable DataFrame attrs."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        df.attrs = {"func": lambda x: x}  # Not JSON-serializable
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="JSON-serializable"):
            ar.write_parquet(frame, tmp_path / "data.parquet", preserve_attrs=True)

    def test_write_parquet_zero_column_frame(self, tmp_path):
        """Reject zero-column frames with rows."""
        df = pd.DataFrame(index=[0, 1, 2])  # 0 columns, 3 rows
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="Cannot write a zero-column ArFrame"):
            ar.write_parquet(frame, tmp_path / "data.parquet")


class TestWriteParquetEdgeCases:
    """Edge cases and special data types."""

    def test_write_parquet_empty_frame(self, tmp_path):
        """Write and roundtrip an empty frame."""
        frame = ar.from_pandas(pd.DataFrame({"col1": [], "col2": []}))
        
        output = tmp_path / "empty.parquet"
        ar.write_parquet(frame, output)
        
        result = pd.read_parquet(output)
        assert len(result) == 0
        assert list(result.columns) == ["col1", "col2"]

    def test_write_parquet_with_nulls(self, tmp_path):
        """Null values are preserved in roundtrip."""
        df = pd.DataFrame({
            "int_col": [1, None, 3],
            "str_col": ["a", None, "c"],
            "float_col": [1.1, 2.2, None],
        })
        frame = ar.from_pandas(df)
        
        output = tmp_path / "nulls.parquet"
        ar.write_parquet(frame, output)
        
        result = pd.read_parquet(output)
        assert result["int_col"].isna().sum() == 1
        assert result["str_col"].isna().sum() == 1
        assert result["float_col"].isna().sum() == 1

    def test_write_parquet_large_data(self, tmp_path):
        """Handle reasonably large datasets."""
        large_string = "x" * 1000
        frame = ar.from_pandas(pd.DataFrame({
            "id": range(500),
            "text": [large_string] * 500,
            "value": np.random.randn(500),
        }))
        
        output = tmp_path / "large.parquet"
        ar.write_parquet(frame, output)
        
        result = pd.read_parquet(output)
        assert len(result) == 500
        assert len(result.iloc[0]["text"]) == 1000


class TestWriteParquetIntegration:
    """Integration with other arnio workflows."""

    def test_pipeline_cleanup_to_parquet(self, tmp_path):
        """Clean data with pipeline and export to Parquet."""
        df = pd.DataFrame({
            "name": [" Alice ", " Bob ", "  Charlie  "],
            "age": ["30", "25", "35"],
            "score": [95.5, None, 88.2],
        })
        frame = ar.from_pandas(df)
        
        # Apply cleaning pipeline
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("fill_nulls", {"value": 0.0, "subset": ["score"]}),
            ("cast_types", {"age": "int64"}),
        ])
        
        output = tmp_path / "cleaned.parquet"
        ar.write_parquet(clean, output)
        
        result = pd.read_parquet(output)
        assert result["name"].iloc[0] == "Alice"
        assert result["score"].iloc[1] == 0.0
        assert result["age"].dtype == "int64"

    def test_validate_then_export_to_parquet(self, tmp_path):
        """Validate a frame and export to Parquet (valid or invalid data)."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "age": [25, 30, 35, None, 50],
        })
        frame = ar.from_pandas(df)
        
        # Set up schema
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "age": ar.Int64(nullable=True, min=0, max=150),
        })
        
        result = ar.validate(frame, schema)
        # Whether validation passes or fails, we can still export the frame
        output = tmp_path / "exported.parquet"
        ar.write_parquet(frame, output)
        
        assert output.exists()
        read_back = pd.read_parquet(output)
        pd.testing.assert_frame_equal(read_back, df)

    def test_read_csv_clean_write_parquet(self, tmp_path):
        """Full end-to-end: read CSV, clean, write Parquet."""
        # Create a sample CSV
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("id,name,score\n1, Alice ,95.5\n2, Bob ,\n3, Charlie ,92.1\n")
        
        # Read and clean
        frame = ar.read_csv(csv_file)
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("fill_nulls", {"value": 0.0, "subset": ["score"]}),
        ])
        
        # Export to Parquet
        parquet_file = tmp_path / "output.parquet"
        ar.write_parquet(clean, parquet_file, compression="gzip")
        
        result = pd.read_parquet(parquet_file)
        assert len(result) == 3
        assert result["name"].iloc[0] == "Alice"
        assert result["score"].iloc[1] == 0.0

    def test_pandas_accessor_write_parquet(self, tmp_path):
        """Use pandas .arnio accessor to clean and write to Parquet."""
        df = pd.DataFrame({
            "value": [" 10 ", " 20 ", " 30 "],
        })
        
        # Use the .arnio accessor
        clean_df = df.arnio.clean(strip_whitespace=True)
        frame = ar.from_pandas(clean_df)
        
        output = tmp_path / "accessor_output.parquet"
        ar.write_parquet(frame, output)
        
        result = pd.read_parquet(output)
        assert result["value"].iloc[0] == "10"

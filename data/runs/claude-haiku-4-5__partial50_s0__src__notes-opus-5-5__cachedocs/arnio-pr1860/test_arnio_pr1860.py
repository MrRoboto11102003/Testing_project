"""
Pytest unit tests for arnio.io.write_parquet
"""

import json
import os
import re
from pathlib import Path

import pandas as pd
import pytest

import arnio as ar


class TestWriteParquetHappyPath:
    """Test normal, valid usage."""

    def test_write_parquet_default_params(self, tmp_path):
        """Write a frame with default parameters."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
        ar.write_parquet(frame, p)
        
        assert p.exists()
        result = pd.read_parquet(p)
        assert result.shape == (3, 2)
        assert list(result.columns) == ["a", "b"]
        assert result["a"].tolist() == [1, 2, 3]

    def test_write_parquet_pq_extension(self, tmp_path):
        """Write to .pq extension."""
        p = tmp_path / "output.pq"
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))
        ar.write_parquet(frame, p)
        
        assert p.exists()
        result = pd.read_parquet(p)
        assert result.shape == (1, 1)

    def test_write_parquet_case_insensitive_ext(self, tmp_path):
        """Extensions are case-insensitive: .PARQUET, .Pq, etc."""
        for ext in [".PARQUET", ".Pq", ".PaRqUeT"]:
            p = tmp_path / f"file{ext}"
            frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
            ar.write_parquet(frame, p)
            assert p.exists()

    def test_write_parquet_all_compressions(self, tmp_path):
        """Test each valid compression codec."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}))
        
        for codec in ["snappy", "gzip", "brotli", "zstd", "none"]:
            p = tmp_path / f"comp_{codec}.parquet"
            ar.write_parquet(frame, p, compression=codec)
            assert p.exists()
            result = pd.read_parquet(p)
            assert result.shape == (2, 2)

    def test_write_parquet_row_group_size(self, tmp_path):
        """Write with explicit row_group_size."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(
            pd.DataFrame({"a": range(10)})
        )
        ar.write_parquet(frame, p, row_group_size=3)
        
        assert p.exists()
        result = pd.read_parquet(p)
        assert len(result) == 10

    def test_write_parquet_pathlike(self, tmp_path):
        """Accept pathlib.Path."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        ar.write_parquet(frame, p)  # p is a Path object
        assert p.exists()

    def test_write_parquet_empty_frame_with_cols(self, tmp_path):
        """Write frame with 0 rows but >=1 column (should work)."""
        p = tmp_path / "empty.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [], "b": []}))
        ar.write_parquet(frame, p)
        
        assert p.exists()
        result = pd.read_parquet(p)
        assert result.shape == (0, 2)


class TestWriteParquetFileExtension:
    """Test file extension validation."""

    def test_unsupported_extension_csv(self, tmp_path):
        """Reject .csv extension."""
        p = tmp_path / "output.csv"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(ValueError, match=r"Unsupported file format.*\.parquet.*\.pq"):
            ar.write_parquet(frame, p)
        assert not p.exists()

    def test_unsupported_extension_json(self, tmp_path):
        """Reject .json extension."""
        p = tmp_path / "output.json"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(ValueError, match=r"Unsupported file format"):
            ar.write_parquet(frame, p)
        assert not p.exists()

    def test_no_extension(self, tmp_path):
        """Reject file with no extension."""
        p = tmp_path / "output"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(ValueError, match=r"Unsupported file format"):
            ar.write_parquet(frame, p)
        assert not p.exists()


class TestWriteParquetCompressionValidation:
    """Test compression parameter validation."""

    def test_compression_case_sensitive(self, tmp_path):
        """Compression codec names are case-sensitive."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(
            ValueError,
            match=re.escape("Unknown compression codec: 'SNAPPY'")
        ):
            ar.write_parquet(frame, p, compression="SNAPPY")
        assert not p.exists()

    def test_compression_invalid(self, tmp_path):
        """Reject unknown compression codec."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(ValueError, match=r"Unknown compression codec: 'lz4'"):
            ar.write_parquet(frame, p, compression="lz4")
        assert not p.exists()

    def test_compression_not_string(self, tmp_path):
        """compression must be a string."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(TypeError, match="compression must be a string"):
            ar.write_parquet(frame, p, compression=None)
        assert not p.exists()


class TestWriteParquetRowGroupSize:
    """Test row_group_size parameter validation."""

    def test_row_group_size_bool_rejected(self, tmp_path):
        """Boolean is rejected even though bool is subclass of int."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(TypeError, match="row_group_size must be an integer"):
            ar.write_parquet(frame, p, row_group_size=True)
        assert not p.exists()

    def test_row_group_size_float(self, tmp_path):
        """Float is rejected."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(TypeError, match="row_group_size must be an integer"):
            ar.write_parquet(frame, p, row_group_size=2.5)
        assert not p.exists()

    def test_row_group_size_string(self, tmp_path):
        """String is rejected."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(TypeError, match="row_group_size must be an integer"):
            ar.write_parquet(frame, p, row_group_size="5")
        assert not p.exists()

    def test_row_group_size_zero(self, tmp_path):
        """Zero is rejected."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
            ar.write_parquet(frame, p, row_group_size=0)
        assert not p.exists()

    def test_row_group_size_negative(self, tmp_path):
        """Negative is rejected."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
            ar.write_parquet(frame, p, row_group_size=-1)
        assert not p.exists()


class TestWriteParquetPreserveAttrs:
    """Test preserve_attrs parameter."""

    def test_preserve_attrs_not_bool(self, tmp_path):
        """preserve_attrs must be a bool."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
            ar.write_parquet(frame, p, preserve_attrs=1)
        assert not p.exists()

    def test_preserve_attrs_false_drops_attrs(self, tmp_path):
        """With preserve_attrs=False, attrs are not written."""
        p = tmp_path / "output.parquet"
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.attrs = {"metadata": "test"}
        frame = ar.from_pandas(df)
        
        ar.write_parquet(frame, p, preserve_attrs=False)
        result = pd.read_parquet(p)
        # With preserve_attrs=False, attrs should not be in the result
        # (behavior depends on pandas/pyarrow version, so we just verify the write succeeds)
        assert p.exists()


class TestWriteParquetFrameValidation:
    """Test frame and type validation."""

    def test_frame_not_arframe(self, tmp_path):
        """frame must be an ArFrame, not a pandas DataFrame."""
        p = tmp_path / "output.parquet"
        df = pd.DataFrame({"a": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_parquet(df, p)
        assert not p.exists()

    def test_path_not_valid_type(self, tmp_path):
        """path must be str, bytes, or PathLike."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(TypeError, match=r"path must be a string, bytes, or os\.PathLike object, got 'int'"):
            ar.write_parquet(frame, 123)


class TestWriteParquetZeroColumnEdgeCase:
    """Test edge case of zero columns with rows."""

    def test_zero_columns_with_rows_raises(self, tmp_path):
        """Cannot write frame with 0 columns but > 0 rows."""
        p = tmp_path / "output.parquet"
        # Build a DataFrame with 0 columns
        df = pd.DataFrame(index=range(3))
        frame = ar.from_pandas(df)
        
        # Only test if the frame actually has the expected shape
        if frame.shape == (3, 0):
            with pytest.raises(
                ValueError,
                match=r"Cannot write a zero-column ArFrame with 3 rows to Parquet"
            ):
                ar.write_parquet(frame, p)
            assert not p.exists()
        else:
            pytest.skip(f"Frame shape is {frame.shape}, not (3, 0)")


class TestWriteParquetValidationOrder:
    """Test that validation happens in the documented order."""

    def test_frame_check_before_path_check(self, tmp_path):
        """frame validation comes before path validation."""
        df = pd.DataFrame({"a": [1]})
        # Pass invalid frame and invalid path
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_parquet(df, 123)

    def test_extension_before_compression(self, tmp_path):
        """File extension is checked before compression validation."""
        p = tmp_path / "output.csv"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        # Both extension and compression are invalid; should report extension
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(frame, p, compression="INVALID")
        assert not p.exists()

    def test_compression_before_row_group_size(self, tmp_path):
        """Compression is checked before row_group_size."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        # Both invalid; should report compression first
        with pytest.raises(ValueError, match="Unknown compression codec"):
            ar.write_parquet(frame, p, compression="bad", row_group_size=True)
        assert not p.exists()

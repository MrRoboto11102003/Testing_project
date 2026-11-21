import sys
import json
import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

import arnio as ar


class TestWriteParquetE2E:
    """End-to-end tests for arnio.io.write_parquet"""

    def test_basic_write_and_read_workflow(self, tmp_path):
        """Basic workflow: create frame, write parquet, read back"""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "value": [10.5, 20.3, 30.1, 40.7, 50.2],
            "category": ["A", "B", "A", "C", "B"]
        })
        frame = ar.frame(df)
        
        path = tmp_path / "output.parquet"
        ar.write_parquet(frame, str(path))
        
        assert path.exists()
        assert path.stat().st_size > 0
        result = pd.read_parquet(path)
        pd.testing.assert_frame_equal(result, df)

    def test_write_with_pq_extension(self, tmp_path):
        """Write with .pq file extension"""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        frame = ar.frame(df)
        
        path = tmp_path / "data.pq"
        ar.write_parquet(frame, path)
        
        assert path.exists()
        result = pd.read_parquet(path)
        pd.testing.assert_frame_equal(result, df)

    def test_write_with_compression_codecs(self, tmp_path):
        """Test various compression codecs"""
        df = pd.DataFrame({"col": np.random.randn(100)})
        frame = ar.frame(df)
        
        for codec in ["snappy", "gzip", "zstd", "none"]:
            path = tmp_path / f"data_{codec}.parquet"
            ar.write_parquet(frame, str(path), compression=codec)
            assert path.exists()
            result = pd.read_parquet(path)
            pd.testing.assert_frame_equal(result, df, check_dtype=False)

    def test_write_with_row_group_size(self, tmp_path):
        """Write large frame with custom row group size"""
        df = pd.DataFrame({
            "id": range(1000),
            "value": np.random.randn(1000),
            "category": np.random.choice(["A", "B", "C"], 1000)
        })
        frame = ar.frame(df)
        
        path = tmp_path / "row_groups.parquet"
        ar.write_parquet(frame, str(path), row_group_size=250)
        
        assert path.exists()
        result = pd.read_parquet(path)
        assert len(result) == 1000
        pd.testing.assert_frame_equal(result, df, check_dtype=False)

    def test_preserve_attrs_true(self, tmp_path):
        """Preserve JSON-serializable DataFrame attributes"""
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.attrs = {
            "source": "test_data",
            "version": 1,
            "quality_score": 0.95,
            "is_validated": True,
            "tags": ["important", "test"]
        }
        frame = ar.frame(df)
        
        path = tmp_path / "with_attrs.parquet"
        ar.write_parquet(frame, str(path), preserve_attrs=True)
        
        result = pd.read_parquet(path)
        assert result.attrs.get("source") == "test_data"
        assert result.attrs.get("version") == 1
        assert result.attrs.get("quality_score") == 0.95
        assert result.attrs.get("is_validated") is True

    def test_preserve_attrs_false(self, tmp_path):
        """Drop attributes when preserve_attrs=False"""
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.attrs = {"metadata": "should_be_dropped", "count": 42}
        frame = ar.frame(df)
        
        path = tmp_path / "no_attrs.parquet"
        ar.write_parquet(frame, str(path), preserve_attrs=False)
        
        result = pd.read_parquet(path)
        assert len(result.attrs) == 0

    def test_pathlib_path_input(self, tmp_path):
        """Accept pathlib.Path as path argument"""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        frame = ar.frame(df)
        
        path = tmp_path / "pathlib_test.parquet"
        ar.write_parquet(frame, path)
        
        assert path.exists()
        result = pd.read_parquet(path)
        pd.testing.assert_frame_equal(result, df)

    def test_error_invalid_file_extension(self, tmp_path):
        """Error raised for unsupported file extension"""
        df = pd.DataFrame({"a": [1]})
        frame = ar.frame(df)
        
        path = tmp_path / "data.csv"
        with pytest.raises(ValueError) as exc:
            ar.write_parquet(frame, str(path))
        assert "Unsupported file format" in str(exc.value)

    def test_error_invalid_compression_codec(self, tmp_path):
        """Error raised for invalid compression codec"""
        df = pd.DataFrame({"a": [1]})
        frame = ar.frame(df)
        
        path = tmp_path / "data.parquet"
        with pytest.raises(ValueError) as exc:
            ar.write_parquet(frame, str(path), compression="invalid_codec")
        assert "Unknown compression codec" in str(exc.value)

    def test_error_not_arframe_type(self, tmp_path):
        """Error raised when frame is not an ArFrame"""
        df = pd.DataFrame({"a": [1]})
        path = tmp_path / "data.parquet"
        
        with pytest.raises(TypeError) as exc:
            ar.write_parquet(df, str(path))
        assert "ArFrame" in str(exc.value)

    def test_error_non_serializable_attrs(self, tmp_path):
        """Error raised for non-JSON-serializable attrs"""
        df = pd.DataFrame({"a": [1]})
        df.attrs = {"obj": object()}
        frame = ar.frame(df)
        
        path = tmp_path / "data.parquet"
        with pytest.raises(TypeError) as exc:
            ar.write_parquet(frame, str(path), preserve_attrs=True)
        assert "JSON-serializable" in str(exc.value)

    def test_error_invalid_row_group_size_zero(self, tmp_path):
        """Error raised for zero or negative row_group_size"""
        df = pd.DataFrame({"a": [1]})
        frame = ar.frame(df)
        path = tmp_path / "data.parquet"
        
        with pytest.raises(ValueError) as exc:
            ar.write_parquet(frame, str(path), row_group_size=0)
        assert "positive integer" in str(exc.value)

    def test_error_invalid_row_group_size_float(self, tmp_path):
        """Error raised for non-integer row_group_size"""
        df = pd.DataFrame({"a": [1]})
        frame = ar.frame(df)
        path = tmp_path / "data.parquet"
        
        with pytest.raises(TypeError) as exc:
            ar.write_parquet(frame, str(path), row_group_size=10.5)
        assert "integer" in str(exc.value)

    def test_error_invalid_preserve_attrs_type(self, tmp_path):
        """Error raised for non-boolean preserve_attrs"""
        df = pd.DataFrame({"a": [1]})
        frame = ar.frame(df)
        path = tmp_path / "data.parquet"
        
        with pytest.raises(TypeError) as exc:
            ar.write_parquet(frame, str(path), preserve_attrs="true")
        assert "bool" in str(exc.value)

    def test_multiple_data_types_and_null_values(self, tmp_path):
        """Write frame with mixed data types and null values"""
        df = pd.DataFrame({
            "int_col": [1, 2, None, 4],
            "float_col": [1.1, None, 3.3, 4.4],
            "str_col": ["a", "b", "c", None],
            "bool_col": [True, False, True, None]
        })
        frame = ar.frame(df)
        
        path = tmp_path / "mixed_types.parquet"
        ar.write_parquet(frame, str(path), compression="gzip", preserve_attrs=False)
        
        assert path.exists()
        result = pd.read_parquet(path)
        assert len(result) == 4
        assert len(result.columns) == 4

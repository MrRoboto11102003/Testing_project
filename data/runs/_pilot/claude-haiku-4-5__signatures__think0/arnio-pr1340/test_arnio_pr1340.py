import pytest
import pandas as pd
import numpy as np
import io
import tempfile
import os
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import arnio as ar


class TestGetFeaturesOut:
    """Test ArnioCleaner.get_feature_names_out"""

    def test_get_feature_names_out_with_input_features(self):
        """Test get_feature_names_out returns input_features when provided"""
        cleaner = ar.integrations.sklearn.ArnioCleaner()
        features = ["col1", "col2", "col3"]
        result = cleaner.get_feature_names_out(input_features=features)
        assert list(result) == features

    def test_get_feature_names_out_without_input_features(self):
        """Test get_feature_names_out works without input_features"""
        cleaner = ar.integrations.sklearn.ArnioCleaner()
        result = cleaner.get_feature_names_out(input_features=None)
        assert result is not None

    def test_get_feature_names_out_returns_array(self):
        """Test get_feature_names_out returns array-like"""
        cleaner = ar.integrations.sklearn.ArnioCleaner()
        features = ["a", "b", "c"]
        result = cleaner.get_feature_names_out(input_features=features)
        assert hasattr(result, '__iter__')

    def test_get_feature_names_out_empty_features(self):
        """Test get_feature_names_out with empty feature list"""
        cleaner = ar.integrations.sklearn.ArnioCleaner()
        result = cleaner.get_feature_names_out(input_features=[])
        assert len(result) == 0

    def test_get_feature_names_out_single_feature(self):
        """Test get_feature_names_out with single feature"""
        cleaner = ar.integrations.sklearn.ArnioCleaner()
        result = cleaner.get_feature_names_out(input_features=["single"])
        assert list(result) == ["single"]


class TestReadCSV:
    """Test arnio.io.read_csv"""

    def test_read_csv_basic(self):
        """Test basic CSV reading"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(path, index=False)
            result = ar.io.read_csv(path)
            assert isinstance(result, ar.ArFrame)
            assert list(result.columns) == ["a", "b"]
            assert len(result) == 3

    def test_read_csv_with_delimiter(self):
        """Test CSV reading with custom delimiter"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            with open(path, 'w') as f:
                f.write("a;b;c\n1;2;3\n4;5;6\n")
            result = ar.io.read_csv(path, delimiter=";")
            assert list(result.columns) == ["a", "b", "c"]
            assert len(result) == 2

    def test_read_csv_no_header(self):
        """Test CSV reading without header"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            with open(path, 'w') as f:
                f.write("1,2,3\n4,5,6\n")
            result = ar.io.read_csv(path, has_header=False)
            assert len(result) == 2

    def test_read_csv_usecols(self):
        """Test CSV reading with column selection"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}).to_csv(path, index=False)
            result = ar.io.read_csv(path, usecols=["a", "c"])
            assert list(result.columns) == ["a", "c"]

    def test_read_csv_nrows(self):
        """Test CSV reading with row limit"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            pd.DataFrame({"a": range(100), "b": range(100, 200)}).to_csv(path, index=False)
            result = ar.io.read_csv(path, nrows=10)
            assert len(result) == 10

    def test_read_csv_skiprows(self):
        """Test CSV reading with row skipping"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            with open(path, 'w') as f:
                f.write("skip1\nskip2\na,b\n1,2\n3,4\n")
            result = ar.io.read_csv(path, skiprows=2)
            assert list(result.columns) == ["a", "b"]
            assert len(result) == 2

    def test_read_csv_with_encoding(self):
        """Test CSV reading with specific encoding"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            pd.DataFrame({"a": [1, 2], "b": ["café", "naïve"]}).to_csv(path, index=False, encoding="utf-8")
            result = ar.io.read_csv(path, encoding="utf-8")
            assert len(result) == 2

    def test_read_csv_with_pathlib(self):
        """Test CSV reading with pathlib.Path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            pd.DataFrame({"a": [1, 2, 3]}).to_csv(path, index=False)
            result = ar.io.read_csv(path)
            assert isinstance(result, ar.ArFrame)

    def test_read_csv_trim_headers(self):
        """Test CSV reading with header trimming"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            with open(path, 'w') as f:
                f.write(" a , b \n1,2\n")
            result = ar.io.read_csv(path, trim_headers=True)
            assert "a" in result.columns
            assert "b" in result.columns


class TestReadCSVChunked:
    """Test arnio.io.read_csv_chunked"""

    def test_read_csv_chunked_basic(self):
        """Test basic chunked CSV reading"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            pd.DataFrame({"a": range(100), "b": range(100, 200)}).to_csv(path, index=False)
            chunks = list(ar.io.read_csv_chunked(path, chunksize=10))
            assert len(chunks) == 10
            assert all(isinstance(c, ar.ArFrame) for c in chunks)

    def test_read_csv_chunked_chunk_size(self):
        """Test chunked CSV reading respects chunksize"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            pd.DataFrame({"a": range(100), "b": range(100, 200)}).to_csv(path, index=False)
            chunks = list(ar.io.read_csv_chunked(path, chunksize=25))
            assert all(len(c) <= 25 for c in chunks)
            total_rows = sum(len(c) for c in chunks)
            assert total_rows == 100

    def test_read_csv_chunked_returns_iterator(self):
        """Test that read_csv_chunked returns an iterator"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            pd.DataFrame({"a": [1, 2, 3]}).to_csv(path, index=False)
            result = ar.io.read_csv_chunked(path, chunksize=1)
            assert hasattr(result, '__iter__')

    def test_read_csv_chunked_skip_rows(self):
        """Test chunked CSV reading with skip_rows"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            pd.DataFrame({"a": range(100), "b": range(100, 200)}).to_csv(path, index=False)
            chunks = list(ar.io.read_csv_chunked(path, chunksize=10, skip_rows=5))
            total_rows = sum(len(c) for c in chunks)
            assert total_rows == 95

    def test_read_csv_chunked_with_nrows(self):
        """Test chunked CSV reading with nrows limit"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            pd.DataFrame({"a": range(100), "b": range(100, 200)}).to_csv(path, index=False)
            chunks = list(ar.io.read_csv_chunked(path, chunksize=10, nrows=30))
            total_rows = sum(len(c) for c in chunks)
            assert total_rows == 30

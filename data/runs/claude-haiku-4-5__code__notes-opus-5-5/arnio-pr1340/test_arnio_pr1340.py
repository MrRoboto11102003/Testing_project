import pytest
import pandas as pd
import numpy as np
import io
from pathlib import Path

# Skip tests if sklearn not installed
pytest.importorskip("sklearn")

import arnio as ar
from arnio.integrations.sklearn import ArnioCleaner
from arnio.exceptions import CsvReadError


class TestArnioCleaner:
    """Tests for ArnioCleaner.get_feature_names_out"""

    def test_not_fitted_raises_error(self):
        """Test NotFittedError when not fitted"""
        est = ArnioCleaner()
        with pytest.raises(Exception, match="not fitted"):
            est.get_feature_names_out()

    def test_returns_fitted_names_with_none(self):
        """Test returns fitted names when input_features=None"""
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        est = ArnioCleaner().fit(df)
        result = est.get_feature_names_out()
        assert list(result) == ["a", "b"]
        assert result is est.feature_names_out_

    def test_string_input_raises_typeerror(self):
        """Test string input raises TypeError"""
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        est = ArnioCleaner().fit(df)
        with pytest.raises(TypeError, match="sequence of strings, not str"):
            est.get_feature_names_out("ab")

    def test_bytes_input_raises_typeerror(self):
        """Test bytes input raises TypeError"""
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        est = ArnioCleaner().fit(df)
        with pytest.raises(TypeError, match="sequence of strings, not str"):
            est.get_feature_names_out(b"ab")

    def test_non_string_elements_raise_typeerror(self):
        """Test non-string elements raise TypeError before length check"""
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        est = ArnioCleaner().fit(df)
        with pytest.raises(TypeError, match="All input_features must be strings"):
            est.get_feature_names_out([1, 2])

    def test_wrong_names_raise_valueerror(self):
        """Test that wrong names raise ValueError"""
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        est = ArnioCleaner().fit(df)
        with pytest.raises(ValueError, match="must match columns seen during fit"):
            est.get_feature_names_out(["b", "a"])


class TestReadCsv:
    """Tests for arnio.io.read_csv"""

    def test_basic_csv_reading(self, tmp_path):
        """Test basic CSV reading"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n", newline="")
        frame = ar.read_csv(str(p))
        assert frame.shape == (2, 2)
        assert list(frame.columns) == ["a", "b"]

    def test_tsv_auto_detection(self, tmp_path):
        """Test .tsv files auto-detect tab delimiter"""
        p = tmp_path / "data.tsv"
        p.write_text("a\tb\n1\tx\n2\ty\n", newline="")
        frame = ar.read_csv(str(p))
        assert frame.shape == (2, 2)
        assert list(frame.columns) == ["a", "b"]

    def test_explicit_delimiter_overrides_tsv(self, tmp_path):
        """Test explicit delimiter overrides .tsv detection"""
        p = tmp_path / "data.tsv"
        p.write_text("a,b\n1,x\n2,y\n", newline="")
        frame = ar.read_csv(str(p), delimiter=",")
        assert frame.shape == (2, 2)

    def test_usecols_parameter(self, tmp_path):
        """Test usecols parameter"""
        p = tmp_path / "data.csv"
        p.write_text("a,b,c\n1,x,10\n2,y,20\n", newline="")
        frame = ar.read_csv(str(p), usecols=["a", "c"])
        assert frame.shape == (2, 2)
        assert list(frame.columns) == ["a", "c"]

    def test_nrows_parameter(self, tmp_path):
        """Test nrows parameter"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n3,z\n", newline="")
        frame = ar.read_csv(str(p), nrows=2)
        assert frame.shape == (2, 2)


class TestReadCsvChunked:
    """Tests for arnio.io.read_csv_chunked"""

    def test_basic_chunked_reading(self, tmp_path):
        """Test basic chunked reading"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n3,z\n", newline="")
        chunks = list(ar.read_csv_chunked(str(p), chunksize=2))
        assert len(chunks) == 2
        assert chunks[0].shape == (2, 2)
        assert chunks[1].shape == (1, 2)

    def test_chunksize_validation_on_iteration(self, tmp_path):
        """Test that chunksize validation happens on first iteration"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n", newline="")
        gen = ar.read_csv_chunked(str(p), chunksize=0)
        with pytest.raises(ValueError, match="chunksize must be a positive integer"):
            next(gen)

    def test_skiprows_conflict_raises(self, tmp_path):
        """Test conflicting skiprows and skip_rows raise ValueError"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n", newline="")
        gen = ar.read_csv_chunked(str(p), skip_rows=1, skiprows=2)
        with pytest.raises(ValueError, match="Conflicting values"):
            next(gen)

    def test_skiprows_as_alias(self, tmp_path):
        """Test skiprows as alias for skip_rows"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n3,z\n", newline="")
        chunks = list(ar.read_csv_chunked(str(p), chunksize=10, skiprows=1))
        assert chunks[0].shape == (2, 2)

import pytest
import pandas as pd
import numpy as np
import io
import gzip
import os
from pathlib import Path
import re

pytest.importorskip("arnio")
pytest.importorskip("sklearn")

import arnio as ar
from arnio.exceptions import CsvReadError
from sklearn.exceptions import NotFittedError
from arnio.integrations.sklearn import ArnioCleaner


class TestArnicCleanerGetFeatureNamesOut:
    """Tests for ArnioCleaner.get_feature_names_out."""

    def test_not_fitted_raises_error(self):
        """Should raise NotFittedError when estimator is not fitted."""
        cleaner = ArnioCleaner()
        with pytest.raises(NotFittedError, match="not fitted"):
            cleaner.get_feature_names_out()

    def test_returns_fitted_names_with_none_input(self):
        """Should return fitted feature names when input_features is None."""
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        cleaner = ArnioCleaner().fit(df)
        
        result = cleaner.get_feature_names_out()
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == object
        assert list(result) == ["a", "b"]
        assert result is cleaner.feature_names_out_

    def test_validates_input_features_sequentially(self):
        """Should validate input_features in correct order."""
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        cleaner = ArnioCleaner().fit(df)
        
        # String input raises first check
        with pytest.raises(TypeError, match="must be a sequence of strings, not str"):
            cleaner.get_feature_names_out("ab")
        
        # Bytes input raises same check
        with pytest.raises(TypeError, match="must be a sequence of strings, not str"):
            cleaner.get_feature_names_out(b"ab")
        
        # Non-sequence (checked before length)
        with pytest.raises(TypeError, match="must be a sequence of strings"):
            cleaner.get_feature_names_out(5)
        
        # Non-string element (checked before length)
        with pytest.raises(TypeError, match="All input_features must be strings"):
            cleaner.get_feature_names_out([1, 2])
        
        # Wrong length
        with pytest.raises(ValueError, match=re.escape("should have length 2")):
            cleaner.get_feature_names_out(["a"])
        
        # Wrong names or order
        with pytest.raises(ValueError, match="must match columns seen during fit"):
            cleaner.get_feature_names_out(["b", "a"])

    def test_accepts_various_sequence_types(self):
        """Should accept list, tuple, and numpy array of strings."""
        df = pd.DataFrame({"x": [1], "y": [2]})
        cleaner = ArnioCleaner().fit(df)
        
        # List
        assert list(cleaner.get_feature_names_out(["x", "y"])) == ["x", "y"]
        # Tuple
        assert list(cleaner.get_feature_names_out(("x", "y"))) == ["x", "y"]
        # Numpy array
        assert list(cleaner.get_feature_names_out(np.array(["x", "y"]))) == ["x", "y"]


class TestReadCsv:
    """Tests for ar.read_csv function."""

    def test_reads_basic_csv(self, tmp_path):
        """Should read a basic CSV file."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n", newline="")
        
        frame = ar.read_csv(str(csv_file))
        
        assert frame.shape == (2, 2)
        assert frame.columns == ["a", "b"]

    def test_tsv_auto_detection(self, tmp_path):
        """Should auto-detect tab delimiter for .tsv files."""
        tsv_file = tmp_path / "data.tsv"
        tsv_file.write_text("a\tb\n1\t2\n3\t4\n", newline="")
        
        frame = ar.read_csv(str(tsv_file))
        
        assert frame.shape == (2, 2)
        assert frame.columns == ["a", "b"]

    def test_explicit_delimiter_overrides_extension(self, tmp_path):
        """Should honor explicit delimiter over .tsv extension."""
        tsv_file = tmp_path / "data.tsv"
        tsv_file.write_text("a,b\n1,2\n", newline="")
        
        # Override tab detection with comma
        frame = ar.read_csv(str(tsv_file), delimiter=",")
        
        assert frame.shape == (1, 2)
        assert frame.columns == ["a", "b"]

    def test_empty_file_raises_error(self, tmp_path):
        """Should raise CsvReadError for empty files."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("", newline="")
        
        with pytest.raises(CsvReadError, match="CSV file is empty"):
            ar.read_csv(str(csv_file))

    def test_gz_file_reading(self, tmp_path):
        """Should read gzipped CSV files."""
        gz_file = tmp_path / "data.csv.gz"
        with gzip.open(str(gz_file), "wt") as f:
            f.write("a,b\n1,2\n3,4\n")
        
        frame = ar.read_csv(str(gz_file))
        
        assert frame.shape == (2, 2)
        assert frame.columns == ["a", "b"]

    def test_file_like_object_reading(self):
        """Should read from StringIO file-like objects."""
        csv_io = io.StringIO("a,b\n1,2\n3,4\n")
        
        frame = ar.read_csv(csv_io)
        
        assert frame.shape == (2, 2)
        assert frame.columns == ["a", "b"]

    def test_parameter_validation_has_header(self, tmp_path):
        """Should validate has_header as bool (not int)."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n", newline="")
        
        with pytest.raises(TypeError, match="has_header must be True or False"):
            ar.read_csv(str(csv_file), has_header=1)

    def test_parameter_validation_encoding(self, tmp_path):
        """Should raise error for unknown encoding."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n", newline="")
        
        with pytest.raises(ValueError, match="Unknown encoding"):
            ar.read_csv(str(csv_file), encoding="bogus")

    def test_usecols_filtering(self, tmp_path):
        """Should filter to specified columns."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n", newline="")
        
        frame = ar.read_csv(str(csv_file), usecols=["a", "c"])
        
        assert frame.columns == ["a", "c"]

    def test_nrows_limiting(self, tmp_path):
        """Should respect nrows parameter."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n5,6\n", newline="")
        
        frame = ar.read_csv(str(csv_file), nrows=2)
        
        assert frame.shape[0] == 2

    def test_pathlike_input(self, tmp_path):
        """Should accept os.PathLike input."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n", newline="")
        
        frame = ar.read_csv(csv_file)  # Pass Path object directly
        
        assert frame.shape == (1, 2)


class TestReadCsvChunked:
    """Tests for ar.read_csv_chunked function."""

    def test_chunked_reading_basic(self, tmp_path):
        """Should yield ArFrame chunks with specified chunksize."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n5,6\n7,8\n", newline="")
        
        chunks = list(ar.read_csv_chunked(str(csv_file), chunksize=2))
        
        assert len(chunks) == 2
        assert chunks[0].shape == (2, 2)
        assert chunks[1].shape == (2, 2)

    def test_chunked_errors_on_first_iteration(self, tmp_path):
        """Should raise validation errors on first next(), not on function call."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n", newline="")
        
        # Invalid chunksize should not raise immediately
        gen = ar.read_csv_chunked(str(csv_file), chunksize=0)
        
        # But should raise on first next()
        with pytest.raises(ValueError, match="chunksize must be a positive integer"):
            next(gen)

    def test_chunked_with_gz_file(self, tmp_path):
        """Should read gzipped CSV files in chunks."""
        gz_file = tmp_path / "data.csv.gz"
        with gzip.open(str(gz_file), "wt") as f:
            f.write("a,b\n1,2\n3,4\n5,6\n")
        
        chunks = list(ar.read_csv_chunked(str(gz_file), chunksize=2))
        
        assert len(chunks) == 2
        assert all(c.shape[1] == 2 for c in chunks)

    def test_chunked_bad_extension_raises_error(self, tmp_path):
        """Should raise ValueError for unsupported file extensions."""
        bad_file = tmp_path / "data.dat"
        bad_file.write_text("a,b\n1,2\n", newline="")
        
        gen = ar.read_csv_chunked(str(bad_file), chunksize=10)
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            next(gen)

    def test_chunked_skiprows_skiprows_conflict(self, tmp_path):
        """Should raise ValueError when skip_rows and skiprows conflict."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n", newline="")
        
        gen = ar.read_csv_chunked(
            str(csv_file), chunksize=10, skip_rows=1, skiprows=2
        )
        
        with pytest.raises(ValueError, match="Conflicting values"):
            next(gen)

    def test_chunked_file_like_object(self):
        """Should handle StringIO file-like objects in chunks."""
        csv_io = io.StringIO("a,b\n1,2\n3,4\n5,6\n")
        
        chunks = list(ar.read_csv_chunked(csv_io, chunksize=2))
        
        assert len(chunks) == 2
        assert chunks[0].shape[0] == 2
        assert chunks[1].shape[0] == 2

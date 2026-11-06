"""
Pytest unit tests for arnio library functions:
- arnio.io.read_csv
- arnio.io.read_csv_chunked
- arnio.integrations.sklearn.ArnioCleaner.get_feature_names_out
"""

import pytest
import pandas as pd
import numpy as np
import arnio as ar
from arnio.integrations.sklearn import ArnioCleaner
import io
import tempfile
import os
from pathlib import Path


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def csv_file_simple():
    """Simple 3-row CSV file."""
    data = "id,name,age\n1,Alice,30\n2,Bob,25\n3,Charlie,35\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(data)
        fname = f.name
    yield fname
    if os.path.exists(fname):
        os.unlink(fname)


@pytest.fixture
def tsv_file():
    """Simple TSV file with tab delimiter."""
    data = "id\tname\tage\n1\tAlice\t30\n2\tBob\t25\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write(data)
        fname = f.name
    yield fname
    if os.path.exists(fname):
        os.unlink(fname)


@pytest.fixture
def csv_file_large():
    """Larger CSV file (150 rows) for chunking tests."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("id,value\n")
        for i in range(150):
            f.write(f"{i},{i*10}\n")
        fname = f.name
    yield fname
    if os.path.exists(fname):
        os.unlink(fname)


@pytest.fixture
def fitted_cleaner():
    """Create and fit an ArnioCleaner instance."""
    df = pd.DataFrame({
        "feat_a": ["x", "y", "z"],
        "feat_b": [1, 2, 3],
        "feat_c": [1.0, 2.0, 3.0]
    })
    cleaner = ArnioCleaner(steps=[("strip_whitespace",)])
    cleaner.fit(df)
    return cleaner


# ============================================================================
# Tests: read_csv
# ============================================================================

class TestReadCsv:
    """Test arnio.io.read_csv function."""

    def test_read_csv_basic_from_path(self, csv_file_simple):
        """Test basic CSV reading from file path."""
        frame = ar.read_csv(csv_file_simple)
        assert frame.shape == (3, 3)
        assert frame.columns == ["id", "name", "age"]

    def test_read_csv_with_usecols(self, csv_file_simple):
        """Test reading with column selection via usecols."""
        frame = ar.read_csv(csv_file_simple, usecols=["id", "name"])
        assert frame.shape == (3, 2)
        assert set(frame.columns) == {"id", "name"}

    def test_read_csv_with_nrows_limit(self, csv_file_simple):
        """Test limiting number of rows read."""
        frame = ar.read_csv(csv_file_simple, nrows=2)
        assert frame.shape[0] == 2
        assert frame.shape[1] == 3

    def test_read_csv_tsv_auto_detection(self, tsv_file):
        """Test automatic TSV delimiter detection from .tsv extension."""
        frame = ar.read_csv(tsv_file)
        assert frame.shape == (2, 3)
        assert frame.columns == ["id", "name", "age"]

    def test_read_csv_explicit_delimiter_override(self, tsv_file):
        """Test explicit delimiter parameter overrides file extension."""
        frame = ar.read_csv(tsv_file, delimiter="\t")
        assert frame.shape[1] == 3

    def test_read_csv_pathlib_support(self, csv_file_simple):
        """Test reading from pathlib.Path object."""
        frame = ar.read_csv(Path(csv_file_simple))
        assert frame.shape == (3, 3)

    def test_read_csv_file_object_support(self, csv_file_simple):
        """Test reading from file-like object."""
        with open(csv_file_simple) as f:
            frame = ar.read_csv(f)
        assert frame.shape == (3, 3)

    def test_read_csv_invalid_nrows_raises_error(self, csv_file_simple):
        """Test that negative nrows raises ValueError or TypeError."""
        with pytest.raises((ValueError, TypeError)):
            ar.read_csv(csv_file_simple, nrows=-1)


# ============================================================================
# Tests: read_csv_chunked
# ============================================================================

class TestReadCsvChunked:
    """Test arnio.io.read_csv_chunked function."""

    def test_read_csv_chunked_basic_iteration(self, csv_file_large):
        """Test basic chunked reading yields correct number of chunks."""
        chunks = list(ar.read_csv_chunked(csv_file_large, chunksize=50))
        # 150 rows / 50 per chunk = 3 chunks
        assert len(chunks) == 3
        assert all(chunk.shape[0] == 50 for chunk in chunks)

    def test_read_csv_chunked_column_consistency(self, csv_file_large):
        """Test all chunks have same columns and dtypes."""
        chunks = list(ar.read_csv_chunked(csv_file_large, chunksize=60))
        first_cols = chunks[0].columns
        for chunk in chunks:
            assert chunk.columns == first_cols

    def test_read_csv_chunked_with_usecols(self, csv_file_large):
        """Test chunked reading with column subset."""
        chunks = list(ar.read_csv_chunked(csv_file_large, usecols=["id"], chunksize=50))
        for chunk in chunks:
            assert chunk.columns == ["id"]
            assert chunk.shape[0] == 50

    def test_read_csv_chunked_nrows_limit(self, csv_file_large):
        """Test total row limit across all chunks."""
        chunks = list(ar.read_csv_chunked(csv_file_large, nrows=75, chunksize=50))
        total_rows = sum(c.shape[0] for c in chunks)
        assert total_rows == 75

    def test_read_csv_chunked_tsv_auto_detection(self, tsv_file):
        """Test TSV auto-detection works in chunked mode."""
        chunks = list(ar.read_csv_chunked(tsv_file, chunksize=10))
        assert len(chunks) >= 1
        assert chunks[0].shape[1] == 3


# ============================================================================
# Tests: ArnioCleaner.get_feature_names_out
# ============================================================================

class TestGetFeatureNamesOut:
    """Test ArnioCleaner.get_feature_names_out method."""

    def test_get_feature_names_out_none_input(self, fitted_cleaner):
        """Test default behavior: returns fitted feature names when input is None."""
        result = fitted_cleaner.get_feature_names_out()
        assert len(result) == 3

    def test_get_feature_names_out_matching_features(self, fitted_cleaner):
        """Test with matching feature names in correct order."""
        result = fitted_cleaner.get_feature_names_out(
            input_features=["feat_a", "feat_b", "feat_c"]
        )
        assert len(result) == 3

    def test_get_feature_names_out_string_input_raises_error(self, fitted_cleaner):
        """Test that passing a string raises TypeError."""
        with pytest.raises(TypeError, match="sequence of strings"):
            fitted_cleaner.get_feature_names_out(input_features="feat_a")

    def test_get_feature_names_out_wrong_length_raises_error(self, fitted_cleaner):
        """Test mismatched feature count raises ValueError."""
        with pytest.raises(ValueError, match="should have length"):
            fitted_cleaner.get_feature_names_out(
                input_features=["feat_a", "feat_b"]
            )

    def test_get_feature_names_out_mismatched_names_raises_error(self, fitted_cleaner):
        """Test mismatched feature names raise ValueError."""
        with pytest.raises(ValueError, match="must match columns"):
            fitted_cleaner.get_feature_names_out(
                input_features=["feat_a", "feat_b", "wrong_name"]
            )

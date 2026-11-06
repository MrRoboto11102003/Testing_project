import pytest
import pandas as pd
import numpy as np
import io
import tempfile
import os
from pathlib import Path

import arnio as ar
from arnio.exceptions import CsvReadError


class TestGetFeatureNamesOut:
    """Test ArnioCleaner.get_feature_names_out compatibility."""
    
    def setup_method(self):
        """Create a fitted cleaner for testing."""
        self.df = pd.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})
        self.frame = ar.from_pandas(self.df)
    
    def test_returns_fitted_names_when_none(self):
        """When input_features is None, return fitted feature names."""
        from arnio.integrations.sklearn import ArnioCleaner
        
        cleaner = ArnioCleaner(steps=[])
        cleaner.fit(self.frame)
        result = cleaner.get_feature_names_out(None)
        np.testing.assert_array_equal(result, ["id", "name"])
    
    def test_accepts_matching_input_features(self):
        """Accept input_features when they match fitted columns."""
        from arnio.integrations.sklearn import ArnioCleaner
        
        cleaner = ArnioCleaner(steps=[])
        cleaner.fit(self.frame)
        result = cleaner.get_feature_names_out(["id", "name"])
        np.testing.assert_array_equal(result, ["id", "name"])
    
    def test_raises_typeerror_on_string_input(self):
        """TypeError when input_features is a string instead of sequence."""
        from arnio.integrations.sklearn import ArnioCleaner
        
        cleaner = ArnioCleaner(steps=[])
        cleaner.fit(self.frame)
        with pytest.raises(TypeError, match="sequence of strings"):
            cleaner.get_feature_names_out("id")
    
    def test_raises_typeerror_on_non_sequence(self):
        """TypeError when input_features is not a sequence."""
        from arnio.integrations.sklearn import ArnioCleaner
        
        cleaner = ArnioCleaner(steps=[])
        cleaner.fit(self.frame)
        with pytest.raises(TypeError, match="sequence of strings"):
            cleaner.get_feature_names_out(123)
    
    def test_raises_typeerror_on_non_string_elements(self):
        """TypeError when input_features contains non-strings."""
        from arnio.integrations.sklearn import ArnioCleaner
        
        cleaner = ArnioCleaner(steps=[])
        cleaner.fit(self.frame)
        with pytest.raises(TypeError, match="All input_features must be strings"):
            cleaner.get_feature_names_out(["id", 1])
    
    def test_raises_valueerror_on_length_mismatch(self):
        """ValueError when input_features length doesn't match n_features_in_."""
        from arnio.integrations.sklearn import ArnioCleaner
        
        cleaner = ArnioCleaner(steps=[])
        cleaner.fit(self.frame)
        with pytest.raises(ValueError, match="should have length 2"):
            cleaner.get_feature_names_out(["id"])
    
    def test_raises_valueerror_on_mismatched_names(self):
        """ValueError when input_features don't match fitted columns."""
        from arnio.integrations.sklearn import ArnioCleaner
        
        cleaner = ArnioCleaner(steps=[])
        cleaner.fit(self.frame)
        with pytest.raises(ValueError, match="must match columns seen during fit"):
            cleaner.get_feature_names_out(["x", "y"])


class TestReadCsv:
    """Test arnio.read_csv functionality."""
    
    def test_basic_csv_read(self):
        """Read a basic CSV file successfully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,name\n1,alice\n2,bob\n")
            path = f.name
        
        try:
            frame = ar.read_csv(path)
            assert frame.shape == (2, 2)
            assert frame.columns == ["id", "name"]
        finally:
            os.unlink(path)
    
    def test_read_with_usecols_filter(self):
        """Read only specified columns via usecols."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,name,age\n1,alice,30\n2,bob,25\n")
            path = f.name
        
        try:
            frame = ar.read_csv(path, usecols=["id", "name"])
            assert frame.columns == ["id", "name"]
            assert frame.shape == (2, 2)
        finally:
            os.unlink(path)
    
    def test_read_with_nrows_limit(self):
        """Read limited number of rows via nrows parameter."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id\n1\n2\n3\n4\n")
            path = f.name
        
        try:
            frame = ar.read_csv(path, nrows=2)
            assert frame.shape[0] == 2
        finally:
            os.unlink(path)
    
    def test_tsv_auto_delimiter(self):
        """Auto-detect tab delimiter for .tsv file extension."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("id\tname\n1\talice\n2\tbob\n")
            path = f.name
        
        try:
            frame = ar.read_csv(path)
            assert frame.shape == (2, 2)
            assert "id" in frame.columns and "name" in frame.columns
        finally:
            os.unlink(path)
    
    def test_explicit_delimiter_precedence(self):
        """Explicit delimiter parameter overrides file extension auto-detection."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("id,name\n1,alice\n")
            path = f.name
        
        try:
            frame = ar.read_csv(path, delimiter=",")
            assert frame.shape == (1, 2)
            assert "id" in frame.columns
        finally:
            os.unlink(path)
    
    def test_skiprows_preamble_skip(self):
        """Skip lines before header using skiprows parameter."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("# Comment\nid,name\n1,alice\n")
            path = f.name
        
        try:
            frame = ar.read_csv(path, skiprows=1)
            assert frame.shape == (1, 2)
            assert frame.columns == ["id", "name"]
        finally:
            os.unlink(path)
    
    def test_null_values_custom_sentinels(self):
        """Recognize custom null value sentinels."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,status\n1,MISSING\n2,active\n")
            path = f.name
        
        try:
            frame = ar.read_csv(path, null_values=["MISSING"])
            df = ar.to_pandas(frame)
            assert pd.isna(df.iloc[0, 1])
            assert df.iloc[1, 1] == "active"
        finally:
            os.unlink(path)
    
    def test_permissive_mode_fills_trailing_nulls(self):
        """Permissive mode fills missing trailing fields with nulls."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("a,b,c\n1,2,3\n4,5\n")
            path = f.name
        
        try:
            frame = ar.read_csv(path, mode="permissive")
            assert frame.shape == (2, 3)
            df = ar.to_pandas(frame)
            assert pd.isna(df.iloc[1, 2])
        finally:
            os.unlink(path)
    
    def test_strict_mode_rejects_short_rows(self):
        """Strict mode raises CsvReadError on inconsistent row widths."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("a,b,c\n1,2,3\n4,5\n")
            path = f.name
        
        try:
            with pytest.raises(CsvReadError):
                ar.read_csv(path, mode="strict")
        finally:
            os.unlink(path)
    
    def test_pathlib_path_input(self):
        """Accept pathlib.Path objects as input."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("x\n1\n")
            path = Path(f.name)
        
        try:
            frame = ar.read_csv(path)
            assert frame.shape == (1, 1)
        finally:
            os.unlink(path)
    
    def test_stringio_file_like_input(self):
        """Accept io.StringIO file-like objects."""
        csv_text = "a,b\n1,2\n3,4\n"
        frame = ar.read_csv(io.StringIO(csv_text))
        assert frame.shape == (2, 2)


class TestReadCsvChunked:
    """Test arnio.read_csv_chunked functionality."""
    
    def test_chunked_iteration_with_chunksize(self):
        """Read CSV file in bounded chunks."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id\n")
            for i in range(25):
                f.write(f"{i}\n")
            path = f.name
        
        try:
            chunks = list(ar.read_csv_chunked(path, chunksize=10))
            assert len(chunks) == 3
            assert chunks[0].shape[0] == 10
            assert chunks[1].shape[0] == 10
            assert chunks[2].shape[0] == 5
        finally:
            os.unlink(path)
    
    def test_chunked_tsv_auto_detection(self):
        """Auto-detect tab delimiter in chunked reader."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("id\tval\n")
            for i in range(15):
                f.write(f"{i}\t{i*2}\n")
            path = f.name
        
        try:
            chunks = list(ar.read_csv_chunked(path, chunksize=10))
            assert chunks[0].columns == ["id", "val"]
            assert len(chunks) == 2
        finally:
            os.unlink(path)
    
    def test_conflicting_skip_params_raises_error(self):
        """ValueError when skip_rows and skiprows parameters conflict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id\n1\n2\n")
            path = f.name
        
        try:
            with pytest.raises(ValueError, match="Conflicting values"):
                list(ar.read_csv_chunked(path, skip_rows=1, skiprows=2))
        finally:
            os.unlink(path)

import pytest
import arnio as ar
import pandas as pd
import numpy as np


class TestArnioCleaner:
    """Tests for ArnioCleaner.get_feature_names_out"""
    
    @pytest.fixture
    def fitted_cleaner(self):
        """Create a fitted ArnioCleaner"""
        try:
            from arnio.integrations.sklearn import ArnioCleaner
        except ImportError:
            pytest.skip("ArnioCleaner not available")
        
        df = pd.DataFrame({
            'feat_1': [1, 2, 3],
            'feat_2': [4, 5, 6],
            'feat_3': [7, 8, 9]
        })
        cleaner = ArnioCleaner()
        cleaner.fit(df)
        return cleaner
    
    def test_none_input_returns_fitted_names(self, fitted_cleaner):
        """Returns fitted feature names when input_features is None"""
        result = fitted_cleaner.get_feature_names_out(None)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
    
    def test_matching_input_returns_names(self, fitted_cleaner):
        """Returns feature names when input matches fitted"""
        result = fitted_cleaner.get_feature_names_out(['feat_1', 'feat_2', 'feat_3'])
        assert isinstance(result, np.ndarray)
    
    def test_string_input_raises_typeerror(self, fitted_cleaner):
        """Raises TypeError for string input"""
        with pytest.raises(TypeError, match="sequence"):
            fitted_cleaner.get_feature_names_out('feat_1')
    
    def test_wrong_length_raises_valueerror(self, fitted_cleaner):
        """Raises ValueError for wrong length"""
        with pytest.raises(ValueError, match="length"):
            fitted_cleaner.get_feature_names_out(['feat_1', 'feat_2'])
    
    def test_wrong_names_raises_valueerror(self, fitted_cleaner):
        """Raises ValueError for non-matching names"""
        with pytest.raises(ValueError, match="must match"):
            fitted_cleaner.get_feature_names_out(['wrong_1', 'wrong_2', 'wrong_3'])


class TestReadCsv:
    """Tests for arnio.io.read_csv"""
    
    @pytest.fixture
    def csv_file(self, tmp_path):
        """Create a test CSV file"""
        path = tmp_path / "data.csv"
        path.write_text("name,age,score\nAlice,30,95\nBob,25,87\nCharlie,35,92\n")
        return str(path)
    
    @pytest.fixture
    def tsv_file(self, tmp_path):
        """Create a test TSV file"""
        path = tmp_path / "data.tsv"
        path.write_text("name\tage\tscore\nAlice\t30\t95\nBob\t25\t87\n")
        return str(path)
    
    def test_read_csv_basic(self, csv_file):
        """Reads CSV into ArFrame with correct shape"""
        frame = ar.read_csv(csv_file)
        df = ar.to_pandas(frame)
        assert df.shape == (3, 3)
        assert set(df.columns) == {'name', 'age', 'score'}
    
    def test_tsv_delimiter_autodetection(self, tsv_file):
        """Auto-detects TSV delimiter from .tsv extension"""
        frame = ar.read_csv(tsv_file)
        df = ar.to_pandas(frame)
        assert df.shape == (2, 3)
    
    def test_usecols_filters_columns(self, csv_file):
        """Filters columns when usecols specified"""
        frame = ar.read_csv(csv_file, usecols=['name', 'age'])
        df = ar.to_pandas(frame)
        assert list(df.columns) == ['name', 'age']
        assert df.shape[1] == 2
    
    def test_nrows_limits_rows(self, csv_file):
        """Limits rows when nrows specified"""
        frame = ar.read_csv(csv_file, nrows=2)
        df = ar.to_pandas(frame)
        assert df.shape[0] == 2


class TestReadCsvChunked:
    """Tests for arnio.io.read_csv_chunked"""
    
    @pytest.fixture
    def chunked_csv(self, tmp_path):
        """Create CSV with 100 rows for chunking"""
        path = tmp_path / "chunked.csv"
        lines = ["id,value"]
        for i in range(100):
            lines.append(f"{i},{i*10}")
        path.write_text("\n".join(lines))
        return str(path)
    
    def test_read_csv_chunked_yields_arframes(self, chunked_csv):
        """Yields ArFrame objects in iterator"""
        chunks = list(ar.read_csv_chunked(chunked_csv, chunksize=30))
        assert len(chunks) >= 2
        assert all(isinstance(c, ar.ArFrame) for c in chunks)
    
    def test_read_csv_chunked_respects_chunksize(self, chunked_csv):
        """Produces chunks of requested size"""
        chunks = list(ar.read_csv_chunked(chunked_csv, chunksize=30))
        for chunk in chunks[:-1]:
            df = ar.to_pandas(chunk)
            assert df.shape[0] == 30
    
    def test_read_csv_chunked_with_usecols(self, chunked_csv):
        """Respects usecols parameter in chunked mode"""
        chunks = list(ar.read_csv_chunked(chunked_csv, usecols=['id'], chunksize=30))
        df = ar.to_pandas(chunks[0])
        assert list(df.columns) == ['id']

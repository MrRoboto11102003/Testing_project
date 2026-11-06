import pytest
import os
import tempfile
from pathlib import Path
import pandas as pd
import arnio as ar


class TestWriteCsv:
    """Test arnio.write_csv functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 88.0, 92.5]
        })
        return ar.from_pandas(df)

    def test_write_csv_basic_default_parameters(self, temp_dir, sample_frame):
        """Test basic CSV writing with default parameters."""
        output_path = os.path.join(temp_dir, "output.csv")
        ar.write_csv(sample_frame, output_path)
        assert os.path.exists(output_path)
        
        df = pd.read_csv(output_path)
        assert list(df.columns) == ["id", "name", "score"]
        assert len(df) == 3

    def test_write_csv_supported_extensions(self, temp_dir, sample_frame):
        """Test writing to .csv, .txt, and .tsv file extensions."""
        for ext in ["output.csv", "output.txt", "output.tsv"]:
            path = os.path.join(temp_dir, ext)
            ar.write_csv(sample_frame, path)
            assert os.path.exists(path)

    def test_write_csv_custom_delimiter(self, temp_dir, sample_frame):
        """Test writing with custom delimiter."""
        output_path = os.path.join(temp_dir, "output.csv")
        ar.write_csv(sample_frame, output_path, delimiter="|")
        
        df = pd.read_csv(output_path, delimiter="|")
        assert list(df.columns) == ["id", "name", "score"]

    def test_write_csv_header_control(self, temp_dir, sample_frame):
        """Test write_header parameter."""
        output_path = os.path.join(temp_dir, "output.csv")
        ar.write_csv(sample_frame, output_path, write_header=False)
        
        df = pd.read_csv(output_path, header=None)
        assert len(df) == 3
        assert df.iloc[0, 0] == 1  # First row should be data

    def test_write_csv_pathlib_path_object(self, temp_dir, sample_frame):
        """Test writing with pathlib.Path object."""
        output_path = Path(temp_dir) / "output.csv"
        ar.write_csv(sample_frame, output_path)
        assert output_path.exists()

    def test_write_csv_line_terminators(self, temp_dir, sample_frame):
        """Test different line terminators."""
        for terminator in ["\n", "\r\n", "\r"]:
            path = os.path.join(temp_dir, f"output_{repr(terminator)}.csv")
            ar.write_csv(sample_frame, path, line_terminator=terminator)
            assert os.path.exists(path)

    def test_write_csv_escape_formulas_feature(self, temp_dir):
        """Test escape_formulas parameter with formula trigger characters."""
        df = pd.DataFrame({"col": ["=SUM(A1)", "+1", "-5", "@indirect", "normal"]})
        frame = ar.from_pandas(df)
        output_path = os.path.join(temp_dir, "output.csv")
        
        ar.write_csv(frame, output_path, escape_formulas=True)
        assert os.path.exists(output_path)

    def test_write_csv_encoding_utf8(self, temp_dir, sample_frame):
        """Test UTF-8 encoding (default fast path)."""
        output_path = os.path.join(temp_dir, "output.csv")
        ar.write_csv(sample_frame, output_path, encoding="utf-8")
        
        df = pd.read_csv(output_path, encoding="utf-8")
        assert len(df) == 3

    def test_write_csv_encoding_latin1(self, temp_dir, sample_frame):
        """Test non-UTF-8 encoding (transcoding path)."""
        output_path = os.path.join(temp_dir, "output.csv")
        ar.write_csv(sample_frame, output_path, encoding="latin-1")
        
        df = pd.read_csv(output_path, encoding="latin-1")
        assert len(df) == 3

    def test_write_csv_encoding_errors_modes(self, temp_dir, sample_frame):
        """Test encoding_errors parameter modes."""
        for mode in ["strict", "replace", "ignore"]:
            path = os.path.join(temp_dir, f"output_{mode}.csv")
            ar.write_csv(sample_frame, path, encoding="utf-8", encoding_errors=mode)
            assert os.path.exists(path)

    def test_write_csv_error_invalid_frame_type(self, temp_dir):
        """Test TypeError when frame is not ArFrame."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        path = os.path.join(temp_dir, "output.csv")
        
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_csv(df, path)

    def test_write_csv_error_unsupported_extension(self, temp_dir, sample_frame):
        """Test ValueError for unsupported file extension."""
        path = os.path.join(temp_dir, "output.json")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_csv(sample_frame, path)

    def test_write_csv_error_invalid_line_terminator(self, temp_dir, sample_frame):
        """Test ValueError for invalid line terminator."""
        path = os.path.join(temp_dir, "output.csv")
        
        with pytest.raises(ValueError, match="line_terminator must be"):
            ar.write_csv(sample_frame, path, line_terminator="\n\n")

    def test_write_csv_error_invalid_encoding(self, temp_dir, sample_frame):
        """Test ValueError for unknown encoding codec."""
        path = os.path.join(temp_dir, "output.csv")
        
        with pytest.raises(ValueError):
            ar.write_csv(sample_frame, path, encoding="unknown-codec-xyz")

    def test_write_csv_error_invalid_encoding_errors(self, temp_dir, sample_frame):
        """Test ValueError for invalid encoding_errors option."""
        path = os.path.join(temp_dir, "output.csv")
        
        with pytest.raises(ValueError):
            ar.write_csv(sample_frame, path, encoding_errors="invalid")

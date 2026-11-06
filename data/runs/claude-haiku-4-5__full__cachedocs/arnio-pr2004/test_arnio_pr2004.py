import pytest
import os
import tempfile
from pathlib import Path
import arnio as ar
import pandas as pd


class TestWriteCsv:
    """Tests for ar.write_csv function."""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "score": [95.5, 88.0, 92.3],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def temp_csv(self):
        """Temporary directory for output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_write_csv_basic(self, sample_frame, temp_csv):
        """Test basic CSV write with default parameters."""
        output_path = os.path.join(temp_csv, "output.csv")
        ar.write_csv(sample_frame, output_path)

        assert os.path.exists(output_path)
        df = pd.read_csv(output_path)
        assert list(df.columns) == ["name", "age", "score"]
        assert len(df) == 3

    def test_write_csv_custom_delimiter_semicolon(self, sample_frame, temp_csv):
        """Test CSV write with semicolon delimiter."""
        output_path = os.path.join(temp_csv, "output.csv")
        ar.write_csv(sample_frame, output_path, delimiter=";")

        df = pd.read_csv(output_path, delimiter=";")
        assert list(df.columns) == ["name", "age", "score"]
        assert len(df) == 3

    def test_write_csv_tab_delimiter(self, sample_frame, temp_csv):
        """Test TSV write with tab delimiter."""
        output_path = os.path.join(temp_csv, "output.tsv")
        ar.write_csv(sample_frame, output_path, delimiter="\t")

        df = pd.read_csv(output_path, delimiter="\t")
        assert list(df.columns) == ["name", "age", "score"]

    def test_write_csv_without_header(self, sample_frame, temp_csv):
        """Test CSV write without header row."""
        output_path = os.path.join(temp_csv, "output.csv")
        ar.write_csv(sample_frame, output_path, write_header=False)

        df = pd.read_csv(output_path, header=None)
        assert len(df) == 3
        assert df.iloc[0, 0] == "Alice"

    def test_write_csv_windows_line_terminator(self, sample_frame, temp_csv):
        """Test CSV write with Windows-style line terminator."""
        output_path = os.path.join(temp_csv, "output.csv")
        ar.write_csv(sample_frame, output_path, line_terminator="\r\n")

        with open(output_path, "rb") as f:
            content = f.read()
        assert b"\r\n" in content

    def test_write_csv_formula_escape_equals(self, temp_csv):
        """Test formula escaping for cells starting with '='."""
        df = pd.DataFrame({"formula": ["=SUM(A1:A10)", "normal"]})
        frame = ar.from_pandas(df)
        output_path = os.path.join(temp_csv, "output.csv")

        ar.write_csv(frame, output_path, escape_formulas=True)

        with open(output_path, "r") as f:
            content = f.read()
        # Formula should be escaped with single quote
        assert "'=SUM" in content or "=\"=" in content

    def test_write_csv_formula_escape_plus(self, temp_csv):
        """Test formula escaping for cells starting with '+'."""
        df = pd.DataFrame({"formula": ["+EXTERNAL", "normal"]})
        frame = ar.from_pandas(df)
        output_path = os.path.join(temp_csv, "output.csv")

        ar.write_csv(frame, output_path, escape_formulas=True)

        with open(output_path, "r") as f:
            content = f.read()
        assert "'+EXTERNAL" in content or "\"++" in content

    def test_write_csv_encoding_utf8(self, temp_csv):
        """Test CSV write with UTF-8 encoding."""
        df = pd.DataFrame({"name": ["Αλίκη", "Бобу", "Charlie"]})
        frame = ar.from_pandas(df)
        output_path = os.path.join(temp_csv, "output.csv")

        ar.write_csv(frame, output_path, encoding="utf-8")

        df_read = pd.read_csv(output_path, encoding="utf-8")
        assert df_read["name"].iloc[0] == "Αλίκη"

    def test_write_csv_encoding_latin1(self, temp_csv):
        """Test CSV write with latin-1 encoding."""
        df = pd.DataFrame({"name": ["Alice", "Café", "Bob"]})
        frame = ar.from_pandas(df)
        output_path = os.path.join(temp_csv, "output.csv")

        ar.write_csv(frame, output_path, encoding="latin-1")

        df_read = pd.read_csv(output_path, encoding="latin-1")
        assert df_read["name"].iloc[1] == "Café"

    def test_write_csv_encoding_errors_replace(self, temp_csv):
        """Test encoding_errors='replace' handles unencodable chars."""
        df = pd.DataFrame({"text": ["hello", "世界"]})
        frame = ar.from_pandas(df)
        output_path = os.path.join(temp_csv, "output.csv")

        # ASCII can't encode Chinese; should replace with placeholder
        ar.write_csv(frame, output_path, encoding="ascii", encoding_errors="replace")
        assert os.path.exists(output_path)

    def test_write_csv_encoding_errors_ignore(self, temp_csv):
        """Test encoding_errors='ignore' drops unencodable chars."""
        df = pd.DataFrame({"text": ["hello", "café"]})
        frame = ar.from_pandas(df)
        output_path = os.path.join(temp_csv, "output.csv")

        ar.write_csv(frame, output_path, encoding="ascii", encoding_errors="ignore")
        assert os.path.exists(output_path)

    def test_write_csv_invalid_encoding_raises_value_error(self, sample_frame, temp_csv):
        """Test that unknown encoding codec raises ValueError."""
        output_path = os.path.join(temp_csv, "output.csv")

        with pytest.raises(ValueError):
            ar.write_csv(sample_frame, output_path, encoding="not-a-real-codec")

    def test_write_csv_invalid_encoding_errors_raises_value_error(self, sample_frame, temp_csv):
        """Test that invalid encoding_errors value raises ValueError."""
        output_path = os.path.join(temp_csv, "output.csv")

        with pytest.raises(ValueError):
            ar.write_csv(sample_frame, output_path, encoding_errors="invalid")

    def test_write_csv_non_string_encoding_raises_type_error(self, sample_frame, temp_csv):
        """Test that non-string encoding raises TypeError."""
        output_path = os.path.join(temp_csv, "output.csv")

        with pytest.raises(TypeError):
            ar.write_csv(sample_frame, output_path, encoding=123)

    def test_write_csv_non_string_encoding_errors_raises_type_error(self, sample_frame, temp_csv):
        """Test that non-string encoding_errors raises TypeError."""
        output_path = os.path.join(temp_csv, "output.csv")

        with pytest.raises(TypeError):
            ar.write_csv(sample_frame, output_path, encoding_errors=True)

    def test_write_csv_with_pathlib_path(self, sample_frame, temp_csv):
        """Test CSV write accepts pathlib.Path objects."""
        output_path = Path(temp_csv) / "output.csv"
        ar.write_csv(sample_frame, output_path)

        assert output_path.exists()

    def test_write_csv_roundtrip_verification(self, sample_frame, temp_csv):
        """Test that written CSV can be read back with matching data."""
        output_path = os.path.join(temp_csv, "output.csv")
        ar.write_csv(sample_frame, output_path)

        frame_read = ar.read_csv(output_path)
        df_original = ar.to_pandas(sample_frame)
        df_read = ar.to_pandas(frame_read)

        pd.testing.assert_frame_equal(df_original, df_read)

    def test_write_csv_empty_frame(self, temp_csv):
        """Test writing an empty frame with columns but no rows."""
        df = pd.DataFrame({"col1": pd.Series(dtype="float64"), "col2": pd.Series(dtype="int64")})
        frame = ar.from_pandas(df)
        output_path = os.path.join(temp_csv, "output.csv")

        ar.write_csv(frame, output_path)

        df_read = pd.read_csv(output_path)
        assert list(df_read.columns) == ["col1", "col2"]
        assert len(df_read) == 0

import pytest
import csv
import pandas as pd
import arnio as ar


class TestArFrameDropColumns:
    """System tests for ArFrame.drop_columns end-to-end workflows."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV with multiple columns for testing."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "email", "age", "city"])
            writer.writerow([1, "Alice", "alice@example.com", 30, "NYC"])
            writer.writerow([2, "Bob", "bob@example.com", 25, "LA"])
            writer.writerow([3, "Charlie", "charlie@example.com", 35, "Chicago"])
        return csv_file

    @pytest.fixture
    def sample_frame(self, sample_csv):
        """Load sample CSV into an ArFrame."""
        return ar.read_csv(sample_csv)

    def test_drop_single_column_basic(self, sample_frame):
        """Test dropping a single column from ArFrame."""
        result = sample_frame.drop_columns(["email"])
        assert result.columns == ["id", "name", "age", "city"]
        assert "email" not in result.columns
        df = ar.to_pandas(result)
        assert df.shape == (3, 4)

    def test_drop_multiple_columns(self, sample_frame):
        """Test dropping multiple columns at once."""
        result = sample_frame.drop_columns(["email", "age"])
        assert result.columns == ["id", "name", "city"]
        assert "email" not in result.columns
        assert "age" not in result.columns

    def test_drop_columns_accepts_tuple(self, sample_frame):
        """Test that tuple input is accepted alongside list."""
        result = sample_frame.drop_columns(("email", "age"))
        assert result.columns == ["id", "name", "city"]

    def test_drop_columns_preserves_original_order(self, sample_frame):
        """Test that remaining columns maintain original order."""
        result = sample_frame.drop_columns(["name"])
        assert result.columns == ["id", "email", "age", "city"]
        assert result.columns[0] == "id"
        assert result.columns[1] == "email"

    def test_drop_columns_handles_duplicates_silently(self, sample_frame):
        """Test that duplicate column names in list are silently ignored."""
        result = sample_frame.drop_columns(["email", "email", "age", "age"])
        assert result.columns == ["id", "name", "city"]
        assert len(result.columns) == 3

    def test_drop_columns_empty_list_returns_copy(self, sample_frame):
        """Test that empty list returns a new unchanged copy."""
        result = sample_frame.drop_columns([])
        assert result.columns == sample_frame.columns
        assert result is not sample_frame
        df_original = ar.to_pandas(sample_frame)
        df_result = ar.to_pandas(result)
        pd.testing.assert_frame_equal(df_original, df_result)

    def test_drop_columns_nonexistent_raises_valueerror(self, sample_frame):
        """Test that ValueError is raised for non-existent columns."""
        with pytest.raises(ValueError, match="Unknown column"):
            sample_frame.drop_columns(["nonexistent"])

    def test_drop_columns_partial_nonexistent_raises_valueerror(self, sample_frame):
        """Test that ValueError is raised if any column doesn't exist."""
        with pytest.raises(ValueError, match="Unknown column"):
            sample_frame.drop_columns(["email", "badcol", "age"])

    def test_drop_columns_all_columns_raises_valueerror(self, sample_frame):
        """Test that ValueError is raised when attempting to drop all columns."""
        with pytest.raises(ValueError, match="cannot remove all columns"):
            sample_frame.drop_columns(sample_frame.columns)

    def test_drop_columns_wrong_type_raises_typeerror(self, sample_frame):
        """Test that TypeError is raised for non-list/tuple input."""
        with pytest.raises(TypeError, match="must be a list or tuple"):
            sample_frame.drop_columns("email")

    def test_drop_columns_non_string_element_raises_typeerror(self, sample_frame):
        """Test that TypeError is raised for non-string column names."""
        with pytest.raises(TypeError, match="must be strings"):
            sample_frame.drop_columns(["email", 123])

    def test_drop_columns_with_pandas_interop(self, sample_csv):
        """Test drop_columns in pandas interoperability workflow."""
        df = pd.read_csv(sample_csv)
        frame = ar.from_pandas(df)
        result = frame.drop_columns(["age", "city"])
        df_result = ar.to_pandas(result)
        
        assert list(df_result.columns) == ["id", "name", "email"]
        assert df_result.shape == (3, 3)

    def test_drop_columns_with_pipeline_integration(self, sample_frame):
        """Test drop_columns integrated with cleaning pipeline."""
        frame = sample_frame.drop_columns(["email"])
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])
        
        df = ar.to_pandas(clean)
        assert "email" not in df.columns
        assert set(df.columns) == {"id", "name", "age", "city"}

    def test_drop_columns_write_read_roundtrip(self, sample_frame, tmp_path):
        """Test drop_columns in CSV write-read roundtrip workflow."""
        result = sample_frame.drop_columns(["age", "city"])
        output_csv = tmp_path / "output.csv"
        ar.write_csv(result, str(output_csv))
        reloaded = ar.read_csv(output_csv)
        
        assert reloaded.columns == ["id", "name", "email"]
        df = ar.to_pandas(reloaded)
        assert df.shape == (3, 3)

    def test_drop_columns_data_integrity(self, sample_csv):
        """Test that drop_columns preserves data integrity in remaining columns."""
        frame = ar.read_csv(sample_csv)
        result = frame.drop_columns(["name"])
        df = ar.to_pandas(result)
        
        assert list(df["id"]) == [1, 2, 3]
        assert list(df["email"]) == ["alice@example.com", "bob@example.com", "charlie@example.com"]
        assert list(df["age"]) == [30, 25, 35]

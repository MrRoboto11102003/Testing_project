import pytest
import tempfile
import pandas as pd
import arnio as ar
from pathlib import Path


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a sample CSV file for testing."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "id,name,salary,department,active\n"
        "1,Alice,50000,Engineering,true\n"
        "2,Bob,45000,Sales,false\n"
        "3,Charlie,60000,Engineering,true\n"
        "4,Diana,55000,Marketing,true\n"
    )
    return csv_path


@pytest.fixture
def mixed_dtypes_csv(tmp_path):
    """Create a CSV with various data types."""
    csv_path = tmp_path / "mixed.csv"
    csv_path.write_text(
        "user_id,score,rating,name,is_active\n"
        "101,95,4.5,Alice,true\n"
        "102,87,3.8,Bob,false\n"
        "103,92,4.2,Charlie,true\n"
    )
    return csv_path


class TestDropColumnsBasic:
    """End-to-end tests for ArFrame.drop_columns()"""

    def test_drop_columns_single_column_preserves_data(self, sample_csv_file, tmp_path):
        """Test dropping a single column and writing result to CSV."""
        frame = ar.read_csv(sample_csv_file)
        assert frame.shape == (4, 5)

        # Drop department column
        frame = frame.drop_columns(["department"])
        assert frame.columns == ["id", "name", "salary", "active"]
        assert frame.shape == (4, 4)

        # Write to CSV and verify
        output_file = tmp_path / "result.csv"
        ar.write_csv(frame, str(output_file))

        # Read back and verify
        result_frame = ar.read_csv(output_file)
        assert result_frame.columns == ["id", "name", "salary", "active"]
        assert result_frame.shape == (4, 4)

    def test_drop_columns_multiple_with_tuple_and_duplicates(self, sample_csv_file):
        """Test dropping multiple columns with tuple input and duplicate handling."""
        frame = ar.read_csv(sample_csv_file)

        # Drop with tuple and duplicate name (should be silently ignored)
        frame = frame.drop_columns(("department", "active", "department"))
        assert frame.columns == ["id", "name", "salary"]

        # Verify data
        df = ar.to_pandas(frame)
        assert len(df) == 4
        assert len(df.columns) == 3

    def test_drop_columns_error_nonexistent_column(self, sample_csv_file):
        """Test that dropping non-existent column raises ValueError."""
        frame = ar.read_csv(sample_csv_file)

        with pytest.raises(ValueError, match="Unknown column"):
            frame.drop_columns(["nonexistent"])

    def test_drop_columns_error_remove_all(self, sample_csv_file):
        """Test that attempting to drop all columns raises ValueError."""
        frame = ar.read_csv(sample_csv_file)

        with pytest.raises(ValueError, match="cannot remove all columns"):
            frame.drop_columns(frame.columns)

    def test_drop_columns_error_invalid_type(self, sample_csv_file):
        """Test that invalid cols type raises TypeError."""
        frame = ar.read_csv(sample_csv_file)

        with pytest.raises(TypeError, match="must be a list or tuple"):
            frame.drop_columns("name")

    def test_drop_columns_error_non_string_elements(self, sample_csv_file):
        """Test that non-string column names raise TypeError."""
        frame = ar.read_csv(sample_csv_file)

        with pytest.raises(TypeError, match="must be strings"):
            frame.drop_columns(["name", 123])

    def test_drop_columns_empty_list_returns_copy(self, sample_csv_file):
        """Test that empty list returns a copy unchanged."""
        frame = ar.read_csv(sample_csv_file)
        original_df = ar.to_pandas(frame)

        result = frame.drop_columns([])

        result_df = ar.to_pandas(result)
        pd.testing.assert_frame_equal(original_df, result_df)
        assert result.columns == frame.columns


class TestSelectColumnsBasic:
    """End-to-end tests for ArFrame.select_columns()"""

    def test_select_columns_subset_and_export(self, sample_csv_file, tmp_path):
        """Test selecting columns and exporting to CSV."""
        frame = ar.read_csv(sample_csv_file)

        # Select specific columns
        frame = frame.select_columns(["id", "name", "salary"])
        assert frame.columns == ["id", "name", "salary"]
        assert frame.shape == (4, 3)

        # Write and verify
        output_file = tmp_path / "selected.csv"
        ar.write_csv(frame, str(output_file))

        result = ar.read_csv(output_file)
        assert result.columns == ["id", "name", "salary"]
        assert result.shape == (4, 3)

    def test_select_columns_reorders_output(self, sample_csv_file):
        """Test that select_columns preserves requested column order."""
        frame = ar.read_csv(sample_csv_file)

        # Select in different order than original
        frame = frame.select_columns(["salary", "id", "name"])
        assert frame.columns == ["salary", "id", "name"]

        df = ar.to_pandas(frame)
        assert list(df.columns) == ["salary", "id", "name"]
        assert len(df) == 4

    def test_select_columns_error_empty_selection(self, sample_csv_file):
        """Test that empty column selection raises ValueError."""
        frame = ar.read_csv(sample_csv_file)

        with pytest.raises(ValueError, match="cannot be empty"):
            frame.select_columns([])

    def test_select_columns_error_unknown_column(self, sample_csv_file):
        """Test that unknown column name raises ValueError."""
        frame = ar.read_csv(sample_csv_file)

        with pytest.raises(ValueError, match="Unknown columns"):
            frame.select_columns(["nonexistent"])

    def test_select_columns_error_duplicates(self, sample_csv_file):
        """Test that duplicate column names raise ValueError."""
        frame = ar.read_csv(sample_csv_file)

        with pytest.raises(ValueError, match="Duplicate"):
            frame.select_columns(["id", "name", "id"])

    def test_select_columns_error_string_instead_of_list(self, sample_csv_file):
        """Test that passing string instead of list raises TypeError."""
        frame = ar.read_csv(sample_csv_file)

        with pytest.raises(TypeError, match="must be a sequence"):
            frame.select_columns("id")

    def test_select_columns_error_non_string_elements(self, sample_csv_file):
        """Test that non-string column names raise TypeError."""
        frame = ar.read_csv(sample_csv_file)

        with pytest.raises(TypeError, match="must be strings"):
            frame.select_columns(["id", 123])


class TestSelectDtypesBasic:
    """End-to-end tests for ArFrame.select_dtypes()"""

    def test_select_dtypes_include_numeric(self, mixed_dtypes_csv, tmp_path):
        """Test selecting numeric columns with include parameter."""
        frame = ar.read_csv(mixed_dtypes_csv)

        # Select numeric types
        numeric = frame.select_dtypes(include=["int64", "float64"])
        assert "name" not in numeric.columns
        assert "is_active" not in numeric.columns
        assert set(numeric.columns) == {"user_id", "score", "rating"}

        # Write and verify
        output_file = tmp_path / "numeric.csv"
        ar.write_csv(numeric, str(output_file))

        result = ar.read_csv(output_file)
        assert len(result.columns) == 3
        assert result.shape == (3, 3)

    def test_select_dtypes_exclude_string(self, mixed_dtypes_csv):
        """Test excluding string type with exclude parameter."""
        frame = ar.read_csv(mixed_dtypes_csv)

        result = frame.select_dtypes(exclude="string")
        assert "name" not in result.columns

        df = ar.to_pandas(result)
        assert "name" not in df.columns
        assert len(df) == 3

    def test_select_dtypes_error_no_args(self, mixed_dtypes_csv):
        """Test that neither include nor exclude raises ValueError."""
        frame = ar.read_csv(mixed_dtypes_csv)

        with pytest.raises(ValueError, match="at least one of"):
            frame.select_dtypes()

    def test_select_dtypes_error_overlapping_args(self, mixed_dtypes_csv):
        """Test that overlapping include and exclude raises ValueError."""
        frame = ar.read_csv(mixed_dtypes_csv)

        with pytest.raises(ValueError, match="overlap"):
            frame.select_dtypes(include=["int64"], exclude=["int64"])

    def test_select_dtypes_error_invalid_dtype(self, mixed_dtypes_csv):
        """Test that invalid dtype string raises ValueError."""
        frame = ar.read_csv(mixed_dtypes_csv)

        with pytest.raises(ValueError, match="Unrecognised dtype"):
            frame.select_dtypes(include=["invalid_dtype"])

    def test_select_dtypes_error_invalid_argument_type(self, mixed_dtypes_csv):
        """Test that non-string/list/tuple argument raises TypeError."""
        frame = ar.read_csv(mixed_dtypes_csv)

        with pytest.raises(TypeError, match="must be a string"):
            frame.select_dtypes(include=123)

    def test_select_dtypes_with_tuple_input(self, mixed_dtypes_csv):
        """Test that include and exclude accept tuple input."""
        frame = ar.read_csv(mixed_dtypes_csv)

        result = frame.select_dtypes(include=("int64", "float64"))
        assert set(result.columns) == {"user_id", "score", "rating"}


class TestChainedOperations:
    """Integration tests combining multiple column operations."""

    def test_drop_and_select_chained(self, sample_csv_file):
        """Test chaining drop_columns and select_columns operations."""
        frame = ar.read_csv(sample_csv_file)

        # Drop one column, then select from remaining
        frame = frame.drop_columns(["department"])
        frame = frame.select_columns(["id", "name", "salary"])

        assert frame.columns == ["id", "name", "salary"]
        assert frame.shape == (4, 3)

        df = ar.to_pandas(frame)
        assert len(df) == 4

    def test_select_dtypes_then_drop_then_select(
        self, mixed_dtypes_csv, tmp_path
    ):
        """Test complex workflow with select_dtypes, drop, and select."""
        frame = ar.read_csv(mixed_dtypes_csv)

        # Select numeric and string types
        frame = frame.select_dtypes(include=["int64", "float64", "string"])
        # Drop user_id
        frame = frame.drop_columns(["user_id"])
        # Select final subset in specific order
        frame = frame.select_columns(["score", "name", "rating"])

        assert frame.columns == ["score", "name", "rating"]

        # Write and read back
        output_file = tmp_path / "final.csv"
        ar.write_csv(frame, str(output_file))

        result = ar.read_csv(output_file)
        assert result.columns == ["score", "name", "rating"]
        assert result.shape == (3, 3)

    def test_operations_with_pipeline_and_pandas_accessor(
        self, sample_csv_file
    ):
        """Test column operations followed by pipeline and pandas accessor."""
        frame = ar.read_csv(sample_csv_file)

        # Drop columns
        frame = frame.drop_columns(["department"])
        # Select subset
        frame = frame.select_columns(["id", "name", "salary", "active"])

        # Apply pipeline
        clean = ar.pipeline(frame, [("strip_whitespace",)])

        # Convert to pandas and use accessor
        df = ar.to_pandas(clean)
        final_df = df.arnio.clean(strip_whitespace=True)

        assert set(final_df.columns) == {"id", "name", "salary", "active"}
        assert len(final_df) == 4

    def test_select_columns_then_profile_and_validate(self, sample_csv_file):
        """Test selecting columns then profiling and validating."""
        frame = ar.read_csv(sample_csv_file)

        # Select columns
        frame = frame.select_columns(["id", "name", "salary"])

        # Profile
        report = ar.profile(frame)
        assert report.row_count == 4
        assert len(report.columns) == 3

        # Validate
        schema = ar.Schema(
            {
                "id": ar.String(nullable=False),
                "name": ar.String(nullable=False),
                "salary": ar.String(nullable=False),
            }
        )
        result = ar.validate(frame, schema)
        # Just verify validation runs without error
        assert isinstance(result.passed, bool)

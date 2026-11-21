import pytest
import pandas as pd
import arnio as ar


class TestDropColumns:
    """Tests for ArFrame.drop_columns() method."""

    def test_drop_columns_basic_workflow(self, tmp_path):
        """Read CSV, drop columns, convert to pandas, verify output."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "email": ["a@example.com", "b@example.com", "c@example.com"]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result_frame = frame.drop_columns(["email", "age"])
        result_df = ar.to_pandas(result_frame)

        assert list(result_df.columns) == ["id", "name"]
        assert result_df.shape == (3, 2)
        assert result_df["id"].tolist() == [1, 2, 3]
        assert result_df["name"].tolist() == ["Alice", "Bob", "Charlie"]

    def test_drop_columns_preserves_original_order(self, tmp_path):
        """Dropped columns preserve original column order."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({"z": [1], "a": [3], "m": [5], "b": [7]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result_frame = frame.drop_columns(["a"])
        result_df = ar.to_pandas(result_frame)

        assert list(result_df.columns) == ["z", "m", "b"]

    def test_drop_columns_deduplicates_list(self, tmp_path):
        """Duplicates in drop list are silently ignored."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1], "b": [3], "c": [5]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result_frame = frame.drop_columns(["a", "a", "b"])
        result_df = ar.to_pandas(result_frame)

        assert list(result_df.columns) == ["c"]

    def test_drop_columns_empty_list_returns_copy(self, tmp_path):
        """Empty drop list returns unchanged copy of frame."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result_frame = frame.drop_columns([])
        result_df = ar.to_pandas(result_frame)

        assert list(result_df.columns) == ["a", "b"]
        assert result_df.shape == (2, 2)

    def test_drop_columns_error_missing_column(self, tmp_path):
        """Raises ValueError when column does not exist."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1], "b": [2]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(ValueError, match="Unknown column"):
            frame.drop_columns(["nonexistent"])

    def test_drop_columns_error_all_columns(self, tmp_path):
        """Raises ValueError when attempting to drop all columns."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1], "b": [2]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(ValueError, match="cannot remove all columns"):
            frame.drop_columns(["a", "b"])

    def test_drop_columns_error_invalid_type_string(self, tmp_path):
        """Raises TypeError when passing string instead of list."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(TypeError, match="must be a list or tuple"):
            frame.drop_columns("a")

    def test_drop_columns_accepts_tuple(self, tmp_path):
        """drop_columns accepts tuple as well as list."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result_frame = frame.drop_columns(("a", "c"))
        result_df = ar.to_pandas(result_frame)

        assert list(result_df.columns) == ["b"]


class TestSelectColumns:
    """Tests for ArFrame.select_columns() method."""

    def test_select_columns_basic_csv_workflow(self, tmp_path):
        """Read CSV, select columns, write to new CSV, verify."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "salary": [50000, 60000, 70000]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result_frame = frame.select_columns(["name", "salary"])

        output_csv = tmp_path / "output.csv"
        ar.write_csv(result_frame, str(output_csv))

        written_df = pd.read_csv(output_csv)
        assert list(written_df.columns) == ["name", "salary"]
        assert written_df.shape == (3, 2)
        assert written_df["salary"].tolist() == [50000, 60000, 70000]

    def test_select_columns_reorders_to_specified_order(self, tmp_path):
        """select_columns reorders columns to match specified order."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result_frame = frame.select_columns(["d", "b", "a"])
        result_df = ar.to_pandas(result_frame)

        assert list(result_df.columns) == ["d", "b", "a"]

    def test_select_columns_error_empty_selection(self, tmp_path):
        """Raises ValueError for empty column selection."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1], "b": [2]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(ValueError, match="cannot be empty"):
            frame.select_columns([])

    def test_select_columns_error_duplicate_names(self, tmp_path):
        """Raises ValueError for duplicate column names."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1], "b": [2]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(ValueError, match="Duplicate"):
            frame.select_columns(["a", "a"])

    def test_select_columns_error_missing_column(self, tmp_path):
        """Raises ValueError when column does not exist."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1], "b": [2]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(ValueError, match="Unknown columns"):
            frame.select_columns(["a", "nonexistent"])

    def test_select_columns_error_string_instead_of_list(self, tmp_path):
        """Raises TypeError when passing string instead of list."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1], "b": [2]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(TypeError, match="must be a sequence"):
            frame.select_columns("ab")


class TestIntegration:
    """Integration tests combining drop_columns and select_columns."""

    def test_select_then_drop_then_convert(self, tmp_path):
        """Combined workflow: select columns, drop one, convert to pandas."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "age": [25, 30],
            "salary": [50000, 60000],
            "debug": ["a", "b"]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        frame = frame.select_columns(["id", "name", "age", "salary"])
        frame = frame.drop_columns(["age"])

        result_df = ar.to_pandas(frame)
        assert list(result_df.columns) == ["id", "name", "salary"]
        assert result_df.shape == (2, 3)

    def test_drop_columns_then_validation(self, tmp_path):
        """Drop columns, then validate remaining columns with schema."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "email": ["a@example.com", "b@example.com", "c@example.com"],
            "debug": ["x", "y", "z"]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        frame = frame.drop_columns(["debug"])

        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
        })

        result = ar.validate(frame, schema)
        assert result.passed

    def test_select_columns_then_profile(self, tmp_path):
        """Select specific columns, then profile the subset."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "extra_col": [None, None, None]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        frame = frame.select_columns(["id", "name", "age"])

        report = ar.profile(frame)
        summary = report.summary()

        assert summary["column_count"] == 3
        assert summary["row_count"] == 3

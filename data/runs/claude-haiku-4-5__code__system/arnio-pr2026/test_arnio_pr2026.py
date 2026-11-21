import pytest
import pandas as pd
import arnio as ar


class TestDropConstantColumns:
    """E2E tests for drop_constant_columns."""

    def test_drop_single_constant_column(self, tmp_path):
        """Remove a single column with one unique value."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "const_col": [1, 1, 1],
            "varying": ["x", "y", "z"]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_constant_columns(frame)
        result_df = pd.DataFrame(result)

        assert list(result_df.columns) == ["varying"]
        assert len(result_df) == 3

    def test_drop_multiple_constant_columns(self, tmp_path):
        """Remove multiple constant columns."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "const1": [42, 42, 42],
            "const2": ["same", "same", "same"],
            "varying": [1, 2, 3]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_constant_columns(frame)
        result_df = pd.DataFrame(result)

        assert list(result_df.columns) == ["varying"]

    def test_keep_null_with_value(self, tmp_path):
        """Keep columns with null and one value (2 unique values)."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "has_null": [1.0, 1.0, None],
            "const_null": [None, None, None]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_constant_columns(frame)
        result_df = pd.DataFrame(result)

        assert list(result_df.columns) == ["has_null"]

    def test_drop_all_null_column(self, tmp_path):
        """All-null column is constant (1 unique value: null)."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "all_null": [None, None, None],
            "normal": [1, 2, 3]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_constant_columns(frame)
        result_df = pd.DataFrame(result)

        assert list(result_df.columns) == ["normal"]

    def test_zero_row_frame_preserves_columns(self, tmp_path):
        """Zero-row frames keep all columns (0 unique values, not 1)."""
        csv_file = tmp_path / "empty.csv"
        df = pd.DataFrame({
            "col_a": pd.Series([], dtype=int),
            "col_b": pd.Series([], dtype=str)
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_constant_columns(frame)
        result_df = pd.DataFrame(result)

        assert len(result_df) == 0
        assert list(result_df.columns) == ["col_a", "col_b"]

    def test_all_columns_constant_preserves_rows(self, tmp_path):
        """When all columns are dropped, preserve row count."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "const1": [5, 5, 5],
            "const2": ["a", "a", "a"]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_constant_columns(frame)
        result_df = pd.DataFrame(result)

        assert len(result_df) == 3
        assert len(result_df.columns) == 0

    def test_pandas_dataframe_input(self):
        """drop_constant_columns works with plain pandas DataFrame."""
        df = pd.DataFrame({
            "const": [1, 1, 1],
            "varying": [1, 2, 3]
        })

        result = ar.drop_constant_columns(df)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["varying"]


class TestDropEmptyColumns:
    """E2E tests for drop_empty_columns."""

    def test_drop_all_null_column(self, tmp_path):
        """Remove column with all nulls."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "empty": [None, None, None],
            "filled": ["a", "b", "c"]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        result_df = pd.DataFrame(result)

        assert list(result_df.columns) == ["filled"]

    def test_drop_whitespace_strings(self, tmp_path):
        """Whitespace-only strings are treated as empty."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "spaces": ["  ", " ", "  "],
            "normal": ["x", "y", "z"]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        result_df = pd.DataFrame(result)

        assert list(result_df.columns) == ["normal"]

    def test_drop_mixed_null_whitespace(self, tmp_path):
        """Column with mix of null and whitespace is dropped."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "mixed": [None, "  ", None],
            "data": ["a", "b", "c"]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        result_df = pd.DataFrame(result)

        assert list(result_df.columns) == ["data"]

    def test_keep_non_empty_values(self, tmp_path):
        """Columns with any non-empty content are kept."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "sparse": ["a", None, None],
            "normal": [1, 2, 3]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        result_df = pd.DataFrame(result)

        assert set(result_df.columns) == {"sparse", "normal"}

    def test_zero_row_frame(self, tmp_path):
        """Zero-row frame preserves all columns."""
        csv_file = tmp_path / "empty.csv"
        df = pd.DataFrame({
            "col_a": pd.Series([], dtype='object'),
            "col_b": pd.Series([], dtype='object')
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        result_df = pd.DataFrame(result)

        assert len(result_df) == 0
        assert set(result_df.columns) == {"col_a", "col_b"}

    def test_all_columns_empty(self, tmp_path):
        """All columns removed if all are empty."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "empty1": [None, None],
            "empty2": ["", ""]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        result_df = pd.DataFrame(result)

        assert len(result_df.columns) == 0
        assert len(result_df) == 2


class TestIntegration:
    """Integration tests combining both functions."""

    def test_chained_cleaning_operations(self, tmp_path):
        """Chain drop_constant_columns and drop_empty_columns."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "const": ["x", "x", "x"],
            "empty": [None, None, None],
            "name": ["Alice", "Bob", "Charlie"]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_constant_columns(frame)
        result = ar.drop_empty_columns(result)
        result_df = pd.DataFrame(result)

        assert set(result_df.columns) == {"id", "name"}
        assert len(result_df) == 3

    def test_full_pipeline_csv_roundtrip(self, tmp_path):
        """Full workflow: load -> clean -> save -> reload."""
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"

        df = pd.DataFrame({
            "id": [1, 2, 3],
            "constant": [42, 42, 42],
            "empty": [None, None, None],
            "name": ["A", "B", "C"],
            "whitespace": ["   ", " ", "  "]
        })
        df.to_csv(input_file, index=False)

        frame = ar.read_csv(str(input_file))
        result = ar.drop_constant_columns(frame)
        result = ar.drop_empty_columns(result)
        result_df = pd.DataFrame(result)
        result_df.to_csv(output_file, index=False)

        reloaded = pd.read_csv(output_file)
        assert set(reloaded.columns) == {"id", "name"}
        assert len(reloaded) == 3

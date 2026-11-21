import pytest
import pandas as pd
import arnio as ar
import csv
from pathlib import Path


class TestDropConstantColumnsBasic:
    """E2E tests for drop_constant_columns: basic functionality."""

    def test_drop_single_constant_column_arframe(self, tmp_path):
        """Read CSV with constant column, drop it via ArFrame."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,status\n1,Alice,active\n2,Bob,active\n3,Charlie,active")

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_constant_columns(frame)

        assert result.columns == ["id", "name"]
        assert result.shape == (3, 2)
        df = ar.to_pandas(result)
        assert df["id"].tolist() == [1, 2, 3]

    def test_drop_constant_column_via_pipeline(self, tmp_path):
        """Use drop_constant_columns as pipeline step."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,const_col,value\n1,X,100\n2,X,200\n3,X,300")

        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [
            ("drop_constant_columns",),
        ])

        assert result.columns == ["id", "value"]
        df = ar.to_pandas(result)
        assert df["value"].tolist() == [100, 200, 300]

    def test_keep_mixed_columns_with_null_and_value(self, tmp_path):
        """Keep columns with multiple unique values (including null)."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,score,active\n1,100,yes\n2,,no\n3,100,yes")

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_constant_columns(frame)

        # score has [100, None, 100] = 2 unique; active has 2 unique -> keep both
        assert set(result.columns) == {"id", "score", "active"}

    def test_drop_all_constant_columns_preserves_rows(self, tmp_path):
        """When all columns constant, return 0-column frame with original row count."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\nactive,yes\nactive,yes\nactive,yes")

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_constant_columns(frame)

        assert result.columns == []
        assert result.shape[0] == 3  # Rows preserved

    def test_empty_frame_zero_rows_keeps_columns(self, tmp_path):
        """0-row frames keep all columns (0 unique values ≠ 1 unique)."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,value")

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_constant_columns(frame)

        assert result.columns == ["id", "name", "value"]
        assert result.shape == (0, 3)


class TestDropConstantColumnsPandas:
    """E2E tests for drop_constant_columns with pd.DataFrame."""

    def test_pandas_dataframe_input_preserves_type(self):
        """drop_constant_columns should preserve pd.DataFrame type."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "const": ["x", "x", "x"],
            "name": ["Alice", "Bob", "Charlie"]
        })

        result = ar.drop_constant_columns(df)

        assert isinstance(result, pd.DataFrame)
        assert result.columns.tolist() == ["id", "name"]
        assert len(result) == 3

    def test_pandas_mixed_dtypes(self):
        """Test pd.DataFrame with various data types."""
        df = pd.DataFrame({
            "int_col": [1, 1, 1],
            "float_col": [1.5, 2.5, 3.5],
            "str_const": ["same", "same", "same"],
            "bool_col": [True, False, True]
        })

        result = ar.drop_constant_columns(df)

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"float_col", "bool_col"}

    def test_arframe_to_pandas_roundtrip(self, tmp_path):
        """Convert ArFrame to pandas, clean, convert back."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,const,value\n1,X,100\n2,X,200")

        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        clean_df = ar.drop_constant_columns(df)
        clean_frame = ar.from_pandas(clean_df)

        assert clean_frame.columns == ["id", "value"]
        assert clean_frame.shape == (2, 2)


class TestDropEmptyColumnsBasic:
    """E2E tests for drop_empty_columns: basic functionality."""

    def test_drop_all_null_column(self, tmp_path):
        """Remove columns with all null values."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,empty_col,name\n1,,Alice\n2,,Bob\n3,,Charlie")

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)

        assert result.columns == ["id", "name"]
        assert result.shape == (3, 2)
        df = ar.to_pandas(result)
        assert df["name"].tolist() == ["Alice", "Bob", "Charlie"]

    def test_drop_whitespace_only_column(self, tmp_path):
        """Remove columns containing only whitespace."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "spaces_col", "name"])
            writer.writerow([1, "    ", "Alice"])
            writer.writerow([2, "  \t  ", "Bob"])

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)

        assert result.columns == ["id", "name"]
        df = ar.to_pandas(result)
        assert df["name"].tolist() == ["Alice", "Bob"]

    def test_keep_column_with_one_data_value(self, tmp_path):
        """Keep columns with at least one non-empty, non-whitespace value."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,mixed,name\n1,data,Alice\n2,,Bob")

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)

        # 'mixed' has one non-empty value -> keep it
        assert "mixed" in result.columns
        assert result.shape == (2, 3)

    def test_drop_all_empty_preserves_rows(self, tmp_path):
        """When all columns are empty, return 0-column frame with original row count."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["col1", "col2"])
            writer.writerow(["", ""])
            writer.writerow(["", ""])
            writer.writerow(["", ""])

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)

        assert result.columns == []
        assert result.shape[0] == 3  # Rows preserved

    def test_zero_row_frame_keeps_all_columns(self, tmp_path):
        """0-row frames keep all columns (can't determine if truly empty)."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,empty_col,name")

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)

        assert result.columns == ["id", "empty_col", "name"]
        assert result.shape == (0, 3)


class TestDropEmptyColumnsPipeline:
    """E2E tests for drop_empty_columns in pipelines."""

    def test_pipeline_strip_then_drop_empty(self, tmp_path):
        """Combine strip_whitespace and drop_empty_columns in pipeline."""
        csv_file = tmp_path / "messy.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "empty_col", "name"])
            writer.writerow([1, "", "  Alice  "])
            writer.writerow([2, "", "  Bob  "])

        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_empty_columns",),
        ])

        assert result.columns == ["id", "name"]
        df = ar.to_pandas(result)
        assert df["name"].tolist() == ["Alice", "Bob"]

    def test_write_csv_after_drop_empty(self, tmp_path):
        """Read, drop empty columns, write to new CSV file."""
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"

        with open(input_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "empty_col", "name"])
            writer.writerow([1, "", "Alice"])
            writer.writerow([2, "", "Bob"])

        frame = ar.read_csv(str(input_csv))
        clean = ar.drop_empty_columns(frame)
        ar.write_csv(clean, str(output_csv))

        # Verify output file
        output_text = output_csv.read_text()
        lines = output_text.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        assert "user_id" in lines[0]
        assert "name" in lines[0]
        assert "empty_col" not in lines[0]


class TestCombinedCleaningWorkflow:
    """E2E tests combining both functions in realistic workflows."""

    def test_combined_drop_empty_then_constant(self, tmp_path):
        """Full workflow: drop empty columns, then drop constant columns."""
        csv_file = tmp_path / "messy.csv"
        output_csv = tmp_path / "clean.csv"

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "empty_col", "status", "name", "version"])
            writer.writerow([1, "", "active", "Alice", "1.0"])
            writer.writerow([2, "", "active", "Bob", "1.0"])

        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [
            ("drop_empty_columns",),
            ("drop_constant_columns",),
        ])
        ar.write_csv(result, str(output_csv))

        # Should remove: empty_col, status (const), version (const)
        assert set(result.columns) == {"user_id", "name"}
        assert result.shape == (2, 2)

        # Verify output file content
        output_text = output_csv.read_text()
        assert "user_id,name" in output_text
        assert "Alice" in output_text
        assert "Bob" in output_text

    def test_profile_before_and_after_cleaning(self, tmp_path):
        """Compare data profiles before and after cleaning."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "empty", "const", "value"])
            writer.writerow([1, "", "X", 100])
            writer.writerow([2, "", "X", 200])
            writer.writerow([3, "", "X", 300])

        frame = ar.read_csv(str(csv_file))
        report_before = ar.profile(frame)

        assert report_before.row_count == 3
        assert report_before.column_count == 4

        clean = ar.pipeline(frame, [
            ("drop_empty_columns",),
            ("drop_constant_columns",),
        ])
        report_after = ar.profile(clean)

        assert report_after.column_count == 2
        assert report_after.row_count == 3

    def test_multiple_constant_columns_removed(self, tmp_path):
        """Remove multiple constant columns in one operation."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,status,priority,environment,name\n"
            "1,active,high,prod,Alice\n"
            "2,active,high,prod,Bob\n"
            "3,active,high,prod,Charlie"
        )

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_constant_columns(frame)

        # status, priority, environment all constant -> dropped
        assert set(result.columns) == {"id", "name"}
        assert result.shape == (3, 2)

    def test_edge_case_single_row_all_constant(self, tmp_path):
        """Single-row frame: all columns are constant."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n100,200,300")

        frame = ar.read_csv(str(csv_file))
        result = ar.drop_constant_columns(frame)

        # Single row -> each column has 1 unique value -> all dropped
        assert result.columns == []
        assert result.shape[0] == 1

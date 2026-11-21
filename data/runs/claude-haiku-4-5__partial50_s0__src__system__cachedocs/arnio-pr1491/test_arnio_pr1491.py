import pytest
import csv
import re
import tempfile
import os
import sys
import pandas as pd
import numpy as np
import arnio as ar


class TestDropColumnsMatching:
    """End-to-end tests for drop_columns_matching functionality."""

    def test_drop_columns_matching_basic(self, tmp_path):
        """Test basic column dropping with simple regex pattern."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "temp_1", "name", "temp_2"])
            writer.writerow([1, "x", "Alice", "y"])
            writer.writerow([2, "z", "Bob", "w"])

        frame = ar.read_csv(str(csv_path))
        clean = ar.drop_columns_matching(frame, r"^temp_")
        result_df = ar.to_pandas(clean)

        assert list(result_df.columns) == ["id", "name"]
        assert result_df.shape == (2, 2)

    def test_drop_columns_matching_no_matches(self, tmp_path):
        """Test when pattern matches no columns - all columns preserved."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "email"])
            writer.writerow([1, "Alice", "alice@example.com"])
            writer.writerow([2, "Bob", "bob@example.com"])

        frame = ar.read_csv(str(csv_path))
        clean = ar.drop_columns_matching(frame, r"^temp_")
        result_df = ar.to_pandas(clean)

        assert list(result_df.columns) == ["id", "name", "email"]
        assert len(result_df) == 2

    def test_drop_columns_matching_all_columns_error(self, tmp_path):
        """Test that ValueError is raised when pattern matches all columns."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["a", "b", "c"])
            writer.writerow([1, 2, 3])

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(ValueError, match="Pattern matches all columns"):
            ar.drop_columns_matching(frame, r".*")

    def test_drop_columns_matching_invalid_regex(self, tmp_path):
        """Test that re.error is raised for invalid regex pattern."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name"])
            writer.writerow([1, "Alice"])

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(re.error, match="Invalid regex pattern"):
            ar.drop_columns_matching(frame, r"[invalid")

    def test_drop_columns_matching_pattern_not_string(self, tmp_path):
        """Test that TypeError is raised when pattern is not a string."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name"])
            writer.writerow([1, "Alice"])

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(TypeError, match="pattern must be a string"):
            ar.drop_columns_matching(frame, 123)

        with pytest.raises(TypeError, match="pattern must be a string"):
            ar.drop_columns_matching(frame, None)

    def test_drop_columns_matching_with_pandas_dataframe(self):
        """Test that function works with pandas DataFrames directly."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "temp_col": ["a", "b", "c"],
            "name": ["Alice", "Bob", "Charlie"]
        })

        result = ar.drop_columns_matching(df, r"^temp_")

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["id", "name"]
        assert len(result) == 3

    def test_drop_columns_matching_complex_pattern(self, tmp_path):
        """Test with complex regex patterns using alternation and anchors."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "user_name", "debug_flag", "config_value"])
            writer.writerow([1, "Alice", "false", "xyz"])

        frame = ar.read_csv(str(csv_path))

        # Pattern: columns ending with _id or _flag
        clean = ar.drop_columns_matching(frame, r"(_id|_flag)$")
        result_df = ar.to_pandas(clean)

        assert list(result_df.columns) == ["user_name", "config_value"]

    def test_drop_columns_matching_write_csv(self, tmp_path):
        """Test that cleaned frame can be written to CSV file."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "temp_debug", "name"])
            writer.writerow([1, "x", "Alice"])
            writer.writerow([2, "y", "Bob"])

        frame = ar.read_csv(str(csv_path))
        clean = ar.drop_columns_matching(frame, r"^temp_")

        output_path = tmp_path / "output.csv"
        ar.write_csv(clean, str(output_path))

        # Verify output file contents
        output_df = pd.read_csv(output_path)
        assert list(output_df.columns) == ["id", "name"]
        assert len(output_df) == 2
        assert output_df.loc[0, "name"] == "Alice"

    def test_drop_columns_matching_in_pipeline(self, tmp_path):
        """Test drop_columns_matching as part of cleaning pipeline."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "temp_1", "name", "temp_2"])
            writer.writerow([1, "x", "  Alice  ", "y"])
            writer.writerow([2, "z", "  Bob  ", "w"])

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_columns_matching", {"pattern": r"^temp_"}),
        ])
        result_df = ar.to_pandas(clean)

        assert list(result_df.columns) == ["id", "name"]
        assert result_df.loc[0, "name"] == "Alice"
        assert result_df.loc[1, "name"] == "Bob"

    def test_drop_columns_matching_with_validation(self, tmp_path):
        """Test that validation works on result after dropping columns."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "debug_col", "email"])
            writer.writerow([1, "x", "alice@example.com"])
            writer.writerow([2, "y", "bob@example.com"])

        frame = ar.read_csv(str(csv_path))
        clean = ar.drop_columns_matching(frame, r"^debug_")

        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email(nullable=False)
        })

        result = ar.validate(clean, schema)
        assert result.passed

    def test_drop_columns_matching_multiple_matches(self, tmp_path):
        """Test dropping multiple columns that match the pattern."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["debug_1", "debug_2", "debug_3", "value"])
            writer.writerow(["x", "y", "z", "100"])

        frame = ar.read_csv(str(csv_path))
        clean = ar.drop_columns_matching(frame, r"^debug_")
        result_df = ar.to_pandas(clean)

        assert list(result_df.columns) == ["value"]
        assert len(result_df) == 1

    def test_drop_columns_matching_case_sensitive(self, tmp_path):
        """Test that pattern matching is case-sensitive by default."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Temp_Col", "temp_col", "name"])
            writer.writerow(["a", "b", "Alice"])

        frame = ar.read_csv(str(csv_path))
        clean = ar.drop_columns_matching(frame, r"^temp_")
        result_df = ar.to_pandas(clean)

        # Only lowercase temp_col should be dropped; Temp_Col preserved
        assert list(result_df.columns) == ["Temp_Col", "name"]

    def test_drop_columns_matching_special_characters(self, tmp_path):
        """Test dropping columns with special characters in names."""
        df = pd.DataFrame({
            "id": [1, 2],
            "col-temp": ["a", "b"],
            "col.temp": ["c", "d"],
            "name": ["Alice", "Bob"]
        })

        # Match columns containing "temp"
        result = ar.drop_columns_matching(df, r"temp")

        assert list(result.columns) == ["id", "name"]
        assert len(result) == 2

    def test_drop_columns_matching_preserves_data(self, tmp_path):
        """Test that data in remaining columns is preserved correctly."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "temp_x", "value", "temp_y"])
            writer.writerow([1, "drop1", "100", "drop2"])
            writer.writerow([2, "drop3", "200", "drop4"])

        frame = ar.read_csv(str(csv_path))
        clean = ar.drop_columns_matching(frame, r"^temp_")
        result_df = ar.to_pandas(clean)

        # Verify data integrity
        assert result_df["id"].tolist() == ["1", "2"]
        assert result_df["value"].tolist() == ["100", "200"]
        assert len(result_df) == 2

    def test_drop_columns_matching_profiling(self, tmp_path):
        """Test that profiling works on result after dropping columns."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "cache_key", "name"])
            writer.writerow([1, "cache1", "Alice"])
            writer.writerow([2, "cache2", "Bob"])

        frame = ar.read_csv(str(csv_path))
        clean = ar.drop_columns_matching(frame, r"^cache_")
        result_df = ar.to_pandas(clean)

        # Profile the result
        report = result_df.arnio.profile()

        assert report.row_count == 2
        assert len(report.columns) == 2
        assert "id" in report.columns
        assert "name" in report.columns
        assert "cache_key" not in report.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

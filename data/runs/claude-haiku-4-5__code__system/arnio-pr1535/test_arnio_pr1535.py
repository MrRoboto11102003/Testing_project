import sys
import subprocess
import pandas as pd
import numpy as np
import pytest
import arnio as ar


class TestFilterRowsOperators:
    """Test filter_rows with all supported operators."""

    def test_filter_rows_greater_than(self, tmp_path):
        """Test filtering with > operator."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("age,name\n20,Alice\n15,Bob\n25,Charlie\n")
        
        frame = ar.read_csv(str(csv_file))
        filtered = ar.filter_rows(frame, column="age", op=">", value=18)
        
        assert len(filtered) == 2
        assert list(filtered["age"].values) == [20, 25]

    def test_filter_rows_less_than(self, tmp_path):
        """Test filtering with < operator."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("age,name\n20,Alice\n15,Bob\n25,Charlie\n")
        
        frame = ar.read_csv(str(csv_file))
        filtered = ar.filter_rows(frame, column="age", op="<", value=20)
        
        assert len(filtered) == 1
        assert filtered["age"].values[0] == 15

    def test_filter_rows_equal(self, tmp_path):
        """Test filtering with == operator."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("status,count\nactive,10\ninactive,5\nactive,8\n")
        
        frame = ar.read_csv(str(csv_file))
        filtered = ar.filter_rows(frame, column="status", op="==", value="active")
        
        assert len(filtered) == 2
        assert all(filtered["status"] == "active")

    def test_filter_rows_not_equal(self, tmp_path):
        """Test filtering with != operator."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("status,count\nactive,10\ninactive,5\nactive,8\n")
        
        frame = ar.read_csv(str(csv_file))
        filtered = ar.filter_rows(frame, column="status", op="!=", value="active")
        
        assert len(filtered) == 1

    def test_filter_rows_greater_equal(self, tmp_path):
        """Test filtering with >= operator."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("score,player\n100,Alice\n100,Bob\n90,Charlie\n")
        
        frame = ar.read_csv(str(csv_file))
        filtered = ar.filter_rows(frame, column="score", op=">=", value=100)
        
        assert len(filtered) == 2

    def test_filter_rows_less_equal(self, tmp_path):
        """Test filtering with <= operator."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("temp,location\n32,Chicago\n68,Miami\n75,Phoenix\n")
        
        frame = ar.read_csv(str(csv_file))
        filtered = ar.filter_rows(frame, column="temp", op="<=", value=68)
        
        assert len(filtered) == 2


class TestFilterRowsEdgeCases:
    """Test edge cases."""

    def test_filter_rows_empty_result(self, tmp_path):
        """Test filtering that returns no rows."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("age,name\n20,Alice\n15,Bob\n25,Charlie\n")
        
        frame = ar.read_csv(str(csv_file))
        filtered = ar.filter_rows(frame, column="age", op=">", value=100)
        
        assert len(filtered) == 0

    def test_filter_rows_with_nan_values(self, tmp_path):
        """Test filtering excludes NaN values."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("value,name\n1.5,Alice\n,Bob\n2.5,Charlie\n")
        
        frame = ar.read_csv(str(csv_file))
        filtered = ar.filter_rows(frame, column="value", op=">", value=1.0)
        
        assert len(filtered) == 2

    def test_filter_rows_preserves_frame_type(self, tmp_path):
        """Test that return type matches input type."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("age,name\n20,Alice\n15,Bob\n25,Charlie\n")
        
        frame = ar.read_csv(str(csv_file))
        filtered = ar.filter_rows(frame, column="age", op=">", value=18)
        
        assert type(frame) == type(filtered)


class TestFilterRowsValidation:
    """Test input validation and error handling."""

    def test_filter_rows_column_not_found(self, tmp_path):
        """Test error when column does not exist."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("age,name\n20,Alice\n15,Bob\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError, match="Unknown column"):
            ar.filter_rows(frame, column="nonexistent", op=">", value=18)

    def test_filter_rows_unsupported_operator(self, tmp_path):
        """Test error for unsupported operator."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("age,name\n20,Alice\n15,Bob\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError, match="Unsupported operator"):
            ar.filter_rows(frame, column="age", op="~", value=18)

    def test_filter_rows_invalid_column_name(self, tmp_path):
        """Test error for invalid column name."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("age,name\n20,Alice\n15,Bob\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(frame, column="", op=">", value=18)

    def test_filter_rows_invalid_operator_type(self, tmp_path):
        """Test error for non-string operator."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("age,name\n20,Alice\n15,Bob\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(TypeError, match="op must be a string"):
            ar.filter_rows(frame, column="age", op=5, value=18)

    def test_filter_rows_non_scalar_value(self, tmp_path):
        """Test error for non-scalar value."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("age,name\n20,Alice\n15,Bob\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(TypeError, match="filter_rows value must be a scalar"):
            ar.filter_rows(frame, column="age", op=">", value=[18, 20])

    def test_filter_rows_type_incompatibility(self, tmp_path):
        """Test error for type incompatibility in comparison."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("age,name\n20,Alice\n15,Bob\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(TypeError, match="cannot compare column"):
            ar.filter_rows(frame, column="age", op=">", value="hello")


class TestFilterRowsPandas:
    """Test filtering with pandas DataFrames."""

    def test_filter_rows_pandas_dataframe(self):
        """Test filtering returns pd.DataFrame when given pd.DataFrame."""
        df = pd.DataFrame({
            "age": [20, 15, 25],
            "name": ["Alice", "Bob", "Charlie"]
        })
        
        filtered = ar.filter_rows(df, column="age", op=">", value=18)
        
        assert isinstance(filtered, pd.DataFrame)
        assert len(filtered) == 2


class TestFilterRowsIntegration:
    """Test integration with other arnio operations."""

    def test_filter_rows_chained_save_reload(self, tmp_path):
        """Test chaining filters, saving, and reloading."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("age,score,name\n20,85,Alice\n15,90,Bob\n25,70,Charlie\n30,95,David\n")
        
        output_file = tmp_path / "filtered.csv"
        
        frame = ar.read_csv(str(csv_file))
        filtered1 = ar.filter_rows(frame, column="age", op=">", value=18)
        filtered2 = ar.filter_rows(filtered1, column="score", op=">=", value=80)
        
        filtered2.to_csv(str(output_file), index=False)
        
        assert output_file.exists()
        reloaded = ar.read_csv(str(output_file))
        assert len(reloaded) == 2

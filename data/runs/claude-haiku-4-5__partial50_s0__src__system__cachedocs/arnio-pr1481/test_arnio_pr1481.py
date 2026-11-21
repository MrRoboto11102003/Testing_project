import pytest
import pandas as pd
import numpy as np
import arnio as ar
import tempfile
from pathlib import Path


class TestFilterRowsEndToEnd:
    """End-to-end system tests for arnio.cleaning.filter_rows"""

    def test_filter_rows_numeric_greater_than_basic(self, tmp_path):
        """Filter rows with > operator on numeric column from CSV"""
        csv_file = tmp_path / "sales.csv"
        csv_file.write_text("product,revenue,quarter\nWidget,15000,Q1\nGadget,8000,Q1\nGizmo,25000,Q1\nDoohickey,12000,Q2")
        
        frame = ar.read_csv(str(csv_file))
        filtered = ar.filter_rows(frame, column="revenue", op=">", value=10000)
        result_df = ar.to_pandas(filtered)
        
        assert len(result_df) == 3
        assert all(result_df["revenue"] > 10000)
        assert set(result_df["product"]) == {"Widget", "Gizmo", "Doohickey"}

    def test_filter_rows_multiple_operators(self, tmp_path):
        """Test all supported comparison operators"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,score\n1,50\n2,75\n3,100\n4,60\n5,85")
        
        frame = ar.read_csv(str(csv_file))
        
        # Test <
        result_lt = ar.to_pandas(ar.filter_rows(frame, column="score", op="<", value=70))
        assert len(result_lt) == 2
        
        # Test <=
        result_le = ar.to_pandas(ar.filter_rows(frame, column="score", op="<=", value=70))
        assert len(result_le) == 3
        
        # Test >=
        result_ge = ar.to_pandas(ar.filter_rows(frame, column="score", op=">=", value=70))
        assert len(result_ge) == 4
        
        # Test ==
        result_eq = ar.to_pandas(ar.filter_rows(frame, column="score", op="==", value=75))
        assert len(result_eq) == 1
        assert result_eq.iloc[0]["score"] == 75
        
        # Test !=
        result_ne = ar.to_pandas(ar.filter_rows(frame, column="score", op="!=", value=75))
        assert len(result_ne) == 4

    def test_filter_rows_string_equality(self, tmp_path):
        """Filter string columns with == and != operators"""
        csv_file = tmp_path / "customers.csv"
        csv_file.write_text("customer_id,status,country\n1,active,USA\n2,inactive,USA\n3,active,Canada\n4,active,UK")
        
        frame = ar.read_csv(str(csv_file))
        
        # Filter for specific status
        active = ar.to_pandas(ar.filter_rows(frame, column="status", op="==", value="active"))
        assert len(active) == 3
        assert all(active["status"] == "active")
        
        # Filter for specific country using !=
        not_usa = ar.to_pandas(ar.filter_rows(frame, column="country", op="!=", value="USA"))
        assert len(not_usa) == 2
        assert all(not_usa["country"] != "USA")

    def test_filter_rows_in_pipeline(self, tmp_path):
        """Use filter_rows as a step in ar.pipeline()"""
        csv_file = tmp_path / "orders.csv"
        csv_file.write_text("order_id,amount,status\n1, 500 ,pending\n2, 1200 ,shipped\n3, 300 ,pending\n4, 950 ,shipped")
        
        frame = ar.read_csv(str(csv_file))
        
        # Apply pipeline with filtering
        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("filter_rows", {"column": "amount", "op": ">=", "value": 500}),
        ])
        
        result_df = ar.to_pandas(result)
        assert len(result_df) == 3
        assert all(result_df["amount"] >= 500)

    def test_filter_rows_preserves_arframe_type(self, tmp_path):
        """Verify ArFrame input returns ArFrame output"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,value\n1,100\n2,200\n3,150\n4,250")
        
        frame = ar.read_csv(str(csv_file))
        assert isinstance(frame, ar.ArFrame)
        
        filtered = ar.filter_rows(frame, column="value", op=">", value=120)
        
        assert isinstance(filtered, ar.ArFrame)
        assert len(ar.to_pandas(filtered)) == 3

    def test_filter_rows_preserves_pandas_type(self, tmp_path):
        """Verify pd.DataFrame input returns pd.DataFrame output"""
        df = pd.DataFrame({
            "product": ["A", "B", "C", "D"],
            "inventory": [50, 150, 30, 200]
        })
        
        filtered = ar.filter_rows(df, column="inventory", op="<", value=100)
        
        assert isinstance(filtered, pd.DataFrame)
        assert len(filtered) == 2
        assert list(filtered["product"]) == ["A", "C"]

    def test_filter_rows_with_nulls_handled(self, tmp_path):
        """Null values are treated as False in filter mask"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,score\n1,90\n2,\n3,85\n4,95\n5,")
        
        frame = ar.read_csv(str(csv_file))
        
        # Filter > 88 should exclude nulls
        filtered = ar.to_pandas(ar.filter_rows(frame, column="score", op=">", value=88))
        
        assert len(filtered) == 2
        assert list(filtered["score"]) == [90.0, 95.0]

    def test_filter_rows_roundtrip_to_csv(self, tmp_path):
        """Filter rows and write result back to CSV"""
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"
        input_csv.write_text("employee_id,salary\n1,45000\n2,65000\n3,55000\n4,75000\n5,50000")
        
        frame = ar.read_csv(str(input_csv))
        filtered = ar.filter_rows(frame, column="salary", op=">=", value=55000)
        ar.write_csv(filtered, str(output_csv))
        
        # Read back and verify
        result_df = pd.read_csv(output_csv)
        assert len(result_df) == 3
        assert all(result_df["salary"] >= 55000)

    def test_filter_rows_error_invalid_operator(self, tmp_path):
        """Raises ValueError for unsupported operator"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,val\n1,10\n2,20")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError, match="Unsupported operator"):
            ar.filter_rows(frame, column="val", op=">>>", value=15)

    def test_filter_rows_error_column_not_found(self, tmp_path):
        """Raises ValueError when column does not exist"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError, match="Unknown column"):
            ar.filter_rows(frame, column="nonexistent_col", op=">", value=10)

    def test_filter_rows_error_non_scalar_value(self, tmp_path):
        """Raises TypeError when value is not a scalar"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,val\n1,10\n2,20\n3,30")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(TypeError, match="scalar"):
            ar.filter_rows(frame, column="val", op=">", value=[10, 20])

    def test_filter_rows_error_invalid_column_type(self, tmp_path):
        """Raises TypeError when column is not a string"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,val\n1,10\n2,20")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(TypeError, match="must be a non-empty string"):
            ar.filter_rows(frame, column=123, op=">", value=10)

    def test_filter_rows_error_empty_column_name(self, tmp_path):
        """Raises TypeError when column name is empty string"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,val\n1,10\n2,20")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(TypeError, match="must be a non-empty string"):
            ar.filter_rows(frame, column="", op=">", value=10)

    def test_filter_rows_type_mismatch_error(self, tmp_path):
        """Raises TypeError when value type is incompatible with column"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob\n3,Charlie")
        
        frame = ar.read_csv(str(csv_file))
        
        # Trying to compare string column with numeric value using > should fail
        with pytest.raises(TypeError, match="cannot compare"):
            ar.filter_rows(frame, column="name", op=">", value=100)

    def test_filter_rows_all_pass_all_fail_cases(self, tmp_path):
        """Test cases where filter matches all or no rows"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,value\n1,100\n2,200\n3,150")
        
        frame = ar.read_csv(str(csv_file))
        
        # All rows pass filter
        all_pass = ar.to_pandas(ar.filter_rows(frame, column="value", op=">", value=50))
        assert len(all_pass) == 3
        
        # No rows pass filter
        none_pass = ar.to_pandas(ar.filter_rows(frame, column="value", op=">", value=1000))
        assert len(none_pass) == 0
        assert list(none_pass.columns) == ["id", "value"]  # Schema preserved

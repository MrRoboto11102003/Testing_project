"""
End-to-end pytest tests for arnio.cleaning.filter_rows

Tests drive complete workflows through arnio's public API:
- CSV I/O with real files (tmp_path)
- ArFrame and pandas DataFrame handling
- Error cases with specific assertions
- Integration with pipelines and profiling
"""

import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestFilterRowsEndToEnd:
    """Complete end-to-end tests for filter_rows functionality"""

    def test_filter_rows_arframe_greater_than(self, tmp_path):
        """Test filtering ArFrame with > operator on numeric column"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,age,name\n1,25,Alice\n2,30,Bob\n3,20,Charlie\n")

        # Workflow: read CSV -> filter -> convert to pandas
        frame = ar.read_csv(str(csv_path))
        filtered = ar.filter_rows(frame, column="age", op=">", value=22)
        df = ar.to_pandas(filtered)

        # Assert user-visible results
        assert len(df) == 2
        assert df["age"].tolist() == [25, 30]
        assert df["name"].tolist() == ["Alice", "Bob"]

    def test_filter_rows_pandas_dataframe_less_equal(self):
        """Test filtering pandas DataFrame directly with <= operator"""
        df = pd.DataFrame({"id": [1, 2, 3], "score": [85, 90, 75]})

        # filter_rows returns same type as input (pandas)
        filtered = ar.filter_rows(df, column="score", op="<=", value=85)

        assert isinstance(filtered, pd.DataFrame)
        assert len(filtered) == 2
        assert filtered["score"].tolist() == [85, 75]

    def test_filter_rows_all_comparison_operators(self, tmp_path):
        """Test all six supported comparison operators: >, <, >=, <=, ==, !="""
        csv_path = tmp_path / "numbers.csv"
        csv_path.write_text("id,value\n1,10\n2,20\n3,30\n4,40\n")

        frame = ar.read_csv(str(csv_path))

        # Test each operator produces expected row counts
        assert len(ar.to_pandas(ar.filter_rows(frame, "value", ">", 25))) == 2
        assert len(ar.to_pandas(ar.filter_rows(frame, "value", "<", 25))) == 2
        assert len(ar.to_pandas(ar.filter_rows(frame, "value", ">=", 30))) == 2
        assert len(ar.to_pandas(ar.filter_rows(frame, "value", "<=", 20))) == 2
        assert len(ar.to_pandas(ar.filter_rows(frame, "value", "==", 20))) == 1
        assert len(ar.to_pandas(ar.filter_rows(frame, "value", "!=", 20))) == 3

    def test_filter_rows_string_column_equality(self, tmp_path):
        """Test filtering on string columns with == operator"""
        csv_path = tmp_path / "products.csv"
        csv_path.write_text("id,category,price\n1,Electronics,100\n2,Books,15\n3,Electronics,200\n")

        frame = ar.read_csv(str(csv_path))
        filtered = ar.filter_rows(frame, column="category", op="==", value="Electronics")
        df = ar.to_pandas(filtered)

        assert len(df) == 2
        assert df["id"].tolist() == [1, 3]
        assert all(df["category"] == "Electronics")

    def test_filter_rows_edge_case_no_rows_match(self):
        """Test edge case where filter removes all rows (empty result)"""
        df = pd.DataFrame({"value": [1, 2, 3]})
        filtered = ar.filter_rows(df, column="value", op=">", value=1000)

        # Result is empty but preserves structure
        assert len(filtered) == 0
        assert list(filtered.columns) == ["value"]

    def test_filter_rows_edge_case_all_rows_match(self):
        """Test edge case where filter keeps all rows"""
        df = pd.DataFrame({"value": [1, 2, 3]})
        filtered = ar.filter_rows(df, column="value", op=">", value=0)

        assert len(filtered) == 3
        assert filtered["value"].tolist() == [1, 2, 3]

    def test_filter_rows_in_pipeline_workflow(self, tmp_path):
        """Test filter_rows integrated into a cleaning pipeline"""
        csv_path = tmp_path / "sales.csv"
        csv_path.write_text("id,amount,status\n1,100,completed\n2,50,pending\n3,200,completed\n")

        frame = ar.read_csv(str(csv_path))

        # Use filter_rows as pipeline step
        filtered = ar.pipeline(
            frame,
            [("filter_rows", {"column": "amount", "op": ">", "value": 75})]
        )
        df = ar.to_pandas(filtered)

        assert len(df) == 2
        assert df["amount"].tolist() == [100, 200]

    def test_filter_rows_complete_roundtrip_csv(self, tmp_path):
        """Test full workflow: read CSV -> filter -> write CSV -> verify"""
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"

        input_path.write_text(
            "id,revenue,region\n1,5000,North\n2,8000,South\n3,3000,East\n4,12000,West\n"
        )

        # Read and filter
        frame = ar.read_csv(str(input_path))
        filtered = ar.filter_rows(frame, column="revenue", op=">=", value=5000)

        # Write to output file
        ar.write_csv(filtered, str(output_path))

        # Verify output file exists and has correct content
        assert output_path.exists()
        output_df = pd.read_csv(str(output_path))
        assert len(output_df) == 3
        assert output_df["revenue"].tolist() == [5000, 8000, 12000]

    def test_filter_rows_handles_null_values(self):
        """Test that filtering properly handles NaN/null values in comparison"""
        df = pd.DataFrame({"value": [10.0, np.nan, 30.0, 20.0]})
        filtered = ar.filter_rows(df, column="value", op=">", value=15)

        # Nulls are treated as False in the mask
        assert len(filtered) == 2
        assert filtered["value"].tolist() == [30.0, 20.0]

    def test_filter_rows_preserves_dtypes(self):
        """Test that filtering preserves column data types"""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"]
        })

        filtered = ar.filter_rows(df, column="int_col", op=">", value=1)

        assert filtered["int_col"].dtype == np.int64
        assert filtered["float_col"].dtype == np.float64
        assert filtered["str_col"].dtype == object

    def test_filter_rows_error_column_not_found(self):
        """Test error case: requested column does not exist"""
        df = pd.DataFrame({"id": [1, 2, 3]})

        with pytest.raises(ValueError, match="Unknown column"):
            ar.filter_rows(df, column="nonexistent_col", op=">", value=1)

    def test_filter_rows_error_invalid_operator(self):
        """Test error case: unsupported comparison operator"""
        df = pd.DataFrame({"id": [1, 2, 3]})

        with pytest.raises(ValueError, match="Unsupported operator"):
            ar.filter_rows(df, column="id", op=">>>", value=1)

    def test_filter_rows_error_column_param_not_string(self):
        """Test error case: column parameter must be a string"""
        df = pd.DataFrame({"id": [1, 2, 3]})

        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(df, column=123, op=">", value=1)

    def test_filter_rows_error_non_scalar_value(self):
        """Test error case: comparison value must be scalar, not list/array"""
        df = pd.DataFrame({"id": [1, 2, 3]})

        with pytest.raises(TypeError, match="must be a scalar"):
            ar.filter_rows(df, column="id", op=">", value=[1, 2, 3])

    def test_filter_rows_affects_data_quality_profile(self, tmp_path):
        """Test that filtering improves quality metrics captured in profile"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,score,status\n1,100,valid\n2,-50,invalid\n3,85,valid\n4,-20,invalid\n")

        frame = ar.read_csv(str(csv_path))

        # Profile raw data
        report_before = ar.profile(frame)

        # Filter to positive scores only
        filtered = ar.filter_rows(frame, column="score", op=">", value=0)

        # Profile filtered data
        report_after = ar.profile(filtered)

        # Row count should decrease, reflecting the filter
        assert report_before.row_count == 4
        assert report_after.row_count == 2
        # Filtered data has no negative scores
        assert report_after.columns["score"].min >= 0

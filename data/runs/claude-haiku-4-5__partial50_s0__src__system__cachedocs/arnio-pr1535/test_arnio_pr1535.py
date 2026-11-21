import arnio as ar
import pandas as pd
import numpy as np
import pytest
from pathlib import Path


class TestFilterRowsBasicArFrame:
    """Basic filter_rows functionality with ArFrame."""

    def test_filter_rows_greater_than(self, tmp_path):
        """Filter ArFrame with > operator on numeric column."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("age,name\n18,Alice\n25,Bob\n17,Charlie\n30,Diana\n")

        frame = ar.read_csv(str(csv_file))
        filtered = ar.filter_rows(frame, column="age", op=">", value=20)

        result_df = ar.to_pandas(filtered)
        assert len(result_df) == 2
        assert set(result_df["name"]) == {"Bob", "Diana"}

    def test_filter_rows_equal_string(self, tmp_path):
        """Filter ArFrame with == operator on string column."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("status,id\nactive,1\ninactive,2\nactive,3\ninactive,4\n")

        frame = ar.read_csv(str(csv_file))
        filtered = ar.filter_rows(frame, column="status", op="==", value="active")

        result_df = ar.to_pandas(filtered)
        assert len(result_df) == 2
        assert list(result_df["id"]) == ["1", "3"]

    def test_filter_rows_not_equal(self, tmp_path):
        """Filter ArFrame with != operator."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("region,sales\nUS,1000\nCA,500\nUS,1500\nUK,800\n")

        frame = ar.read_csv(str(csv_file))
        filtered = ar.filter_rows(frame, column="region", op="!=", value="US")

        result_df = ar.to_pandas(filtered)
        assert len(result_df) == 2
        assert set(result_df["region"]) == {"CA", "UK"}

    def test_filter_rows_all_operators(self, tmp_path):
        """Test all supported operators (<=, >=, <, >, ==, !=)."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("value\n10\n20\n30\n40\n50\n")

        frame = ar.read_csv(str(csv_file))

        # Test >=
        result = ar.to_pandas(ar.filter_rows(frame, column="value", op=">=", value=30))
        assert len(result) == 3

        # Test <=
        result = ar.to_pandas(ar.filter_rows(frame, column="value", op="<=", value=20))
        assert len(result) == 2

        # Test <
        result = ar.to_pandas(ar.filter_rows(frame, column="value", op="<", value=25))
        assert len(result) == 2

        # Test >
        result = ar.to_pandas(ar.filter_rows(frame, column="value", op=">", value=35))
        assert len(result) == 2


class TestFilterRowsPandas:
    """Filter rows with pandas DataFrame input."""

    def test_filter_rows_pandas_dataframe_returns_pandas(self, tmp_path):
        """Filter pandas DataFrame returns pandas DataFrame."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("score,player\n85,Alice\n92,Bob\n78,Charlie\n")

        df = pd.read_csv(str(csv_file))
        filtered = ar.filter_rows(df, column="score", op=">", value=80)

        assert isinstance(filtered, pd.DataFrame)
        assert len(filtered) == 2
        assert list(filtered["player"]) == ["Alice", "Bob"]

    def test_filter_rows_pandas_accessor(self, tmp_path):
        """Use pandas .arnio.filter_rows() accessor."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("revenue,product\n1000,A\n500,B\n2000,C\n")

        df = pd.read_csv(str(csv_file))
        filtered = df.arnio.filter_rows(column="revenue", op=">", value=600)

        assert isinstance(filtered, pd.DataFrame)
        assert len(filtered) == 2
        assert list(filtered["product"]) == ["A", "C"]


class TestFilterRowsInPipeline:
    """Filter rows as part of an ar.pipeline()."""

    def test_filter_rows_in_pipeline_basic(self, tmp_path):
        """Use filter_rows in ar.pipeline()."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,age,name\n1,25,Alice\n2,30,Bob\n3,22,Charlie\n"
        )

        frame = ar.read_csv(str(csv_file))
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("filter_rows", {"column": "age", "op": ">", "value": 24}),
        ])

        result_df = ar.to_pandas(clean)
        assert len(result_df) == 2
        assert list(result_df["name"]) == ["Alice", "Bob"]

    def test_filter_rows_chained_in_pipeline(self, tmp_path):
        """Chain multiple filter_rows steps in a pipeline."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,age,salary\n1,25,50000\n2,30,60000\n3,22,45000\n4,35,70000\n"
        )

        frame = ar.read_csv(str(csv_file))
        clean = ar.pipeline(frame, [
            ("filter_rows", {"column": "age", "op": ">=", "value": 25}),
            ("filter_rows", {"column": "salary", "op": "<=", "value": 65000}),
        ])

        result_df = ar.to_pandas(clean)
        assert len(result_df) == 1
        assert result_df["age"].iloc[0] == "30"

    def test_filter_rows_pipeline_with_validation(self, tmp_path):
        """Filter rows in pipeline then validate result."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("age,email\n25,alice@test.com\n30,bob@test.com\n20,invalid\n")

        frame = ar.read_csv(str(csv_file))
        filtered = ar.pipeline(frame, [
            ("filter_rows", {"column": "age", "op": ">=", "value": 25}),
        ])

        schema = ar.Schema({
            "age": ar.Int64(min=25),
            "email": ar.Email(),
        })

        result = ar.validate(filtered, schema)
        assert result.passed


class TestFilterRowsEdgeCases:
    """Edge cases and special scenarios."""

    def test_filter_rows_empty_result(self, tmp_path):
        """Filter that results in an empty frame."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("value\n5\n10\n15\n")

        frame = ar.read_csv(str(csv_file))
        filtered = ar.filter_rows(frame, column="value", op=">", value=100)

        result_df = ar.to_pandas(filtered)
        assert len(result_df) == 0
        assert list(result_df.columns) == ["value"]

    def test_filter_rows_all_pass(self, tmp_path):
        """Filter where all rows pass the condition."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("value\n100\n200\n300\n")

        frame = ar.read_csv(str(csv_file))
        filtered = ar.filter_rows(frame, column="value", op=">", value=50)

        result_df = ar.to_pandas(filtered)
        assert len(result_df) == 3

    def test_filter_rows_with_nulls(self, tmp_path):
        """Filter rows containing null/empty values."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("value,name\n10,Alice\n,Bob\n20,Charlie\n")

        frame = ar.read_csv(str(csv_file))
        filtered = ar.filter_rows(frame, column="value", op=">", value=15)

        result_df = ar.to_pandas(filtered)
        assert len(result_df) == 1
        assert result_df["name"].iloc[0] == "Charlie"

    def test_filter_rows_roundtrip_csv(self, tmp_path):
        """Filter and write result back to CSV file."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("id,score\n1,80\n2,95\n3,70\n4,88\n")
        output_csv = tmp_path / "output.csv"

        frame = ar.read_csv(str(input_csv))
        filtered = ar.filter_rows(frame, column="score", op=">=", value=85)
        ar.write_csv(filtered, str(output_csv))

        result_df = pd.read_csv(str(output_csv))
        assert len(result_df) == 2
        assert list(result_df["id"]) == [2, 4]

    def test_filter_rows_from_dict(self):
        """Filter rows using ArFrame created from dictionary."""
        frame = ar.from_dict({
            "product": ["A", "B", "C", "D"],
            "price": [100, 50, 200, 75],
        })

        filtered = ar.filter_rows(frame, column="price", op="<", value=100)
        result_df = ar.to_pandas(filtered)

        assert len(result_df) == 2
        assert list(result_df["product"]) == ["B", "D"]


class TestFilterRowsErrorHandling:
    """Error handling and input validation."""

    def test_filter_rows_invalid_operator(self, tmp_path):
        """Raise ValueError for unsupported operator."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("value\n10\n20\n")

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(ValueError, match="Unsupported operator"):
            ar.filter_rows(frame, column="value", op=">>", value=15)

    def test_filter_rows_unknown_column(self, tmp_path):
        """Raise ValueError for unknown column."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("value\n10\n20\n")

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(ValueError, match="Unknown column"):
            ar.filter_rows(frame, column="nonexistent", op=">", value=15)

    def test_filter_rows_invalid_column_type(self, tmp_path):
        """Raise TypeError for non-string column parameter."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("value\n10\n20\n")

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(frame, column=123, op=">", value=15)

    def test_filter_rows_invalid_op_type(self, tmp_path):
        """Raise TypeError for non-string operator."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("value\n10\n20\n")

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(TypeError, match="op must be a string"):
            ar.filter_rows(frame, column="value", op=123, value=15)

    def test_filter_rows_non_scalar_value(self, tmp_path):
        """Raise TypeError for non-scalar value parameter."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("value\n10\n20\n")

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(TypeError, match="filter_rows value must be a scalar"):
            ar.filter_rows(frame, column="value", op=">", value=[1, 2, 3])

    def test_filter_rows_type_mismatch_comparison(self, tmp_path):
        """Raise TypeError for incompatible type comparison."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name\nAlice\nBob\n")

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(TypeError, match="cannot compare column"):
            ar.filter_rows(frame, column="name", op=">", value=100)


class TestFilterRowsIntegration:
    """Integration with other arnio features and workflows."""

    def test_filter_rows_then_profile(self, tmp_path):
        """Filter rows then profile the result."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "age,salary,active\n25,50000,yes\n30,60000,yes\n22,45000,no\n35,70000,yes\n"
        )

        frame = ar.read_csv(str(csv_file))
        filtered = ar.filter_rows(frame, column="age", op=">=", value=25)

        report = ar.profile(filtered)
        assert report.row_count == 3
        assert len(report.columns) == 3

    def test_filter_rows_then_drop_duplicates(self, tmp_path):
        """Filter rows then apply drop_duplicates."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "category,value\nA,100\nA,100\nB,200\nA,100\n"
        )

        frame = ar.read_csv(str(csv_file))
        clean = ar.pipeline(frame, [
            ("filter_rows", {"column": "value", "op": ">=", "value": 100}),
            ("drop_duplicates",),
        ])

        result_df = ar.to_pandas(clean)
        assert len(result_df) == 2

    def test_filter_rows_with_from_dict(self):
        """Filter rows on ArFrame created from Python dict."""
        frame = ar.from_dict({
            "employee": ["Alice", "Bob", "Charlie", "Diana"],
            "salary": [50000, 65000, 55000, 70000],
        })

        filtered = ar.filter_rows(frame, column="salary", op=">", value=60000)
        result_df = ar.to_pandas(filtered)

        assert len(result_df) == 2
        assert set(result_df["employee"]) == {"Bob", "Diana"}

    def test_filter_rows_multiple_conditions(self, tmp_path):
        """Apply filter_rows with multiple independent conditions."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "year,revenue,region\n2020,1000,US\n2021,1500,US\n2020,500,CA\n2021,2000,CA\n"
        )

        frame = ar.read_csv(str(csv_file))
        result1 = ar.filter_rows(frame, column="year", op="==", value=2021)
        result2 = ar.filter_rows(result1, column="revenue", op=">", value=1000)

        result_df = ar.to_pandas(result2)
        assert len(result_df) == 2
        assert all(result_df["year"] == "2021")
        assert all(result_df["revenue"].astype(int) > 1000)

import pytest
import pandas as pd
import arnio as ar


class TestFilterRowsBasic:
    """Basic filter_rows functionality with all supported operators."""

    @pytest.fixture
    def sample_frame(self, tmp_path):
        """Create and load a sample CSV."""
        csv_path = tmp_path / "sample.csv"
        csv_path.write_text(
            "id,name,age,salary\n"
            "1,Alice,25,50000\n"
            "2,Bob,30,60000\n"
            "3,Charlie,35,75000\n"
        )
        return ar.read_csv(csv_path)

    def test_filter_greater_than(self, sample_frame):
        """Test > operator."""
        result = ar.to_pandas(ar.filter_rows(sample_frame, "age", ">", 28))
        assert len(result) == 2
        assert list(result["age"]) == [30, 35]

    def test_filter_less_than(self, sample_frame):
        """Test < operator."""
        result = ar.to_pandas(ar.filter_rows(sample_frame, "age", "<", 35))
        assert len(result) == 2
        assert list(result["age"]) == [25, 30]

    def test_filter_greater_equal(self, sample_frame):
        """Test >= operator."""
        result = ar.to_pandas(ar.filter_rows(sample_frame, "age", ">=", 30))
        assert len(result) == 2
        assert list(result["age"]) == [30, 35]

    def test_filter_less_equal(self, sample_frame):
        """Test <= operator."""
        result = ar.to_pandas(ar.filter_rows(sample_frame, "age", "<=", 30))
        assert len(result) == 2
        assert list(result["age"]) == [25, 30]

    def test_filter_equal(self, sample_frame):
        """Test == operator."""
        result = ar.to_pandas(ar.filter_rows(sample_frame, "age", "==", 30))
        assert len(result) == 1
        assert result["name"].iloc[0] == "Bob"

    def test_filter_not_equal(self, sample_frame):
        """Test != operator."""
        result = ar.to_pandas(ar.filter_rows(sample_frame, "age", "!=", 30))
        assert len(result) == 2
        assert set(result["name"]) == {"Alice", "Charlie"}


class TestFilterRowsReturnType:
    """Verify input type is preserved in output."""

    def test_arframe_returns_arframe(self, tmp_path):
        """ArFrame input should return ArFrame."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x,y\n1,10\n2,20\n3,30\n")
        frame = ar.read_csv(csv_path)
        result = ar.filter_rows(frame, "y", ">", 15)
        assert isinstance(result, ar.ArFrame)

    def test_dataframe_returns_dataframe(self):
        """DataFrame input should return DataFrame."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        result = ar.filter_rows(df, "y", ">", 15)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


class TestFilterRowsEdgeCases:
    """Test boundary conditions and special cases."""

    def test_filter_returns_empty_frame(self, tmp_path):
        """Filter that matches no rows returns empty frame with preserved schema."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("value\n10\n20\n30\n")
        frame = ar.read_csv(csv_path)
        result = ar.to_pandas(ar.filter_rows(frame, "value", ">", 100))
        assert len(result) == 0
        assert list(result.columns) == ["value"]

    def test_filter_returns_all_rows(self, tmp_path):
        """Filter that matches all rows returns all rows."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("value\n10\n20\n30\n")
        frame = ar.read_csv(csv_path)
        result = ar.to_pandas(ar.filter_rows(frame, "value", ">", 0))
        assert len(result) == 3

    def test_filter_string_column(self, tmp_path):
        """Filter on string column with equality."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("name\nAlice\nBob\nCharlie\n")
        frame = ar.read_csv(csv_path)
        result = ar.to_pandas(ar.filter_rows(frame, "name", "==", "Bob"))
        assert len(result) == 1
        assert result["name"].iloc[0] == "Bob"

    def test_filter_with_nulls_excluded(self, tmp_path):
        """Null values in filtered column are treated as False."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,value\n1,10\n2,\n3,30\n")
        frame = ar.read_csv(csv_path)
        result = ar.to_pandas(ar.filter_rows(frame, "value", ">", 20))
        assert len(result) == 1
        assert result["id"].iloc[0] == 3


class TestFilterRowsErrors:
    """Test error handling and input validation."""

    @pytest.fixture
    def sample_frame(self, tmp_path):
        """Sample frame for error tests."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,age\n1,25\n2,30\n")
        return ar.read_csv(csv_path)

    def test_column_not_found(self, sample_frame):
        """ValueError when column doesn't exist."""
        with pytest.raises(ValueError, match="Unknown column"):
            ar.filter_rows(sample_frame, "nonexistent", ">", 20)

    def test_invalid_operator(self, sample_frame):
        """ValueError for unsupported operator."""
        with pytest.raises(ValueError, match="Unsupported operator"):
            ar.filter_rows(sample_frame, "age", ">>", 20)

    def test_non_string_column(self, sample_frame):
        """TypeError when column is not a string."""
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(sample_frame, 123, ">", 20)

    def test_empty_column_name(self, sample_frame):
        """TypeError when column name is empty."""
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(sample_frame, "", ">", 20)

    def test_non_string_operator(self, sample_frame):
        """TypeError when operator is not a string."""
        with pytest.raises(TypeError, match="op must be a string"):
            ar.filter_rows(sample_frame, "age", 123, 20)

    def test_non_scalar_value(self, sample_frame):
        """TypeError when value is not scalar."""
        with pytest.raises(TypeError, match="must be a scalar"):
            ar.filter_rows(sample_frame, "age", ">", [1, 2])

    def test_incompatible_type_comparison(self, sample_frame):
        """TypeError when comparing incompatible types."""
        with pytest.raises(TypeError):
            ar.filter_rows(sample_frame, "age", ">", "text")


class TestFilterRowsPipeline:
    """Test filter_rows within cleaning pipelines."""

    def test_filter_in_pipeline(self, tmp_path):
        """filter_rows as a single pipeline step."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("name,score\nAlice,85\nBob,92\nCharlie,78\n")
        frame = ar.read_csv(csv_path)
        
        clean = ar.pipeline(frame, [
            ("filter_rows", {"column": "score", "op": ">=", "value": 85}),
        ])
        result = ar.to_pandas(clean)
        assert len(result) == 2
        assert set(result["name"]) == {"Alice", "Bob"}

    def test_multiple_filters_in_pipeline(self, tmp_path):
        """Multiple filter_rows steps chained together."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "id,age,salary\n"
            "1,25,50000\n"
            "2,30,60000\n"
            "3,35,75000\n"
            "4,40,80000\n"
        )
        frame = ar.read_csv(csv_path)
        
        clean = ar.pipeline(frame, [
            ("filter_rows", {"column": "age", "op": ">", "value": 28}),
            ("filter_rows", {"column": "salary", "op": "<", "value": 80000}),
        ])
        result = ar.to_pandas(clean)
        assert len(result) == 2
        assert list(result["id"]) == [2, 3]

    def test_filter_with_other_steps(self, tmp_path):
        """filter_rows combined with other cleaning steps."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,score\n1, Alice ,85\n2, Bob ,92\n3, Charlie ,78\n")
        frame = ar.read_csv(csv_path)
        
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("filter_rows", {"column": "score", "op": ">=", "value": 85}),
        ])
        result = ar.to_pandas(clean)
        assert len(result) == 2
        assert result["name"].iloc[0] == "Alice"


class TestFilterRowsIntegration:
    """Integration tests with I/O and profiling."""

    def test_filter_and_write_csv(self, tmp_path):
        """Filter data, write to CSV, and verify results."""
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"
        input_csv.write_text("product,revenue\nA,1000\nB,2000\nC,3000\n")
        
        frame = ar.read_csv(input_csv)
        filtered = ar.filter_rows(frame, "revenue", ">=", 2000)
        ar.write_csv(filtered, output_csv)
        
        result = pd.read_csv(output_csv)
        assert len(result) == 2
        assert list(result["product"]) == ["B", "C"]

    def test_filter_and_profile(self, tmp_path):
        """Filter data and generate quality profile report."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("score\n50\n75\n85\n95\n")
        
        frame = ar.read_csv(csv_path)
        filtered = ar.filter_rows(frame, "score", ">=", 75)
        report = ar.profile(filtered)
        
        assert report.row_count == 3
        assert report.columns["score"]["min"] == 75.0
        assert report.columns["score"]["max"] == 95.0

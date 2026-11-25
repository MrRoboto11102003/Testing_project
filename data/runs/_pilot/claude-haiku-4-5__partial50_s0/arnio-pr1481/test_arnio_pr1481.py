import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestFilterRows:
    """Tests for arnio.cleaning.filter_rows"""

    def test_filter_greater_than_int(self):
        """Test > operator with integers"""
        frame = ar.from_dict({"age": [18, 25, 30, 35]})
        result = ar.filter_rows(frame, column="age", op=">", value=25)
        df = ar.to_pandas(result)
        assert df["age"].tolist() == [30, 35]

    def test_filter_less_than_int(self):
        """Test < operator with integers"""
        frame = ar.from_dict({"age": [18, 25, 30, 35]})
        result = ar.filter_rows(frame, column="age", op="<", value=25)
        df = ar.to_pandas(result)
        assert df["age"].tolist() == [18]

    def test_filter_greater_equal_int(self):
        """Test >= operator with integers"""
        frame = ar.from_dict({"age": [18, 25, 30, 35]})
        result = ar.filter_rows(frame, column="age", op=">=", value=25)
        df = ar.to_pandas(result)
        assert df["age"].tolist() == [25, 30, 35]

    def test_filter_less_equal_int(self):
        """Test <= operator with integers"""
        frame = ar.from_dict({"age": [18, 25, 30, 35]})
        result = ar.filter_rows(frame, column="age", op="<=", value=25)
        df = ar.to_pandas(result)
        assert df["age"].tolist() == [18, 25]

    def test_filter_equal_int(self):
        """Test == operator with integers"""
        frame = ar.from_dict({"age": [18, 25, 30, 25, 35]})
        result = ar.filter_rows(frame, column="age", op="==", value=25)
        df = ar.to_pandas(result)
        assert df["age"].tolist() == [25, 25]

    def test_filter_not_equal_int(self):
        """Test != operator with integers"""
        frame = ar.from_dict({"age": [18, 25, 30]})
        result = ar.filter_rows(frame, column="age", op="!=", value=25)
        df = ar.to_pandas(result)
        assert df["age"].tolist() == [18, 30]

    def test_filter_float_values(self):
        """Test filtering with float values"""
        frame = ar.from_dict({"score": [1.5, 2.3, 3.7, 4.1]})
        result = ar.filter_rows(frame, column="score", op=">", value=2.5)
        df = ar.to_pandas(result)
        assert len(df) == 2

    def test_filter_string_values(self):
        """Test filtering with string values"""
        frame = ar.from_dict({"name": ["alice", "bob", "charlie"]})
        result = ar.filter_rows(frame, column="name", op="==", value="bob")
        df = ar.to_pandas(result)
        assert len(df) == 1
        assert df["name"].iloc[0] == "bob"

    def test_filter_string_comparison_lex(self):
        """Test lexicographic string comparison"""
        frame = ar.from_dict({"letter": ["a", "b", "c", "d"]})
        result = ar.filter_rows(frame, column="letter", op="<", value="c")
        df = ar.to_pandas(result)
        assert set(df["letter"].tolist()) == {"a", "b"}

    def test_filter_preserves_other_columns(self):
        """Test that other columns are preserved after filtering"""
        frame = ar.from_dict({
            "age": [18, 25, 30],
            "name": ["alice", "bob", "charlie"]
        })
        result = ar.filter_rows(frame, column="age", op=">", value=20)
        df = ar.to_pandas(result)
        assert set(df.columns) == {"age", "name"}
        assert len(df) == 2

    def test_filter_arframe_returns_arframe(self):
        """Test that ArFrame input returns ArFrame output"""
        frame = ar.from_dict({"age": [18, 25, 30]})
        result = ar.filter_rows(frame, column="age", op=">", value=20)
        assert isinstance(result, ar.ArFrame)

    def test_filter_dataframe_returns_dataframe(self):
        """Test that DataFrame input returns DataFrame output"""
        df = pd.DataFrame({"age": [18, 25, 30]})
        result = ar.filter_rows(df, column="age", op=">", value=20)
        assert isinstance(result, pd.DataFrame)

    def test_filter_empty_result(self):
        """Test filtering that results in empty frame"""
        frame = ar.from_dict({"age": [18, 20, 22]})
        result = ar.filter_rows(frame, column="age", op=">", value=100)
        df = ar.to_pandas(result)
        assert len(df) == 0

    def test_filter_all_rows_match(self):
        """Test filtering where all rows match condition"""
        frame = ar.from_dict({"age": [25, 30, 35]})
        result = ar.filter_rows(frame, column="age", op=">", value=20)
        df = ar.to_pandas(result)
        assert len(df) == 3

    def test_filter_in_pipeline(self):
        """Test using filter_rows as a pipeline step"""
        frame = ar.from_dict({"revenue": [100, 500, 1000, 2000]})
        result = ar.pipeline(frame, [
            ("filter_rows", {"column": "revenue", "op": ">=", "value": 500})
        ])
        df = ar.to_pandas(result)
        assert len(df) == 3
        assert df["revenue"].min() >= 500

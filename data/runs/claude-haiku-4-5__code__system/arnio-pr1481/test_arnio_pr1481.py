import pytest
import pandas as pd
import numpy as np
import sys
import subprocess
from pathlib import Path

import arnio as ar


class TestFilterRows:
    """End-to-end tests for arnio.cleaning.filter_rows"""
    
    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV file for testing"""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "age": [15, 20, 25, 30, 35, 40],
            "salary": [30000, 40000, 50000, 60000, 70000, 80000],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
            "active": [True, False, True, True, False, True]
        })
        df.to_csv(csv_file, index=False)
        return csv_file
    
    def test_filter_rows_greater_than(self, sample_csv):
        """Test > operator: load file, filter, verify results"""
        frame = ar.read_csv(sample_csv)
        filtered = ar.filter_rows(frame, column="age", op=">", value=25)
        
        result = pd.DataFrame(filtered)
        assert len(result) == 3
        assert all(result["age"] > 25)
        assert result["age"].tolist() == [30, 35, 40]
    
    def test_filter_rows_less_than(self, sample_csv):
        """Test < operator"""
        frame = ar.read_csv(sample_csv)
        filtered = ar.filter_rows(frame, column="salary", op="<", value=50000)
        
        result = pd.DataFrame(filtered)
        assert len(result) == 2
        assert all(result["salary"] < 50000)
    
    def test_filter_rows_equal(self, sample_csv):
        """Test == operator"""
        frame = ar.read_csv(sample_csv)
        filtered = ar.filter_rows(frame, column="active", op="==", value=True)
        
        result = pd.DataFrame(filtered)
        assert len(result) == 4
        assert all(result["active"] == True)
    
    def test_filter_rows_not_equal(self, sample_csv):
        """Test != operator"""
        frame = ar.read_csv(sample_csv)
        filtered = ar.filter_rows(frame, column="name", op="!=", value="Alice")
        
        result = pd.DataFrame(filtered)
        assert len(result) == 5
        assert "Alice" not in result["name"].values
    
    def test_filter_rows_greater_equal(self, sample_csv):
        """Test >= operator"""
        frame = ar.read_csv(sample_csv)
        filtered = ar.filter_rows(frame, column="age", op=">=", value=30)
        
        result = pd.DataFrame(filtered)
        assert len(result) == 3
        assert all(result["age"] >= 30)
        assert result["age"].tolist() == [30, 35, 40]
    
    def test_filter_rows_less_equal(self, sample_csv):
        """Test <= operator"""
        frame = ar.read_csv(sample_csv)
        filtered = ar.filter_rows(frame, column="salary", op="<=", value=50000)
        
        result = pd.DataFrame(filtered)
        assert len(result) == 3
        assert all(result["salary"] <= 50000)
    
    def test_filter_rows_unsupported_operator_error(self, sample_csv):
        """Test ValueError on unsupported operator"""
        frame = ar.read_csv(sample_csv)
        
        with pytest.raises(ValueError, match="Unsupported operator"):
            ar.filter_rows(frame, column="age", op=">>", value=25)
    
    def test_filter_rows_unknown_column_error(self, sample_csv):
        """Test ValueError on unknown column"""
        frame = ar.read_csv(sample_csv)
        
        with pytest.raises(ValueError, match="Unknown column"):
            ar.filter_rows(frame, column="nonexistent", op=">", value=25)
    
    def test_filter_rows_invalid_column_type_error(self, sample_csv):
        """Test TypeError when column is not a string"""
        frame = ar.read_csv(sample_csv)
        
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(frame, column=123, op=">", value=25)
    
    def test_filter_rows_invalid_operator_type_error(self, sample_csv):
        """Test TypeError when op is not a string"""
        frame = ar.read_csv(sample_csv)
        
        with pytest.raises(TypeError, match="op must be a string"):
            ar.filter_rows(frame, column="age", op=5, value=25)
    
    def test_filter_rows_non_scalar_value_error(self, sample_csv):
        """Test TypeError when value is not scalar"""
        frame = ar.read_csv(sample_csv)
        
        with pytest.raises(TypeError, match="value must be a scalar"):
            ar.filter_rows(frame, column="age", op=">", value=[20, 30])
    
    def test_filter_rows_type_incompatibility_error(self, sample_csv):
        """Test TypeError when comparing incompatible types"""
        frame = ar.read_csv(sample_csv)
        
        with pytest.raises(TypeError, match="cannot compare column"):
            ar.filter_rows(frame, column="name", op=">", value=100)
    
    def test_filter_rows_with_pandas_dataframe_input(self, sample_csv):
        """Test that pd.DataFrame input returns pd.DataFrame output"""
        df = pd.read_csv(sample_csv)
        result = ar.filter_rows(df, column="age", op=">", value=25)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert all(result["age"] > 25)
    
    def test_filter_rows_empty_result(self, sample_csv):
        """Test filtering that returns no rows"""
        frame = ar.read_csv(sample_csv)
        filtered = ar.filter_rows(frame, column="age", op=">", value=100)
        
        result = pd.DataFrame(filtered)
        assert len(result) == 0
        assert list(result.columns) == ["age", "salary", "name", "active"]
    
    def test_filter_rows_with_nan_values(self, tmp_path):
        """Test filtering handles NaN values correctly by filtering them out"""
        csv_file = tmp_path / "data_nan.csv"
        df = pd.DataFrame({
            "value": [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(csv_file)
        filtered = ar.filter_rows(frame, column="value", op=">", value=2.0)
        
        result = pd.DataFrame(filtered)
        assert len(result) == 2
        assert result["value"].tolist() == [3.0, 5.0]
        assert not result["value"].isna().any()

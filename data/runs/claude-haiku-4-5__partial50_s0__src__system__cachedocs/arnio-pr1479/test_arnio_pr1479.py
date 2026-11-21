import pytest
import json
import pandas as pd
import numpy as np
import arnio as ar


class TestDataQualityReportToDict:
    """End-to-end tests for DataQualityReport.to_dict()"""

    def test_basic_to_dict_structure(self, tmp_path):
        """Test basic to_dict() conversion returns required keys"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1,Alice,95\n2,Bob,87\n3,Charlie,92\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        result = report.to_dict()
        
        # Verify structure and required keys
        assert isinstance(result, dict)
        required_keys = {
            "row_count", "column_count", "memory_usage",
            "duplicate_rows", "duplicate_ratio", "quality_score",
            "score_components", "columns", "suggestions"
        }
        assert required_keys.issubset(result.keys())
        
        # Verify basic types
        assert result["row_count"] == 3
        assert result["column_count"] == 3
        assert isinstance(result["memory_usage"], int)
        assert isinstance(result["columns"], dict)
        assert isinstance(result["suggestions"], list)

    def test_to_dict_redact_sample_values_true(self, tmp_path):
        """Test to_dict() with redact_sample_values=True"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("email,phone\nalice@example.com,555-1234\nbob@example.com,555-5678\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        result_no_redact = report.to_dict(redact_sample_values=False)
        result_redact = report.to_dict(redact_sample_values=True)
        
        # Both should have structure
        assert "columns" in result_no_redact
        assert "columns" in result_redact
        assert len(result_no_redact["columns"]) == 2
        assert len(result_redact["columns"]) == 2

    def test_to_dict_exclude_columns_list(self, tmp_path):
        """Test excluding columns using list"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,email\n1,Alice,alice@example.com\n2,Bob,bob@example.com\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        result = report.to_dict(exclude_columns=["email"])
        
        assert "id" in result["columns"]
        assert "name" in result["columns"]
        assert "email" not in result["columns"]

    def test_to_dict_exclude_columns_tuple(self, tmp_path):
        """Test excluding columns using tuple"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        result = report.to_dict(exclude_columns=("b", "c"))
        
        assert "a" in result["columns"]
        assert "b" not in result["columns"]
        assert "c" not in result["columns"]

    def test_to_dict_exclude_columns_set(self, tmp_path):
        """Test excluding columns using set"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x,y,z\n1,2,3\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        result = report.to_dict(exclude_columns={"y"})
        
        assert "x" in result["columns"]
        assert "y" not in result["columns"]
        assert "z" in result["columns"]

    def test_to_dict_exclude_nonexistent_column_raises_keyerror(self, tmp_path):
        """Test error when excluding non-existent column"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        with pytest.raises(KeyError) as exc_info:
            report.to_dict(exclude_columns=["nonexistent"])
        
        error_msg = str(exc_info.value)
        assert "Unknown exclude_columns" in error_msg
        assert "nonexistent" in error_msg

    def test_to_dict_exclude_columns_wrong_type_raises_typeerror(self, tmp_path):
        """Test error when exclude_columns is wrong type"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        with pytest.raises(TypeError) as exc_info:
            report.to_dict(exclude_columns="a")
        
        assert "must be a list, tuple, set, or None" in str(exc_info.value)

    def test_to_dict_exclude_columns_non_string_elements_raises_typeerror(self, tmp_path):
        """Test error when exclude_columns contains non-strings"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        with pytest.raises(TypeError) as exc_info:
            report.to_dict(exclude_columns=[1, 2])
        
        assert "must contain only string column names" in str(exc_info.value)

    def test_to_dict_with_duplicates_includes_metrics(self, tmp_path):
        """Test to_dict() captures duplicate metrics"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,value\n1,100\n1,100\n2,200\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        result = report.to_dict()
        
        assert "duplicate_rows" in result
        assert "duplicate_ratio" in result
        assert isinstance(result["duplicate_rows"], (int, float))
        assert 0 <= result["duplicate_ratio"] <= 1

    def test_to_dict_with_missing_values_in_columns(self, tmp_path):
        """Test to_dict() on data with null/missing values"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1,Alice,95\n2,,\n3,Charlie,\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        result = report.to_dict()
        
        assert result["column_count"] == 3
        assert "name" in result["columns"]
        assert "score" in result["columns"]

    def test_to_dict_exclude_all_columns_empty_dict(self, tmp_path):
        """Test excluding all columns results in empty columns dict"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        result = report.to_dict(exclude_columns=["a", "b", "c"])
        
        assert result["columns"] == {}
        assert result["column_count"] == 3  # original count preserved

    def test_to_dict_json_serializable(self, tmp_path):
        """Test that to_dict() output is JSON serializable"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,value\n1,100\n2,200\n3,300\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        result = report.to_dict()
        
        # Should serialize to JSON without error
        json_str = json.dumps(result, default=str)
        assert isinstance(json_str, str)
        
        # Should deserialize back
        deserialized = json.loads(json_str)
        assert deserialized["row_count"] == 3

    def test_to_dict_through_pandas_accessor(self, tmp_path):
        """Test to_dict() workflow using pandas accessor"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age,salary\nAlice,30,50000\nBob,25,45000\n")
        
        df = pd.read_csv(str(csv_file))
        report = df.arnio.profile()
        result = report.to_dict(exclude_columns=["age"])
        
        # Verify structure
        assert "name" in result["columns"]
        assert "age" not in result["columns"]
        assert "salary" in result["columns"]
        assert result["row_count"] == 2

    def test_to_dict_multiple_exclude_columns_nonexistent(self, tmp_path):
        """Test error message includes all unknown columns"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        with pytest.raises(KeyError) as exc_info:
            report.to_dict(exclude_columns=["x", "y", "z"])
        
        error_msg = str(exc_info.value)
        assert "Unknown exclude_columns" in error_msg

    def test_to_dict_quality_score_present(self, tmp_path):
        """Test that quality_score and components are in output"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,value\n1,100\n2,200\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        result = report.to_dict()
        
        assert "quality_score" in result
        assert "score_components" in result
        assert isinstance(result["quality_score"], (int, float))
        assert isinstance(result["score_components"], dict)

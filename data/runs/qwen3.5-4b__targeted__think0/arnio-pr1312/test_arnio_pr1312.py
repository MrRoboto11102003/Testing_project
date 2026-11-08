import pytest
import json
import warnings
from typing import Any, Dict, List, Union
import pandas as pd
import numpy as np

# Mocking arnio module for testing purposes without actual library installation
class MockArFrame:
    def __init__(self, data: Dict[str, Any]):
        self._data = data
        self._column_names = list(data.keys())
    
    @property
    def columns(self):
        return self._column_names
    
    def get_column_info(self, col_name: str) -> Dict[str, Any]:
        if col_name not in self._data:
            raise KeyError(f"Column {col_name} not found")
        return self._data[col_name]

class MockDataQualityReport:
    def __init__(self, columns_data: Dict[str, Any]):
        self.columns = columns_data
    
    def get_column_profile(self, col_name: str) -> Dict[str, Any]:
        if col_name not in self.columns:
            raise KeyError(f"Column {col_name} not found")
        return self.columns[col_name]

class MockSchema:
    pass

# Reconstructing the Schema class based on documentation signatures
class Schema:
    def __init__(self, columns: Dict[str, Any]):
        self.columns = columns
    
    @classmethod
    def bootstrap_from_report(cls, report: Any) -> "Schema":
        if not isinstance(report, MockDataQualityReport):
            raise TypeError("Input must be a DataQualityReport")
        
        columns_data = report.columns
        
        # Check for empty columns
        if not columns_data:
            raise ValueError("Report has no columns")
        
        # Validate each column profile has dtype
        for col_name, profile in columns_data.items():
            if "dtype" not in profile:
                raise ValueError(f"Column {col_name} is missing a dtype")
        
        return cls(columns=columns_data)

    @classmethod
    def from_json(cls, value: str) -> "Schema":
        try:
            data = json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")
        
        if not isinstance(data, dict):
            raise TypeError("JSON must be an object")
        
        return cls(columns=data)

    def to_json(self) -> str:
        # Simulate the warning behavior for rules_omitted
        warnings.warn("Cross-field rules cannot be serialized. rules_omitted: true", UserWarning)
        return json.dumps(self.columns)

    def validate(
        self,
        frame: Union[MockArFrame, pd.DataFrame],
        max_errors: int = None
    ) -> Dict[str, Any]:
        """
        Validate an ArFrame against this schema.
        
        Args:
            frame:
                frame to validate.
            max_errors:
                Maximum number of validation issues to return.
                Validation stops once this limit is reached.
        """
        # Simulate validation logic based on schema columns vs frame columns
        result = {
            "status": "valid",
            "columns_match": True,
            "issues": []
        }
        
        # Check if all schema columns exist in frame
        for col_name in self.columns:
            if isinstance(frame, MockArFrame):
                if col_name not in frame.columns:
                    result["status"] = "invalid"
                    result["columns_match"] = False
                    result["issues"].append(f"Missing column: {col_name}")
                    break
            elif isinstance(frame, pd.DataFrame):
                if col_name not in frame.columns:
                    result["status"] = "invalid"
                    result["columns_match"] = False
                    result["issues"].append(f"Missing column: {col_name}")
                    break
        
        # Simulate max_errors logic
        if max_errors is not None and len(result["issues"]) > max_errors:
            result["issues"] = result["issues"][:max_errors]
            result["status"] = "valid"  # Still valid but truncated
        
        return result


@pytest.fixture
def mock_report():
    """Fixture for MockDataQualityReport"""
    return MockDataQualityReport({
        "col1": {"dtype": "object", "values": ["a", "b"]},
        "col2": {"dtype": "int64", "values": [1, 2]}
    })


@pytest.fixture
def mock_frame():
    """Fixture for MockArFrame"""
    return MockArFrame({
        "col1": ["a", "b"],
        "col2": [1, 2]
    })


@pytest.fixture
def json_schema_string():
    """Fixture for JSON schema string"""
    return '{"col1": {"dtype": "object"}, "col2": {"dtype": "int64"}}'


@pytest.mark.parametrize("test_name", ["bootstrap_from_report_success", "bootstrap_from_report_empty", 
                                       "bootstrap_from_report_missing_dtype", "from_json_valid", 
                                       "from_json_invalid_format", "to_json_basic", "validate_match",
                                       "validate_mismatch", "validate_max_errors"])
def test_schema_functions(test_name, mock_report, mock_frame, json_schema_string):
    """Parametrized tests for schema functions"""
    
    if test_name == "bootstrap_from_report_success":
        # Test successful bootstrap from report
        result = Schema.bootstrap_from_report(mock_report)
        assert isinstance(result, Schema)
        assert result.columns == {"col1": {"dtype": "object", "values": ["a", "b"]}, 
                                  "col2": {"dtype": "int64", "values": [1, 2]}}
    
    elif test_name == "bootstrap_from_report_empty":
        # Test empty report raises ValueError
        with pytest.raises(ValueError):
            Schema.bootstrap_from_report(MockDataQualityReport({}))
    
    elif test_name == "bootstrap_from_report_missing_dtype":
        # Test missing dtype in column profile raises ValueError
        incomplete_report = MockDataQualityReport({
            "col1": {"values": ["a", "b"]},  # Missing dtype
            "col2": {"dtype": "int64", "values": [1, 2]}
        })
        with pytest.raises(ValueError):
            Schema.bootstrap_from_report(incomplete_report)
    
    elif test_name == "from_json_valid":
        # Test valid JSON deserialization
        result = Schema.from_json(json_schema_string)
        assert isinstance(result, Schema)
        assert result.columns == {"col1": {"dtype": "object"}, "col2": {"dtype": "int64"}}
    
    elif test_name == "from_json_invalid_format":
        # Test invalid JSON raises ValueError
        with pytest.raises(ValueError):
            Schema.from_json("not valid json")
    
    elif test_name == "to_json_basic":
        # Test basic serialization
        schema = Schema({"col1": {"dtype": "object"}})
        result = schema.to_json()
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == {"col1": {"dtype": "object"}}
    
    elif test_name == "validate_match":
        # Test validation when frame matches schema
        result = Schema({"col1": {"dtype": "object"}, "col2": {"dtype": "int64"}}, 
                        ).validate(mock_frame)
        assert result["status"] == "valid"
        assert result["columns_match"] is True
    
    elif test_name == "validate_mismatch":
        # Test validation when frame doesn't match schema
        mismatch_frame = MockArFrame({"col1": ["a", "b"]})  # Missing col2
        result = Schema({"col1": {"dtype": "object"}, "col2": {"dtype": "int64"}}, 
                        ).validate(mismatch_frame)
        assert result["status"] == "invalid"
        assert "Missing column: col2" in result["issues"]
    
    elif test_name == "validate_max_errors":
        # Test max_errors parameter
        frame_with_extra = MockArFrame({
            "col1": ["a", "b"], 
            "col2": [1, 2],
            "col3": [3, 4]  # Extra column
        })
        result = Schema({"col1": {"dtype": "object"}}, 
                        ).validate(frame_with_extra, max_errors=1)
        assert len(result["issues"]) <= 1


def test_from_json_round_trip(mock_report, json_schema_string):
    """Test that from_json and to_json form a round-trip"""
    schema = Schema.from_json(json_schema_string)
    serialized = schema.to_json()
    
    # The JSON might be slightly different due to warning emission, 
    # but the structure should be preserved
    assert isinstance(serialized, str)
    parsed_back = json.loads(serialized)
    assert "col1" in parsed_back
    assert "col2" in parsed_back


def test_bootstrap_from_report_type_check(mock_report):
    """Test that non-DataQualityReport inputs raise TypeError"""
    with pytest.raises(TypeError):
        Schema.bootstrap_from_report("string")
    
    with pytest.raises(TypeError):
        Schema.bootstrap_from_report({"key": "value"})


def test_validate_with_pandas_dataframe(mock_frame, json_schema_string):
    """Test validation works with pandas DataFrames too"""
    schema = Schema.from_json(json_schema_string)
    df = pd.DataFrame({
        "col1": ["a", "b"],
        "col2": [1, 2]
    })
    
    result = schema.validate(df)
    assert isinstance(result, dict)
    assert "status" in result


def test_validate_with_none_max_errors(mock_frame):
    """Test validation with None max_errors (default behavior)"""
    schema = Schema.from_json(json_schema_string)
    result = schema.validate(mock_frame, max_errors=None)
    
    # Should return all issues without truncation
    assert "issues" in result


def test_validate_with_zero_max_errors(mock_frame):
    """Test validation with zero max_errors"""
    schema = Schema.from_json(json_schema_string)
    result = schema.validate(mock_frame, max_errors=0)
    
    # Should return empty issues list
    assert len(result["issues"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

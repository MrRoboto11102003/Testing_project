import pytest
import pandas as pd
import arnio as ar
from arnio.schema import Schema, ValidationResult
from arnio.schema import String, Int64, Float64
from arnio.convert import from_pandas


class TestSchemaValidate:
    """Pytest tests for arnio.schema.Schema.validate method"""

    def test_validate_returns_validation_result(self):
        """validate() returns a ValidationResult object"""
        df = pd.DataFrame({'col': [1, 2]})
        frame = from_pandas(df)
        schema = Schema(fields={'col': Int64()})
        
        result = schema.validate(frame)
        
        assert isinstance(result, ValidationResult)

    def test_validate_passed_true_for_valid_frame(self):
        """validate() returns passed=True for valid data"""
        df = pd.DataFrame({'age': [25, 30]})
        frame = from_pandas(df)
        schema = Schema(fields={'age': Int64()})
        
        result = schema.validate(frame)
        
        assert result.passed is True
        assert result.issue_count == 0

    def test_validate_passed_false_for_invalid_frame(self):
        """validate() returns passed=False when constraints violated"""
        df = pd.DataFrame({'age': [25, None]})
        frame = from_pandas(df)
        schema = Schema(fields={'age': Int64(nullable=False)})
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert result.issue_count > 0

    def test_validate_detects_null_violation_when_nullable_false(self):
        """validate() detects nulls when field has nullable=False"""
        df = pd.DataFrame({'required': ['a', None]})
        frame = from_pandas(df)
        schema = Schema(fields={'required': String(nullable=False)})
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert len(result.issues) > 0

    def test_validate_allows_nulls_when_nullable_true(self):
        """validate() allows nulls when field has nullable=True"""
        df = pd.DataFrame({'optional': ['a', None, 'c']})
        frame = from_pandas(df)
        schema = Schema(fields={'optional': String(nullable=True)})
        
        result = schema.validate(frame)
        
        assert result.passed is True

    def test_validate_enforces_min_constraint_on_numeric(self):
        """validate() enforces minimum value for numeric fields"""
        df = pd.DataFrame({'score': [5, 20]})
        frame = from_pandas(df)
        schema = Schema(fields={'score': Int64(min=10)})
        
        result = schema.validate(frame)
        
        assert result.passed is False

    def test_validate_enforces_max_constraint_on_numeric(self):
        """validate() enforces maximum value for numeric fields"""
        df = pd.DataFrame({'score': [50, 150]})
        frame = from_pandas(df)
        schema = Schema(fields={'score': Int64(max=100)})
        
        result = schema.validate(frame)
        
        assert result.passed is False

    def test_validate_checks_allowed_values_constraint(self):
        """validate() validates against allowed value set"""
        df = pd.DataFrame({'status': ['active', 'invalid']})
        frame = from_pandas(df)
        schema = Schema(fields={
            'status': String(allowed={'active', 'inactive'})
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False

    def test_validate_detects_duplicates_when_unique_true(self):
        """validate() detects duplicates when unique=True"""
        df = pd.DataFrame({'id': [1, 2, 2]})
        frame = from_pandas(df)
        schema = Schema(fields={'id': Int64(unique=True)})
        
        result = schema.validate(frame)
        
        assert result.passed is False

    def test_validate_respects_max_errors_parameter(self):
        """validate() respects max_errors parameter to limit issues"""
        df = pd.DataFrame({'val': ['a', 'b', 'c', 'd']})
        frame = from_pandas(df)
        schema = Schema(fields={'val': Int64()})
        
        result = schema.validate(frame, max_errors=2)
        
        assert result.issue_count <= 2

    def test_validate_enforces_string_min_length(self):
        """validate() checks minimum string length constraint"""
        df = pd.DataFrame({'code': ['', 'ab']})
        frame = from_pandas(df)
        schema = Schema(fields={'code': String(min_length=2)})
        
        result = schema.validate(frame)
        
        assert result.passed is False

    def test_validate_enforces_string_max_length(self):
        """validate() checks maximum string length constraint"""
        df = pd.DataFrame({'code': ['ab', 'abcd']})
        frame = from_pandas(df)
        schema = Schema(fields={'code': String(max_length=3)})
        
        result = schema.validate(frame)
        
        assert result.passed is False

    def test_validate_checks_regex_pattern_on_strings(self):
        """validate() validates regex patterns on string columns"""
        df = pd.DataFrame({'email': ['a@b.c', 'invalid']})
        frame = from_pandas(df)
        schema = Schema(fields={
            'email': String(pattern=r'^[a-z]+@[a-z]+\.[a-z]+$')
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False

    def test_validate_checks_multiple_columns_simultaneously(self):
        """validate() validates all columns defined in schema"""
        df = pd.DataFrame({'name': [None], 'age': [99]})
        frame = from_pandas(df)
        schema = Schema(fields={
            'name': String(nullable=False),
            'age': Int64(max=100)
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert result.issue_count >= 2

    def test_validate_handles_empty_frame(self):
        """validate() handles empty frames without error"""
        df = pd.DataFrame({'col': pd.Series([], dtype='int64')})
        frame = from_pandas(df)
        schema = Schema(fields={'col': Int64()})
        
        result = schema.validate(frame)
        
        assert result.row_count == 0
        assert isinstance(result, ValidationResult)

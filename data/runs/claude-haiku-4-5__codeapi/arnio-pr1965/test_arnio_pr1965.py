import pytest
import pandas as pd
import arnio as ar
from arnio.schema import ValidationResult, ValidationIssue, Schema


class TestValidate:
    """Tests for arnio.schema.validate function"""

    def test_validate_returns_validation_result(self):
        """validate returns a ValidationResult object"""
        frame = ar.from_dict({"col": [1, 2, 3]})
        schema = {"col": ar.Int64()}
        result = ar.validate(frame, schema)
        assert isinstance(result, ValidationResult)

    def test_validate_passes_with_valid_data(self):
        """Validation passes when data matches schema"""
        frame = ar.from_dict({"name": ["Alice", "Bob"], "age": [30, 25]})
        schema = ar.Schema({
            "name": ar.String(),
            "age": ar.Int64()
        })
        result = ar.validate(frame, schema)
        assert result.passed
        assert result.issue_count == 0

    def test_validate_missing_column_detected(self):
        """Missing required column generates issue"""
        frame = ar.from_dict({"name": ["Alice"]})
        schema = ar.Schema({
            "name": ar.String(),
            "age": ar.Int64()
        })
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.rule == "required_column" and issue.column == "age" 
                   for issue in result.issues)

    def test_validate_dict_schema_converted(self):
        """Dict[str, Field] schema is accepted and converted"""
        frame = ar.from_dict({"col": [1, 2]})
        schema_dict = {"col": ar.Int64()}
        result = ar.validate(frame, schema_dict)
        assert isinstance(result, ValidationResult)

    def test_validate_non_arframe_raises_type_error(self):
        """Only ArFrame input is accepted"""
        df = pd.DataFrame({"col": [1, 2]})
        schema = {"col": ar.Int64()}
        with pytest.raises(TypeError, match="ArFrame"):
            ar.validate(df, schema)

    def test_validate_max_errors_zero_raises(self):
        """max_errors=0 raises ValueError"""
        frame = ar.from_dict({"col": [1]})
        schema = {"col": ar.Int64()}
        with pytest.raises(ValueError, match="max_errors must be >= 1"):
            ar.validate(frame, schema, max_errors=0)

    def test_validate_max_errors_negative_raises(self):
        """max_errors negative raises ValueError"""
        frame = ar.from_dict({"col": [1]})
        schema = {"col": ar.Int64()}
        with pytest.raises(ValueError, match="max_errors must be >= 1"):
            ar.validate(frame, schema, max_errors=-1)

    def test_validate_max_errors_non_int_raises(self):
        """max_errors non-int raises TypeError"""
        frame = ar.from_dict({"col": [1]})
        schema = {"col": ar.Int64()}
        with pytest.raises(TypeError, match="max_errors must be an int or None"):
            ar.validate(frame, schema, max_errors="5")

    def test_validate_max_errors_bool_raises(self):
        """max_errors bool raises TypeError"""
        frame = ar.from_dict({"col": [1]})
        schema = {"col": ar.Int64()}
        with pytest.raises(TypeError, match="max_errors must be an int or None"):
            ar.validate(frame, schema, max_errors=True)

    def test_validate_max_errors_limits_returned_issues(self):
        """max_errors parameter limits the number of issues returned"""
        frame = ar.from_dict({"col": [1]})
        schema = ar.Schema({
            "col": ar.Int64(),
            "missing1": ar.String(),
            "missing2": ar.Email(),
            "missing3": ar.URL()
        })
        result = ar.validate(frame, schema, max_errors=2)
        assert result.issue_count <= 2

    def test_validate_strict_mode_unexpected_column(self):
        """strict=True detects unexpected columns"""
        frame = ar.from_dict({"col1": [1], "extra": ["x"]})
        schema = ar.Schema(
            {"col1": ar.Int64()},
            strict=True
        )
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.rule == "unexpected_column" for issue in result.issues)

    def test_validate_unique_single_column_no_duplicates(self):
        """unique constraint passes with no duplicates"""
        frame = ar.from_dict({"id": [1, 2, 3]})
        schema = ar.Schema(
            {"id": ar.Int64()},
            unique=["id"]
        )
        result = ar.validate(frame, schema)
        assert result.passed

    def test_validate_unique_composite_with_duplicates(self):
        """unique constraint detects duplicate composite keys"""
        frame = ar.from_dict({
            "first": ["A", "A"],
            "last": ["B", "B"]
        })
        schema = ar.Schema(
            {"first": ar.String(), "last": ar.String()},
            unique=["first", "last"]
        )
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.rule == "composite_unique" for issue in result.issues)

    def test_validate_unique_missing_column_in_constraint(self):
        """unique constraint on missing column reports error"""
        frame = ar.from_dict({"col1": [1]})
        schema = ar.Schema(
            {"col1": ar.Int64()},
            unique=["col1", "missing"]
        )
        result = ar.validate(frame, schema)
        assert not result.passed

    def test_validate_unique_must_be_list_or_tuple(self):
        """unique must be a list or tuple"""
        frame = ar.from_dict({"col": [1]})
        schema = ar.Schema({"col": ar.Int64()}, unique={"col"})
        with pytest.raises(TypeError, match="unique.*list or tuple"):
            ar.validate(frame, schema)

    def test_validate_unique_elements_must_be_strings(self):
        """unique list elements must be strings"""
        frame = ar.from_dict({"col": [1]})
        schema = ar.Schema({"col": ar.Int64()}, unique=["col", 123])
        with pytest.raises(TypeError, match="unique members must be strings"):
            ar.validate(frame, schema)

    def test_validate_unique_empty_list_generates_issue(self):
        """empty unique list generates composite_unique issue"""
        frame = ar.from_dict({"col": [1, 2]})
        schema = ar.Schema({"col": ar.Int64()}, unique=[])
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.rule == "composite_unique" for issue in result.issues)

    def test_validate_custom_rules_executed(self):
        """Custom validation rules are executed"""
        frame = ar.from_dict({"value": [5, 15, 25]})
        
        def rule_max_20(df):
            issues = []
            for idx, row in df.iterrows():
                if row["value"] >= 20:
                    issues.append(
                        ValidationIssue(
                            column="value",
                            rule="custom_max",
                            message="must be < 20",
                            row_index=int(idx) + 1
                        )
                    )
            return issues
        
        schema = ar.Schema(
            {"value": ar.Int64()},
            rules=[rule_max_20]
        )
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(i.rule == "custom_max" for i in result.issues)

    def test_validate_bad_rows_populated(self):
        """bad_rows list contains correct row indices"""
        frame = ar.from_dict({"x": [1, 2, 3]})
        
        def mark_rows(df):
            return [
                ValidationIssue(column="x", rule="test", message="test", row_index=2)
            ]
        
        schema = ar.Schema({"x": ar.Int64()}, rules=[mark_rows])
        result = ar.validate(frame, schema)
        assert 2 in result.bad_rows

    def test_validate_empty_frame(self):
        """Validation succeeds on empty frame"""
        frame = ar.from_dict({"col": []})
        schema = {"col": ar.Int64()}
        result = ar.validate(frame, schema)
        assert result.row_count == 0
        assert result.passed

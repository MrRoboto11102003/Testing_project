import pytest
import pandas as pd
import arnio as ar


class TestValidateBasic:
    """Basic validation functionality."""
    
    def test_validate_passing_schema(self):
        """Passing validation returns empty issues."""
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int64(), "name": ar.String()})
        
        result = ar.validate(frame, schema)
        
        assert result.passed is True
        assert len(result.issues) == 0
    
    def test_validate_with_dict_schema(self):
        """Dictionary schema works the same as Schema object."""
        df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        frame = ar.from_pandas(df)
        
        result = ar.validate(frame, {"id": ar.Int64(), "name": ar.String()})
        
        assert result.passed is True
    
    def test_validate_failing_dtype(self):
        """Wrong dtype generates validation issues."""
        df = pd.DataFrame({"id": ["1", "2", "3"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int64()})
        
        result = ar.validate(frame, schema)
        
        assert result.passed is False
        assert len(result.issues) > 0
        assert result.issues[0].column == "id"


class TestValidateConstraints:
    """Field constraint validation."""
    
    def test_nullable_false_rejects_null(self):
        """nullable=False detects null values."""
        df = pd.DataFrame({"email": ["test@example.com", None, "user@example.com"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        
        result = ar.validate(frame, schema)
        
        assert result.passed is False
        assert any(issue.column == "email" for issue in result.issues)
    
    def test_nullable_true_allows_null(self):
        """nullable=True accepts null values."""
        df = pd.DataFrame({"email": ["test@example.com", None]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email(nullable=True)})
        
        result = ar.validate(frame, schema)
        
        assert result.passed is True
    
    def test_int64_min_constraint(self):
        """Int64 min constraint is enforced."""
        df = pd.DataFrame({"age": [18, 25, 15]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"age": ar.Int64(min=18)})
        
        result = ar.validate(frame, schema)
        
        assert result.passed is False
        assert any(issue.rule == "min" for issue in result.issues)
    
    def test_int64_max_constraint(self):
        """Int64 max constraint is enforced."""
        df = pd.DataFrame({"age": [18, 25, 150]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"age": ar.Int64(max=100)})
        
        result = ar.validate(frame, schema)
        
        assert result.passed is False
        assert any(issue.rule == "max" for issue in result.issues)
    
    def test_string_min_length(self):
        """String min_length constraint is enforced."""
        df = pd.DataFrame({"code": ["ABC", "XY", "DEF"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"code": ar.String(min_length=3)})
        
        result = ar.validate(frame, schema)
        
        assert result.passed is False
        assert any(issue.rule == "min_length" for issue in result.issues)
    
    def test_string_max_length(self):
        """String max_length constraint is enforced."""
        df = pd.DataFrame({"code": ["AB", "TOOLONG", "CD"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"code": ar.String(max_length=5)})
        
        result = ar.validate(frame, schema)
        
        assert result.passed is False
        assert any(issue.rule == "max_length" for issue in result.issues)


class TestValidateEmail:
    """Email field validation."""
    
    def test_email_valid(self):
        """Valid emails pass validation."""
        df = pd.DataFrame({
            "email": ["user@example.com", "test+tag@domain.co.uk"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email()})
        
        result = ar.validate(frame, schema)
        
        assert result.passed is True
    
    def test_email_invalid(self):
        """Invalid emails fail validation."""
        df = pd.DataFrame({"email": ["not-an-email", "user@", "@example.com"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email()})
        
        result = ar.validate(frame, schema)
        
        assert result.passed is False
        assert all(i.column == "email" for i in result.issues)


class TestValidateUnique:
    """Unique constraint validation."""
    
    def test_unique_single_column_duplicates(self):
        """unique=True detects duplicate values."""
        df = pd.DataFrame({"id": [1, 2, 2, 3]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int64(unique=True)})
        
        result = ar.validate(frame, schema)
        
        assert result.passed is False
        assert any(issue.rule == "unique" for issue in result.issues)
    
    def test_unique_composite_key(self):
        """Composite key uniqueness is enforced."""
        df = pd.DataFrame({
            "user_id": [1, 1, 2],
            "course_id": [101, 101, 101]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema(
            {"user_id": ar.Int64(), "course_id": ar.Int64()},
            unique=["user_id", "course_id"]
        )
        
        result = ar.validate(frame, schema)
        
        assert result.passed is False


class TestValidateMaxErrors:
    """max_errors parameter limiting."""
    
    def test_max_errors_limits_issue_count(self):
        """max_errors stops validation early."""
        df = pd.DataFrame({"value": [-10, -20, -30, -40, -50]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"value": ar.Int64(min=0)})
        
        result = ar.validate(frame, schema, max_errors=2)
        
        assert len(result.issues) <= 2
    
    def test_max_errors_none_returns_all(self):
        """max_errors=None returns all validation issues."""
        df = pd.DataFrame({"value": [-10, -20, -30]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"value": ar.Int64(min=0)})
        
        result = ar.validate(frame, schema, max_errors=None)
        
        assert len(result.issues) == 3


class TestValidationResult:
    """ValidationResult properties and methods."""
    
    def test_validation_result_issue_properties(self):
        """ValidationResult issues have required attributes."""
        df = pd.DataFrame({"email": ["not-email", None]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        
        result = ar.validate(frame, schema)
        
        assert result.passed is False
        assert hasattr(result, "issues")
        assert hasattr(result, "issue_count")
        for issue in result.issues:
            assert hasattr(issue, "column")
            assert hasattr(issue, "rule")
            assert hasattr(issue, "message")
    
    def test_row_index_one_based_convention(self):
        """Row index is 1-based (header not counted)."""
        df = pd.DataFrame({"age": [18, -5, 25]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"age": ar.Int64(min=0)})
        
        result = ar.validate(frame, schema)
        
        assert len(result.issues) > 0
        assert any(i.row_index == 2 for i in result.issues)
    
    def test_validation_result_to_pandas_export(self):
        """ValidationResult can export issues as pandas DataFrame."""
        df = pd.DataFrame({"val": [10, -1]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"val": ar.Int64(min=0)})
        
        result = ar.validate(frame, schema)
        issues_df = result.to_pandas()
        
        assert isinstance(issues_df, pd.DataFrame)
        assert len(issues_df) > 0


class TestCrossFieldValidation:
    """Cross-field validation rules."""
    
    def test_cross_field_rule_evaluation(self):
        """Custom cross-field rules are evaluated correctly."""
        df = pd.DataFrame({
            "start_date": ["2024-01-01", "2024-02-01"],
            "end_date": ["2024-12-31", "2024-01-15"]
        })
        frame = ar.from_pandas(df)
        
        def end_after_start(df):
            issues = []
            for i, row in df.iterrows():
                if str(row["end_date"]) < str(row["start_date"]):
                    issues.append(
                        ar.ValidationIssue(
                            column="end_date",
                            rule="cross_field",
                            message="end_date must be >= start_date",
                            row_index=int(i) + 1,
                        )
                    )
            return issues
        
        schema = ar.Schema(
            {"start_date": ar.String(), "end_date": ar.String()},
            rules=[end_after_start],
        )
        
        result = ar.validate(frame, schema)
        
        assert result.passed is False
        assert any(i.rule == "cross_field" for i in result.issues)


class TestValidateErrors:
    """Error handling and validation constraints."""
    
    def test_invalid_unique_parameter_type(self):
        """TypeError when schema.unique is not a sequence of strings."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError):
            schema = ar.Schema({"id": ar.Int64()}, unique=123)
            ar.validate(frame, schema)

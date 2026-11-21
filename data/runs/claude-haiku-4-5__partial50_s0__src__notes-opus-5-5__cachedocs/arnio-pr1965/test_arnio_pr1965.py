import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestValidateInputValidation:
    """Test input type validation for frame, schema, and max_errors."""
    
    def test_frame_must_be_arframe(self):
        """Validate() must reject non-ArFrame inputs."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(TypeError, match="validate\\(\\) expects a single ArFrame"):
            ar.validate(df, {"a": ar.Int64()})
    
    def test_schema_field_must_be_field_instance(self):
        """Schema dict values must be Field instances."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        with pytest.raises(TypeError, match="must be a Field instance"):
            ar.validate(frame, {"a": 1})
    
    def test_max_errors_rejects_bool_and_string(self):
        """max_errors must be int or None, not bool or str."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        schema = ar.Schema({"a": ar.Int64()})
        
        with pytest.raises(TypeError, match="max_errors must be an int or None"):
            ar.validate(frame, schema, max_errors=True)
        
        with pytest.raises(TypeError, match="max_errors must be an int or None"):
            ar.validate(frame, schema, max_errors="1")
    
    def test_max_errors_must_be_positive(self):
        """max_errors must be >= 1."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        schema = ar.Schema({"a": ar.Int64()})
        
        with pytest.raises(ValueError, match="max_errors must be >= 1"):
            ar.validate(frame, schema, max_errors=0)
        
        with pytest.raises(ValueError, match="max_errors must be >= 1"):
            ar.validate(frame, schema, max_errors=-1)


class TestValidationBasics:
    """Test basic validation results and structure."""
    
    def test_valid_data_passes(self):
        """Valid data returns passed=True with no issues."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        schema = ar.Schema({"a": ar.Int64()})
        
        result = ar.validate(frame, schema)
        assert result.passed is True
        assert result.issue_count == 0
        assert result.issues == []
        assert result.bad_rows == []
    
    def test_missing_required_column(self):
        """Missing schema columns are detected with required_column rule."""
        frame = ar.from_pandas(pd.DataFrame({"b": [1, 2, 3]}))
        schema = ar.Schema({"a": ar.Int64()})
        
        result = ar.validate(frame, schema)
        assert result.passed is False
        assert result.issue_count == 1
        assert result.issues[0].rule == "required_column"
        assert result.issues[0].column == "a"
        assert result.issues[0].row_index is None


class TestDtypeAndNullable:
    """Test dtype mismatch and nullable constraint validation."""
    
    def test_dtype_mismatch_string_vs_int(self):
        """dtype mismatch is detected with helpful message."""
        frame = ar.from_pandas(pd.DataFrame({"a": ["1", "2", "3"]}))
        schema = ar.Schema({"a": ar.Int64()})
        
        result = ar.validate(frame, schema)
        assert result.passed is False
        assert result.issues[0].rule == "dtype"
        assert "has dtype" in result.issues[0].message
        assert "expected" in result.issues[0].message
    
    def test_nullable_false_detects_null_values(self):
        """nullable=False detects null values with 1-based row index."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, None, 3]}))
        schema = ar.Schema({"a": ar.Int64(nullable=False)})
        
        result = ar.validate(frame, schema)
        assert result.passed is False
        assert result.issues[0].rule == "nullable"
        assert result.issues[0].row_index == 2  # 1-based: second row


class TestNumericConstraints:
    """Test min/max constraint validation."""
    
    def test_min_max_violations(self):
        """Min and max constraints produce separate issues with correct row indices."""
        frame = ar.from_pandas(pd.DataFrame({"age": [5, -1, 25, 200]}))
        schema = ar.Schema({"age": ar.Int64(min=0, max=150)})
        
        result = ar.validate(frame, schema)
        assert result.passed is False
        assert result.issue_count == 2
        
        min_issues = [i for i in result.issues if i.rule == "min"]
        max_issues = [i for i in result.issues if i.rule == "max"]
        
        assert len(min_issues) == 1 and min_issues[0].row_index == 2
        assert len(max_issues) == 1 and max_issues[0].row_index == 4
        assert sorted(result.bad_rows) == [2, 4]


class TestUniqueAndComposite:
    """Test field-level and schema-level unique constraints."""
    
    def test_field_unique_detects_duplicates(self):
        """Field unique=True flags all duplicate occurrences."""
        frame = ar.from_pandas(pd.DataFrame({"id": [1, 1, 2, 3]}))
        schema = ar.Schema({"id": ar.Int64(unique=True)})
        
        result = ar.validate(frame, schema)
        assert result.passed is False
        assert result.issue_count == 2
        assert all(i.rule == "unique" for i in result.issues)
        assert sorted(result.bad_rows) == [1, 2]
    
    def test_composite_unique_violation(self):
        """Composite unique detects duplicate rows across specified columns."""
        frame = ar.from_pandas(pd.DataFrame({
            "user_id": [1, 1, 2],
            "course_id": [10, 10, 20]
        }))
        schema = ar.Schema(
            {"user_id": ar.Int64(), "course_id": ar.Int64()},
            unique=["user_id", "course_id"]
        )
        
        result = ar.validate(frame, schema)
        assert result.passed is False
        assert all(i.rule == "composite_unique" for i in result.issues)
        assert sorted(result.bad_rows) == [1, 2]


class TestSemanticValidation:
    """Test semantic validators and pattern matching."""
    
    def test_email_validation(self):
        """Email semantic validator detects invalid email formats."""
        frame = ar.from_pandas(pd.DataFrame({
            "email": ["valid@example.com", "invalid.email", "test@example.org"]
        }))
        schema = ar.Schema({"email": ar.Email()})
        
        result = ar.validate(frame, schema)
        assert result.passed is False
        assert result.issues[0].rule == "email"
        assert result.issues[0].row_index == 2
    
    def test_regex_pattern_validation(self):
        """Regex pattern validator flags non-matching values."""
        frame = ar.from_pandas(pd.DataFrame({
            "code": ["USR-0001", "INVALID-CODE", "USR-0002"]
        }))
        schema = ar.Schema({
            "code": ar.Regex(r"^USR-\d{4}$")
        })
        
        result = ar.validate(frame, schema)
        assert result.passed is False
        assert result.issues[0].rule == "pattern"
        assert result.issues[0].row_index == 2


class TestMaxErrorsAndStrictMode:
    """Test max_errors truncation and schema strict mode."""
    
    def test_max_errors_truncates_result(self):
        """max_errors limits the number of issues returned and stops evaluation."""
        frame = ar.from_pandas(pd.DataFrame({"age": [-1, -2, -3, -4]}))
        schema = ar.Schema({"age": ar.Int64(min=0)})
        
        result = ar.validate(frame, schema, max_errors=2)
        assert result.issue_count == 2
        assert len(result.issues) == 2
        assert len(result.bad_rows) == 2
    
    def test_strict_mode_rejects_extra_columns(self):
        """strict=True makes unexpected columns raise unexpected_column issues."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        schema = ar.Schema({"a": ar.Int64()}, strict=True)
        
        result = ar.validate(frame, schema)
        assert result.passed is False
        extra_col_issues = [i for i in result.issues if i.rule == "unexpected_column"]
        assert len(extra_col_issues) == 1
        assert extra_col_issues[0].column == "b"


class TestRequiredIfAndWhitespace:
    """Test conditional constraints and null semantics."""
    
    def test_required_if_condition(self):
        """required_if detects missing values when condition is true."""
        frame = ar.from_pandas(pd.DataFrame({
            "user_type": ["standard", "international", "standard"],
            "country": ["USA", None, "Canada"]
        }))
        schema = ar.Schema({
            "user_type": ar.String(),
            "country": ar.String(required_if=("user_type", "international"))
        })
        
        result = ar.validate(frame, schema)
        assert result.passed is False
        assert result.issues[0].rule == "required_if"
        assert result.issues[0].row_index == 2
    
    def test_whitespace_string_counts_as_null(self):
        """Whitespace-only strings are treated as null values."""
        frame = ar.from_pandas(pd.DataFrame({"name": ["Alice", "   ", "Bob"]}))
        schema = ar.Schema({"name": ar.String(nullable=False)})
        
        result = ar.validate(frame, schema)
        assert result.passed is False
        assert result.issues[0].rule == "nullable"
        assert result.issues[0].row_index == 2


class TestCustomValidatorsAndRules:
    """Test custom validators and schema-level rules."""
    
    def test_custom_validator_success_and_failure(self):
        """Custom validators execute and produce expected issues on failure."""
        ar.register_validator("positive", lambda v: v > 0)
        
        frame = ar.from_pandas(pd.DataFrame({"revenue": [100, -50, 75]}))
        schema = ar.Schema({"revenue": ar.Custom("positive")})
        
        result = ar.validate(frame, schema)
        assert result.passed is False
        assert result.issues[0].row_index == 2
        
        ar.reset_steps()
    
    def test_schema_rule_validation(self):
        """Schema rules are executed and can produce custom issues."""
        def cross_field_rule(df):
            issues = []
            bad = df[df["a"] > df["b"]].index
            for idx in bad:
                issues.append(ar.ValidationIssue(
                    column="a",
                    rule="cross_field",
                    message="a must be <= b",
                    row_index=int(idx) + 1
                ))
            return issues
        
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 5, 2], "b": [3, 2, 4]}))
        schema = ar.Schema(
            {"a": ar.Int64(), "b": ar.Int64()},
            rules=[cross_field_rule]
        )
        
        result = ar.validate(frame, schema)
        assert result.passed is False
        assert result.issues[0].row_index == 2

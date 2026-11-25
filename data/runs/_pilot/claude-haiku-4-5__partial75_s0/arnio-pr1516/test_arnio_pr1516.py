import pytest
import pandas as pd
import json
import arnio as ar


# ============ Fixtures ============

@pytest.fixture
def simple_frame():
    """Simple valid test frame."""
    return ar.from_pandas(pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35]
    }))


@pytest.fixture
def frame_with_nulls():
    """Frame containing null values."""
    return ar.from_pandas(pd.DataFrame({
        "id": [1, 2, None],
        "email": ["alice@example.com", None, "charlie@example.com"]
    }))


# ============ Schema.to_json() Tests ============

class TestSchemaToJson:
    """Tests for Schema.to_json() method."""
    
    def test_to_json_returns_string(self):
        """to_json() should return a string."""
        schema = ar.Schema({"id": ar.Int64()})
        result = schema.to_json()
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_to_json_is_valid_json(self):
        """to_json() output should be valid JSON."""
        schema = ar.Schema({"id": ar.Int64(), "name": ar.String()})
        json_str = schema.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
    
    def test_to_json_roundtrip_simple_schema(self):
        """to_json() and from_json() should preserve a simple schema."""
        original = ar.Schema({"id": ar.Int64()})
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        assert restored is not None
    
    def test_to_json_roundtrip_with_field_constraints(self):
        """to_json() should preserve field-level constraints."""
        original = ar.Schema({
            "id": ar.Int64(nullable=False, unique=True),
            "age": ar.Int64(min=0, max=150),
            "email": ar.Email(nullable=False)
        })
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        assert restored is not None
    
    def test_to_json_roundtrip_with_datetime_and_schema_unique(self):
        """to_json() should preserve DateTime formats and schema-level unique."""
        original = ar.Schema(
            {
                "user_id": ar.Int64(),
                "created_at": ar.DateTime(format="%Y-%m-%dT%H:%M:%S"),
                "course_id": ar.Int64()
            },
            unique=["user_id", "course_id"]
        )
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        assert restored is not None


# ============ Schema.validate() Tests ============

class TestSchemaValidate:
    """Tests for Schema.validate() method."""
    
    def test_validate_returns_validation_result(self, simple_frame):
        """validate() should return a ValidationResult object."""
        schema = ar.Schema({"id": ar.Int64()})
        result = schema.validate(simple_frame)
        
        assert result is not None
        assert hasattr(result, 'passed')
        assert hasattr(result, 'issues')
    
    def test_validate_passing_data(self, simple_frame):
        """validate() should mark result.passed=True for valid data."""
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
            "age": ar.Int64()
        })
        result = schema.validate(simple_frame)
        assert result.passed is True
    
    def test_validate_dtype_mismatch(self):
        """validate() should detect dtype mismatches."""
        frame = ar.from_pandas(pd.DataFrame({"age": ["not", "numbers", "here"]}))
        schema = ar.Schema({"age": ar.Int64()})
        result = schema.validate(frame)
        
        assert result.passed is False
        assert result.issue_count > 0
    
    def test_validate_nullable_constraint(self, frame_with_nulls):
        """validate() should detect nullable=False violations."""
        schema = ar.Schema({"email": ar.String(nullable=False)})
        result = schema.validate(frame_with_nulls)
        
        assert result.passed is False
    
    def test_validate_row_index_one_based(self):
        """validate() row_index should be 1-based (header excluded)."""
        frame = ar.from_pandas(pd.DataFrame({"age": [25, -5, 35]}))
        schema = ar.Schema({"age": ar.Int64(min=0)})
        result = schema.validate(frame)
        
        assert result.passed is False
        assert any(issue.row_index == 2 for issue in result.issues)
    
    def test_validate_numeric_constraints(self):
        """validate() should enforce min/max constraints."""
        frame = ar.from_pandas(pd.DataFrame({"age": [10, 25, 999]}))
        schema = ar.Schema({"age": ar.Int64(min=18, max=120)})
        result = schema.validate(frame)
        
        assert result.passed is False
        assert result.issue_count >= 2
    
    def test_validate_unique_constraint(self):
        """validate() should detect unique constraint violations."""
        frame = ar.from_pandas(pd.DataFrame({"id": [1, 2, 1]}))
        schema = ar.Schema({"id": ar.Int64(unique=True)})
        result = schema.validate(frame)
        
        assert result.passed is False
    
    def test_validate_with_cross_field_rule(self):
        """validate() should execute cross-field validation rules."""
        def date_rule(df):
            issues = []
            for i, row in df.iterrows():
                if str(row["end"]) < str(row["start"]):
                    issues.append(
                        ar.ValidationIssue(
                            column="end",
                            rule="cross_field",
                            message="end must be >= start",
                            row_index=int(i) + 1
                        )
                    )
            return issues
        
        frame = ar.from_pandas(pd.DataFrame({
            "start": ["2024-01-01", "2024-02-01"],
            "end": ["2024-01-31", "2024-01-15"]
        }))
        
        schema = ar.Schema(
            {"start": ar.String(), "end": ar.String()},
            rules=[date_rule]
        )
        
        result = schema.validate(frame)
        assert result.passed is False
        assert any(issue.rule == "cross_field" for issue in result.issues)
    
    def test_validate_multiple_field_types(self):
        """validate() should check multiple field types correctly."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "email": ["test@example.com", "invalid", "another@example.com"],
            "active": [True, False, True]
        }))
        schema = ar.Schema({
            "id": ar.Int64(unique=True),
            "email": ar.Email(),
            "active": ar.Bool()
        })
        result = schema.validate(frame)
        
        assert result.passed is False
        assert any(issue.column == "email" for issue in result.issues)
    
    def test_validate_max_errors_parameter(self):
        """validate() should respect max_errors limit."""
        frame = ar.from_pandas(pd.DataFrame({
            "val": ["a", "b", "c", "d", "e"]
        }))
        schema = ar.Schema({"val": ar.Int64()})
        
        result_limited = schema.validate(frame, max_errors=2)
        
        assert result_limited.issue_count <= 2
    
    def test_validate_issue_contains_required_fields(self):
        """validate() issues should contain column, rule, message, and row_index."""
        frame = ar.from_pandas(pd.DataFrame({"value": [999]}))
        schema = ar.Schema({"value": ar.Int64(max=100)})
        result = schema.validate(frame)
        
        assert len(result.issues) > 0
        issue = result.issues[0]
        assert hasattr(issue, 'column')
        assert hasattr(issue, 'rule')
        assert hasattr(issue, 'message')
        assert hasattr(issue, 'row_index')
    
    def test_validate_multiple_violations_same_row(self):
        """validate() should report multiple violations in the same row."""
        frame = ar.from_pandas(pd.DataFrame({
            "age": [-1],
            "email": ["not-an-email"]
        }))
        schema = ar.Schema({
            "age": ar.Int64(min=0, max=150),
            "email": ar.Email()
        })
        result = schema.validate(frame)
        
        assert result.passed is False
        assert result.issue_count >= 2

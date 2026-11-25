import pytest
import pandas as pd
import arnio as ar


class TestValidate:
    """Test arnio.schema.validate function"""
    
    def test_validate_passing_schema(self):
        """Test validation passes for valid data"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False)
        })
        result = ar.validate(frame, schema)
        assert result.passed
        assert result.issue_count == 0
    
    def test_validate_failing_dtype(self):
        """Test validation fails for incompatible dtype"""
        df = pd.DataFrame({
            "age": ["not_a_number", "thirty", "forty"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "age": ar.Int64(nullable=False)
        })
        result = ar.validate(frame, schema)
        assert not result.passed
        assert result.issue_count > 0
        assert len(result.issues) > 0
    
    def test_validate_nullable_false_fails_with_nulls(self):
        """Test nullable=False rejects null values"""
        df = pd.DataFrame({
            "id": [1, 2, None]
        })
        frame = ar.from_pandas(df)
        
        schema = ar.Schema({
            "id": ar.Int64(nullable=False)
        })
        result = ar.validate(frame, schema)
        assert not result.passed
        assert result.issue_count > 0
    
    def test_validate_nullable_true_accepts_nulls(self):
        """Test nullable=True accepts null values"""
        df = pd.DataFrame({
            "id": [1.0, 2.0, None]
        })
        frame = ar.from_pandas(df)
        
        schema = ar.Schema({
            "id": ar.Float64(nullable=True)
        })
        result = ar.validate(frame, schema)
        assert result.passed
    
    def test_validate_unique_constraint(self):
        """Test unique constraint detects duplicates"""
        df = pd.DataFrame({
            "user_id": [1, 2, 1]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "user_id": ar.Int64(unique=True)
        })
        result = ar.validate(frame, schema)
        assert not result.passed
    
    def test_validate_int_min_constraint(self):
        """Test Int64 min constraint"""
        df = pd.DataFrame({
            "age": [5, 25, 30]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "age": ar.Int64(min=18)
        })
        result = ar.validate(frame, schema)
        assert not result.passed
    
    def test_validate_int_max_constraint(self):
        """Test Int64 max constraint"""
        df = pd.DataFrame({
            "age": [20, 25, 150]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "age": ar.Int64(max=100)
        })
        result = ar.validate(frame, schema)
        assert not result.passed
    
    def test_validate_float_range_constraints(self):
        """Test Float64 min and max constraints"""
        df = pd.DataFrame({
            "score": [0.5, 75.0, 150.5]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "score": ar.Float64(min=1.0, max=100.0)
        })
        result = ar.validate(frame, schema)
        assert not result.passed
    
    def test_validate_string_min_length(self):
        """Test String min_length constraint"""
        df = pd.DataFrame({
            "username": ["ab", "valid_user"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "username": ar.String(min_length=3)
        })
        result = ar.validate(frame, schema)
        assert not result.passed
    
    def test_validate_string_max_length(self):
        """Test String max_length constraint"""
        df = pd.DataFrame({
            "username": ["valid", "x" * 20]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "username": ar.String(max_length=15)
        })
        result = ar.validate(frame, schema)
        assert not result.passed
    
    def test_validate_email_field(self):
        """Test Email field type validation"""
        df = pd.DataFrame({
            "email": ["invalid_email", "test@example.com"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "email": ar.Email(nullable=False)
        })
        result = ar.validate(frame, schema)
        assert not result.passed
    
    def test_validate_url_field(self):
        """Test URL field type validation"""
        df = pd.DataFrame({
            "website": ["not_a_url", "https://example.com"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "website": ar.URL()
        })
        result = ar.validate(frame, schema)
        assert not result.passed
    
    def test_validate_regex_field(self):
        """Test Regex field type validation"""
        df = pd.DataFrame({
            "code": ["USR-0001", "BAD-CODE"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "code": ar.Regex(r"^USR-\d{4}$", nullable=False)
        })
        result = ar.validate(frame, schema)
        assert not result.passed
    
    def test_validate_composite_unique_key(self):
        """Test unique constraint across multiple columns"""
        df = pd.DataFrame({
            "user_id": [1, 1, 2],
            "course_id": [100, 100, 200]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "user_id": ar.Int64(nullable=False),
            "course_id": ar.Int64(nullable=False)
        }, unique=["user_id", "course_id"])
        result = ar.validate(frame, schema)
        assert not result.passed
    
    def test_validate_schema_from_dict(self):
        """Test validate accepts schema as dict"""
        df = pd.DataFrame({
            "id": [1, 2, 3]
        })
        frame = ar.from_pandas(df)
        schema_dict = {
            "id": ar.Int64(nullable=False)
        }
        result = ar.validate(frame, schema_dict)
        assert result.passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

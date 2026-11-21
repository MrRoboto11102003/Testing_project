import pytest
import csv
import sys
import subprocess
import tempfile
from pathlib import Path
import pandas as pd
import arnio as ar
from arnio.schema import Schema, ValidationIssue


class TestValidateBasicWorkflow:
    """Test basic validation workflows with real files"""
    
    def test_validate_valid_data(self, tmp_path):
        """Complete workflow: load CSV, validate against schema, check result"""
        csv_file = tmp_path / "valid_data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["email", "age"])
            writer.writerow(["alice@example.com", "25"])
            writer.writerow(["bob@example.com", "30"])
        
        frame = ar.read_csv(csv_file)
        schema = Schema({
            "email": ar.Email(),
            "age": ar.Integer()
        })
        
        result = ar.validate(frame, schema)
        assert result.passed
        assert result.issue_count == 0
        assert result.row_count == 2
        assert result.bad_rows == []
    
    def test_validate_invalid_email_field(self, tmp_path):
        """Validation catches invalid email format"""
        csv_file = tmp_path / "invalid_emails.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["email"])
            writer.writerow(["valid@example.com"])
            writer.writerow(["not-an-email"])
        
        frame = ar.read_csv(csv_file)
        schema = Schema({"email": ar.Email()})
        
        result = ar.validate(frame, schema)
        assert not result.passed
        assert result.issue_count > 0
        assert any(issue.column == "email" for issue in result.issues)
    
    def test_validate_missing_required_column(self, tmp_path):
        """Validation fails when required column is missing from data"""
        csv_file = tmp_path / "incomplete_cols.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name"])
            writer.writerow(["Alice"])
        
        frame = ar.read_csv(csv_file)
        schema = Schema({
            "name": ar.String(),
            "email": ar.Email()
        })
        
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.rule == "required_column" and issue.column == "email" 
                   for issue in result.issues)
    
    def test_validate_dict_schema_input(self, tmp_path):
        """Validate accepts dict[str, Field] in addition to Schema object"""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["email"])
            writer.writerow(["test@example.com"])
        
        frame = ar.read_csv(csv_file)
        schema_dict = {"email": ar.Email()}
        
        result = ar.validate(frame, schema_dict)
        assert result.passed


class TestValidateNullability:
    """Test nullable field handling"""
    
    def test_nullable_field_accepts_empty_string(self, tmp_path):
        """Nullable=True allows empty/null values in column"""
        csv_file = tmp_path / "nulls.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["email", "phone"])
            writer.writerow(["alice@example.com", "555-1234"])
            writer.writerow(["bob@example.com", ""])
        
        frame = ar.read_csv(csv_file)
        schema = Schema({
            "email": ar.Email(nullable=False),
            "phone": ar.String(nullable=True)
        })
        
        result = ar.validate(frame, schema)
        assert result.passed
    
    def test_non_nullable_field_rejects_empty_values(self, tmp_path):
        """Nullable=False rejects empty/null values"""
        csv_file = tmp_path / "null_violation.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["email"])
            writer.writerow(["valid@example.com"])
            writer.writerow([""])
        
        frame = ar.read_csv(csv_file)
        schema = Schema({"email": ar.Email(nullable=False)})
        
        result = ar.validate(frame, schema)
        assert not result.passed


class TestValidateStrictMode:
    """Test strict schema mode"""
    
    def test_strict_schema_rejects_unexpected_columns(self, tmp_path):
        """strict=True makes schema reject unexpected columns"""
        csv_file = tmp_path / "extra_cols.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["email", "extra_col"])
            writer.writerow(["test@example.com", "unexpected"])
        
        frame = ar.read_csv(csv_file)
        schema = Schema(
            {"email": ar.Email()},
            strict=True
        )
        
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.rule == "unexpected_column" for issue in result.issues)
    
    def test_non_strict_schema_allows_extra_columns(self, tmp_path):
        """strict=False (default) allows unexpected columns"""
        csv_file = tmp_path / "extra_ok.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["email", "extra"])
            writer.writerow(["test@example.com", "data"])
        
        frame = ar.read_csv(csv_file)
        schema = Schema(
            {"email": ar.Email()},
            strict=False
        )
        
        result = ar.validate(frame, schema)
        assert result.passed


class TestValidateCompositeUnique:
    """Test composite unique constraints"""
    
    def test_unique_constraint_valid_distinct_rows(self, tmp_path):
        """unique=[cols] passes when all row combinations are unique"""
        csv_file = tmp_path / "unique_valid.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "email"])
            writer.writerow(["1", "alice@example.com"])
            writer.writerow(["2", "bob@example.com"])
        
        frame = ar.read_csv(csv_file)
        schema = Schema(
            {
                "user_id": ar.Integer(),
                "email": ar.Email()
            },
            unique=["user_id", "email"]
        )
        
        result = ar.validate(frame, schema)
        assert result.passed
    
    def test_unique_constraint_detects_duplicates(self, tmp_path):
        """unique=[cols] fails when duplicate row combinations exist"""
        csv_file = tmp_path / "unique_duplicate.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "email"])
            writer.writerow(["1", "alice@example.com"])
            writer.writerow(["1", "alice@example.com"])
        
        frame = ar.read_csv(csv_file)
        schema = Schema(
            {
                "user_id": ar.Integer(),
                "email": ar.Email()
            },
            unique=["user_id", "email"]
        )
        
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.rule == "composite_unique" for issue in result.issues)
        assert len(result.bad_rows) > 0
    
    def test_unique_missing_column_in_constraint(self, tmp_path):
        """unique=[cols] reports error if column doesn't exist in data"""
        csv_file = tmp_path / "unique_missing_col.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id"])
            writer.writerow(["1"])
        
        frame = ar.read_csv(csv_file)
        schema = Schema(
            {"id": ar.Integer()},
            unique=["id", "missing_col"]
        )
        
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any("missing_col" in str(issue.message) for issue in result.issues)


class TestValidateMaxErrors:
    """Test max_errors parameter limits issue collection"""
    
    def test_max_errors_stops_collection_at_limit(self, tmp_path):
        """max_errors=N stops validation after N issues found"""
        csv_file = tmp_path / "many_errors.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["email"])
            for i in range(10):
                writer.writerow(["invalid" + str(i)])
        
        frame = ar.read_csv(csv_file)
        schema = Schema({"email": ar.Email()})
        
        result_no_limit = ar.validate(frame, schema)
        result_limited = ar.validate(frame, schema, max_errors=3)
        
        assert len(result_limited.issues) <= 3
        assert len(result_limited.issues) <= len(result_no_limit.issues)
    
    def test_max_errors_invalid_type_raises_typeerror(self, tmp_path):
        """max_errors with non-int type raises TypeError"""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["email"])
            writer.writerow(["test@example.com"])
        
        frame = ar.read_csv(csv_file)
        schema = Schema({"email": ar.Email()})
        
        with pytest.raises(TypeError, match="max_errors must be an int or None"):
            ar.validate(frame, schema, max_errors="invalid")
    
    def test_max_errors_zero_raises_valueerror(self, tmp_path):
        """max_errors=0 raises ValueError (must be >= 1)"""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["email"])
            writer.writerow(["test@example.com"])
        
        frame = ar.read_csv(csv_file)
        schema = Schema({"email": ar.Email()})
        
        with pytest.raises(ValueError, match="max_errors must be >= 1"):
            ar.validate(frame, schema, max_errors=0)


class TestValidateCustomRules:
    """Test custom validation rule functions"""
    
    def test_custom_rule_function_called_and_issues_collected(self, tmp_path):
        """Custom rule function is called and its ValidationIssues collected"""
        csv_file = tmp_path / "age_data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["age"])
            writer.writerow(["25"])
            writer.writerow(["18"])
        
        def age_minimum(df):
            issues = []
            for idx, row in df.iterrows():
                if row["age"] < 21:
                    issues.append(
                        ValidationIssue(
                            column="age",
                            rule="age_minimum",
                            message="Age must be at least 21",
                            row_index=idx + 1
                        )
                    )
            return issues
        
        frame = ar.read_csv(csv_file)
        schema = Schema(
            {"age": ar.Integer()},
            rules=[age_minimum]
        )
        
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.rule == "age_minimum" for issue in result.issues)
    
    def test_custom_rule_missing_column_caught(self, tmp_path):
        """KeyError in custom rule (missing column) reported as validation issue"""
        csv_file = tmp_path / "simple.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name"])
            writer.writerow(["Alice"])
        
        def bad_rule(df):
            # This will raise KeyError when column doesn't exist
            return [ValidationIssue(
                column="missing",
                message=str(df["missing_column"].sum())
            )]
        
        frame = ar.read_csv(csv_file)
        schema = Schema({"name": ar.String()}, rules=[bad_rule])
        
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.rule == "missing_column" for issue in result.issues)
    
    def test_custom_rule_invalid_return_type_raises_typeerror(self, tmp_path):
        """Custom rule returning non-list raises TypeError"""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id"])
            writer.writerow(["1"])
        
        def bad_return_rule(df):
            return "not a list"
        
        frame = ar.read_csv(csv_file)
        schema = Schema({"id": ar.Integer()}, rules=[bad_return_rule])
        
        with pytest.raises(TypeError):
            ar.validate(frame, schema)


class TestValidateErrorHandling:
    """Test error types and input validation"""
    
    def test_non_arframe_input_raises_typeerror(self):
        """validate() with non-ArFrame raises TypeError"""
        df = pd.DataFrame({"email": ["test@example.com"]})
        schema = Schema({"email": ar.Email()})
        
        with pytest.raises(TypeError, match="single ArFrame"):
            ar.validate(df, schema)
    
    def test_unique_not_list_or_tuple_raises_typeerror(self, tmp_path):
        """unique parameter as non-list/tuple raises TypeError"""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id"])
            writer.writerow(["1"])
        
        frame = ar.read_csv(csv_file)
        schema = Schema(
            {"id": ar.Integer()},
            unique="id"  # Should be list/tuple
        )
        
        with pytest.raises(TypeError, match="unique.*list or tuple"):
            ar.validate(frame, schema)
    
    def test_unique_members_non_string_raises_typeerror(self, tmp_path):
        """unique=[...] with non-string members raises TypeError"""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id"])
            writer.writerow(["1"])
        
        frame = ar.read_csv(csv_file)
        schema = Schema(
            {"id": ar.Integer()},
            unique=["id", 123]
        )
        
        with pytest.raises(TypeError, match="unique.*members must be strings"):
            ar.validate(frame, schema)
    
    def test_empty_unique_list_reported_as_issue(self, tmp_path):
        """unique=[] is reported as validation issue"""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id"])
            writer.writerow(["1"])
        
        frame = ar.read_csv(csv_file)
        schema = Schema(
            {"id": ar.Integer()},
            unique=[]
        )
        
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any("empty" in issue.message.lower() for issue in result.issues)


class TestValidateBadRowTracking:
    """Test bad row index tracking in results"""
    
    def test_bad_rows_indexed_correctly(self, tmp_path):
        """bad_rows contains correct row indices for failing validations"""
        csv_file = tmp_path / "mixed_valid_invalid.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["email"])
            writer.writerow(["valid@example.com"])      # row 2
            writer.writerow(["invalid"])                 # row 3 - bad
            writer.writerow(["another@example.com"])     # row 4
            writer.writerow(["also-invalid"])            # row 5 - bad
        
        frame = ar.read_csv(csv_file)
        schema = Schema({"email": ar.Email()})
        
        result = ar.validate(frame, schema)
        assert len(result.bad_rows) > 0
        assert all(isinstance(idx, int) for idx in result.bad_rows)
        assert result.bad_rows == sorted(result.bad_rows)
    
    def test_validation_result_properties(self, tmp_path):
        """ValidationResult has expected properties and types"""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["email"])
            writer.writerow(["test@example.com"])
            writer.writerow(["invalid"])
        
        frame = ar.read_csv(csv_file)
        schema = Schema({"email": ar.Email()})
        
        result = ar.validate(frame, schema)
        assert hasattr(result, "passed") and isinstance(result.passed, bool)
        assert hasattr(result, "issues") and isinstance(result.issues, list)
        assert hasattr(result, "issue_count") and isinstance(result.issue_count, int)
        assert hasattr(result, "row_count") and isinstance(result.row_count, int)
        assert hasattr(result, "bad_rows") and isinstance(result.bad_rows, list)
        assert result.row_count == 2
        assert result.issue_count == len(result.issues)


class TestValidateEdgeCases:
    """Test edge cases and corner scenarios"""
    
    def test_empty_dataframe_validates(self, tmp_path):
        """Validate empty dataframe (no rows) passes schema check"""
        csv_file = tmp_path / "empty.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["email"])
        
        frame = ar.read_csv(csv_file)
        schema = Schema({"email": ar.Email()})
        
        result = ar.validate(frame, schema)
        assert result.row_count == 0
        assert result.passed
        assert result.bad_rows == []
    
    def test_validate_large_number_of_columns(self, tmp_path):
        """Validate works with many columns"""
        csv_file = tmp_path / "many_cols.csv"
        columns = [f"col_{i}" for i in range(50)]
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerow([str(i) for i in range(50)])
        
        frame = ar.read_csv(csv_file)
        schema = Schema({col: ar.String() for col in columns})
        
        result = ar.validate(frame, schema)
        assert result.passed
        assert result.row_count == 1
    
    def test_validate_multiple_field_types(self, tmp_path):
        """Validate with multiple different field types"""
        csv_file = tmp_path / "mixed_types.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "email", "name", "age"])
            writer.writerow(["1", "user@example.com", "Alice", "25"])
            writer.writerow(["2", "bob@example.com", "Bob", "30"])
        
        frame = ar.read_csv(csv_file)
        schema = Schema({
            "id": ar.Integer(),
            "email": ar.Email(),
            "name": ar.String(),
            "age": ar.Integer()
        })
        
        result = ar.validate(frame, schema)
        assert result.passed
        assert result.row_count == 2

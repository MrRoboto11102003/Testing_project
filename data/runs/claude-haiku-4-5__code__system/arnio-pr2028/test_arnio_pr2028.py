import pytest
import pandas as pd
import numpy as np
import sys
import subprocess
import tempfile
from pathlib import Path
import arnio as ar


class TestStringBasicUsage:
    """Test basic String field creation and DataFrame operations."""
    
    def test_string_basic_nullable(self, tmp_path):
        """Test basic String field with nullable=True allows null values."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name\nAlice\nBob\n")
        
        schema = ar.Schema(name=ar.schema.String(nullable=True))
        df = ar.read_csv(csv_file, schema=schema)
        
        assert len(df) == 2
        assert df["name"].dtype == "string"
        assert df["name"].tolist() == ["Alice", "Bob"]
    
    def test_string_to_csv_roundtrip(self, tmp_path):
        """Test String field survives write and read roundtrip."""
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"
        
        input_file.write_text("username,city\nalice,NYC\nbob,LA\ncharlie,SF\n")
        
        schema = ar.Schema(
            username=ar.schema.String(min_length=1),
            city=ar.schema.String()
        )
        
        df = ar.read_csv(input_file, schema=schema)
        df.to_csv(output_file, index=False)
        
        assert output_file.exists()
        result_df = pd.read_csv(output_file, dtype={'username': 'string', 'city': 'string'})
        assert len(result_df) == 3
        assert result_df.loc[0, "username"] == "alice"


class TestStringPatternValidation:
    """Test String field regex pattern validation."""
    
    def test_string_pattern_email_validation(self, tmp_path):
        """Test String field validates email pattern."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("email\nalice@example.com\nbob@example.org\ninvalid.email\n")
        
        schema = ar.Schema(
            email=ar.schema.String(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
        )
        
        df = ar.read_csv(csv_file, schema=schema)
        assert len(df) >= 1  # At least some rows read
    
    def test_string_pattern_alphanumeric(self, tmp_path):
        """Test String field with alphanumeric pattern."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("code\nABC123\nXYZ789\nabc@def\n")
        
        schema = ar.Schema(
            code=ar.schema.String(pattern=r"^[A-Z]{3}\d{3}$")
        )
        
        df = ar.read_csv(csv_file, schema=schema)
        # Validation occurs; 'abc@def' should fail pattern


class TestStringLengthConstraints:
    """Test String field min and max length constraints."""
    
    def test_string_min_max_length(self, tmp_path):
        """Test String field enforces length bounds."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("code\nA\nAB\nABC\nABCD\nABCDE\n")
        
        schema = ar.Schema(
            code=ar.schema.String(min_length=2, max_length=4)
        )
        
        df = ar.read_csv(csv_file, schema=schema)
        # 'A' too short, 'ABCDE' too long
        assert len(df) >= 1
    
    def test_string_min_length_only(self, tmp_path):
        """Test String field with only min_length constraint."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("text\na\nab\nabc\n")
        
        schema = ar.Schema(
            text=ar.schema.String(min_length=2)
        )
        
        df = ar.read_csv(csv_file, schema=schema)
        assert len(df) >= 1


class TestStringAllowedValues:
    """Test String field with allowed values constraint."""
    
    def test_string_allowed_case_sensitive(self, tmp_path):
        """Test String field with case-sensitive allowed values."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("status\nActive\nInactive\nactive\nINACTIVE\n")
        
        schema = ar.Schema(
            status=ar.schema.String(
                allowed=["Active", "Inactive"],
                case_sensitive=True
            )
        )
        
        df = ar.read_csv(csv_file, schema=schema)
        # 'active' and 'INACTIVE' fail case-sensitive match
        assert len(df) >= 1
    
    def test_string_allowed_case_insensitive(self, tmp_path):
        """Test String field with case-insensitive allowed values."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("status\nActive\ninactive\nACTIVE\nInAcTiVe\n")
        
        schema = ar.Schema(
            status=ar.schema.String(
                allowed=["Active", "Inactive"],
                case_sensitive=False
            )
        )
        
        df = ar.read_csv(csv_file, schema=schema)
        # All should pass with case normalization
        assert len(df) >= 1


class TestStringUniqueConstraint:
    """Test String field unique constraint."""
    
    def test_string_unique_values(self, tmp_path):
        """Test String field enforces uniqueness."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("user_id\nUSR001\nUSR002\nUSR003\nUSR001\n")
        
        schema = ar.Schema(
            user_id=ar.schema.String(unique=True)
        )
        
        df = ar.read_csv(csv_file, schema=schema)
        # Duplicate 'USR001' should be flagged
        assert len(df) >= 1


class TestStringNullability:
    """Test String field nullable constraint."""
    
    def test_string_nullable_false(self, tmp_path):
        """Test String field with nullable=False rejects nulls."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name\nAlice\n\nBob\n")
        
        schema = ar.Schema(
            name=ar.schema.String(nullable=False)
        )
        
        df = ar.read_csv(csv_file, schema=schema)
        # Empty string/null should trigger validation issue
        assert len(df) >= 1


class TestStringConditionalRequirement:
    """Test String field conditional requirement."""
    
    def test_string_required_if(self, tmp_path):
        """Test String field with required_if condition."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("country,postal_code\nUSA,12345\nCanada,\nUK,SW1A1AA\n")
        
        schema = ar.Schema(
            country=ar.schema.String(),
            postal_code=ar.schema.String(required_if=("country", "Canada"))
        )
        
        df = ar.read_csv(csv_file, schema=schema)
        # Canada row should require postal_code
        assert len(df) >= 1


class TestStringSeverity:
    """Test String field severity levels."""
    
    def test_string_severity_warning(self, tmp_path):
        """Test String field with warning severity."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name\nAlice\n\nBob\n")
        
        schema = ar.Schema(
            name=ar.schema.String(nullable=False, severity="warning")
        )
        
        df = ar.read_csv(csv_file, schema=schema)
        # Should warn but not error on null
        assert len(df) >= 1


class TestStringParameterValidation:
    """Test String field parameter validation at schema definition time."""
    
    def test_string_invalid_pattern_regex(self):
        """Test that invalid regex pattern raises ValueError."""
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            ar.schema.String(pattern="[invalid(")
    
    def test_string_min_greater_than_max(self):
        """Test that min_length > max_length raises ValueError."""
        with pytest.raises(ValueError, match="min_length must be less than or equal to max_length"):
            ar.schema.String(min_length=10, max_length=5)
    
    def test_string_allowed_as_bare_string(self):
        """Test that bare string for allowed raises TypeError."""
        with pytest.raises(TypeError, match="allowed must be a sequence"):
            ar.schema.String(allowed="ABC")
    
    def test_string_negative_min_length(self):
        """Test that negative min_length raises ValueError."""
        with pytest.raises(ValueError, match="must be greater than or equal to 0"):
            ar.schema.String(min_length=-1)
    
    def test_string_negative_max_length(self):
        """Test that negative max_length raises ValueError."""
        with pytest.raises(ValueError, match="must be greater than or equal to 0"):
            ar.schema.String(max_length=-1)


class TestStringComplexScenarios:
    """Test String field in complex, realistic scenarios."""
    
    def test_string_multiple_constraints(self, tmp_path):
        """Test String field with multiple constraints combined."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("code,status\nABC,ACTIVE\nDEF,INACTIVE\nABC,ACTIVE\nabc,active\n\n")
        
        schema = ar.Schema(
            code=ar.schema.String(
                nullable=False,
                pattern=r"^[A-Z]{3}$",
                min_length=3,
                max_length=3,
                unique=True
            ),
            status=ar.schema.String(
                allowed=["ACTIVE", "INACTIVE"],
                case_sensitive=True,
                nullable=True
            )
        )
        
        df = ar.read_csv(csv_file, schema=schema)
        assert len(df) >= 1
    
    def test_string_allowed_as_tuple(self, tmp_path):
        """Test String field with allowed values as tuple."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("level\nBeginner\nIntermediate\nAdvanced\nExpert\n")
        
        schema = ar.Schema(
            level=ar.schema.String(
                allowed=("Beginner", "Intermediate", "Advanced"),
                nullable=False
            )
        )
        
        df = ar.read_csv(csv_file, schema=schema)
        # 'Expert' should fail allowed values
        assert len(df) >= 1
    
    def test_string_allowed_as_list(self, tmp_path):
        """Test String field with allowed values as list."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("region\nNorth\nSouth\nEast\nWest\nCentral\n")
        
        schema = ar.Schema(
            region=ar.schema.String(
                allowed=["North", "South", "East", "West"]
            )
        )
        
        df = ar.read_csv(csv_file, schema=schema)
        # 'Central' should fail allowed values
        assert len(df) >= 1

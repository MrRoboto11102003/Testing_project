import pytest
import pandas as pd
import numpy as np
import arnio as ar
from pathlib import Path


class TestCleanColumnNames:
    """End-to-end tests for arnio.cleaning.clean_column_names"""

    def test_basic_cleanup_default_lower(self, tmp_path):
        """Test basic column name cleanup with default lower case."""
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text("First Name,Last-Name,Email@Address\nalice,smith,alice@example.com\n")

        frame = ar.read_csv(str(csv_file))
        cleaned = ar.clean_column_names(frame)

        assert cleaned.columns == ["first_name", "last_name", "email_address"]
        df = ar.to_pandas(cleaned)
        assert len(df) == 1
        assert df.iloc[0]["first_name"] == "alice"

    def test_case_type_upper(self, tmp_path):
        """Test clean_column_names with case_type='upper'."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("first name,last name\nalice,smith\n")

        frame = ar.read_csv(str(csv_file))
        cleaned = ar.clean_column_names(frame, case_type="upper")

        assert cleaned.columns == ["FIRST_NAME", "LAST_NAME"]

    def test_case_type_title(self, tmp_path):
        """Test clean_column_names with case_type='title'."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("first_name,last_name\nalice,smith\n")

        frame = ar.read_csv(str(csv_file))
        cleaned = ar.clean_column_names(frame, case_type="title")

        assert cleaned.columns == ["First_Name", "Last_Name"]

    def test_case_type_camel(self, tmp_path):
        """Test clean_column_names with case_type='camel'."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("first_name,last_name\nalice,smith\n")

        frame = ar.read_csv(str(csv_file))
        cleaned = ar.clean_column_names(frame, case_type="camel")

        assert cleaned.columns == ["firstName", "lastName"]

    def test_case_type_none(self, tmp_path):
        """Test clean_column_names with case_type='none' preserves original case."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("First_Name,Last_Name\nalice,smith\n")

        frame = ar.read_csv(str(csv_file))
        cleaned = ar.clean_column_names(frame, case_type="none")

        assert cleaned.columns == ["First_Name", "Last_Name"]

    def test_special_characters_and_spaces_to_underscores(self, tmp_path):
        """Test that special characters and spaces become underscores."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col@name!,val-ue$,spa ce here\na,b,c\n")

        frame = ar.read_csv(str(csv_file))
        cleaned = ar.clean_column_names(frame)

        assert cleaned.columns == ["col_name", "val_ue", "spa_ce_here"]

    def test_consecutive_underscores_collapsed(self, tmp_path):
        """Test that consecutive underscores are collapsed into single underscore."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col__with__underscores,another___col\na,b\n")

        frame = ar.read_csv(str(csv_file))
        cleaned = ar.clean_column_names(frame)

        assert cleaned.columns == ["col_with_underscores", "another_col"]

    def test_leading_trailing_underscores_trimmed(self, tmp_path):
        """Test that leading and trailing underscores are removed."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("_col_name_,__another__\na,b\n")

        frame = ar.read_csv(str(csv_file))
        cleaned = ar.clean_column_names(frame)

        assert cleaned.columns == ["col_name", "another"]

    def test_only_special_chars_become_default_column(self, tmp_path):
        """Test that columns with only special characters become 'column'."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("@#$,valid_name,!!!!\na,b,c\n")

        frame = ar.read_csv(str(csv_file))
        cleaned = ar.clean_column_names(frame)

        assert cleaned.columns == ["column", "valid_name", "column_1"]

    def test_duplicate_names_raises_error(self, tmp_path):
        """Test that duplicate cleaned names raise ValueError."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col-name,col_name\na,b\n")

        frame = ar.read_csv(str(csv_file))

        with pytest.raises(ValueError, match="duplicate"):
            ar.clean_column_names(frame)

    def test_invalid_case_type_raises_error(self, tmp_path):
        """Test that invalid case_type value raises ValueError."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col_name\na\n")

        frame = ar.read_csv(str(csv_file))

        with pytest.raises(ValueError, match="case_type must be one of"):
            ar.clean_column_names(frame, case_type="invalid")

    def test_non_string_case_type_raises_error(self, tmp_path):
        """Test that non-string case_type raises TypeError."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col_name\na\n")

        frame = ar.read_csv(str(csv_file))

        with pytest.raises(TypeError, match="case_type must be a string"):
            ar.clean_column_names(frame, case_type=123)

    def test_write_cleaned_csv(self, tmp_path):
        """Test full workflow: read messy CSV, clean names, write result."""
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"
        input_file.write_text(
            "First Name,Email@Address,Phone-Number\n"
            "alice,alice@example.com,555-1234\n"
            "bob,bob@example.com,555-5678\n"
        )

        # Read, clean, write
        frame = ar.read_csv(str(input_file))
        cleaned = ar.clean_column_names(frame)
        ar.write_csv(cleaned, str(output_file))

        # Verify output
        assert output_file.exists()
        output_text = output_file.read_text()
        assert "first_name" in output_text
        assert "email_address" in output_text
        assert "phone_number" in output_text

        # Reload and verify
        reloaded = ar.read_csv(str(output_file))
        df = ar.to_pandas(reloaded)
        assert len(df) == 2
        assert list(df.columns) == ["first_name", "email_address", "phone_number"]

    def test_pandas_roundtrip(self, tmp_path):
        """Test clean_column_names with pandas DataFrame roundtrip."""
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text("First Name,Last-Name,Age (years)\nalice,smith,30\nbob,jones,25\n")

        # Read CSV to pandas, convert to ArFrame, clean, convert back
        df = pd.read_csv(str(csv_file))
        frame = ar.from_pandas(df)
        cleaned = ar.clean_column_names(frame)
        result_df = ar.to_pandas(cleaned)

        assert list(result_df.columns) == ["first_name", "last_name", "age_years"]
        assert len(result_df) == 2
        assert result_df.iloc[0]["first_name"] == "alice"
        assert result_df.iloc[1]["age_years"] == 25

    def test_numbers_preserved_in_column_names(self, tmp_path):
        """Test that numbers within column names are preserved."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col-123,val_456_name,id2product\na,b,c\n")

        frame = ar.read_csv(str(csv_file))
        cleaned = ar.clean_column_names(frame)

        assert cleaned.columns == ["col_123", "val_456_name", "id2product"]

    def test_mixed_case_and_special_chars_comprehensive(self, tmp_path):
        """Test comprehensive workflow with mixed case, spaces, and special chars."""
        input_file = tmp_path / "complex.csv"
        output_file = tmp_path / "cleaned.csv"
        input_file.write_text(
            "User  ID,eMail-ADDRESS,Account@Status,ZIP_Code\n"
            "101,test@ex.com,Active,12345\n"
            "102,user@ex.com,Inactive,67890\n"
        )

        # Read, clean with title case, write, reload
        frame = ar.read_csv(str(input_file))
        cleaned = ar.clean_column_names(frame, case_type="title")
        ar.write_csv(cleaned, str(output_file))

        # Verify written output
        reloaded = ar.read_csv(str(output_file))
        df = ar.to_pandas(reloaded)

        assert "User_Id" in df.columns
        assert "Email_Address" in df.columns
        assert len(df) == 2
        assert df.iloc[0]["Email_Address"] == "test@ex.com"

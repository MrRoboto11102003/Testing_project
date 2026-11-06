import pytest
import pandas as pd
import arnio as ar


class TestCleanColumnNames:
    """Tests for arnio.cleaning.clean_column_names function."""

    def test_default_lowercase_normalization(self):
        """Test cleaning with default lowercase case normalization."""
        df = pd.DataFrame({
            "First Name": [1, 2],
            "LAST NAME": [3, 4],
            "Email@Address": [5, 6]
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)
        
        assert "first_name" in result_df.columns
        assert "last_name" in result_df.columns
        assert "email_address" in result_df.columns

    def test_uppercase_normalization(self):
        """Test uppercase case normalization."""
        df = pd.DataFrame({"hello_world": [1]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="upper")
        result_df = ar.to_pandas(result)
        
        assert "HELLO_WORLD" in result_df.columns

    def test_title_case_normalization(self):
        """Test title case normalization."""
        df = pd.DataFrame({"hello_world": [1]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="title")
        result_df = ar.to_pandas(result)
        
        assert "Hello_World" in result_df.columns

    def test_camel_case_normalization(self):
        """Test camel case normalization."""
        df = pd.DataFrame({"hello_world_test": [1]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="camel")
        result_df = ar.to_pandas(result)
        
        assert "helloWorldTest" in result_df.columns

    def test_none_case_no_normalization(self):
        """Test case_type='none' preserves original case."""
        df = pd.DataFrame({"Hello_World": [1]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="none")
        result_df = ar.to_pandas(result)
        
        assert "Hello_World" in result_df.columns

    def test_replace_special_characters(self):
        """Test that non-alphanumeric characters are replaced with underscores."""
        df = pd.DataFrame({
            "user@email.com": [1],
            "price$amount": [2],
            "user#id": [3],
            "name-first": [4],
            "col.with.dots": [5]
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)
        
        # All special characters replaced with underscores
        assert "user_email_com" in result_df.columns
        assert "price_amount" in result_df.columns
        assert "user_id" in result_df.columns
        assert "name_first" in result_df.columns
        assert "col_with_dots" in result_df.columns

    def test_collapse_consecutive_underscores(self):
        """Test that consecutive underscores are collapsed into one."""
        df = pd.DataFrame({
            "hello___world": [1],
            "test____name": [2]
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)
        
        assert "hello_world" in result_df.columns
        assert "test_name" in result_df.columns

    def test_trim_boundary_underscores(self):
        """Test that leading and trailing underscores are trimmed."""
        df = pd.DataFrame({
            "_hello_world_": [1],
            "___test___": [2]
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)
        
        assert "hello_world" in result_df.columns
        assert "test" in result_df.columns

    def test_empty_column_name_becomes_column(self):
        """Test that empty/blank column names become 'column'."""
        df = pd.DataFrame({"___": [1], "hello": [2]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)
        
        assert "column" in result_df.columns
        assert "hello" in result_df.columns

    def test_numeric_characters_preserved(self):
        """Test that numeric characters are preserved in column names."""
        df = pd.DataFrame({
            "col1": [1],
            "col_2_name": [2],
            "123test": [3],
            "test123": [4]
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)
        
        assert "col1" in result_df.columns
        assert "col_2_name" in result_df.columns
        assert "123test" in result_df.columns
        assert "test123" in result_df.columns

    def test_invalid_case_type_raises_value_error(self):
        """Test that invalid case_type raises ValueError."""
        df = pd.DataFrame({"col": [1]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="case_type must be one of"):
            ar.clean_column_names(frame, case_type="invalid")

    def test_non_string_case_type_raises_type_error(self):
        """Test that non-string case_type raises TypeError."""
        df = pd.DataFrame({"col": [1]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="case_type must be a string"):
            ar.clean_column_names(frame, case_type=123)
        
        with pytest.raises(TypeError, match="case_type must be a string"):
            ar.clean_column_names(frame, case_type=None)

    def test_duplicate_column_names_raise_value_error(self):
        """Test that cleaning that creates duplicate names raises ValueError."""
        df = pd.DataFrame({
            "hello-world": [1],
            "hello_world": [2]
        })
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="duplicate"):
            ar.clean_column_names(frame)

    def test_no_changes_returns_copy(self):
        """Test that when no changes are needed, a new copy is returned."""
        df = pd.DataFrame({"hello_world": [1, 2, 3], "test_col": [4, 5, 6]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        
        # Check columns are identical
        assert result.columns == frame.columns
        # Check it's a different frame object
        assert result is not frame
        # Verify data is preserved
        result_df = ar.to_pandas(result)
        assert list(result_df["hello_world"]) == [1, 2, 3]

    def test_complex_real_world_scenario(self):
        """Test complex real-world column name cleaning scenario."""
        df = pd.DataFrame({
            "First & Last Name!!!": [1],
            "ACCOUNT@@@ID": [2],
            "  Email  Address  ": [3],
            "Purchase$$Price": [4],
            "_customer_number_": [5]
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="lower")
        result_df = ar.to_pandas(result)
        
        expected_cols = {
            "first_last_name",
            "account_id",
            "email_address",
            "purchase_price",
            "customer_number"
        }
        assert set(result_df.columns) == expected_cols
        # Verify data is preserved
        assert len(result_df) == 5

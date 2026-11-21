import pytest
import pandas as pd
import arnio as ar


class TestNormalizeWhitespace:
    """End-to-end tests for arnio.cleaning.normalize_whitespace"""

    def test_normalize_whitespace_basic_csv_workflow(self, tmp_path):
        """Test basic workflow: CSV with whitespace issues → normalize → pandas."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "name,city,notes\n"
            "  Alice  ,  New York  , Great  student  \n"
            "Bob,London,Good  work\n"
            "  Charlie  ,  Paris  ,Excellent\n"
        )

        # Load and normalize
        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame)
        df = ar.to_pandas(clean)

        # Verify: leading/trailing stripped, internal spaces collapsed
        assert df.loc[0, "name"] == "Alice"
        assert df.loc[0, "city"] == "New York"
        assert df.loc[0, "notes"] == "Great student"
        assert df.loc[1, "notes"] == "Good work"
        assert df.loc[2, "name"] == "Charlie"

    def test_normalize_whitespace_in_pipeline(self, tmp_path):
        """Test normalize_whitespace as a step in ar.pipeline()."""
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text(
            "id,product,description\n"
            "1,  Widget A  ,  High   quality   item  \n"
            "2,Widget B,  Good  price  \n"
        )

        frame = ar.read_csv(str(csv_file))
        clean = ar.pipeline(
            frame,
            [
                ("normalize_whitespace",),
                ("normalize_case", {"case_type": "lower"}),
            ],
        )
        df = ar.to_pandas(clean)

        assert df.loc[0, "product"] == "widget a"
        assert df.loc[0, "description"] == "high quality item"
        assert df.loc[1, "description"] == "good price"

    def test_normalize_whitespace_specific_columns(self, tmp_path):
        """Test columns parameter to normalize only specified columns."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,name,description\n"
            "1,  Alice  ,  Extra  spaces  \n"
            "2,  Bob  ,  Multiple  words  \n"
        )

        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame, columns=["name"])
        df = ar.to_pandas(clean)

        # name normalized
        assert df.loc[0, "name"] == "Alice"
        assert df.loc[1, "name"] == "Bob"
        # description untouched (still has double spaces)
        assert "Extra  spaces" in df.loc[0, "description"]

    def test_normalize_whitespace_tabs_and_mixed_whitespace(self, tmp_path):
        """Test that tabs and multiple whitespace types collapse to single space."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w") as f:
            f.write("text\n")
            f.write("  word1\t\tword2   word3  \n")

        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame)
        df = ar.to_pandas(clean)

        assert df.loc[0, "text"] == "word1 word2 word3"

    def test_normalize_whitespace_empty_and_whitespace_only_strings(self, tmp_path):
        """Test handling of empty strings and whitespace-only content."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,text\n"
            "1,\n"
            "2,  spaces  \n"
            "3,   \n"
        )

        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame)
        df = ar.to_pandas(clean)

        assert df.loc[1, "text"] == "spaces"
        # Whitespace-only becomes empty string after strip()
        assert df.loc[2, "text"] == ""

    def test_normalize_whitespace_null_values_preserved(self, tmp_path):
        """Test that null/missing values are preserved."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,name\n"
            "1,  Alice  \n"
            "2,\n"
            "3,  Bob  \n"
        )

        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame)
        df = ar.to_pandas(clean)

        assert df.loc[0, "name"] == "Alice"
        assert pd.isna(df.loc[1, "name"])
        assert df.loc[2, "name"] == "Bob"

    def test_normalize_whitespace_numeric_columns_unaffected(self, tmp_path):
        """Test that numeric columns are not affected by normalization."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,name,score\n"
            "1,  Alice  ,95.5\n"
            "2,Bob,88.0\n"
        )

        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame)
        df = ar.to_pandas(clean)

        assert df.loc[0, "name"] == "Alice"
        assert df.loc[0, "score"] == 95.5
        assert df.loc[1, "score"] == 88.0

    def test_normalize_whitespace_multiple_internal_spaces(self, tmp_path):
        """Test collapsing many consecutive internal spaces to one."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "phrase\n"
            "  hello     world     test  \n"
        )

        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame)
        df = ar.to_pandas(clean)

        assert df.loc[0, "phrase"] == "hello world test"

    def test_normalize_whitespace_nonexistent_column_error(self, tmp_path):
        """Test that specifying a non-existent column raises ValueError."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n")

        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError) as exc_info:
            ar.normalize_whitespace(frame, columns=["nonexistent"])
        
        assert "Missing columns" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)

    def test_normalize_whitespace_all_string_columns_by_default(self, tmp_path):
        """Test that columns=None (default) normalizes all string/object columns."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,first_name,last_name,city\n"
            "1,  John  ,  Doe  ,  New York  \n"
        )

        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame)  # columns=None
        df = ar.to_pandas(clean)

        # All text columns should be normalized
        assert df.loc[0, "first_name"] == "John"
        assert df.loc[0, "last_name"] == "Doe"
        assert df.loc[0, "city"] == "New York"

    def test_normalize_whitespace_csv_write_round_trip(self, tmp_path):
        """Test normalize_whitespace with CSV write round-trip."""
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"
        
        input_csv.write_text(
            "id,product,notes\n"
            "1,  Widget A  ,  High  quality  \n"
            "2,Widget B,Good\n"
        )

        # Load, clean, and write
        frame = ar.read_csv(str(input_csv))
        clean = ar.normalize_whitespace(frame)
        ar.write_csv(clean, str(output_csv))

        # Verify written file
        assert output_csv.exists()
        result_df = pd.read_csv(output_csv)
        assert result_df.loc[0, "product"] == "Widget A"
        assert result_df.loc[0, "notes"] == "High quality"

    def test_normalize_whitespace_from_dict(self):
        """Test normalize_whitespace with ArFrame created from dictionary."""
        data = {
            "name": ["  Alice  ", "  Bob  ", "Charlie"],
            "email": ["  alice@test.com  ", "bob@test.com", "  charlie@test.com  "],
        }
        frame = ar.from_dict(data)
        clean = ar.normalize_whitespace(frame)
        df = ar.to_pandas(clean)

        assert df.loc[0, "name"] == "Alice"
        assert df.loc[0, "email"] == "alice@test.com"
        assert df.loc[1, "email"] == "bob@test.com"
        assert df.loc[2, "name"] == "Charlie"

    def test_normalize_whitespace_subset_mixed_dtypes(self, tmp_path):
        """Test normalizing subset preserves numeric and other columns."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,name,description,score\n"
            "1,  Alice  ,  Extra  spaces  ,95.5\n"
        )

        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame, columns=["name", "description"])
        df = ar.to_pandas(clean)

        assert df.loc[0, "name"] == "Alice"
        assert df.loc[0, "description"] == "Extra spaces"
        assert df.loc[0, "score"] == 95.5

    def test_normalize_whitespace_leading_trailing_only(self, tmp_path):
        """Test leading/trailing space trimming on words without internal spaces."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "word\n"
            "  hello  \n"
            "world\n"
            "  test  \n"
        )

        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame)
        df = ar.to_pandas(clean)

        assert df.loc[0, "word"] == "hello"
        assert df.loc[1, "word"] == "world"
        assert df.loc[2, "word"] == "test"

    def test_normalize_whitespace_idempotent(self, tmp_path):
        """Test that applying normalize_whitespace twice yields same result."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "text\n"
            "  hello   world  \n"
        )

        frame = ar.read_csv(str(csv_file))
        clean_once = ar.normalize_whitespace(frame)
        clean_twice = ar.normalize_whitespace(clean_once)
        
        df_once = ar.to_pandas(clean_once)
        df_twice = ar.to_pandas(clean_twice)

        assert df_once.loc[0, "text"] == df_twice.loc[0, "text"]
        assert df_once.loc[0, "text"] == "hello world"

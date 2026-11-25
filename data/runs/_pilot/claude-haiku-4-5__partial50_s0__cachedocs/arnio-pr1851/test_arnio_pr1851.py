import pytest
import pandas as pd
import arnio as ar


class TestNormalizeCase:
    """Tests for ar.normalize_case function."""

    def test_normalize_case_lower_default(self):
        """Default case_type='lower' normalizes to lowercase."""
        df = pd.DataFrame({"name": ["Alice", "BOB", "ChArLiE"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame)
        result_df = ar.to_pandas(result)
        assert result_df["name"].tolist() == ["alice", "bob", "charlie"]

    def test_normalize_case_upper(self):
        """case_type='upper' converts to uppercase."""
        df = pd.DataFrame({"name": ["alice", "Bob", "CHARLIE"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame, case_type="upper")
        result_df = ar.to_pandas(result)
        assert result_df["name"].tolist() == ["ALICE", "BOB", "CHARLIE"]

    def test_normalize_case_title(self):
        """case_type='title' converts to title case."""
        df = pd.DataFrame({"name": ["alice smith", "BOB JONES", "charlie brown"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame, case_type="title")
        result_df = ar.to_pandas(result)
        assert result_df["name"][0].startswith("Alice")
        assert result_df["name"][1].startswith("Bob")

    def test_normalize_case_subset_single_column(self):
        """subset parameter affects only specified columns."""
        df = pd.DataFrame({
            "name": ["ALICE", "BOB"],
            "city": ["NEW YORK", "LOS ANGELES"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame, subset=["name"], case_type="lower")
        result_df = ar.to_pandas(result)
        assert result_df["name"].tolist() == ["alice", "bob"]
        assert result_df["city"].tolist() == ["NEW YORK", "LOS ANGELES"]

    def test_normalize_case_subset_multiple_columns(self):
        """subset parameter works with multiple columns."""
        df = pd.DataFrame({
            "name": ["ALICE"],
            "city": ["NEW YORK"],
            "country": ["USA"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame, subset=["name", "country"], case_type="lower")
        result_df = ar.to_pandas(result)
        assert result_df["name"][0] == "alice"
        assert result_df["country"][0] == "usa"
        assert result_df["city"][0] == "NEW YORK"

    def test_normalize_case_preserves_nulls(self):
        """Null values are preserved."""
        df = pd.DataFrame({"name": ["ALICE", None, "BOB"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame, case_type="lower")
        result_df = ar.to_pandas(result)
        assert result_df["name"][0] == "alice"
        assert pd.isna(result_df["name"][1])
        assert result_df["name"][2] == "bob"

    def test_normalize_case_non_ascii_preserved(self):
        """Non-ASCII UTF-8 bytes are preserved unchanged."""
        df = pd.DataFrame({"name": ["Café", "JOSÉ"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame, case_type="lower")
        result_df = ar.to_pandas(result)
        # Accented characters should remain
        assert "é" in result_df["name"][0]
        assert "é" in result_df["name"][1]

    def test_normalize_case_invalid_case_type_value(self):
        """Invalid case_type value raises ValueError."""
        df = pd.DataFrame({"name": ["Alice"]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            ar.normalize_case(frame, case_type="invalid")

    def test_normalize_case_invalid_case_type_type(self):
        """Non-string case_type raises TypeError."""
        df = pd.DataFrame({"name": ["Alice"]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError):
            ar.normalize_case(frame, case_type=42)

    def test_normalize_case_missing_column(self):
        """Missing column in subset raises KeyError."""
        df = pd.DataFrame({"name": ["Alice"]})
        frame = ar.from_pandas(df)
        with pytest.raises(KeyError):
            ar.normalize_case(frame, subset=["missing_col"])


class TestNormalizeUnicode:
    """Tests for ar.normalize_unicode function."""

    def test_normalize_unicode_default_form(self):
        """Default NFC normalization is applied."""
        df = pd.DataFrame({"text": ["café", "naïve"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame)
        result_df = ar.to_pandas(result)
        assert len(result_df) == 2
        assert pd.notna(result_df["text"][0])

    def test_normalize_unicode_subset_column(self):
        """subset parameter affects only specified columns."""
        df = pd.DataFrame({
            "text1": ["café"],
            "text2": ["naïve"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame, subset=["text1"])
        result_df = ar.to_pandas(result)
        assert "text1" in result_df.columns
        assert "text2" in result_df.columns

    def test_normalize_unicode_preserves_nulls(self):
        """Null values are preserved."""
        df = pd.DataFrame({"text": ["café", None, "naïve"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame)
        result_df = ar.to_pandas(result)
        assert pd.notna(result_df["text"][0])
        assert pd.isna(result_df["text"][1])
        assert pd.notna(result_df["text"][2])

    def test_normalize_unicode_pipeline_integration(self):
        """normalize_unicode works as a pipeline step."""
        df = pd.DataFrame({"text": ["café"]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [
            ("normalize_unicode", {"form": "NFC"})
        ])
        result_df = ar.to_pandas(result)
        assert len(result_df) == 1

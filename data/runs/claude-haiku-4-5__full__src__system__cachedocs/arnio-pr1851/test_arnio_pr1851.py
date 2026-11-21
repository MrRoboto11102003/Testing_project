"""
System (end-to-end) tests for arnio.cleaning.normalize_case and
arnio.cleaning.normalize_unicode functions.

These tests drive complete user workflows: loading CSVs, cleaning, converting
to pandas, writing results, and verifying output. They test both happy paths
and documented error cases.
"""

import pytest
import pandas as pd
import numpy as np
import arnio as ar
import unicodedata
from pathlib import Path


class TestNormalizeCase:
    """System tests for ar.normalize_case()."""

    def test_normalize_case_lowercase_basic(self, tmp_path):
        """Test lowercase conversion on string columns."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "name": ["Alice", "BOB", "ChArLiE"],
            "city": ["PARIS", "London", "ToKyO"],
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.normalize_case(frame, case_type="lower")
        result_df = ar.to_pandas(result)

        assert result_df["name"].tolist() == ["alice", "bob", "charlie"]
        assert result_df["city"].tolist() == ["paris", "london", "tokyo"]

    def test_normalize_case_uppercase(self, tmp_path):
        """Test uppercase conversion."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"name": ["alice", "Bob", "CHARLIE"]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.normalize_case(frame, case_type="upper")
        result_df = ar.to_pandas(result)

        assert result_df["name"].tolist() == ["ALICE", "BOB", "CHARLIE"]

    def test_normalize_case_title(self, tmp_path):
        """Test title case conversion."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"phrase": ["hello world", "MIXED case", "tItLe"]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.normalize_case(frame, case_type="title")
        result_df = ar.to_pandas(result)

        assert result_df["phrase"].tolist() == ["Hello World", "Mixed Case", "Title"]

    def test_normalize_case_subset_parameter(self, tmp_path):
        """Test that subset only normalizes specified columns."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "name": ["ALICE", "BOB"],
            "city": ["paris", "LONDON"],
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.normalize_case(frame, subset=["name"], case_type="lower")
        result_df = ar.to_pandas(result)

        # name is normalized, city unchanged
        assert result_df["name"].tolist() == ["alice", "bob"]
        assert result_df["city"].tolist() == ["paris", "LONDON"]

    def test_normalize_case_preserves_non_ascii(self, tmp_path):
        """Test that non-ASCII bytes (accents, emoji, CJK) are preserved."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "text": ["Café", "NAÏVE", "Zürich", "🎉HELLO", "日本語"],
        })
        df.to_csv(csv_file, index=False, encoding="utf-8")

        frame = ar.read_csv(str(csv_file), encoding="utf-8")
        result = ar.normalize_case(frame, case_type="lower")
        result_df = ar.to_pandas(result)

        # Accents, emoji, CJK preserved; ASCII lowercased
        assert result_df["text"][0] == "café"
        assert result_df["text"][1] == "naïve"
        assert result_df["text"][2] == "zürich"
        assert "🎉" in result_df["text"][3]
        assert "hello" in result_df["text"][3].lower()
        assert result_df["text"][4] == "日本語"  # Unchanged (no ASCII)

    def test_normalize_case_invalid_case_type_value(self, tmp_path):
        """Test that unsupported case_type raises ValueError."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"name": ["Alice"]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(ValueError, match="case_type must be one of"):
            ar.normalize_case(frame, case_type="mixed")

    def test_normalize_case_invalid_case_type_type(self, tmp_path):
        """Test that non-string case_type raises TypeError."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"name": ["Alice"]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(TypeError, match="case_type must be a string"):
            ar.normalize_case(frame, case_type=123)

    def test_normalize_case_missing_column_in_subset(self, tmp_path):
        """Test that missing column in subset raises KeyError."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"name": ["Alice"]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(KeyError):
            ar.normalize_case(frame, subset=["nonexistent"], case_type="lower")

    def test_normalize_case_with_nulls_and_pipeline(self, tmp_path):
        """Test normalize_case with nulls and within a pipeline."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "name": ["  ALICE  ", None, "  BOB  "],
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
        ])
        result_df = ar.to_pandas(result)

        assert result_df["name"][0] == "alice"
        assert pd.isna(result_df["name"][1])
        assert result_df["name"][2] == "bob"

        # Write result to new CSV
        output_file = tmp_path / "output.csv"
        ar.write_csv(result, str(output_file))
        reloaded = ar.read_csv(str(output_file))
        reloaded_df = ar.to_pandas(reloaded)
        assert reloaded_df["name"][0] == "alice"


class TestNormalizeUnicode:
    """System tests for ar.normalize_unicode()."""

    def test_normalize_unicode_nfc_normalization(self, tmp_path):
        """Test NFC (composed) Unicode normalization."""
        csv_file = tmp_path / "test.csv"
        # Create both decomposed and composed forms
        decomposed = "e\u0301"  # e + combining acute
        composed = "é"  # precomposed e acute
        
        df = pd.DataFrame({
            "text": [decomposed, composed, "café"],
        })
        df.to_csv(csv_file, index=False, encoding="utf-8")

        frame = ar.read_csv(str(csv_file), encoding="utf-8")
        result = ar.normalize_unicode(frame, form="NFC")
        result_df = ar.to_pandas(result)

        # All should be in composed form
        for val in result_df["text"]:
            normalized = unicodedata.normalize("NFC", val)
            assert val == normalized

    def test_normalize_unicode_nfd_normalization(self, tmp_path):
        """Test NFD (decomposed) Unicode normalization."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "text": ["café", "naïve", "Zürich"],
        })
        df.to_csv(csv_file, index=False, encoding="utf-8")

        frame = ar.read_csv(str(csv_file), encoding="utf-8")
        result = ar.normalize_unicode(frame, form="NFD")
        result_df = ar.to_pandas(result)

        # All should be in decomposed form
        for i, val in enumerate(result_df["text"]):
            expected = unicodedata.normalize("NFD", df["text"].iloc[i])
            assert val == expected

    def test_normalize_unicode_subset_parameter(self, tmp_path):
        """Test that subset only normalizes specified columns."""
        csv_file = tmp_path / "test.csv"
        decomposed = "e\u0301"  # decomposed
        
        df = pd.DataFrame({
            "name": [decomposed],
            "city": [decomposed],
        })
        df.to_csv(csv_file, index=False, encoding="utf-8")

        frame = ar.read_csv(str(csv_file), encoding="utf-8")
        result = ar.normalize_unicode(frame, subset=["name"], form="NFC")
        result_df = ar.to_pandas(result)

        # Only 'name' is normalized to composed
        composed = unicodedata.normalize("NFC", decomposed)
        assert result_df["name"][0] == composed
        assert result_df["city"][0] == decomposed

    def test_normalize_unicode_preserves_non_string_columns(self, tmp_path):
        """Test that non-string columns are preserved unchanged."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["café", "naïve", "zürich"],
            "score": [1.5, 2.5, 3.5],
        })
        df.to_csv(csv_file, index=False, encoding="utf-8")

        frame = ar.read_csv(str(csv_file), encoding="utf-8")
        result = ar.normalize_unicode(frame, form="NFC")
        result_df = ar.to_pandas(result)

        # Numeric columns unchanged
        assert result_df["id"].tolist() == [1, 2, 3]
        assert result_df["score"].tolist() == [1.5, 2.5, 3.5]

    def test_normalize_unicode_invalid_form(self, tmp_path):
        """Test that unsupported normalization form raises ValueError."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"name": ["café"]})
        df.to_csv(csv_file, index=False, encoding="utf-8")

        frame = ar.read_csv(str(csv_file), encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported Unicode normalization form"):
            ar.normalize_unicode(frame, form="INVALID")

    def test_normalize_unicode_invalid_form_type(self, tmp_path):
        """Test that non-string form raises TypeError."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"name": ["café"]})
        df.to_csv(csv_file, index=False, encoding="utf-8")

        frame = ar.read_csv(str(csv_file), encoding="utf-8")
        with pytest.raises(TypeError, match="form must be a string"):
            ar.normalize_unicode(frame, form=42)

    def test_normalize_unicode_missing_column_in_subset(self, tmp_path):
        """Test that missing column in subset raises KeyError."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"name": ["café"]})
        df.to_csv(csv_file, index=False, encoding="utf-8")

        frame = ar.read_csv(str(csv_file), encoding="utf-8")
        with pytest.raises(KeyError):
            ar.normalize_unicode(frame, subset=["missing"], form="NFC")

    def test_normalize_unicode_with_nulls_and_round_trip(self, tmp_path):
        """Test normalize_unicode with nulls and round-trip through CSV."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "text": ["café", None, "naïve", np.nan],
        })
        df.to_csv(csv_file, index=False, encoding="utf-8")

        frame = ar.read_csv(str(csv_file), encoding="utf-8")
        result = ar.normalize_unicode(frame, form="NFC")
        result_df = ar.to_pandas(result)

        assert result_df["text"][0] == "café"
        assert pd.isna(result_df["text"][1])
        assert result_df["text"][2] == "naïve"
        assert pd.isna(result_df["text"][3])

        # Write and reload
        output_file = tmp_path / "output.csv"
        ar.write_csv(result, str(output_file))
        reloaded = ar.read_csv(str(output_file), encoding="utf-8")
        reloaded_df = ar.to_pandas(reloaded)
        assert reloaded_df["text"][0] == "café"

    def test_normalize_case_and_unicode_combined_in_pipeline(self, tmp_path):
        """Test combining normalize_case and normalize_unicode in a pipeline."""
        csv_file = tmp_path / "test.csv"
        decomposed_upper = "CAFÉ"  # e is decomposed in input
        
        df = pd.DataFrame({
            "name": [decomposed_upper],
        })
        df.to_csv(csv_file, index=False, encoding="utf-8")

        frame = ar.read_csv(str(csv_file), encoding="utf-8")
        result = ar.pipeline(frame, [
            ("normalize_case", {"case_type": "lower"}),
            ("normalize_unicode", {"form": "NFC"}),
        ])
        result_df = ar.to_pandas(result)

        # Should be lowercase and in NFC form
        expected = unicodedata.normalize("NFC", "café")
        assert result_df["name"][0] == expected

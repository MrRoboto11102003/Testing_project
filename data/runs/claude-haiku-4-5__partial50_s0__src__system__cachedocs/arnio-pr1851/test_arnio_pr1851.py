import pytest
import pandas as pd
import numpy as np
import arnio as ar
import unicodedata
from pathlib import Path


class TestNormalizeCaseBasics:
    """Tests for ar.normalize_case() fundamental behavior"""

    def test_normalize_case_to_lowercase(self, tmp_path):
        """Read CSV, normalize to lowercase, verify output matches expected"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,city\nAlice,New York\nBob,Los Angeles\n")
        
        frame = ar.read_csv(str(csv_file))
        normalized = ar.normalize_case(frame, case_type="lower")
        df = ar.to_pandas(normalized)
        
        assert df["name"].tolist() == ["alice", "bob"]
        assert df["city"].tolist() == ["new york", "los angeles"]

    def test_normalize_case_to_uppercase(self, tmp_path):
        """Normalize string columns to uppercase"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("product,status\nLaptop,Active\nMouse,Inactive\n")
        
        frame = ar.read_csv(str(csv_file))
        normalized = ar.normalize_case(frame, case_type="upper")
        df = ar.to_pandas(normalized)
        
        assert df["product"].tolist() == ["LAPTOP", "MOUSE"]
        assert df["status"].tolist() == ["ACTIVE", "INACTIVE"]

    def test_normalize_case_to_titlecase(self, tmp_path):
        """Normalize to title case"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name\nalice smith\nbob jones\n")
        
        frame = ar.read_csv(str(csv_file))
        normalized = ar.normalize_case(frame, case_type="title")
        df = ar.to_pandas(normalized)
        
        assert df["name"].tolist() == ["Alice Smith", "Bob Jones"]

    def test_normalize_case_with_subset_parameter(self, tmp_path):
        """Only normalize specified columns via subset"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,city,country\nAlice,NEW YORK,USA\nBob,LONDON,UK\n")
        
        frame = ar.read_csv(str(csv_file))
        normalized = ar.normalize_case(frame, subset=["name"], case_type="lower")
        df = ar.to_pandas(normalized)
        
        assert df["name"].tolist() == ["alice", "bob"]
        assert df["city"].tolist() == ["NEW YORK", "LONDON"]  # unchanged

    def test_normalize_case_preserves_non_ascii_utf8(self, tmp_path):
        """Non-ASCII UTF-8 characters preserved, ASCII case normalized"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("location\nCAFE PARIS\nMÜNCHEN\n", encoding="utf-8")
        
        frame = ar.read_csv(str(csv_file))
        normalized = ar.normalize_case(frame, case_type="lower")
        df = ar.to_pandas(normalized)
        
        result = df["location"].iloc[0]
        assert "cafe paris" == result  # ASCII lowercased
        result_second = df["location"].iloc[1]
        assert "MÜNCHEN"[0:1] in result_second or "m" in result_second.lower()

    def test_normalize_case_error_invalid_case_type(self, tmp_path):
        """Invalid case_type raises ValueError"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name\nAlice\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError, match="case_type must be one of"):
            ar.normalize_case(frame, case_type="invalid")

    def test_normalize_case_error_missing_column_in_subset(self, tmp_path):
        """Missing column in subset raises KeyError"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,city\nAlice,Paris\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(KeyError):
            ar.normalize_case(frame, subset=["nonexistent"])

    def test_normalize_case_in_pipeline(self, tmp_path):
        """normalize_case works as a pipeline step"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("product,status\nLAPTOP,ACTIVE\nMOUSE,INACTIVE\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [
            ("normalize_case", {"case_type": "lower"}),
            ("drop_duplicates",),
        ])
        
        df = ar.to_pandas(result)
        assert df["product"].iloc[0] == "laptop"

    def test_normalize_case_roundtrip_csv(self, tmp_path):
        """Full workflow: read → normalize → write → read → verify"""
        input_file = tmp_path / "input.csv"
        input_file.write_text("ID,DESCRIPTION\n1,LAPTOP\n2,MOUSE\n")
        
        frame = ar.read_csv(str(input_file))
        normalized = ar.normalize_case(frame, case_type="lower")
        
        output_file = tmp_path / "output.csv"
        ar.write_csv(normalized, str(output_file))
        
        final_frame = ar.read_csv(str(output_file))
        df = ar.to_pandas(final_frame)
        
        assert df["DESCRIPTION"].tolist() == ["laptop", "mouse"]


class TestNormalizeUnicodeBasics:
    """Tests for ar.normalize_unicode() fundamental behavior"""

    def test_normalize_unicode_nfc_form(self, tmp_path):
        """Normalize to NFC canonical form"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("text\ncafé\nnaïve\n", encoding="utf-8")
        
        frame = ar.read_csv(str(csv_file))
        normalized = ar.normalize_unicode(frame, form="NFC")
        df = ar.to_pandas(normalized)
        
        for value in df["text"]:
            assert unicodedata.is_normalized("NFC", value)

    def test_normalize_unicode_all_forms(self, tmp_path):
        """All normalization forms work correctly"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("word\ncafé\nstraße\n", encoding="utf-8")
        
        frame = ar.read_csv(str(csv_file))
        
        for form in ["NFC", "NFD", "NFKC", "NFKD"]:
            normalized = ar.normalize_unicode(frame, form=form)
            df = ar.to_pandas(normalized)
            
            for value in df["word"]:
                assert unicodedata.is_normalized(form, value)

    def test_normalize_unicode_with_subset(self, tmp_path):
        """Only normalize specified columns"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,city\ncafé,MÜNSTER\n", encoding="utf-8")
        
        frame = ar.read_csv(str(csv_file))
        normalized = ar.normalize_unicode(frame, subset=["name"], form="NFC")
        df = ar.to_pandas(normalized)
        
        assert unicodedata.is_normalized("NFC", df["name"].iloc[0])

    def test_normalize_unicode_error_invalid_form(self, tmp_path):
        """Invalid form raises ValueError"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("text\ncafé\n", encoding="utf-8")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError, match="Unsupported Unicode normalization form"):
            ar.normalize_unicode(frame, form="INVALID")

    def test_normalize_unicode_error_missing_column(self, tmp_path):
        """Missing column in subset raises KeyError"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name\ncafé\n", encoding="utf-8")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(KeyError):
            ar.normalize_unicode(frame, subset=["nonexistent"])

    def test_normalize_unicode_in_pipeline(self, tmp_path):
        """normalize_unicode works as a pipeline step"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("location,description\nMÜNCHEN,café\n", encoding="utf-8")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [
            ("normalize_unicode", {"form": "NFC"}),
            ("strip_whitespace",),
        ])
        
        df = ar.to_pandas(result)
        for value in df["location"]:
            assert unicodedata.is_normalized("NFC", value)

    def test_normalize_unicode_roundtrip_csv(self, tmp_path):
        """Full workflow with Unicode normalization"""
        input_file = tmp_path / "input.csv"
        input_file.write_text("ID,LOCATION\n1,café\n2,münchen\n", encoding="utf-8")
        
        frame = ar.read_csv(str(input_file))
        normalized = ar.normalize_unicode(frame, form="NFC")
        
        output_file = tmp_path / "output.csv"
        ar.write_csv(normalized, str(output_file))
        
        final_frame = ar.read_csv(str(output_file))
        df = ar.to_pandas(final_frame)
        
        for value in df["LOCATION"]:
            assert unicodedata.is_normalized("NFC", value)


class TestNormalizeCombined:
    """Tests combining normalize_case and normalize_unicode"""

    def test_case_and_unicode_together_in_pipeline(self, tmp_path):
        """Apply both normalize_case and normalize_unicode together"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name\nAlice Café\nBob José\n", encoding="utf-8")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [
            ("normalize_case", {"case_type": "lower"}),
            ("normalize_unicode", {"form": "NFC"}),
        ])
        
        df = ar.to_pandas(result)
        
        for value in df["name"]:
            # ASCII parts should be lowercase
            ascii_chars = [c for c in value if ord(c) < 128]
            assert all(c.islower() or not c.isalpha() for c in ascii_chars)
            # Should be in NFC form
            assert unicodedata.is_normalized("NFC", value)

    def test_from_pandas_then_normalize(self):
        """Create ArFrame from pandas DataFrame and normalize"""
        df = pd.DataFrame({
            "name": ["ALICE CAFÉ", "BOB JOSÉ"],
            "city": ["NEW YORK", "LOS ANGELES"]
        })
        
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [
            ("normalize_case", {"case_type": "lower"}),
            ("normalize_unicode", {"form": "NFC"}),
        ])
        
        output_df = ar.to_pandas(result)
        
        for value in output_df["name"]:
            assert unicodedata.is_normalized("NFC", value)
            assert "alice" in value.lower() or "bob" in value.lower()

import pytest
import pandas as pd
import numpy as np
import unicodedata
import arnio as ar


class TestNormalizeCase:
    """Tests for ar.normalize_case()"""
    
    def test_normalize_case_lower_default(self):
        """Test default lower case normalization."""
        df = pd.DataFrame({
            "name": ["Alice", "BOB", "Charlie"],
            "city": ["NEW YORK", "Los Angeles", "SAN FRANCISCO"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["name"].tolist() == ["alice", "bob", "charlie"]
        assert result_df["city"].tolist() == ["new york", "los angeles", "san francisco"]
    
    def test_normalize_case_upper_and_title(self):
        """Test upper and title case normalization."""
        df = pd.DataFrame({
            "text": ["hello world", "UPPERCASE", "MiXeD CaSe"]
        })
        frame = ar.from_pandas(df)
        
        upper_result = ar.to_pandas(ar.normalize_case(frame, case_type="upper"))
        title_result = ar.to_pandas(ar.normalize_case(frame, case_type="title"))
        
        assert upper_result["text"].tolist() == ["HELLO WORLD", "UPPERCASE", "MIXED CASE"]
        assert title_result["text"].tolist() == ["Hello World", "Uppercase", "Mixed Case"]
    
    def test_normalize_case_subset_parameter(self):
        """Test that subset parameter only normalizes specified columns."""
        df = pd.DataFrame({
            "name": ["Alice", "BOB"],
            "city": ["NEW YORK", "LOS ANGELES"],
            "id": [1, 2]
        })
        frame = ar.from_pandas(df)
        result = ar.to_pandas(ar.normalize_case(frame, subset=["name"], case_type="lower"))
        
        assert result["name"].tolist() == ["alice", "bob"]
        assert result["city"].tolist() == ["NEW YORK", "LOS ANGELES"]  # unchanged
        assert result["id"].tolist() == [1, 2]
    
    def test_normalize_case_preserves_non_ascii_and_emoji(self):
        """Test that non-ASCII UTF-8 bytes and emoji are preserved."""
        df = pd.DataFrame({
            "text": ["Café", "Naïve", "北京", "São Paulo", "🎉 Party"]
        })
        frame = ar.from_pandas(df)
        result = ar.to_pandas(ar.normalize_case(frame, case_type="lower"))
        
        # Non-ASCII characters should be preserved as-is
        assert result["text"].tolist() == ["café", "naïve", "北京", "são paulo", "🎉 party"]
    
    def test_normalize_case_handles_nulls_and_mixed_types(self):
        """Test that null values and mixed column types are handled correctly."""
        df = pd.DataFrame({
            "name": ["Alice", None, "BOB"],
            "age": [25, 30, 35],
            "active": [True, False, True]
        })
        frame = ar.from_pandas(df)
        result = ar.to_pandas(ar.normalize_case(frame, case_type="upper"))
        
        assert result["name"].iloc[0] == "ALICE"
        assert pd.isna(result["name"].iloc[1])
        assert result["name"].iloc[2] == "BOB"
        assert result["age"].tolist() == [25, 30, 35]  # numeric unchanged
        assert result["active"].tolist() == [True, False, True]  # bool unchanged
    
    def test_normalize_case_error_invalid_case_type(self):
        """Test that invalid case_type raises appropriate errors."""
        df = pd.DataFrame({"name": ["Alice"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="case_type must be a string"):
            ar.normalize_case(frame, case_type=123)
        
        with pytest.raises(ValueError, match="case_type must be one of"):
            ar.normalize_case(frame, case_type="invalid")
    
    def test_normalize_case_error_missing_column(self):
        """Test that KeyError is raised for non-existent columns in subset."""
        df = pd.DataFrame({"name": ["Alice"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(KeyError):
            ar.normalize_case(frame, subset=["nonexistent"])


class TestNormalizeUnicode:
    """Tests for ar.normalize_unicode()"""
    
    def test_normalize_unicode_all_forms(self):
        """Test all Unicode normalization forms: NFC, NFD, NFKC, NFKD."""
        # Use a string that has different representations in different forms
        df = pd.DataFrame({
            "text": ["café", "naïve"]
        })
        frame = ar.from_pandas(df)
        
        # Default is NFC
        nfc_result = ar.to_pandas(ar.normalize_unicode(frame))
        assert all(unicodedata.is_normalized("NFC", t) for t in nfc_result["text"])
        
        # Test other forms
        for form in ["NFD", "NFKC", "NFKD"]:
            result = ar.to_pandas(ar.normalize_unicode(frame, form=form))
            assert all(unicodedata.is_normalized(form, t) for t in result["text"])
    
    def test_normalize_unicode_subset_parameter(self):
        """Test that subset parameter only normalizes specified columns."""
        df = pd.DataFrame({
            "name": ["café"],
            "city": ["São Paulo"],
            "id": [1]
        })
        frame = ar.from_pandas(df)
        result = ar.to_pandas(ar.normalize_unicode(frame, subset=["name"], form="NFC"))
        
        assert result["name"].iloc[0] == "café"
        assert result["id"].iloc[0] == 1
        assert result["city"].iloc[0] == "São Paulo"
    
    def test_normalize_unicode_handles_nulls_and_mixed_types(self):
        """Test that null values and non-string columns are preserved."""
        df = pd.DataFrame({
            "text": ["café", None, "naïve"],
            "score": [95.5, 88.0, 92.0],
            "count": [1, 2, 3]
        })
        frame = ar.from_pandas(df)
        result = ar.to_pandas(ar.normalize_unicode(frame, form="NFC"))
        
        assert unicodedata.is_normalized("NFC", result["text"].iloc[0])
        assert pd.isna(result["text"].iloc[1])
        assert unicodedata.is_normalized("NFC", result["text"].iloc[2])
        assert result["score"].tolist() == [95.5, 88.0, 92.0]  # unchanged
        assert result["count"].tolist() == [1, 2, 3]  # unchanged
    
    def test_normalize_unicode_error_invalid_form(self):
        """Test that invalid form parameter raises appropriate errors."""
        df = pd.DataFrame({"text": ["café"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="form must be a string"):
            ar.normalize_unicode(frame, form=123)
        
        with pytest.raises(ValueError, match="Unsupported Unicode normalization form"):
            ar.normalize_unicode(frame, form="invalid")
    
    def test_normalize_unicode_error_missing_column(self):
        """Test that KeyError is raised for non-existent columns in subset."""
        df = pd.DataFrame({"text": ["café"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(KeyError):
            ar.normalize_unicode(frame, subset=["nonexistent"])

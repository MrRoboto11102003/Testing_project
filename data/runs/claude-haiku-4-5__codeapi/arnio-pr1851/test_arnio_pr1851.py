import pytest
import pandas as pd
import arnio as ar
import unicodedata


class TestNormalizeCase:
    """Test cases for arnio.cleaning.normalize_case function."""

    def test_normalize_case_lower(self):
        """Test lowercase normalization."""
        frame = ar.from_dict({"text": ["HELLO", "World", "tEsT"]})
        result = ar.normalize_case(frame, case_type="lower")
        df = ar.to_pandas(result)
        assert df["text"].tolist() == ["hello", "world", "test"]

    def test_normalize_case_upper(self):
        """Test uppercase normalization."""
        frame = ar.from_dict({"text": ["hello", "World", "test"]})
        result = ar.normalize_case(frame, case_type="upper")
        df = ar.to_pandas(result)
        assert df["text"].tolist() == ["HELLO", "WORLD", "TEST"]

    def test_normalize_case_title(self):
        """Test title case normalization."""
        frame = ar.from_dict({"text": ["hello world", "TITLE", "test"]})
        result = ar.normalize_case(frame, case_type="title")
        df = ar.to_pandas(result)
        assert df["text"].tolist() == ["Hello World", "Title", "Test"]

    def test_normalize_case_default_lower(self):
        """Test that default case_type is 'lower'."""
        frame = ar.from_dict({"text": ["HELLO"]})
        result = ar.normalize_case(frame)
        df = ar.to_pandas(result)
        assert df["text"].iloc[0] == "hello"

    def test_normalize_case_with_nulls(self):
        """Test that null values are preserved."""
        frame = ar.from_dict({"text": ["HELLO", None, "WORLD"]})
        result = ar.normalize_case(frame, case_type="lower")
        df = ar.to_pandas(result)
        assert df["text"].iloc[0] == "hello"
        assert pd.isna(df["text"].iloc[1])
        assert df["text"].iloc[2] == "world"

    def test_normalize_case_subset_columns(self):
        """Test normalization on subset of columns."""
        frame = ar.from_dict({"a": ["HELLO"], "b": ["WORLD"], "c": ["TEST"]})
        result = ar.normalize_case(frame, subset=["a", "c"], case_type="lower")
        df = ar.to_pandas(result)
        assert df["a"].iloc[0] == "hello"
        assert df["b"].iloc[0] == "WORLD"
        assert df["c"].iloc[0] == "test"

    def test_normalize_case_preserves_non_ascii(self):
        """Test that non-ASCII UTF-8 bytes are preserved unchanged."""
        frame = ar.from_dict({"text": ["CAFÉ", "naïve", "MÜNCHEN"]})
        result = ar.normalize_case(frame, case_type="lower")
        df = ar.to_pandas(result)
        # Non-ASCII bytes should be preserved; only ASCII letters change
        assert "é" in df["text"].iloc[0]
        assert "ï" in df["text"].iloc[1]
        assert "ü" in df["text"].iloc[2]

    def test_normalize_case_invalid_case_type_value(self):
        """Test that ValueError is raised for invalid case_type."""
        frame = ar.from_dict({"text": ["hello"]})
        with pytest.raises(ValueError, match="case_type must be one of"):
            ar.normalize_case(frame, case_type="invalid")

    def test_normalize_case_invalid_case_type_type(self):
        """Test that TypeError is raised when case_type is not a string."""
        frame = ar.from_dict({"text": ["hello"]})
        with pytest.raises(TypeError, match="case_type must be a string"):
            ar.normalize_case(frame, case_type=123)

    def test_normalize_case_missing_column_in_subset(self):
        """Test that KeyError is raised for missing columns in subset."""
        frame = ar.from_dict({"text": ["hello"]})
        with pytest.raises(KeyError):
            ar.normalize_case(frame, subset=["nonexistent"])


class TestNormalizeUnicode:
    """Test cases for arnio.cleaning.normalize_unicode function."""

    def test_normalize_unicode_nfc_default(self):
        """Test NFC normalization (default form)."""
        # Create a decomposed form: e + combining acute accent
        decomposed = "e\u0301"
        # Expected composed form: precomposed é
        expected = "\u00e9"
        
        frame = ar.from_dict({"text": [decomposed]})
        result = ar.normalize_unicode(frame)
        df = ar.to_pandas(result)
        assert df["text"].iloc[0] == expected

    def test_normalize_unicode_nfc_explicit(self):
        """Test explicit NFC normalization (composition)."""
        decomposed = "e\u0301"
        expected = "\u00e9"
        
        frame = ar.from_dict({"text": [decomposed]})
        result = ar.normalize_unicode(frame, form="NFC")
        df = ar.to_pandas(result)
        assert df["text"].iloc[0] == expected

    def test_normalize_unicode_nfd(self):
        """Test NFD normalization (decomposition)."""
        composed = "\u00e9"
        expected = "e\u0301"
        
        frame = ar.from_dict({"text": [composed]})
        result = ar.normalize_unicode(frame, form="NFD")
        df = ar.to_pandas(result)
        assert df["text"].iloc[0] == expected

    def test_normalize_unicode_nfkc_nfkd(self):
        """Test NFKC and NFKD compatibility forms."""
        frame = ar.from_dict({"text": ["\u00e9"]})
        result_kc = ar.normalize_unicode(frame, form="NFKC")
        result_kd = ar.normalize_unicode(frame, form="NFKD")
        df_kc = ar.to_pandas(result_kc)
        df_kd = ar.to_pandas(result_kd)
        # Both should produce valid Unicode strings
        assert isinstance(df_kc["text"].iloc[0], str)
        assert isinstance(df_kd["text"].iloc[0], str)

    def test_normalize_unicode_with_nulls(self):
        """Test that null values are preserved."""
        decomposed = "e\u0301"
        expected = "\u00e9"
        
        frame = ar.from_dict({"text": [decomposed, None, expected]})
        result = ar.normalize_unicode(frame, form="NFC")
        df = ar.to_pandas(result)
        
        assert df["text"].iloc[0] == expected
        assert pd.isna(df["text"].iloc[1])
        assert df["text"].iloc[2] == expected

    def test_normalize_unicode_subset_columns(self):
        """Test normalization on subset of columns."""
        decomposed = "e\u0301"
        expected = "\u00e9"
        
        frame = ar.from_dict({
            "a": [decomposed],
            "b": [decomposed],
            "c": [decomposed]
        })
        result = ar.normalize_unicode(frame, subset=["a", "c"], form="NFC")
        df = ar.to_pandas(result)
        
        assert df["a"].iloc[0] == expected
        assert df["b"].iloc[0] == decomposed
        assert df["c"].iloc[0] == expected

    def test_normalize_unicode_invalid_form_value(self):
        """Test that ValueError is raised for invalid form."""
        frame = ar.from_dict({"text": ["hello"]})
        with pytest.raises(ValueError, match="Unsupported Unicode normalization form"):
            ar.normalize_unicode(frame, form="INVALID")

    def test_normalize_unicode_invalid_form_type(self):
        """Test that TypeError is raised when form is not a string."""
        frame = ar.from_dict({"text": ["hello"]})
        with pytest.raises(TypeError, match="form must be a string"):
            ar.normalize_unicode(frame, form=123)

    def test_normalize_unicode_missing_column_in_subset(self):
        """Test that KeyError is raised for missing columns in subset."""
        frame = ar.from_dict({"text": ["hello"]})
        with pytest.raises(KeyError):
            ar.normalize_unicode(frame, subset=["nonexistent"])

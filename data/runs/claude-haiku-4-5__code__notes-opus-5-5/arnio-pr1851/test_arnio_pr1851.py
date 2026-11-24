import pytest
import pandas as pd
import arnio as ar


class TestNormalizeCase:
    """Tests for arnio.cleaning.normalize_case"""

    def test_all_case_types(self):
        """Test lower, upper, and title case transformations"""
        df = pd.DataFrame({"text": ["Hello World"]})
        f = ar.from_pandas(df)
        
        lower = ar.to_pandas(ar.normalize_case(f, case_type="lower"))
        upper = ar.to_pandas(ar.normalize_case(f, case_type="upper"))
        title = ar.to_pandas(ar.normalize_case(f, case_type="title"))
        
        assert lower["text"][0] == "hello world"
        assert upper["text"][0] == "HELLO WORLD"
        assert title["text"][0] == "Hello World"

    def test_subset_parameter_limits_columns(self):
        """Test subset parameter only modifies selected columns"""
        df = pd.DataFrame({
            "name": ["ALICE"],
            "city": ["PARIS"],
            "count": [42]
        })
        f = ar.from_pandas(df)
        result = ar.normalize_case(f, subset=["name"], case_type="lower")
        out = ar.to_pandas(result)
        
        assert out["name"][0] == "alice"
        assert out["city"][0] == "PARIS"
        assert out["count"][0] == 42

    def test_null_and_non_ascii_preserved(self):
        """Test null values and non-ASCII UTF-8 are preserved"""
        df = pd.DataFrame({"text": ["ÉCOLE", None, "NAÏVE"]})
        f = ar.from_pandas(df)
        result = ar.normalize_case(f, case_type="lower")
        out = ar.to_pandas(result)
        
        assert out["text"][0] == "école"
        assert pd.isna(out["text"][1])
        assert out["text"][2] == "naïve"

    def test_input_not_mutated(self):
        """Test input frame is not mutated"""
        df = pd.DataFrame({"text": ["HELLO"]})
        f = ar.from_pandas(df)
        ar.normalize_case(f, case_type="lower")
        original = ar.to_pandas(f)
        assert original["text"][0] == "HELLO"

    def test_error_invalid_case_type(self):
        """Test TypeError and ValueError for invalid case_type"""
        df = pd.DataFrame({"text": ["HELLO"]})
        f = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="case_type must be a string"):
            ar.normalize_case(f, case_type=123)
        with pytest.raises(ValueError, match="case_type must be one of"):
            ar.normalize_case(f, case_type="invalid")

    def test_error_missing_subset_column(self):
        """Test KeyError for missing column in subset"""
        df = pd.DataFrame({"text": ["HELLO"]})
        f = ar.from_pandas(df)
        
        with pytest.raises(KeyError):
            ar.normalize_case(f, subset=["missing"], case_type="lower")


class TestNormalizeUnicode:
    """Tests for arnio.cleaning.normalize_unicode"""

    def test_different_unicode_forms(self):
        """Test NFC, NFD, and NFKC normalization forms"""
        df = pd.DataFrame({"text": ["e\u0301", "\u00e9", "\ufb01"]})
        f = ar.from_pandas(df)
        
        nfc = ar.to_pandas(ar.normalize_unicode(f, form="NFC"))
        nfd = ar.to_pandas(ar.normalize_unicode(f, form="NFD"))
        nfkc = ar.to_pandas(ar.normalize_unicode(f, form="NFKC"))
        
        # NFC composes combining characters
        assert nfc["text"][0] == "\u00e9"
        # NFD decomposes characters
        assert nfd["text"][1] == "e\u0301"
        # NFKC transforms ligatures
        assert nfkc["text"][2] == "fi"

    def test_subset_null_and_non_string_handling(self):
        """Test subset parameter, null preservation, and non-string columns"""
        df = pd.DataFrame({
            "s": ["e\u0301", None],
            "t": ["e\u0301"],
            "count": [42]
        })
        f = ar.from_pandas(df)
        result = ar.normalize_unicode(f, form="NFC", subset=["s"])
        out = ar.to_pandas(result)
        
        assert out["s"][0] == "\u00e9"
        assert pd.isna(out["s"][1])
        assert out["t"][0] == "e\u0301"
        assert out["count"][0] == 42

    def test_input_not_mutated(self):
        """Test input frame is not mutated"""
        df = pd.DataFrame({"text": ["e\u0301"]})
        f = ar.from_pandas(df)
        ar.normalize_unicode(f, form="NFC")
        original = ar.to_pandas(f)
        assert original["text"][0] == "e\u0301"

    def test_error_invalid_form(self):
        """Test TypeError and ValueError for invalid form"""
        df = pd.DataFrame({"text": ["hello"]})
        f = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="form must be a string"):
            ar.normalize_unicode(f, form=123)
        with pytest.raises(ValueError, match="Unsupported Unicode normalization form"):
            ar.normalize_unicode(f, form="INVALID")

    def test_error_missing_subset_column(self):
        """Test KeyError for missing column in subset"""
        df = pd.DataFrame({"text": ["hello"]})
        f = ar.from_pandas(df)
        
        with pytest.raises(KeyError):
            ar.normalize_unicode(f, subset=["missing"], form="NFC")

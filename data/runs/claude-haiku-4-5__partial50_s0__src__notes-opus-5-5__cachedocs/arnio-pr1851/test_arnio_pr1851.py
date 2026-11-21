import pytest
import pandas as pd
import numpy as np
import arnio as ar
import re


class TestNormalizeCase:
    """Tests for ar.normalize_case."""

    def test_normalize_case_all_modes(self):
        """Test lower, upper, and title case transformations."""
        df = pd.DataFrame({"name": ["Alice", "BOB", "cAraMel"]})
        frame = ar.from_pandas(df)
        
        # Test lower
        lower_result = ar.normalize_case(frame, case_type="lower")
        assert ar.to_pandas(lower_result)["name"].tolist() == ["alice", "bob", "caramel"]
        
        # Test upper
        upper_result = ar.normalize_case(frame, case_type="upper")
        assert ar.to_pandas(upper_result)["name"].tolist() == ["ALICE", "BOB", "CARAMEL"]
        
        # Test title (default is lower)
        title_result = ar.normalize_case(frame, case_type="title")
        out_title = ar.to_pandas(title_result)["name"].tolist()
        assert out_title[0][0].isupper() and out_title[0][1:].islower()

    def test_normalize_case_default_is_lower(self):
        """Test that default case_type is 'lower'."""
        df = pd.DataFrame({"name": ["HELLO"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame)
        assert ar.to_pandas(result)["name"][0] == "hello"

    def test_normalize_case_subset_behavior(self):
        """Test that subset applies only to specified columns."""
        df = pd.DataFrame({"name": ["ALICE"], "city": ["PARIS"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame, subset=["name"], case_type="lower")
        out = ar.to_pandas(result)
        assert out["name"][0] == "alice"
        assert out["city"][0] == "PARIS"  # unchanged

    def test_normalize_case_subset_tuple_accepted(self):
        """Test that subset accepts tuples."""
        df = pd.DataFrame({"name": ["HELLO"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame, subset=("name",), case_type="lower")
        assert ar.to_pandas(result)["name"][0] == "hello"

    def test_normalize_case_nulls_preserved(self):
        """Test that null/NA values are preserved."""
        df = pd.DataFrame({"name": ["ALICE", None, "BOB"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame, case_type="lower")
        out = ar.to_pandas(result)
        assert out["name"][0] == "alice"
        assert pd.isna(out["name"][1])
        assert out["name"][2] == "bob"

    def test_normalize_case_non_string_columns_untouched(self):
        """Test that non-string columns remain unchanged."""
        df = pd.DataFrame({"name": ["HELLO"], "age": [30], "active": [True]})
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame, case_type="upper")
        out = ar.to_pandas(result)
        assert out["age"][0] == 30
        assert out["active"][0] is True or out["active"][0] == True

    def test_normalize_case_non_ascii_preserved(self):
        """Test that non-ASCII UTF-8 bytes are preserved byte-for-byte."""
        df = pd.DataFrame({"text": ["ÉCOLE"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame, case_type="lower")
        out = ar.to_pandas(result)
        # Non-ASCII should be preserved; ASCII 'E' becomes 'e'
        assert "é" in out["text"][0] or "E" not in out["text"][0]

    def test_normalize_case_attrs_deepcopied(self):
        """Test that attrs are deep-copied, not referenced."""
        df = pd.DataFrame({"name": ["HELLO"]})
        frame = ar.from_pandas(df)
        frame._attrs = {"key": [1, 2, 3]}
        result = ar.normalize_case(frame)
        assert result._attrs == {"key": [1, 2, 3]}
        assert result._attrs is not frame._attrs

    def test_normalize_case_input_unchanged(self):
        """Test that input frame is not mutated."""
        df = pd.DataFrame({"name": ["HELLO", "WORLD"]})
        frame = ar.from_pandas(df)
        original = ar.to_pandas(frame).copy()
        ar.normalize_case(frame, case_type="lower")
        pd.testing.assert_frame_equal(ar.to_pandas(frame), original)

    def test_normalize_case_error_case_type_not_string(self):
        """Test TypeError when case_type is not a string."""
        frame = ar.from_pandas(pd.DataFrame({"name": ["HELLO"]}))
        with pytest.raises(TypeError, match="case_type must be a string"):
            ar.normalize_case(frame, case_type=42)

    def test_normalize_case_error_case_type_invalid_value(self):
        """Test ValueError for unsupported case_type values."""
        frame = ar.from_pandas(pd.DataFrame({"name": ["HELLO"]}))
        with pytest.raises(ValueError, match="case_type must be one of"):
            ar.normalize_case(frame, case_type="reverse")

    def test_normalize_case_error_subset_not_sequence(self):
        """Test TypeError when subset is not a sequence."""
        frame = ar.from_pandas(pd.DataFrame({"name": ["HELLO"]}))
        with pytest.raises(TypeError, match="subset must be a sequence"):
            ar.normalize_case(frame, subset="name")

    def test_normalize_case_error_missing_column_in_subset(self):
        """Test KeyError when subset references missing column."""
        frame = ar.from_pandas(pd.DataFrame({"name": ["HELLO"]}))
        with pytest.raises(KeyError, match=re.escape("Missing columns for normalize_case")):
            ar.normalize_case(frame, subset=["nonexistent"])


class TestNormalizeUnicode:
    """Tests for ar.normalize_unicode."""

    def test_normalize_unicode_all_forms(self):
        """Test NFC, NFD, NFKC, NFKD normalization forms."""
        # NFC: e + combining accent (2 chars) -> é (1 char)
        df = pd.DataFrame({"text": ["e\u0301", "\ufb01"]})  # e+accent, ﬁ ligature
        frame = ar.from_pandas(df)
        
        result_nfc = ar.normalize_unicode(frame, form="NFC")
        out_nfc = ar.to_pandas(result_nfc)
        assert len(out_nfc["text"][0]) == 1  # composed
        
        result_nfd = ar.normalize_unicode(frame, form="NFD")
        out_nfd = ar.to_pandas(result_nfd)
        assert len(out_nfd["text"][0]) == 2  # decomposed
        
        result_nfkc = ar.normalize_unicode(frame, form="NFKC")
        out_nfkc = ar.to_pandas(result_nfkc)
        assert out_nfkc["text"][1] == "fi"  # ligature replaced

    def test_normalize_unicode_default_is_nfc(self):
        """Test that default form is NFC (composed form)."""
        df = pd.DataFrame({"text": ["e\u0301"]})  # 2-char decomposed form
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame)
        out = ar.to_pandas(result)
        assert len(out["text"][0]) == 1  # NFC produces 1-char composed form
        assert out["text"][0] == "\u00e9"

    def test_normalize_unicode_subset_behavior(self):
        """Test that subset applies only to specified columns."""
        df = pd.DataFrame({"s": ["e\u0301"], "t": ["e\u0301"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame, form="NFC", subset=["s"])
        out = ar.to_pandas(result)
        assert len(out["s"][0]) == 1  # normalized
        assert len(out["t"][0]) == 2  # not in subset, unchanged

    def test_normalize_unicode_subset_tuple_accepted(self):
        """Test that subset accepts tuples."""
        df = pd.DataFrame({"text": ["e\u0301"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame, form="NFC", subset=("text",))
        assert len(ar.to_pandas(result)["text"][0]) == 1

    def test_normalize_unicode_nulls_preserved(self):
        """Test that null/NA values are preserved."""
        df = pd.DataFrame({"text": ["e\u0301", None, "\ufb01"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame, form="NFKC")
        out = ar.to_pandas(result)
        assert pd.isna(out["text"][1])
        assert out["text"][2] == "fi"

    def test_normalize_unicode_non_string_untouched(self):
        """Test that non-string columns remain unchanged."""
        df = pd.DataFrame({"text": ["hello"], "num": [42], "flag": [True]})
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame, form="NFC")
        out = ar.to_pandas(result)
        assert out["num"][0] == 42
        assert out["flag"][0] is True or out["flag"][0] == True

    def test_normalize_unicode_non_string_in_subset_ignored(self):
        """Test that non-string columns in subset are silently skipped."""
        df = pd.DataFrame({"text": ["e\u0301"], "num": [42]})
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame, form="NFC", subset=["text", "num"])
        out = ar.to_pandas(result)
        assert len(out["text"][0]) == 1
        assert out["num"][0] == 42

    def test_normalize_unicode_attrs_deepcopied(self):
        """Test that attrs are deep-copied, not referenced."""
        df = pd.DataFrame({"text": ["hello"]})
        frame = ar.from_pandas(df)
        frame._attrs = {"key": [1, 2, 3]}
        result = ar.normalize_unicode(frame)
        assert result._attrs == {"key": [1, 2, 3]}
        assert result._attrs is not frame._attrs

    def test_normalize_unicode_input_unchanged(self):
        """Test that input frame is not mutated."""
        df = pd.DataFrame({"text": ["e\u0301"]})
        frame = ar.from_pandas(df)
        original = ar.to_pandas(frame).copy()
        ar.normalize_unicode(frame, form="NFC")
        pd.testing.assert_frame_equal(ar.to_pandas(frame), original)

    def test_normalize_unicode_error_form_not_string(self):
        """Test TypeError when form is not a string."""
        frame = ar.from_pandas(pd.DataFrame({"text": ["hello"]}))
        with pytest.raises(TypeError, match="form must be a string"):
            ar.normalize_unicode(frame, form=42)

    def test_normalize_unicode_error_form_invalid_value(self):
        """Test ValueError for unsupported normalization forms."""
        frame = ar.from_pandas(pd.DataFrame({"text": ["hello"]}))
        with pytest.raises(ValueError, match="Unsupported Unicode normalization form"):
            ar.normalize_unicode(frame, form="NFX")

    def test_normalize_unicode_error_subset_not_sequence(self):
        """Test TypeError when subset is not a sequence."""
        frame = ar.from_pandas(pd.DataFrame({"text": ["hello"]}))
        with pytest.raises(TypeError, match="subset must be a sequence"):
            ar.normalize_unicode(frame, subset="text")

    def test_normalize_unicode_error_missing_column_in_subset(self):
        """Test KeyError when subset references missing column."""
        frame = ar.from_pandas(pd.DataFrame({"text": ["hello"]}))
        with pytest.raises(KeyError, match=re.escape("Missing columns for normalize_unicode")):
            ar.normalize_unicode(frame, subset=["nonexistent"])

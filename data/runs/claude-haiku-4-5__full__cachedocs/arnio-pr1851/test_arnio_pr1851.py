import pytest
import pandas as pd
import arnio as ar


class TestNormalizeCase:
    """Tests for ar.normalize_case"""

    def test_lower_case(self):
        """Normalize to lower case"""
        df = pd.DataFrame({"name": ["Alice", "BOB", "ChArlie"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame, case_type="lower")
        assert ar.to_pandas(result)["name"].tolist() == ["alice", "bob", "charlie"]

    def test_upper_case(self):
        """Normalize to upper case"""
        df = pd.DataFrame({"name": ["alice", "Bob", "charlie"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame, case_type="upper")
        assert ar.to_pandas(result)["name"].tolist() == ["ALICE", "BOB", "CHARLIE"]

    def test_title_case(self):
        """Normalize to title case"""
        df = pd.DataFrame({"name": ["alice smith", "BOB JONES"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame, case_type="title")
        result_df = ar.to_pandas(result)
        assert result_df["name"].iloc[0][0].isupper()  # First char capitalized
        assert result_df["name"].iloc[1][0].isupper()

    def test_subset_specific_columns(self):
        """Apply normalization only to specified columns"""
        df = pd.DataFrame({"name": ["Alice"], "city": ["NEW YORK"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame, subset=["name"], case_type="lower")
        result_df = ar.to_pandas(result)
        assert result_df["name"].iloc[0] == "alice"
        assert result_df["city"].iloc[0] == "NEW YORK"  # unchanged

    def test_all_string_columns_when_subset_none(self):
        """When subset=None, all string columns are normalized"""
        df = pd.DataFrame({"name": ["Alice"], "city": ["NEW YORK"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame, subset=None, case_type="lower")
        result_df = ar.to_pandas(result)
        assert result_df["name"].iloc[0] == "alice"
        assert result_df["city"].iloc[0] == "new york"

    def test_preserves_non_ascii_utf8(self):
        """Non-ASCII UTF-8 characters are preserved unchanged"""
        df = pd.DataFrame({
            "text": ["Café", "Naïve", "中文", "🚀ABC"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame, case_type="upper")
        result_df = ar.to_pandas(result)
        # Non-ASCII bytes should be preserved
        assert "中文" in result_df["text"].iloc[2]
        assert "🚀" in result_df["text"].iloc[3]

    def test_invalid_case_type_raises_value_error(self):
        """ValueError raised for unsupported case_type"""
        df = pd.DataFrame({"name": ["Alice"]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            ar.normalize_case(frame, case_type="invalid_case")

    def test_missing_column_in_subset_raises_key_error(self):
        """KeyError raised when subset column doesn't exist"""
        df = pd.DataFrame({"name": ["Alice"]})
        frame = ar.from_pandas(df)
        with pytest.raises(KeyError):
            ar.normalize_case(frame, subset=["nonexistent"], case_type="lower")


class TestNormalizeUnicode:
    """Tests for ar.normalize_unicode"""

    def test_nfc_normalization(self):
        """NFC normalization form applied"""
        df = pd.DataFrame({"text": ["café"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame, form="NFC")
        # Should produce valid string (exact form varies by input)
        assert isinstance(ar.to_pandas(result)["text"].iloc[0], str)

    def test_nfd_normalization(self):
        """NFD normalization form applied"""
        df = pd.DataFrame({"text": ["café"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame, form="NFD")
        # Should produce valid string
        assert isinstance(ar.to_pandas(result)["text"].iloc[0], str)

    def test_subset_specific_columns(self):
        """Apply normalization only to specified columns"""
        df = pd.DataFrame({"name": ["Café"], "city": ["Montréal"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame, subset=["name"], form="NFC")
        result_df = ar.to_pandas(result)
        # name should be processed, city unchanged
        assert isinstance(result_df["name"].iloc[0], str)
        assert isinstance(result_df["city"].iloc[0], str)

    def test_all_string_columns_when_subset_none(self):
        """When subset=None, all string columns are normalized"""
        df = pd.DataFrame({"name": ["Café"], "city": ["Montréal"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame, subset=None, form="NFC")
        result_df = ar.to_pandas(result)
        assert isinstance(result_df["name"].iloc[0], str)
        assert isinstance(result_df["city"].iloc[0], str)

    def test_non_string_columns_unchanged(self):
        """Non-string columns are left unchanged"""
        df = pd.DataFrame({
            "text": ["Café"],
            "number": [42],
            "flag": [True]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame, form="NFC")
        result_df = ar.to_pandas(result)
        assert result_df["number"].iloc[0] == 42
        assert result_df["flag"].iloc[0] is True

    def test_invalid_normalization_form_raises_value_error(self):
        """ValueError raised for unsupported normalization form"""
        df = pd.DataFrame({"text": ["café"]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            ar.normalize_unicode(frame, form="INVALID")

    def test_missing_column_in_subset_raises_key_error(self):
        """KeyError raised when subset column doesn't exist"""
        df = pd.DataFrame({"text": ["café"]})
        frame = ar.from_pandas(df)
        with pytest.raises(KeyError):
            ar.normalize_unicode(frame, subset=["nonexistent"], form="NFC")

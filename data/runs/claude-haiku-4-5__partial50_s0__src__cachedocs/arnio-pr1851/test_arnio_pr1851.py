import pytest
import pandas as pd
import arnio as ar


class TestNormalizeCase:
    """Test suite for ar.normalize_case()"""

    @pytest.fixture
    def basic_frame(self):
        """Create a basic test frame with mixed case strings and numeric column"""
        df = pd.DataFrame({
            "name": ["Alice", "BOB", "cHaRLiE"],
            "city": ["New York", "los ANGELES", "SAN francisco"],
            "age": [25, 30, 35]
        })
        return ar.from_pandas(df)

    def test_normalize_case_default_lower(self, basic_frame):
        """Test normalize_case with default case_type='lower'"""
        result = ar.normalize_case(basic_frame)
        df = ar.to_pandas(result)
        assert df["name"].tolist() == ["alice", "bob", "charlie"]
        # Numeric column should be unchanged
        assert df["age"].tolist() == [25, 30, 35]

    def test_normalize_case_upper(self, basic_frame):
        """Test normalize_case with case_type='upper'"""
        result = ar.normalize_case(basic_frame, case_type="upper")
        df = ar.to_pandas(result)
        assert df["name"].tolist() == ["ALICE", "BOB", "CHARLIE"]
        assert df["city"].tolist() == ["NEW YORK", "LOS ANGELES", "SAN FRANCISCO"]

    def test_normalize_case_title(self, basic_frame):
        """Test normalize_case with case_type='title'"""
        result = ar.normalize_case(basic_frame, case_type="title")
        df = ar.to_pandas(result)
        assert df["name"][0] == "Alice"
        assert df["city"][0] == "New York"

    def test_normalize_case_subset_single_column(self, basic_frame):
        """Test normalize_case with subset parameter on single column"""
        result = ar.normalize_case(basic_frame, subset=["name"], case_type="upper")
        df = ar.to_pandas(result)
        assert df["name"].tolist() == ["ALICE", "BOB", "CHARLIE"]
        # City should remain unchanged
        assert df["city"].tolist() == ["New York", "los ANGELES", "SAN francisco"]

    def test_normalize_case_preserves_non_ascii(self):
        """Test that non-ASCII UTF-8 bytes are preserved"""
        df = pd.DataFrame({"text": ["Café", "naïve", "résumé"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame, case_type="lower")
        result_df = ar.to_pandas(result)
        # Accented characters should be preserved
        assert "é" in result_df["text"].values

    def test_normalize_case_with_nulls(self):
        """Test normalize_case handles null values correctly"""
        df = pd.DataFrame({"text": ["Alice", None, "BOB"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_case(frame, case_type="upper")
        result_df = ar.to_pandas(result)
        assert result_df["text"][0] == "ALICE"
        assert pd.isna(result_df["text"][1])
        assert result_df["text"][2] == "BOB"

    def test_normalize_case_invalid_case_type(self, basic_frame):
        """Test that invalid case_type raises ValueError"""
        with pytest.raises(ValueError, match="case_type must be one of"):
            ar.normalize_case(basic_frame, case_type="invalid")

    def test_normalize_case_case_type_not_string(self, basic_frame):
        """Test that non-string case_type raises TypeError"""
        with pytest.raises(TypeError, match="case_type must be a string"):
            ar.normalize_case(basic_frame, case_type=123)

    def test_normalize_case_nonexistent_column_in_subset(self, basic_frame):
        """Test that subset with nonexistent column raises error"""
        with pytest.raises((KeyError, ValueError)):
            ar.normalize_case(basic_frame, subset=["nonexistent"])


class TestNormalizeUnicode:
    """Test suite for ar.normalize_unicode()"""

    @pytest.fixture
    def unicode_frame(self):
        """Create frame with Unicode text and numeric column"""
        df = pd.DataFrame({
            "text": ["café", "naïve", "résumé"],
            "value": [1, 2, 3]
        })
        return ar.from_pandas(df)

    def test_normalize_unicode_default_nfc(self, unicode_frame):
        """Test normalize_unicode with default form='NFC'"""
        result = ar.normalize_unicode(unicode_frame)
        df = ar.to_pandas(result)
        assert df["text"][0] is not None
        # Numeric columns should be unchanged
        assert df["value"].tolist() == [1, 2, 3]

    def test_normalize_unicode_nfd_form(self, unicode_frame):
        """Test normalize_unicode with form='NFD'"""
        result = ar.normalize_unicode(unicode_frame, form="NFD")
        df = ar.to_pandas(result)
        assert df["text"] is not None
        assert df["value"].tolist() == [1, 2, 3]

    def test_normalize_unicode_nfkc_nfkd_forms(self, unicode_frame):
        """Test normalize_unicode with forms='NFKC' and 'NFKD'"""
        result_nfkc = ar.normalize_unicode(unicode_frame, form="NFKC")
        df_nfkc = ar.to_pandas(result_nfkc)
        assert df_nfkc["text"] is not None

        result_nfkd = ar.normalize_unicode(unicode_frame, form="NFKD")
        df_nfkd = ar.to_pandas(result_nfkd)
        assert df_nfkd["text"] is not None

    def test_normalize_unicode_subset_parameter(self, unicode_frame):
        """Test normalize_unicode with subset parameter"""
        result = ar.normalize_unicode(unicode_frame, subset=["text"], form="NFC")
        df = ar.to_pandas(result)
        assert df["value"].tolist() == [1, 2, 3]

    def test_normalize_unicode_with_nulls(self):
        """Test normalize_unicode handles null values correctly"""
        df = pd.DataFrame({"text": ["café", None, "naïve"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame, form="NFC")
        result_df = ar.to_pandas(result)
        assert result_df["text"][0] is not None
        assert pd.isna(result_df["text"][1])
        assert result_df["text"][2] is not None

    def test_normalize_unicode_invalid_form(self, unicode_frame):
        """Test that invalid form raises ValueError"""
        with pytest.raises(ValueError, match="Unsupported Unicode normalization form"):
            ar.normalize_unicode(unicode_frame, form="INVALID")

    def test_normalize_unicode_form_not_string(self, unicode_frame):
        """Test that non-string form raises TypeError"""
        with pytest.raises(TypeError, match="form must be a string"):
            ar.normalize_unicode(unicode_frame, form=123)

    def test_normalize_unicode_nonexistent_column_in_subset(self, unicode_frame):
        """Test that subset with nonexistent column raises error"""
        with pytest.raises((KeyError, ValueError)):
            ar.normalize_unicode(unicode_frame, subset=["nonexistent"])

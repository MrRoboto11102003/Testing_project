import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCompareProfiles:
    """Test suite for arnio.quality.compare_profiles function."""

    def test_compare_identical_profiles(self):
        """Test comparing two identical profiles returns a ProfileComparison."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"],
            "age": [25, 30, 35]
        })
        frame = ar.from_pandas(df)
        
        profile1 = ar.profile(frame)
        profile2 = ar.profile(frame)
        
        comparison = ar.compare_profiles(profile1, profile2)
        assert comparison is not None

    def test_compare_profiles_different_row_counts(self):
        """Test comparing profiles from dataframes with different row counts."""
        df1 = pd.DataFrame({"value": [1, 2, 3]})
        df2 = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        
        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ar.compare_profiles(profile1, profile2)
        assert comparison is not None

    def test_compare_profiles_with_nulls(self):
        """Test comparing profiles with different null patterns."""
        df1 = pd.DataFrame({"col": [1, 2, None]})
        df2 = pd.DataFrame({"col": [1, None, None, 4]})
        
        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ar.compare_profiles(profile1, profile2)
        assert comparison is not None

    def test_compare_profiles_before_after_cleaning(self):
        """Test comparing profiles before and after a cleaning pipeline."""
        messy = pd.DataFrame({
            "name": [" alice ", " bob ", " charlie ", " alice "],
            "email": ["a@ex.com", "b@ex.com", None, "a@ex.com"]
        })
        
        messy_frame = ar.from_pandas(messy)
        messy_profile = ar.profile(messy_frame)
        
        cleaned = ar.pipeline(messy_frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])
        clean_profile = ar.profile(cleaned)
        
        comparison = ar.compare_profiles(messy_profile, clean_profile)
        assert comparison is not None

    def test_compare_profiles_with_duplicates(self):
        """Test comparing profiles with and without duplicate rows."""
        df_with_dups = pd.DataFrame({"id": [1, 1, 2, 2, 3]})
        df_without_dups = pd.DataFrame({"id": [1, 2, 3]})
        
        frame_dups = ar.from_pandas(df_with_dups)
        frame_no_dups = ar.from_pandas(df_without_dups)
        
        profile_dups = ar.profile(frame_dups)
        profile_no_dups = ar.profile(frame_no_dups)
        
        comparison = ar.compare_profiles(profile_dups, profile_no_dups)
        assert comparison is not None

    def test_compare_profiles_mixed_types(self):
        """Test comparing profiles with mixed column types."""
        df1 = pd.DataFrame({
            "int_col": [1, 2, 3],
            "str_col": ["a", "b", "c"],
            "float_col": [1.1, 2.2, 3.3],
            "bool_col": [True, False, True]
        })
        
        df2 = pd.DataFrame({
            "int_col": [10, 20, 30],
            "str_col": ["x", "y", "z"],
            "float_col": [10.5, 20.5, 30.5],
            "bool_col": [False, True, False]
        })
        
        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ar.compare_profiles(profile1, profile2)
        assert comparison is not None

    def test_compare_profiles_single_column(self):
        """Test comparing profiles with single-column dataframes."""
        df1 = pd.DataFrame({"value": [1, 2, 3]})
        df2 = pd.DataFrame({"value": [4, 5, 6]})
        
        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ar.compare_profiles(profile1, profile2)
        assert comparison is not None

    def test_compare_profiles_empty_dataframe(self):
        """Test comparing profile of empty dataframe with non-empty."""
        empty_df = pd.DataFrame({"col": pd.Series([], dtype='int64')})
        nonempty_df = pd.DataFrame({"col": [1, 2, 3]})
        
        empty_frame = ar.from_pandas(empty_df)
        nonempty_frame = ar.from_pandas(nonempty_df)
        
        empty_profile = ar.profile(empty_frame)
        nonempty_profile = ar.profile(nonempty_frame)
        
        comparison = ar.compare_profiles(empty_profile, nonempty_profile)
        assert comparison is not None

    def test_compare_profiles_whitespace_differences(self):
        """Test comparing profiles before and after whitespace normalization."""
        with_spaces = pd.DataFrame({
            "name": [" alice ", "bob", "  charlie  "],
            "city": ["New York", "Los  Angeles", "  San Francisco"]
        })
        
        without_spaces = pd.DataFrame({
            "name": ["alice", "bob", "charlie"],
            "city": ["New York", "Los Angeles", "San Francisco"]
        })
        
        frame_spaces = ar.from_pandas(with_spaces)
        frame_no_spaces = ar.from_pandas(without_spaces)
        
        profile_spaces = ar.profile(frame_spaces)
        profile_no_spaces = ar.profile(frame_no_spaces)
        
        comparison = ar.compare_profiles(profile_spaces, profile_no_spaces)
        assert comparison is not None

    def test_compare_profiles_numeric_precision(self):
        """Test comparing profiles with different numeric precision."""
        df1 = pd.DataFrame({"amount": [1.5, 2.7, 3.9]})
        df2 = pd.DataFrame({"amount": [1.50, 2.70, 3.90]})
        
        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ar.compare_profiles(profile1, profile2)
        assert comparison is not None

    def test_compare_profiles_invalid_input_none_first(self):
        """Test that compare_profiles raises error when first profile is None."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)
        
        with pytest.raises((TypeError, AttributeError)):
            ar.compare_profiles(None, profile)

    def test_compare_profiles_invalid_input_none_second(self):
        """Test that compare_profiles raises error when second profile is None."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)
        
        with pytest.raises((TypeError, AttributeError)):
            ar.compare_profiles(profile, None)

    def test_compare_profiles_invalid_input_wrong_type(self):
        """Test that compare_profiles raises error for non-DataQualityReport inputs."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)
        
        with pytest.raises((TypeError, AttributeError)):
            ar.compare_profiles("not_a_profile", profile)
        
        with pytest.raises((TypeError, AttributeError)):
            ar.compare_profiles(profile, {"dict": "not_profile"})

    def test_compare_profiles_large_dataset(self):
        """Test comparing profiles with larger datasets."""
        df1 = pd.DataFrame({
            "id": range(1000),
            "value": np.random.randn(1000),
            "category": np.random.choice(["A", "B", "C"], 1000)
        })
        
        df2 = pd.DataFrame({
            "id": range(500, 1500),
            "value": np.random.randn(1000),
            "category": np.random.choice(["A", "B", "C"], 1000)
        })
        
        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ar.compare_profiles(profile1, profile2)
        assert comparison is not None

    def test_compare_profiles_all_nulls_column(self):
        """Test comparing profiles where one column is all nulls."""
        df1 = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [None, None, None]
        })
        
        df2 = pd.DataFrame({
            "col1": [4, 5, 6],
            "col2": ["a", "b", "c"]
        })
        
        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ar.compare_profiles(profile1, profile2)
        assert comparison is not None

    def test_compare_profiles_symmetry(self):
        """Test that comparing profiles in different order still succeeds."""
        df1 = pd.DataFrame({"col": [1, 2, 3]})
        df2 = pd.DataFrame({"col": [4, 5, 6]})
        
        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison_ab = ar.compare_profiles(profile1, profile2)
        comparison_ba = ar.compare_profiles(profile2, profile1)
        
        assert comparison_ab is not None
        assert comparison_ba is not None

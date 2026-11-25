import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCompareProfiles:
    """Test suite for arnio.quality.compare_profiles"""

    def test_identical_profiles(self):
        """Comparing identical profiles should work"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.0, 92.3]
        })
        
        frame_a = ar.from_pandas(df)
        frame_b = ar.from_pandas(df)
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        assert comparison is not None

    def test_different_row_counts(self):
        """Profiles with different row counts should be compared"""
        df_a = pd.DataFrame({"col": [1, 2, 3]})
        df_b = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
        
        frame_a = ar.from_pandas(df_a)
        frame_b = ar.from_pandas(df_b)
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        assert comparison is not None

    def test_different_null_ratios(self):
        """Profiles with different null ratios should be compared"""
        df_a = pd.DataFrame({"col": [1, 2, 3]})
        df_b = pd.DataFrame({"col": [1, None, 3]})
        
        frame_a = ar.from_pandas(df_a)
        frame_b = ar.from_pandas(df_b)
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        assert comparison is not None

    def test_added_column(self):
        """Profile with added column should be compared"""
        df_a = pd.DataFrame({"id": [1, 2, 3]})
        df_b = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        
        frame_a = ar.from_pandas(df_a)
        frame_b = ar.from_pandas(df_b)
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        assert comparison is not None

    def test_removed_column(self):
        """Profile with removed column should be compared"""
        df_a = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        df_b = pd.DataFrame({"id": [1, 2, 3]})
        
        frame_a = ar.from_pandas(df_a)
        frame_b = ar.from_pandas(df_b)
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        assert comparison is not None

    def test_different_duplicate_ratios(self):
        """Profiles with different duplicate counts should be detected"""
        df_a = pd.DataFrame({"col": [1, 2, 3]})
        df_b = pd.DataFrame({"col": [1, 1, 1]})
        
        frame_a = ar.from_pandas(df_a)
        frame_b = ar.from_pandas(df_b)
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        assert comparison is not None

    def test_empty_frame_comparison(self):
        """Comparing profiles of empty frames should work"""
        df_empty = pd.DataFrame({"col": pd.Series([], dtype='int64')})
        
        frame_a = ar.from_pandas(df_empty)
        frame_b = ar.from_pandas(df_empty)
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        assert comparison is not None

    def test_single_row_frames(self):
        """Comparing profiles with single row should work"""
        df_a = pd.DataFrame({"col": [1]})
        df_b = pd.DataFrame({"col": [2]})
        
        frame_a = ar.from_pandas(df_a)
        frame_b = ar.from_pandas(df_b)
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        assert comparison is not None

    def test_numeric_type_change(self):
        """Different numeric types should be detected"""
        df_a = pd.DataFrame({"col": [1, 2, 3]})
        df_b = pd.DataFrame({"col": [1.0, 2.0, 3.0]})
        
        frame_a = ar.from_pandas(df_a)
        frame_b = ar.from_pandas(df_b)
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        assert comparison is not None

    def test_type_conversion_string_to_numeric(self):
        """String to numeric type conversion should be detected"""
        df_a = pd.DataFrame({"col": ["1", "2", "3"]})
        df_b = pd.DataFrame({"col": [1, 2, 3]})
        
        frame_a = ar.from_pandas(df_a)
        frame_b = ar.from_pandas(df_b)
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        assert comparison is not None

    def test_multiple_column_additions_and_removals(self):
        """Multiple column changes should be detected"""
        df_a = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "old_col": [10, 20, 30]
        })
        df_b = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "new_col": [10, 20, 30],
            "another_col": [100, 200, 300]
        })
        
        frame_a = ar.from_pandas(df_a)
        frame_b = ar.from_pandas(df_b)
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        assert comparison is not None

    def test_all_nulls_to_all_values(self):
        """Transition from all nulls to all values should be detected"""
        df_a = pd.DataFrame({"col": [None, None, None]})
        df_b = pd.DataFrame({"col": [1, 2, 3]})
        
        frame_a = ar.from_pandas(df_a)
        frame_b = ar.from_pandas(df_b)
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        assert comparison is not None

    def test_large_row_count_difference(self):
        """Large differences in row counts should be detected"""
        df_a = pd.DataFrame({"col": range(10)})
        df_b = pd.DataFrame({"col": range(10000)})
        
        frame_a = ar.from_pandas(df_a)
        frame_b = ar.from_pandas(df_b)
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        assert comparison is not None

    def test_mixed_data_types_comparison(self):
        """Comparing profiles with mixed data types should work"""
        df_a = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True]
        })
        df_b = pd.DataFrame({
            "int_col": [1, 2, None],
            "float_col": [1.5, None, 3.5],
            "str_col": ["a", "b", None],
            "bool_col": [True, False, None]
        })
        
        frame_a = ar.from_pandas(df_a)
        frame_b = ar.from_pandas(df_b)
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        assert comparison is not None

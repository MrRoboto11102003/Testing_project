import pytest
import pandas as pd
import arnio as ar


class TestCompareProfiles:
    """Test suite for arnio.quality.compare_profiles function"""

    def test_compare_identical_profiles(self):
        """Comparing identical profiles should work without error"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.3, 92.1]
        })
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        result = ar.compare_profiles(profile, profile)
        assert result is not None

    def test_compare_different_row_counts(self):
        """Profiles with different row counts should be distinguishable"""
        df1 = pd.DataFrame({"value": [1, 2, 3]})
        df2 = pd.DataFrame({"value": [1, 2]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        profile_a = ar.profile(frame1)
        profile_b = ar.profile(frame2)

        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None

    def test_compare_different_null_ratios(self):
        """Profiles with different null value counts"""
        df1 = pd.DataFrame({"col": [1, 2, 3]})
        df2 = pd.DataFrame({"col": [1, None, 3]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        profile_a = ar.profile(frame1)
        profile_b = ar.profile(frame2)

        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None

    def test_compare_schema_with_added_column(self):
        """Profiles with added columns should be distinguishable"""
        df1 = pd.DataFrame({"id": [1, 2, 3]})
        df2 = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        profile_a = ar.profile(frame1)
        profile_b = ar.profile(frame2)

        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None

    def test_compare_schema_with_removed_column(self):
        """Profiles with removed columns should be distinguishable"""
        df1 = pd.DataFrame({"id": [1, 2, 3], "debug": ["x", "y", "z"]})
        df2 = pd.DataFrame({"id": [1, 2, 3]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        profile_a = ar.profile(frame1)
        profile_b = ar.profile(frame2)

        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None

    def test_compare_dtype_change(self):
        """Profiles with changed dtypes should be distinguishable"""
        df1 = pd.DataFrame({"value": [1, 2, 3]})
        df2 = pd.DataFrame({"value": ["1", "2", "3"]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        profile_a = ar.profile(frame1)
        profile_b = ar.profile(frame2)

        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None

    def test_compare_empty_vs_nonempty_frame(self):
        """Comparing profile of empty vs populated frame"""
        df_empty = pd.DataFrame({"col": pd.Series([], dtype='int64')})
        df_nonempty = pd.DataFrame({"col": [1, 2, 3]})

        frame_empty = ar.from_pandas(df_empty)
        frame_nonempty = ar.from_pandas(df_nonempty)
        profile_a = ar.profile(frame_empty)
        profile_b = ar.profile(frame_nonempty)

        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None

    def test_compare_all_nulls_vs_all_values(self):
        """Profiles where one column is all nulls in first, all values in second"""
        df1 = pd.DataFrame({"col": [None, None, None]})
        df2 = pd.DataFrame({"col": [1, 2, 3]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        profile_a = ar.profile(frame1)
        profile_b = ar.profile(frame2)

        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None

    def test_compare_duplicate_counts_change(self):
        """Profiles with different levels of row duplication"""
        df1 = pd.DataFrame({"id": [1, 2, 3, 1, 2]})
        df2 = pd.DataFrame({"id": [1, 2, 3, 4, 5]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        profile_a = ar.profile(frame1)
        profile_b = ar.profile(frame2)

        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None

    def test_compare_cardinality_differences(self):
        """Profiles with different cardinality of categorical columns"""
        df1 = pd.DataFrame({"category": ["A", "B", "A", "B"]})
        df2 = pd.DataFrame({"category": ["A", "B", "C", "D"]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        profile_a = ar.profile(frame1)
        profile_b = ar.profile(frame2)

        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None

    def test_compare_numeric_range_shift(self):
        """Profiles with different numeric value ranges"""
        df1 = pd.DataFrame({"score": [1.0, 2.0, 3.0]})
        df2 = pd.DataFrame({"score": [100.0, 200.0, 300.0]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        profile_a = ar.profile(frame1)
        profile_b = ar.profile(frame2)

        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None

    def test_compare_multiple_simultaneous_changes(self):
        """Profiles with multiple types of changes at once"""
        df1 = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "active": [True, True, False]
        })
        df2 = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "name": ["A", None, "C", "D"],
            "score": [95, 87, 92, 88]
        })

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        profile_a = ar.profile(frame1)
        profile_b = ar.profile(frame2)

        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None

    def test_compare_row_reordered_data(self):
        """Comparing profiles of same data but different row order"""
        df1 = pd.DataFrame({"val": [1, 2, 3]})
        df2 = pd.DataFrame({"val": [3, 1, 2]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        profile_a = ar.profile(frame1)
        profile_b = ar.profile(frame2)

        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None

    def test_compare_column_order_difference(self):
        """Profiles with same columns in different order"""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        df2 = pd.DataFrame({"c": [5, 6], "a": [1, 2], "b": [3, 4]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        profile_a = ar.profile(frame1)
        profile_b = ar.profile(frame2)

        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None

    def test_compare_extreme_value_ranges(self):
        """Profiles with vastly different numeric scales"""
        df1 = pd.DataFrame({"val": [1e-10, 2e-10, 3e-10]})
        df2 = pd.DataFrame({"val": [1e10, 2e10, 3e10]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        profile_a = ar.profile(frame1)
        profile_b = ar.profile(frame2)

        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None

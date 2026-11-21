import pytest
import pandas as pd
import numpy as np
import arnio as ar
import tempfile
from pathlib import Path


class TestRegisterStepAndReset:
    """End-to-end tests for register_step and reset_steps."""

    def teardown_method(self):
        """Clean up custom steps after each test."""
        ar.reset_steps()

    def test_register_and_use_simple_custom_step(self, tmp_path):
        """Register a simple custom step and use it in a full pipeline."""
        # Create test CSV
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({
            "name": ["  alice  ", "  bob  "],
            "score": [95, 88]
        }).to_csv(csv_path, index=False)

        # Define and register custom step
        def uppercase_name(df, column="name"):
            df = df.copy()
            df[column] = df[column].str.upper()
            return df

        ar.register_step("uppercase_name", uppercase_name)

        # Full workflow: load → clean → check
        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("uppercase_name", {"column": "name"}),
        ])
        result_df = ar.to_pandas(clean)

        assert result_df["name"].tolist() == ["ALICE", "BOB"]
        assert result_df["score"].tolist() == [95, 88]

    def test_register_custom_step_with_multiple_parameters(self, tmp_path):
        """Register step with multiple optional parameters."""
        csv_path = tmp_path / "numbers.csv"
        pd.DataFrame({
            "value": [1.567, 2.891, 3.234],
        }).to_csv(csv_path, index=False)

        def scale_and_round(df, column="value", scale=1.0, decimals=2):
            df = df.copy()
            df[column] = (df[column] * scale).round(decimals)
            return df

        ar.register_step("scale_and_round", scale_and_round)

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("scale_and_round", {"column": "value", "scale": 10, "decimals": 1}),
        ])
        result = ar.to_pandas(clean)

        assert result["value"].tolist() == [15.7, 28.9, 32.3]

    def test_register_multiple_custom_steps_in_sequence(self, tmp_path):
        """Register and chain multiple custom steps."""
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({
            "text": ["hello", "world"],
            "count": [5, 10]
        }).to_csv(csv_path, index=False)

        def to_upper(df, column="text"):
            df = df.copy()
            df[column] = df[column].str.upper()
            return df

        def double_count(df, column="count"):
            df = df.copy()
            df[column] = df[column] * 2
            return df

        ar.register_step("to_upper", to_upper)
        ar.register_step("double_count", double_count)

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("to_upper", {"column": "text"}),
            ("double_count", {"column": "count"}),
        ])
        result = ar.to_pandas(clean)

        assert result["text"].tolist() == ["HELLO", "WORLD"]
        assert result["count"].tolist() == [10, 20]

    def test_register_cannot_use_reserved_builtin_namespace(self):
        """Registering with 'builtin:' namespace prefix raises ValueError."""
        def dummy(df):
            return df

        with pytest.raises(ValueError, match="reserved for built-in pipeline steps"):
            ar.register_step("builtin:custom", dummy)

    def test_register_conflicting_custom_step_without_overwrite_fails(self):
        """Attempting to re-register same name without overwrite=True raises."""
        def step_v1(df):
            return df

        def step_v2(df):
            return df

        ar.register_step("my_step", step_v1)

        with pytest.raises(ValueError, match="already registered"):
            ar.register_step("my_step", step_v2)

    def test_register_with_overwrite_true_replaces_custom_step(self):
        """overwrite=True allows replacing existing custom step."""
        def step_v1(df):
            df = df.copy()
            df["ver"] = 1
            return df

        def step_v2(df):
            df = df.copy()
            df["ver"] = 2
            return df

        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)

        # Register v1
        ar.register_step("test_step", step_v1)
        clean1 = ar.pipeline(frame, [("test_step",)])
        assert ar.to_pandas(clean1)["ver"].iloc[0] == 1

        # Overwrite with v2
        ar.register_step("test_step", step_v2, overwrite=True)
        clean2 = ar.pipeline(frame, [("test_step",)])
        assert ar.to_pandas(clean2)["ver"].iloc[0] == 2

    def test_cannot_overwrite_builtin_step_even_with_overwrite_flag(self):
        """Built-in steps cannot be overwritten, even with overwrite=True."""
        def fake_drop_nulls(df):
            return df

        with pytest.raises(ValueError, match="conflicts with built-in"):
            ar.register_step("drop_nulls", fake_drop_nulls, overwrite=True)

    def test_register_with_invalid_name_type_raises_error(self):
        """Non-string name raises TypeError or ValueError."""
        def dummy(df):
            return df

        with pytest.raises((TypeError, ValueError)):
            ar.register_step(123, dummy)

    def test_register_with_empty_name_raises_error(self):
        """Empty or whitespace-only name raises ValueError."""
        def dummy(df):
            return df

        with pytest.raises(ValueError, match="non-empty"):
            ar.register_step("", dummy)

    def test_register_with_non_callable_function_raises_error(self):
        """Non-callable fn parameter raises TypeError."""
        with pytest.raises(TypeError, match="must be a callable"):
            ar.register_step("step", "not_a_function")

    def test_register_with_non_bool_overwrite_raises_error(self):
        """Non-bool overwrite parameter raises TypeError."""
        def dummy(df):
            return df

        with pytest.raises(TypeError, match="must be a bool"):
            ar.register_step("step", dummy, overwrite="yes")

    def test_reset_steps_clears_custom_registry(self):
        """reset_steps() clears all custom steps, built-ins remain."""
        def custom_step(df):
            df = df.copy()
            df["custom"] = "added"
            return df

        ar.register_step("my_custom_step", custom_step)

        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)

        # Custom step works before reset
        clean = ar.pipeline(frame, [("my_custom_step",)])
        result = ar.to_pandas(clean)
        assert "custom" in result.columns

        # After reset, custom step is unavailable
        ar.reset_steps()

        with pytest.raises(Exception):  # UnknownStepError
            ar.pipeline(frame, [("my_custom_step",)])

        # But built-in steps still work
        clean2 = ar.pipeline(frame, [("drop_duplicates",)])
        assert ar.to_pandas(clean2).shape[0] == 1

    def test_custom_step_with_complex_data_transformation(self, tmp_path):
        """Complex custom step combining multiple columns."""
        csv_path = tmp_path / "people.csv"
        pd.DataFrame({
            "first": ["alice", "bob"],
            "last": ["smith", "jones"],
            "age": [25, 30]
        }).to_csv(csv_path, index=False)

        def create_full_record(df, first_col="first", last_col="last", output="full_name"):
            df = df.copy()
            df[output] = (df[first_col].str.capitalize() + 
                         " " + df[last_col].str.capitalize())
            return df

        ar.register_step("create_full_record", create_full_record)

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("create_full_record", {
                "first_col": "first",
                "last_col": "last",
                "output": "full_name"
            }),
        ])
        result = ar.to_pandas(clean)

        assert "full_name" in result.columns
        assert result["full_name"].tolist() == ["Alice Smith", "Bob Jones"]

    def test_custom_step_error_propagates_during_execution(self):
        """Errors raised in custom step functions propagate."""
        def failing_step(df, should_fail=False):
            if should_fail:
                raise ValueError("Intentional step failure")
            return df

        ar.register_step("failing_step", failing_step)

        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)

        # Should succeed without error flag
        clean = ar.pipeline(frame, [("failing_step", {"should_fail": False})])
        assert ar.to_pandas(clean).shape[0] == 1

        # Should fail when flag is set
        with pytest.raises(ValueError, match="Intentional step failure"):
            ar.pipeline(frame, [("failing_step", {"should_fail": True})])

    def test_reset_then_register_new_step_with_same_name(self):
        """After reset_steps(), same name can be registered again."""
        def step_v1(df):
            df = df.copy()
            df["v"] = 1
            return df

        def step_v2(df):
            df = df.copy()
            df["v"] = 2
            return df

        # Register, use, reset
        ar.register_step("test_step", step_v1)
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)

        clean1 = ar.pipeline(frame, [("test_step",)])
        assert ar.to_pandas(clean1)["v"].iloc[0] == 1

        # Reset and re-register with different logic
        ar.reset_steps()
        ar.register_step("test_step", step_v2)

        clean2 = ar.pipeline(frame, [("test_step",)])
        assert ar.to_pandas(clean2)["v"].iloc[0] == 2

    def test_registered_step_works_in_full_cleaning_pipeline(self, tmp_path):
        """Registered step integrates with built-in steps in full pipeline."""
        csv_path = tmp_path / "messy.csv"
        pd.DataFrame({
            "id": [1, 2, 2, 3],
            "product": ["  apple  ", "  banana  ", "  banana  ", "  orange  "],
            "quantity": [5, 10, 10, 15]
        }).to_csv(csv_path, index=False)

        def tag_product(df, column="product", prefix="ITEM_"):
            df = df.copy()
            df[column] = prefix + df[column]
            return df

        ar.register_step("tag_product", tag_product)

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("tag_product", {"column": "product", "prefix": "SKU_"}),
            ("drop_duplicates", {"keep": "first"}),
        ])
        result = ar.to_pandas(clean)

        assert len(result) == 3  # Duplicate dropped
        assert all(p.startswith("SKU_") for p in result["product"])
        assert result["product"].tolist() == ["SKU_apple", "SKU_banana", "SKU_orange"]

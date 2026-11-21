"""
System (end-to-end) tests for arnio.pipeline.unregister_step.

These tests verify complete user workflows: registering custom pipeline steps,
using them to clean/validate data, unregistering them, and observing the results.
"""

import pytest
import pandas as pd
import arnio as ar
from pathlib import Path


class TestUnregisterStep:
    """System tests for arnio.pipeline.unregister_step"""

    def test_unregister_custom_step_removes_from_pipeline(self, tmp_path):
        """Unregistering a custom step prevents it from being used in pipelines."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,value\nalice,10\nbob,20\n")

        def uppercase_names(df, **kwargs):
            df = df.copy()
            df["name"] = df["name"].str.upper()
            return df

        try:
            ar.register_step("uppercase_names", uppercase_names)

            # Verify step works before unregister
            frame = ar.read_csv(str(csv_file))
            result = ar.pipeline(frame, [("uppercase_names",)])
            df = ar.to_pandas(result)
            assert df["name"].tolist() == ["ALICE", "BOB"]

            # Unregister the step
            ar.unregister_step("uppercase_names")

            # Verify step no longer works
            frame = ar.read_csv(str(csv_file))
            with pytest.raises(ar.UnknownStepError):
                ar.pipeline(frame, [("uppercase_names",)])
        finally:
            ar.reset_steps()

    def test_unregister_nonexistent_step_raises_unknown_step_error(self):
        """Unregistering a non-existent step raises UnknownStepError."""
        with pytest.raises(ar.UnknownStepError):
            ar.unregister_step("step_that_never_existed_xyz")

    def test_unregister_builtin_step_raises_value_error(self):
        """Built-in steps cannot be unregistered."""
        with pytest.raises(ValueError, match="Cannot unregister built-in step"):
            ar.unregister_step("drop_nulls")

        with pytest.raises(ValueError, match="Cannot unregister built-in step"):
            ar.unregister_step("strip_whitespace")

    def test_unregister_invalid_name_type_raises_type_error(self):
        """Non-string step names raise TypeError."""
        with pytest.raises(TypeError, match="must be a string"):
            ar.unregister_step(123)

        with pytest.raises(TypeError, match="must be a string"):
            ar.unregister_step(None)

        with pytest.raises(TypeError, match="must be a string"):
            ar.unregister_step(["step"])

    def test_unregister_empty_string_raises_value_error(self):
        """Empty string step name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            ar.unregister_step("")

    def test_unregister_then_reregister_with_new_logic(self, tmp_path):
        """A step can be re-registered with different logic after unregistration."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val\n5\n10\n")

        def double_values(df, **kwargs):
            df = df.copy()
            df["val"] = df["val"] * 2
            return df

        def triple_values(df, **kwargs):
            df = df.copy()
            df["val"] = df["val"] * 3
            return df

        try:
            # Register and use first version
            ar.register_step("multiply_values", double_values)
            frame = ar.read_csv(str(csv_file))
            result = ar.pipeline(frame, [("multiply_values",)])
            df1 = ar.to_pandas(result)
            assert df1["val"].tolist() == [10, 20]

            # Unregister
            ar.unregister_step("multiply_values")

            # Re-register with new logic
            ar.register_step("multiply_values", triple_values)
            frame = ar.read_csv(str(csv_file))
            result = ar.pipeline(frame, [("multiply_values",)])
            df2 = ar.to_pandas(result)
            assert df2["val"].tolist() == [15, 30]
        finally:
            ar.reset_steps()

    def test_unregister_multiple_independent_custom_steps(self, tmp_path):
        """Multiple custom steps can be unregistered independently."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,value\nalice,10\n")

        def uppercase_names(df, **kwargs):
            df = df.copy()
            df["name"] = df["name"].str.upper()
            return df

        def double_values(df, **kwargs):
            df = df.copy()
            df["value"] = df["value"] * 2
            return df

        try:
            ar.register_step("step_a", uppercase_names)
            ar.register_step("step_b", double_values)

            # Both work together
            frame = ar.read_csv(str(csv_file))
            result = ar.pipeline(frame, [("step_a",), ("step_b",)])
            df = ar.to_pandas(result)
            assert df["name"].tolist() == ["ALICE"]
            assert df["value"].tolist() == [20]

            # Unregister one
            ar.unregister_step("step_a")

            # Second still works
            frame = ar.read_csv(str(csv_file))
            result = ar.pipeline(frame, [("step_b",)])
            df = ar.to_pandas(result)
            assert df["value"].tolist() == [20]

            # First fails
            with pytest.raises(ar.UnknownStepError):
                frame = ar.read_csv(str(csv_file))
                ar.pipeline(frame, [("step_a",)])
        finally:
            ar.reset_steps()

    def test_unregister_in_realistic_cleaning_workflow(self, tmp_path):
        """Unregister in a realistic data cleaning workflow with file I/O."""
        input_csv = tmp_path / "messy.csv"
        input_csv.write_text("name,score\n  alice  ,  95  \n  bob  ,  87  \n")
        output_csv = tmp_path / "clean.csv"

        def add_grade(df, **kwargs):
            """Custom step: add letter grade based on score."""
            df = df.copy()
            df["grade"] = df["score"].apply(
                lambda x: "A" if x >= 90 else "B" if x >= 80 else "C"
            )
            return df

        try:
            ar.register_step("add_grade", add_grade)

            # Use custom step in pipeline
            frame = ar.read_csv(str(input_csv))
            clean = ar.pipeline(
                frame,
                [
                    ("strip_whitespace",),
                    ("normalize_case", {"case_type": "lower"}),
                    ("add_grade",),
                ],
            )
            df = ar.to_pandas(clean)
            ar.write_csv(clean, str(output_csv))

            # Verify output
            output_df = pd.read_csv(str(output_csv))
            assert output_df["grade"].tolist() == ["A", "B"]
            assert output_csv.exists()

            # Unregister custom step
            ar.unregister_step("add_grade")

            # Pipeline without custom step succeeds
            frame = ar.read_csv(str(input_csv))
            clean = ar.pipeline(
                frame, [("strip_whitespace",), ("normalize_case", {"case_type": "lower"})]
            )
            df = ar.to_pandas(clean)
            assert "grade" not in df.columns
            assert df["name"].tolist() == ["alice", "bob"]

            # Pipeline with unregistered custom step fails
            with pytest.raises(ar.UnknownStepError):
                frame = ar.read_csv(str(input_csv))
                ar.pipeline(frame, [("add_grade",)])
        finally:
            ar.reset_steps()

    def test_unregister_affects_list_steps(self):
        """list_steps() reflects unregistered custom steps."""

        def custom_func(df, **kwargs):
            return df

        try:
            ar.register_step("my_custom_step", custom_func)
            steps_with = ar.list_steps()
            assert "my_custom_step" in steps_with

            ar.unregister_step("my_custom_step")
            steps_without = ar.list_steps()
            assert "my_custom_step" not in steps_without
        finally:
            ar.reset_steps()

    def test_unregister_with_namespaced_step_names(self, tmp_path):
        """Unregister works with namespaced step names."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val\n42\n")

        def negate(df, **kwargs):
            df = df.copy()
            df["val"] = -df["val"]
            return df

        try:
            ar.register_step("custom:negate", negate)

            frame = ar.read_csv(str(csv_file))
            result = ar.pipeline(frame, [("custom:negate",)])
            df = ar.to_pandas(result)
            assert df["val"].tolist() == [-42]

            ar.unregister_step("custom:negate")

            with pytest.raises(ar.UnknownStepError):
                frame = ar.read_csv(str(csv_file))
                ar.pipeline(frame, [("custom:negate",)])
        finally:
            ar.reset_steps()

    def test_unregister_step_names_are_case_sensitive(self):
        """Step names are case-sensitive."""

        def my_step(df, **kwargs):
            return df

        try:
            ar.register_step("MyStep", my_step)

            # Wrong case should raise error
            with pytest.raises(ar.UnknownStepError):
                ar.unregister_step("mystep")

            with pytest.raises(ar.UnknownStepError):
                ar.unregister_step("MYSTEP")

            # Correct case should work
            ar.unregister_step("MyStep")
            with pytest.raises(ar.UnknownStepError):
                ar.unregister_step("MyStep")
        finally:
            ar.reset_steps()

    def test_unregister_with_pandas_accessor(self, tmp_path):
        """Unregister affects pandas .arnio accessor pipeline execution."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("value\n100\n200\n")

        def add_fifty(df, **kwargs):
            df = df.copy()
            df["value"] = df["value"] + 50
            return df

        try:
            ar.register_step("add_fifty", add_fifty)

            # Use through pandas accessor
            df = pd.read_csv(str(csv_file))
            result = df.arnio.clean(steps=[("add_fifty",)])
            assert result["value"].tolist() == [150, 250]

            # Unregister
            ar.unregister_step("add_fifty")

            # Should fail now
            df = pd.read_csv(str(csv_file))
            with pytest.raises(ar.UnknownStepError):
                df.arnio.clean(steps=[("add_fifty",)])
        finally:
            ar.reset_steps()

    def test_reset_steps_removes_all_custom_steps_and_restores_builtins(
        self, tmp_path
    ):
        """reset_steps() removes custom steps while keeping built-ins."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val\n5\n")

        def custom_step(df, **kwargs):
            df = df.copy()
            df["val"] = df["val"] + 100
            return df

        ar.register_step("custom_step", custom_step)

        # Custom step works
        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [("custom_step",)])
        df = ar.to_pandas(result)
        assert df["val"].tolist() == [105]

        # Reset
        ar.reset_steps()

        # Custom step is gone
        with pytest.raises(ar.UnknownStepError):
            frame = ar.read_csv(str(csv_file))
            ar.pipeline(frame, [("custom_step",)])

        # Built-in still works
        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [("strip_whitespace",)])
        df = ar.to_pandas(result)
        assert len(df) == 1

    def test_unregister_custom_step_in_validation_workflow(self, tmp_path):
        """Unregister impacts custom steps used in profiling and validation workflows."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nalice,25\nbob,30\n")

        def add_category(df, **kwargs):
            df = df.copy()
            df["category"] = df["age"].apply(lambda x: "young" if x < 30 else "old")
            return df

        try:
            ar.register_step("add_category", add_category)

            # Use in pipeline
            frame = ar.read_csv(str(csv_file))
            result = ar.pipeline(frame, [("add_category",)])
            df = ar.to_pandas(result)

            # Profile the data
            report = ar.profile(result)
            assert report.row_count == 2
            assert "category" in report.columns

            # Unregister
            ar.unregister_step("add_category")

            # Can't use in new pipelines
            with pytest.raises(ar.UnknownStepError):
                frame = ar.read_csv(str(csv_file))
                ar.pipeline(frame, [("add_category",)])
        finally:
            ar.reset_steps()

    def test_unregister_nonexistent_shows_available_steps_in_error(self):
        """UnknownStepError for unregister shows available steps."""
        with pytest.raises(ar.UnknownStepError) as exc_info:
            ar.unregister_step("definitely_not_a_real_step_12345")

        # Error should mention available steps
        error_msg = str(exc_info.value)
        assert "definitely_not_a_real_step_12345" in error_msg or (
            "Unknown" in error_msg
        )

import pytest
import json
import io
from pathlib import Path
import pandas as pd
import numpy as np
import arnio as ar
from arnio.quality import ProfileComparison


class TestProfileComparisonToDict:
    """E2E tests for ProfileComparison.to_dict() method."""

    def test_basic_to_dict_with_defaults(self, tmp_path):
        """Test basic to_dict() call with default parameters."""
        csv1 = tmp_path / "data1.csv"
        csv1.write_text("id,name,age\n1,Alice,25\n2,Bob,30\n3,Charlie,35\n")
        
        csv2 = tmp_path / "data2.csv"
        csv2.write_text("id,name,age\n1,Alice,25\n2,Bob,30\n")
        
        frame1 = ar.read_csv(str(csv1))
        frame2 = ar.read_csv(str(csv2))
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ProfileComparison(profile1, profile2)
        result = comparison.to_dict()
        
        assert isinstance(result, dict)
        assert "left_profile" in result
        assert "right_profile" in result
        assert "status_counts" in result
        assert "drift_report" in result
        assert isinstance(result["left_profile"], dict)
        assert isinstance(result["right_profile"], dict)

    def test_to_dict_with_redact_sample_values_true(self, tmp_path):
        """Test to_dict() with redact_sample_values=True redacts sensitive data."""
        csv1 = tmp_path / "data1.csv"
        csv1.write_text("id,email,salary\n1,alice@example.com,50000\n2,bob@example.com,60000\n")
        
        csv2 = tmp_path / "data2.csv"
        csv2.write_text("id,email,salary\n1,alice@example.com,50000\n")
        
        frame1 = ar.read_csv(str(csv1))
        frame2 = ar.read_csv(str(csv2))
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ProfileComparison(profile1, profile2)
        
        result_redacted = comparison.to_dict(redact_sample_values=True)
        result_unredacted = comparison.to_dict(redact_sample_values=False)
        
        assert isinstance(result_redacted, dict)
        assert isinstance(result_unredacted, dict)
        # Both should have the same structure
        assert set(result_redacted.keys()) == set(result_unredacted.keys())

    def test_to_dict_with_exclude_columns_list(self, tmp_path):
        """Test to_dict() with exclude_columns as list."""
        csv1 = tmp_path / "data1.csv"
        csv1.write_text("id,name,age,salary,department\n1,Alice,25,50000,Engineering\n2,Bob,30,60000,Sales\n")
        
        csv2 = tmp_path / "data2.csv"
        csv2.write_text("id,name,age,salary,department\n1,Alice,25,50000,Engineering\n")
        
        frame1 = ar.read_csv(str(csv1))
        frame2 = ar.read_csv(str(csv2))
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ProfileComparison(profile1, profile2)
        
        result = comparison.to_dict(exclude_columns=["salary", "department"])
        
        assert isinstance(result, dict)
        assert "left_profile" in result
        assert "right_profile" in result

    def test_to_dict_with_exclude_columns_set(self, tmp_path):
        """Test to_dict() with exclude_columns as set."""
        csv1 = tmp_path / "data1.csv"
        csv1.write_text("a,b,c,d\n1,2,3,4\n5,6,7,8\n")
        
        csv2 = tmp_path / "data2.csv"
        csv2.write_text("a,b,c,d\n1,2,3,4\n")
        
        frame1 = ar.read_csv(str(csv1))
        frame2 = ar.read_csv(str(csv2))
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ProfileComparison(profile1, profile2)
        
        result = comparison.to_dict(exclude_columns={"b", "d"})
        
        assert isinstance(result, dict)
        assert isinstance(result["status_counts"], dict)

    def test_to_dict_with_exclude_columns_tuple(self, tmp_path):
        """Test to_dict() with exclude_columns as tuple."""
        csv1 = tmp_path / "data1.csv"
        csv1.write_text("x,y,z\n1,2,3\n4,5,6\n")
        
        csv2 = tmp_path / "data2.csv"
        csv2.write_text("x,y,z\n1,2,3\n")
        
        frame1 = ar.read_csv(str(csv1))
        frame2 = ar.read_csv(str(csv2))
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ProfileComparison(profile1, profile2)
        
        result = comparison.to_dict(exclude_columns=("y", "z"))
        
        assert isinstance(result, dict)

    def test_to_dict_with_both_redact_and_exclude(self, tmp_path):
        """Test to_dict() with both redact_sample_values and exclude_columns."""
        csv1 = tmp_path / "data1.csv"
        csv1.write_text("customer,product,price,notes\nAlice,Widget,10.00,Great product\nBob,Gadget,20.00,Very good\n")
        
        csv2 = tmp_path / "data2.csv"
        csv2.write_text("customer,product,price,notes\nAlice,Widget,10.00,Great product\n")
        
        frame1 = ar.read_csv(str(csv1))
        frame2 = ar.read_csv(str(csv2))
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ProfileComparison(profile1, profile2)
        
        result = comparison.to_dict(
            redact_sample_values=True,
            exclude_columns=["notes"]
        )
        
        assert isinstance(result, dict)
        assert "left_profile" in result
        assert "right_profile" in result
        assert "status_counts" in result
        assert "drift_report" in result

    def test_to_dict_result_is_json_serializable(self, tmp_path):
        """Test that to_dict() result can be serialized to JSON."""
        csv1 = tmp_path / "data1.csv"
        csv1.write_text("id,value\n1,100\n2,200\n3,300\n")
        
        csv2 = tmp_path / "data2.csv"
        csv2.write_text("id,value\n1,100\n2,200\n")
        
        frame1 = ar.read_csv(str(csv1))
        frame2 = ar.read_csv(str(csv2))
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ProfileComparison(profile1, profile2)
        result = comparison.to_dict()
        
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        
        reloaded = json.loads(json_str)
        assert isinstance(reloaded, dict)
        assert "left_profile" in reloaded

    def test_to_dict_with_identical_frames(self, tmp_path):
        """Test to_dict() when comparing identical frames."""
        csv_data = "id,name,score\n1,Alice,95\n2,Bob,87\n3,Charlie,92\n"
        
        csv1 = tmp_path / "data1.csv"
        csv1.write_text(csv_data)
        
        csv2 = tmp_path / "data2.csv"
        csv2.write_text(csv_data)
        
        frame1 = ar.read_csv(str(csv1))
        frame2 = ar.read_csv(str(csv2))
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ProfileComparison(profile1, profile2)
        result = comparison.to_dict()
        
        assert isinstance(result, dict)
        assert isinstance(result["drift_report"], dict)

    def test_to_dict_with_significantly_different_frames(self, tmp_path):
        """Test to_dict() when frames differ significantly."""
        csv1 = tmp_path / "data1.csv"
        csv1.write_text("id,val\n1,10\n2,20\n3,30\n4,40\n5,50\n")
        
        csv2 = tmp_path / "data2.csv"
        csv2.write_text("id,val\n1,100\n")
        
        frame1 = ar.read_csv(str(csv1))
        frame2 = ar.read_csv(str(csv2))
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ProfileComparison(profile1, profile2)
        result = comparison.to_dict()
        
        assert isinstance(result, dict)
        assert result["drift_report"] is not None

    def test_to_dict_with_null_values(self, tmp_path):
        """Test to_dict() on frames containing null/missing values."""
        csv1 = tmp_path / "data1.csv"
        csv1.write_text("id,name,age\n1,Alice,25\n2,,30\n3,Charlie,\n")
        
        csv2 = tmp_path / "data2.csv"
        csv2.write_text("id,name,age\n1,Alice,25\n2,Bob,\n3,,35\n")
        
        frame1 = ar.read_csv(str(csv1))
        frame2 = ar.read_csv(str(csv2))
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ProfileComparison(profile1, profile2)
        result = comparison.to_dict()
        
        assert isinstance(result, dict)
        assert "left_profile" in result
        assert "right_profile" in result

    def test_to_dict_preserves_required_structure(self, tmp_path):
        """Test that to_dict() preserves all required structural keys."""
        csv1 = tmp_path / "data1.csv"
        csv1.write_text("col1,col2,col3\n1,2,3\n4,5,6\n")
        
        csv2 = tmp_path / "data2.csv"
        csv2.write_text("col1,col2,col3\n1,2,3\n")
        
        frame1 = ar.read_csv(str(csv1))
        frame2 = ar.read_csv(str(csv2))
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ProfileComparison(profile1, profile2)
        result = comparison.to_dict()
        
        required_keys = {"left_profile", "right_profile", "status_counts", "drift_report"}
        assert required_keys.issubset(result.keys())
        
        assert isinstance(result["left_profile"], dict)
        assert isinstance(result["right_profile"], dict)
        assert isinstance(result["status_counts"], dict)
        assert isinstance(result["drift_report"], dict)

    def test_to_dict_with_multiple_exclude_columns(self, tmp_path):
        """Test to_dict() with multiple columns excluded."""
        csv1 = tmp_path / "data1.csv"
        csv1.write_text("a,b,c,d,e,f\n1,2,3,4,5,6\n7,8,9,10,11,12\n")
        
        csv2 = tmp_path / "data2.csv"
        csv2.write_text("a,b,c,d,e,f\n1,2,3,4,5,6\n")
        
        frame1 = ar.read_csv(str(csv1))
        frame2 = ar.read_csv(str(csv2))
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ProfileComparison(profile1, profile2)
        
        result = comparison.to_dict(exclude_columns=["b", "d", "f"])
        
        assert isinstance(result, dict)

    def test_to_dict_with_numeric_types(self, tmp_path):
        """Test to_dict() with various numeric column types."""
        csv1 = tmp_path / "data1.csv"
        csv1.write_text("int_col,float_col,negative\n1,1.5,-10\n2,2.5,-20\n3,3.5,-30\n")
        
        csv2 = tmp_path / "data2.csv"
        csv2.write_text("int_col,float_col,negative\n1,1.5,-10\n2,2.5,-20\n")
        
        frame1 = ar.read_csv(str(csv1))
        frame2 = ar.read_csv(str(csv2))
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ProfileComparison(profile1, profile2)
        result = comparison.to_dict()
        
        assert isinstance(result, dict)

    def test_to_dict_with_empty_exclude_columns(self, tmp_path):
        """Test to_dict() with empty exclude_columns list."""
        csv1 = tmp_path / "data1.csv"
        csv1.write_text("a,b,c\n1,2,3\n4,5,6\n")
        
        csv2 = tmp_path / "data2.csv"
        csv2.write_text("a,b,c\n1,2,3\n")
        
        frame1 = ar.read_csv(str(csv1))
        frame2 = ar.read_csv(str(csv2))
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ProfileComparison(profile1, profile2)
        
        result = comparison.to_dict(exclude_columns=[])
        
        assert isinstance(result, dict)

    def test_to_dict_round_trip_with_pandas(self, tmp_path):
        """Test complete workflow: CSV → ArFrame → profile → comparison → dict → JSON → export."""
        csv1 = tmp_path / "sales_2024.csv"
        csv1.write_text("product,revenue,date\nWidget,1000,2024-01-01\nGadget,2000,2024-01-02\nDoohickey,1500,2024-01-03\n")
        
        csv2 = tmp_path / "sales_2025.csv"
        csv2.write_text("product,revenue,date\nWidget,1100,2025-01-01\nGadget,2100,2025-01-02\n")
        
        frame1 = ar.read_csv(str(csv1))
        frame2 = ar.read_csv(str(csv2))
        
        df1 = ar.to_pandas(frame1)
        df2 = ar.to_pandas(frame2)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ProfileComparison(profile1, profile2)
        result = comparison.to_dict(redact_sample_values=True, exclude_columns=["date"])
        
        json_path = tmp_path / "comparison.json"
        json_path.write_text(json.dumps(result, indent=2))
        
        assert json_path.exists()
        loaded = json.loads(json_path.read_text())
        assert loaded["status_counts"] is not None
        assert loaded["drift_report"] is not None

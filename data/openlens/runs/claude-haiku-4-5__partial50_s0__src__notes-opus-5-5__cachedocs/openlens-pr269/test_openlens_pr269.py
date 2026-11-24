import pytest
import math
import dataclasses
from openlens.chromatic_analyzer import ChromaticResult


class TestChromaticResultToDict:
    """Test suite for ChromaticResult.to_dict() method"""

    def test_to_dict_basic_values(self):
        """Test to_dict with normal floating-point values"""
        result = ChromaticResult(
            wavelengths=[486.1, 587.6, 656.3],
            focal_lengths=[97.0, 98.0, 98.5],
            focal_shift=1.5,
            lateral_color=0.02,
            spot_sizes=[0.1, 0.05, 0.08],
            transverse_aberration=[0.2, 0.1, 0.15],
            axial_chromatic_aberration=1.5
        )
        d = result.to_dict()
        
        assert d["wavelengths"] == [486.1, 587.6, 656.3]
        assert d["focal_lengths"] == [97.0, 98.0, 98.5]
        assert d["focal_shift"] == 1.5
        assert d["lateral_color"] == 0.02
        assert d["spot_sizes"] == [0.1, 0.05, 0.08]
        assert d["transverse_aberration"] == [0.2, 0.1, 0.15]
        assert d["axial_chromatic_aberration"] == 1.5

    def test_to_dict_key_order(self):
        """Test that keys appear in exact specified order"""
        result = ChromaticResult(
            wavelengths=[1.0],
            focal_lengths=[2.0],
            focal_shift=3.0,
            lateral_color=4.0,
            spot_sizes=[5.0],
            transverse_aberration=[6.0],
            axial_chromatic_aberration=7.0
        )
        d = result.to_dict()
        
        expected_order = [
            "wavelengths",
            "focal_lengths",
            "focal_shift",
            "lateral_color",
            "spot_sizes",
            "transverse_aberration",
            "axial_chromatic_aberration"
        ]
        assert list(d.keys()) == expected_order

    def test_to_dict_key_count(self):
        """Test that dict has exactly 7 keys"""
        result = ChromaticResult(
            wavelengths=[],
            focal_lengths=[],
            focal_shift=0.0,
            lateral_color=0.0,
            spot_sizes=[],
            transverse_aberration=[],
            axial_chromatic_aberration=0.0
        )
        d = result.to_dict()
        assert len(d) == 7

    def test_to_dict_list_identity(self):
        """Test that list values are same objects (not copied)"""
        wl = [486.1, 587.6]
        fl = [97.0, 98.0]
        ss = [0.1, 0.05]
        ta = [0.2, 0.1]
        
        result = ChromaticResult(
            wavelengths=wl,
            focal_lengths=fl,
            focal_shift=1.5,
            lateral_color=0.02,
            spot_sizes=ss,
            transverse_aberration=ta,
            axial_chromatic_aberration=1.5
        )
        d = result.to_dict()
        
        # Verify list objects are identical, not copies
        assert d["wavelengths"] is wl
        assert d["focal_lengths"] is fl
        assert d["spot_sizes"] is ss
        assert d["transverse_aberration"] is ta

    def test_to_dict_empty_lists(self):
        """Test to_dict with empty lists"""
        result = ChromaticResult(
            wavelengths=[],
            focal_lengths=[],
            focal_shift=0.0,
            lateral_color=0.0,
            spot_sizes=[],
            transverse_aberration=[],
            axial_chromatic_aberration=0.0
        )
        d = result.to_dict()
        
        assert d["wavelengths"] == []
        assert d["focal_lengths"] == []
        assert d["spot_sizes"] == []
        assert d["transverse_aberration"] == []

    def test_to_dict_with_nan_values(self):
        """Test to_dict preserves NaN values without rounding or conversion"""
        nan = float('nan')
        result = ChromaticResult(
            wavelengths=[486.1, nan, 656.3],
            focal_lengths=[97.0, 98.0, 98.5],
            focal_shift=nan,
            lateral_color=0.02,
            spot_sizes=[0.1, nan, 0.08],
            transverse_aberration=[0.2, 0.1, 0.15],
            axial_chromatic_aberration=nan
        )
        d = result.to_dict()
        
        # Verify NaN entries are preserved (compare with math.isnan)
        assert math.isnan(d["wavelengths"][1])
        assert math.isnan(d["focal_shift"])
        assert math.isnan(d["spot_sizes"][1])
        assert math.isnan(d["axial_chromatic_aberration"])

    def test_to_dict_with_none_values(self):
        """Test to_dict preserves None values without conversion"""
        result = ChromaticResult(
            wavelengths=[486.1, None, 656.3],
            focal_lengths=[None],
            focal_shift=None,
            lateral_color=0.02,
            spot_sizes=[0.1],
            transverse_aberration=[0.2],
            axial_chromatic_aberration=None
        )
        d = result.to_dict()
        
        # None values should be preserved
        assert d["wavelengths"][1] is None
        assert d["focal_lengths"][0] is None
        assert d["focal_shift"] is None
        assert d["axial_chromatic_aberration"] is None

    def test_to_dict_with_integer_values(self):
        """Test that integer values are not converted to float"""
        result = ChromaticResult(
            wavelengths=[486, 587, 656],
            focal_lengths=[97, 98, 98],
            focal_shift=1,
            lateral_color=0,
            spot_sizes=[1, 1],
            transverse_aberration=[2, 3],
            axial_chromatic_aberration=1
        )
        d = result.to_dict()
        
        # Integers should remain integers
        assert isinstance(d["focal_shift"], int)
        assert isinstance(d["lateral_color"], int)
        assert isinstance(d["axial_chromatic_aberration"], int)
        assert d["wavelengths"][0] == 486

    def test_to_dict_roundtrip_with_dataclass(self):
        """Test that dict can reconstruct identical ChromaticResult instance"""
        result = ChromaticResult(
            wavelengths=[486.1, 587.6, 656.3],
            focal_lengths=[97.0, 98.0, 98.5],
            focal_shift=1.5,
            lateral_color=0.02,
            spot_sizes=[0.1, 0.05, 0.08],
            transverse_aberration=[0.2, 0.1, 0.15],
            axial_chromatic_aberration=1.5
        )
        d = result.to_dict()
        result2 = ChromaticResult(**d)
        
        # Dataclass equality compares all fields
        assert result == result2

    def test_to_dict_distinct_values_catch_swaps(self):
        """Test with distinct values per field to catch accidental swaps"""
        result = ChromaticResult(
            wavelengths=[1.0],
            focal_lengths=[2.0],
            focal_shift=3.0,
            lateral_color=4.0,
            spot_sizes=[5.0],
            transverse_aberration=[6.0],
            axial_chromatic_aberration=7.0
        )
        d = result.to_dict()
        
        # Each field has a unique value; swaps are caught
        assert d["focal_shift"] == 3.0
        assert d["axial_chromatic_aberration"] == 7.0
        assert d["lateral_color"] == 4.0
        assert d["spot_sizes"][0] == 5.0
        assert d["transverse_aberration"][0] == 6.0

    def test_to_dict_list_mutation_after_construction(self):
        """Test that list mutations after construction are reflected in to_dict"""
        wl = [486.1, 587.6]
        result = ChromaticResult(
            wavelengths=wl,
            focal_lengths=[97.0],
            focal_shift=1.5,
            lateral_color=0.02,
            spot_sizes=[0.1],
            transverse_aberration=[0.2],
            axial_chromatic_aberration=1.5
        )
        
        d1 = result.to_dict()
        # Mutate the original list
        wl.append(656.3)
        d2 = result.to_dict()
        
        # Both dicts reflect the mutation (they reference same list)
        assert len(d1["wavelengths"]) == 3
        assert len(d2["wavelengths"]) == 3
        assert d1["wavelengths"][2] == 656.3

    def test_to_dict_returns_new_dict_each_call(self):
        """Test that each call returns a new dict object"""
        result = ChromaticResult(
            wavelengths=[486.1],
            focal_lengths=[97.0],
            focal_shift=1.5,
            lateral_color=0.02,
            spot_sizes=[0.1],
            transverse_aberration=[0.2],
            axial_chromatic_aberration=1.5
        )
        
        d1 = result.to_dict()
        d2 = result.to_dict()
        
        # Different dict objects
        assert d1 is not d2
        # But with equal content
        assert d1 == d2

    def test_to_dict_long_lists(self):
        """Test with longer lists to verify no truncation or limits"""
        wl = list(range(100, 200))
        fl = [float(x) for x in range(97, 197)]
        ss = [0.1 * i for i in range(100)]
        ta = [0.2 * i for i in range(100)]
        
        result = ChromaticResult(
            wavelengths=wl,
            focal_lengths=fl,
            focal_shift=1.5,
            lateral_color=0.02,
            spot_sizes=ss,
            transverse_aberration=ta,
            axial_chromatic_aberration=1.5
        )
        d = result.to_dict()
        
        assert len(d["wavelengths"]) == 100
        assert len(d["focal_lengths"]) == 100
        assert len(d["spot_sizes"]) == 100
        assert len(d["transverse_aberration"]) == 100

    def test_to_dict_matches_dataclasses_asdict(self):
        """Test that result matches dataclasses.asdict for flat structure"""
        result = ChromaticResult(
            wavelengths=[486.1, 587.6],
            focal_lengths=[97.0, 98.0],
            focal_shift=1.5,
            lateral_color=0.02,
            spot_sizes=[0.1, 0.05],
            transverse_aberration=[0.2, 0.1],
            axial_chromatic_aberration=1.5
        )
        
        d = result.to_dict()
        expected = dataclasses.asdict(result)
        
        assert d == expected

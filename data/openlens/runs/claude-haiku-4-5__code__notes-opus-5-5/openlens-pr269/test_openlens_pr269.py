import pytest
import math
import dataclasses
from openlens.chromatic_analyzer import ChromaticResult


class TestChromaticResultToDict:
    
    def test_basic_conversion(self):
        """Test basic conversion to dict with simple float values"""
        r = ChromaticResult(
            wavelengths=[486.1, 587.6, 656.3],
            focal_lengths=[97.0, 98.0, 98.5],
            focal_shift=1.5,
            lateral_color=0.02,
            spot_sizes=[0.1, 0.05, 0.08],
            transverse_aberration=[0.2, 0.1, 0.15],
            axial_chromatic_aberration=2.5
        )
        d = r.to_dict()
        
        assert d["wavelengths"] == [486.1, 587.6, 656.3]
        assert d["focal_lengths"] == [97.0, 98.0, 98.5]
        assert d["focal_shift"] == 1.5
        assert d["lateral_color"] == 0.02
        assert d["spot_sizes"] == [0.1, 0.05, 0.08]
        assert d["transverse_aberration"] == [0.2, 0.1, 0.15]
        assert d["axial_chromatic_aberration"] == 2.5
    
    def test_dict_has_exactly_7_keys(self):
        """Test that the dict has exactly 7 keys"""
        r = ChromaticResult([], [], 0.0, 0.0, [], [], 0.0)
        d = r.to_dict()
        assert len(d) == 7
    
    def test_key_order_preserved(self):
        """Test that keys are in the correct insertion order"""
        r = ChromaticResult([], [], 0.0, 0.0, [], [], 0.0)
        d = r.to_dict()
        expected_keys = [
            "wavelengths",
            "focal_lengths",
            "focal_shift",
            "lateral_color",
            "spot_sizes",
            "transverse_aberration",
            "axial_chromatic_aberration"
        ]
        assert list(d.keys()) == expected_keys
    
    def test_key_names_singular_and_correct(self):
        """Test key names: transverse_aberration (singular), not plural"""
        r = ChromaticResult([], [], 0.0, 0.0, [], [], 0.0)
        d = r.to_dict()
        
        assert "transverse_aberration" in d  # singular
        assert "transverse_aberrations" not in d
        assert "axial_chromatic_aberration" in d
        assert "axial_color" not in d
    
    def test_list_identity_not_copied(self):
        """Test that list values are identical objects, not copies"""
        wl = [486.1, 587.6]
        fl = [97.0, 98.0]
        ss = [0.1, 0.05]
        ta = [0.2, 0.1]
        
        r = ChromaticResult(
            wavelengths=wl,
            focal_lengths=fl,
            focal_shift=1.5,
            lateral_color=0.02,
            spot_sizes=ss,
            transverse_aberration=ta,
            axial_chromatic_aberration=2.5
        )
        d = r.to_dict()
        
        # Verify same object via identity check
        assert d["wavelengths"] is wl
        assert d["focal_lengths"] is fl
        assert d["spot_sizes"] is ss
        assert d["transverse_aberration"] is ta
    
    def test_mutating_dict_mutates_source(self):
        """Test that mutating returned dict also mutates source object"""
        r = ChromaticResult(
            wavelengths=[486.1],
            focal_lengths=[97.0],
            focal_shift=1.5,
            lateral_color=0.02,
            spot_sizes=[0.1],
            transverse_aberration=[0.2],
            axial_chromatic_aberration=2.5
        )
        d = r.to_dict()
        
        d["spot_sizes"].append(0.5)
        
        # Verify source object is mutated too
        assert 0.5 in r.spot_sizes
        assert len(r.spot_sizes) == 2
    
    def test_empty_lists_valid(self):
        """Test that empty lists are handled correctly"""
        r = ChromaticResult([], [], 0.0, 0.0, [], [], 0.0)
        d = r.to_dict()
        
        assert d["wavelengths"] == []
        assert d["focal_lengths"] == []
        assert d["spot_sizes"] == []
        assert d["transverse_aberration"] == []
    
    def test_nan_values_preserved(self):
        """Test that NaN values are preserved without conversion"""
        nan = float('nan')
        r = ChromaticResult(
            wavelengths=[nan],
            focal_lengths=[nan],
            focal_shift=nan,
            lateral_color=nan,
            spot_sizes=[nan],
            transverse_aberration=[nan],
            axial_chromatic_aberration=nan
        )
        d = r.to_dict()
        
        # Use math.isnan since NaN != NaN
        assert math.isnan(d["focal_shift"])
        assert math.isnan(d["lateral_color"])
        assert math.isnan(d["axial_chromatic_aberration"])
        assert math.isnan(d["wavelengths"][0])
        assert math.isnan(d["focal_lengths"][0])
        assert math.isnan(d["spot_sizes"][0])
        assert math.isnan(d["transverse_aberration"][0])
    
    def test_distinct_scalar_values_no_swap(self):
        """Test distinct scalar values to detect field mapping errors"""
        r = ChromaticResult(
            wavelengths=[486.1],
            focal_lengths=[97.0],
            focal_shift=1.5,
            lateral_color=0.02,
            spot_sizes=[0.1],
            transverse_aberration=[0.2],
            axial_chromatic_aberration=2.5
        )
        d = r.to_dict()
        
        assert d["focal_shift"] == 1.5
        assert d["axial_chromatic_aberration"] == 2.5
        assert d["lateral_color"] == 0.02
        # Verify not swapped
        assert d["focal_shift"] != d["axial_chromatic_aberration"]
    
    def test_distinct_list_values_no_swap(self):
        """Test distinct lists detect swaps between spot_sizes and transverse_aberration"""
        r = ChromaticResult(
            wavelengths=[486.1],
            focal_lengths=[97.0],
            focal_shift=1.5,
            lateral_color=0.02,
            spot_sizes=[0.1, 0.05, 0.08],
            transverse_aberration=[0.2, 0.1, 0.15],
            axial_chromatic_aberration=2.5
        )
        d = r.to_dict()
        
        assert d["spot_sizes"] == [0.1, 0.05, 0.08]
        assert d["transverse_aberration"] == [0.2, 0.1, 0.15]
        assert d["spot_sizes"] != d["transverse_aberration"]
    
    def test_values_read_at_call_time(self):
        """Test that values are read at call time, not stored at construction"""
        r = ChromaticResult(
            wavelengths=[486.1],
            focal_lengths=[97.0],
            focal_shift=1.5,
            lateral_color=0.02,
            spot_sizes=[0.1],
            transverse_aberration=[0.2],
            axial_chromatic_aberration=2.5
        )
        
        d1 = r.to_dict()
        assert d1["focal_shift"] == 1.5
        
        r.focal_shift = 3.0
        d2 = r.to_dict()
        
        assert d2["focal_shift"] == 3.0
    
    def test_new_dict_created_each_call(self):
        """Test that each call returns a new dict object"""
        r = ChromaticResult([], [], 0.0, 0.0, [], [], 0.0)
        
        d1 = r.to_dict()
        d2 = r.to_dict()
        
        # Different dict objects
        assert d1 is not d2
        # But equal values
        assert d1 == d2
    
    def test_round_trip_reconstruction(self):
        """Test round-trip: to_dict then reconstruct via constructor"""
        r_orig = ChromaticResult(
            wavelengths=[486.1, 587.6, 656.3],
            focal_lengths=[97.0, 98.0, 98.5],
            focal_shift=1.5,
            lateral_color=0.02,
            spot_sizes=[0.1, 0.05, 0.08],
            transverse_aberration=[0.2, 0.1, 0.15],
            axial_chromatic_aberration=2.5
        )
        
        d = r_orig.to_dict()
        r_new = ChromaticResult(**d)
        
        assert r_new == r_orig
    
    def test_equivalence_with_dataclasses_asdict(self):
        """Test that to_dict matches dataclasses.asdict output"""
        r = ChromaticResult(
            wavelengths=[486.1, 587.6],
            focal_lengths=[97.0, 98.0],
            focal_shift=1.5,
            lateral_color=0.02,
            spot_sizes=[0.1, 0.05],
            transverse_aberration=[0.2, 0.1],
            axial_chromatic_aberration=2.5
        )
        
        d_to_dict = r.to_dict()
        d_asdict = dataclasses.asdict(r)
        
        assert d_to_dict == d_asdict
    
    def test_integer_and_mixed_types_preserved(self):
        """Test that integer and mixed types are preserved, not converted"""
        r = ChromaticResult(
            wavelengths=[486, 587.6],
            focal_lengths=[97, 98],
            focal_shift=1,
            lateral_color=0.02,
            spot_sizes=[1, 0],
            transverse_aberration=[2.0, 1],
            axial_chromatic_aberration=2
        )
        d = r.to_dict()
        
        # Type not converted
        assert isinstance(d["focal_shift"], int)
        assert isinstance(d["axial_chromatic_aberration"], int)
        assert d["wavelengths"][0] == 486
        assert isinstance(d["wavelengths"][0], int)
        assert all(isinstance(x, int) for x in d["spot_sizes"])

import pytest
import numpy as np
from openlens.chromatic_analyzer import ChromaticResult


class TestChromaticResultToDict:
    """Test suite for ChromaticResult.to_dict method"""

    def test_to_dict_returns_dict(self):
        """Test that to_dict returns a dictionary"""
        result = ChromaticResult(
            wavelengths=[587.6],
            focal_lengths=[50.0],
            focal_shift=0.1,
            lateral_color=0.05,
            spot_sizes=[1.0],
            transverse_aberration=[0.01],
            axial_chromatic_aberration=0.5,
        )
        output = result.to_dict()
        assert isinstance(output, dict)

    def test_to_dict_has_all_required_keys(self):
        """Test that to_dict returns all required keys"""
        result = ChromaticResult(
            wavelengths=[587.6],
            focal_lengths=[50.0],
            focal_shift=0.1,
            lateral_color=0.05,
            spot_sizes=[1.0],
            transverse_aberration=[0.01],
            axial_chromatic_aberration=0.5,
        )
        output = result.to_dict()
        expected_keys = {
            "wavelengths",
            "focal_lengths",
            "focal_shift",
            "lateral_color",
            "spot_sizes",
            "transverse_aberration",
            "axial_chromatic_aberration",
        }
        assert set(output.keys()) == expected_keys

    def test_to_dict_preserves_wavelengths_list(self):
        """Test that wavelengths list is preserved correctly"""
        wavelengths = [486.1, 587.6, 656.3]
        result = ChromaticResult(
            wavelengths=wavelengths,
            focal_lengths=[50.0, 50.1, 50.2],
            focal_shift=0.1,
            lateral_color=0.05,
            spot_sizes=[1.0, 1.1, 1.2],
            transverse_aberration=[0.01, 0.02, 0.03],
            axial_chromatic_aberration=0.5,
        )
        output = result.to_dict()
        assert output["wavelengths"] == wavelengths

    def test_to_dict_preserves_scalar_values(self):
        """Test that scalar values are preserved correctly"""
        result = ChromaticResult(
            wavelengths=[587.6],
            focal_lengths=[50.0],
            focal_shift=0.123,
            lateral_color=0.456,
            spot_sizes=[1.0],
            transverse_aberration=[0.01],
            axial_chromatic_aberration=0.789,
        )
        output = result.to_dict()
        assert output["focal_shift"] == 0.123
        assert output["lateral_color"] == 0.456
        assert output["axial_chromatic_aberration"] == 0.789

    def test_to_dict_with_numpy_arrays(self):
        """Test to_dict preserves numpy arrays"""
        wavelengths = np.array([486.1, 587.6, 656.3])
        focal_lengths = np.array([50.0, 50.1, 50.2])
        spot_sizes = np.array([1.0, 1.1, 1.2])
        transverse_aberration = np.array([0.01, 0.02, 0.03])
        
        result = ChromaticResult(
            wavelengths=wavelengths,
            focal_lengths=focal_lengths,
            focal_shift=0.1,
            lateral_color=0.05,
            spot_sizes=spot_sizes,
            transverse_aberration=transverse_aberration,
            axial_chromatic_aberration=0.5,
        )
        output = result.to_dict()
        
        assert len(output["wavelengths"]) == 3
        assert len(output["focal_lengths"]) == 3
        assert len(output["spot_sizes"]) == 3
        assert len(output["transverse_aberration"]) == 3

    def test_to_dict_with_none_values(self):
        """Test to_dict preserves None values for optional fields"""
        result = ChromaticResult(
            wavelengths=[587.6],
            focal_lengths=[50.0],
            focal_shift=None,
            lateral_color=None,
            spot_sizes=[1.0],
            transverse_aberration=[0.01],
            axial_chromatic_aberration=None,
        )
        output = result.to_dict()
        
        assert output["focal_shift"] is None
        assert output["lateral_color"] is None
        assert output["axial_chromatic_aberration"] is None

    def test_to_dict_with_zero_values(self):
        """Test to_dict correctly handles zero values"""
        result = ChromaticResult(
            wavelengths=[0.0],
            focal_lengths=[0.0],
            focal_shift=0.0,
            lateral_color=0.0,
            spot_sizes=[0.0],
            transverse_aberration=[0.0],
            axial_chromatic_aberration=0.0,
        )
        output = result.to_dict()
        
        assert output["focal_shift"] == 0.0
        assert output["lateral_color"] == 0.0
        assert output["axial_chromatic_aberration"] == 0.0

    def test_to_dict_with_negative_values(self):
        """Test to_dict preserves negative values"""
        result = ChromaticResult(
            wavelengths=[587.6],
            focal_lengths=[50.0],
            focal_shift=-0.5,
            lateral_color=-0.2,
            spot_sizes=[-1.0],
            transverse_aberration=[-0.01],
            axial_chromatic_aberration=-0.3,
        )
        output = result.to_dict()
        
        assert output["focal_shift"] == -0.5
        assert output["lateral_color"] == -0.2
        assert output["axial_chromatic_aberration"] == -0.3

    def test_to_dict_does_not_modify_original(self):
        """Test that to_dict does not mutate the original object"""
        result = ChromaticResult(
            wavelengths=[587.6],
            focal_lengths=[50.0],
            focal_shift=0.1,
            lateral_color=0.05,
            spot_sizes=[1.0],
            transverse_aberration=[0.01],
            axial_chromatic_aberration=0.5,
        )
        
        original_focal_shift = result.focal_shift
        _ = result.to_dict()
        
        assert result.focal_shift == original_focal_shift

    def test_to_dict_with_large_numerical_values(self):
        """Test to_dict with large numerical values"""
        result = ChromaticResult(
            wavelengths=[1e6],
            focal_lengths=[1e9],
            focal_shift=1e3,
            lateral_color=1e2,
            spot_sizes=[1e4],
            transverse_aberration=[1e-3],
            axial_chromatic_aberration=1e5,
        )
        output = result.to_dict()
        
        assert output["focal_shift"] == 1e3
        assert output["lateral_color"] == 1e2
        assert output["axial_chromatic_aberration"] == 1e5

    def test_to_dict_with_small_numerical_values(self):
        """Test to_dict with small numerical values"""
        result = ChromaticResult(
            wavelengths=[1e-6],
            focal_lengths=[1e-9],
            focal_shift=1e-10,
            lateral_color=1e-11,
            spot_sizes=[1e-12],
            transverse_aberration=[1e-13],
            axial_chromatic_aberration=1e-14,
        )
        output = result.to_dict()
        
        assert output["focal_shift"] == pytest.approx(1e-10, abs=1e-20)
        assert output["lateral_color"] == pytest.approx(1e-11, abs=1e-20)

    def test_to_dict_multiple_calls_idempotent(self):
        """Test that multiple calls to to_dict return identical results"""
        result = ChromaticResult(
            wavelengths=[486.1, 587.6, 656.3],
            focal_lengths=[50.0, 50.1, 50.2],
            focal_shift=0.1,
            lateral_color=0.05,
            spot_sizes=[1.0, 1.1, 1.2],
            transverse_aberration=[0.01, 0.02, 0.03],
            axial_chromatic_aberration=0.5,
        )
        
        output1 = result.to_dict()
        output2 = result.to_dict()
        
        assert output1 == output2

    def test_to_dict_with_empty_lists(self):
        """Test to_dict with empty list fields"""
        result = ChromaticResult(
            wavelengths=[],
            focal_lengths=[],
            focal_shift=0.1,
            lateral_color=0.05,
            spot_sizes=[],
            transverse_aberration=[],
            axial_chromatic_aberration=0.5,
        )
        output = result.to_dict()
        
        assert output["wavelengths"] == []
        assert output["focal_lengths"] == []
        assert output["spot_sizes"] == []
        assert output["transverse_aberration"] == []

    def test_to_dict_with_mismatched_array_lengths(self):
        """Test to_dict with arrays of different lengths"""
        result = ChromaticResult(
            wavelengths=[486.1, 587.6, 656.3],
            focal_lengths=[50.0, 50.1],  # Different length
            focal_shift=0.1,
            lateral_color=0.05,
            spot_sizes=[1.0, 1.1, 1.2, 1.3, 1.4],  # Different length
            transverse_aberration=[0.01],  # Different length
            axial_chromatic_aberration=0.5,
        )
        output = result.to_dict()
        
        assert len(output["wavelengths"]) == 3
        assert len(output["focal_lengths"]) == 2
        assert len(output["spot_sizes"]) == 5
        assert len(output["transverse_aberration"]) == 1

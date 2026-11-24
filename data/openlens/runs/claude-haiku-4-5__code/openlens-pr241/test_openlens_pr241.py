import pytest
from openlens.validation import check_physical_feasibility


class TestCheckPhysicalFeasibility:
    """Test suite for check_physical_feasibility validation function."""

    def test_valid_biconvex_lens(self):
        """Test a simple, valid biconvex lens."""
        is_feasible, warning = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=5.0,
            diameter=20.0,
        )
        assert is_feasible is True
        assert warning is None

    def test_valid_biconcave_lens(self):
        """Test a simple, valid biconcave lens."""
        is_feasible, warning = check_physical_feasibility(
            radius1=-50.0,
            radius2=50.0,
            thickness=3.0,
            diameter=15.0,
        )
        assert is_feasible is True
        assert warning is None

    def test_valid_meniscus_lens(self):
        """Test a valid meniscus lens with different curvatures."""
        is_feasible, warning = check_physical_feasibility(
            radius1=100.0,
            radius2=-50.0,
            thickness=4.0,
            diameter=18.0,
        )
        assert is_feasible is True
        assert warning is None

    def test_surfaces_intersect_hard_error(self):
        """Test hard error when surfaces intersect."""
        is_feasible, warning = check_physical_feasibility(
            radius1=10.0,
            radius2=-10.0,
            thickness=50.0,
            diameter=15.0,
        )
        assert is_feasible is False
        assert warning is not None
        assert "intersect" in warning.lower()

    def test_aperture_overhangs_hard_error(self):
        """Test hard error when aperture overhangs a surface."""
        is_feasible, warning = check_physical_feasibility(
            radius1=3.0,
            radius2=3.0,
            thickness=0.5,
            diameter=20.0,
        )
        assert is_feasible is False
        assert warning is not None

    def test_thickness_exceeds_half_min_radius_warning(self):
        """Test warning when thickness exceeds 50% of minimum radius."""
        is_feasible, warning = check_physical_feasibility(
            radius1=10.0,
            radius2=-10.0,
            thickness=5.5,
            diameter=5.0,
        )
        assert is_feasible is False
        assert warning is not None
        assert "thickness" in warning.lower()

    def test_diameter_exceeds_80_percent_min_radius_warning(self):
        """Test warning when diameter exceeds 80% of minimum radius."""
        is_feasible, warning = check_physical_feasibility(
            radius1=20.0,
            radius2=-20.0,
            thickness=2.0,
            diameter=16.5,
        )
        assert is_feasible is False
        assert warning is not None
        assert any(x in warning.lower() for x in ["diameter", "edge"])

    def test_lens_too_thin_relative_to_diameter_warning(self):
        """Test warning when thickness is less than 2% of diameter."""
        is_feasible, warning = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=0.3,
            diameter=20.0,
        )
        assert is_feasible is False
        assert warning is not None
        assert any(x in warning.lower() for x in ["thin", "fragile"])

    def test_parabolic_surface_first_surface(self):
        """Test valid lens with first surface parabolic."""
        is_feasible, warning = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=5.0,
            diameter=20.0,
            is_parabolic_1=True,
            parabolic_sag_1=0.3,
        )
        assert is_feasible is True
        assert warning is None

    def test_parabolic_surface_both_surfaces(self):
        """Test valid lens with both surfaces parabolic."""
        is_feasible, warning = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=5.0,
            diameter=20.0,
            is_parabolic_1=True,
            parabolic_sag_1=0.2,
            is_parabolic_2=True,
            parabolic_sag_2=0.2,
        )
        assert is_feasible is True
        assert warning is None

    def test_conic_constants_valid(self):
        """Test valid lens with non-zero conic constants."""
        is_feasible, warning = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=5.0,
            diameter=20.0,
            conic_constant_1=1.5,
            conic_constant_2=-0.5,
        )
        assert is_feasible is True
        assert warning is None

    def test_clear_apertures_specified(self):
        """Test lens with explicit clear aperture specifications."""
        is_feasible, warning = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=5.0,
            diameter=20.0,
            clear_aperture_1=18.0,
            clear_aperture_2=18.0,
        )
        assert is_feasible is True
        assert warning is None

    def test_zero_thickness_error(self):
        """Test that zero thickness is physically infeasible."""
        is_feasible, warning = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=0.0,
            diameter=20.0,
        )
        assert is_feasible is False
        assert warning is not None

    def test_zero_diameter_error(self):
        """Test that zero diameter is physically infeasible."""
        is_feasible, warning = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=5.0,
            diameter=0.0,
        )
        assert is_feasible is False
        assert warning is not None

    def test_radius_much_smaller_than_diameter_error(self):
        """Test hard error when radius is much smaller than diameter."""
        is_feasible, warning = check_physical_feasibility(
            radius1=0.5,
            radius2=-0.5,
            thickness=0.1,
            diameter=10.0,
        )
        assert is_feasible is False
        assert warning is not None

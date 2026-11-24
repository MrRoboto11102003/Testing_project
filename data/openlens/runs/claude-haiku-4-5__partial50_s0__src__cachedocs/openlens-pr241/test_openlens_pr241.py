import pytest
from openlens.validation import check_physical_feasibility


class TestCheckPhysicalFeasibility:
    """Test suite for check_physical_feasibility function."""

    def test_valid_biconvex_no_warnings(self):
        """Test a valid biconvex lens returns (True, None)."""
        is_feasible, msg = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=5.0,
            diameter=20.0,
        )
        assert is_feasible is True
        assert msg is None

    def test_valid_biconcave_no_warnings(self):
        """Test a valid biconcave lens returns (True, None)."""
        is_feasible, msg = check_physical_feasibility(
            radius1=-100.0,
            radius2=100.0,
            thickness=5.0,
            diameter=20.0,
        )
        assert is_feasible is True
        assert msg is None

    def test_hard_error_surfaces_intersect_negative_edge(self):
        """Test hard error: surfaces intersect (edge <= 0)."""
        is_feasible, msg = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=150.0,  # Too thick
            diameter=20.0,
        )
        assert is_feasible is False
        assert msg is not None
        assert isinstance(msg, str)

    def test_hard_error_aperture_overhangs_surface(self):
        """Test hard error: aperture overhangs surface (|R| < D/2)."""
        is_feasible, msg = check_physical_feasibility(
            radius1=5.0,  # Very small radius
            radius2=-200.0,
            thickness=5.0,
            diameter=50.0,  # Diameter too large for small radius
        )
        assert is_feasible is False
        assert msg is not None

    def test_warning_thickness_large_vs_radius(self):
        """Test warning when thickness > 0.5 * min_radius."""
        is_feasible, msg = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=30.0,  # 30 > 0.5 * 50 = 25
            diameter=15.0,
        )
        assert msg is not None

    def test_warning_diameter_large_vs_radius(self):
        """Test warning when diameter > 0.8 * min_radius."""
        is_feasible, msg = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=5.0,
            diameter=42.0,  # 42 > 0.8 * 50 = 40
        )
        assert msg is not None

    def test_warning_lens_too_thin_relative_to_diameter(self):
        """Test warning when thickness < 0.02 * diameter."""
        is_feasible, msg = check_physical_feasibility(
            radius1=200.0,
            radius2=-200.0,
            thickness=0.1,  # 0.1 < 0.02 * 100 = 2.0
            diameter=100.0,
        )
        assert msg is not None
        assert "thin" in msg.lower()

    def test_valid_planoconvex_lens(self):
        """Test valid plano-convex lens with flat front surface."""
        is_feasible, msg = check_physical_feasibility(
            radius1=float('inf'),
            radius2=-100.0,
            thickness=5.0,
            diameter=20.0,
        )
        assert is_feasible is True
        assert msg is None

    def test_valid_planoplane_lens(self):
        """Test valid plano-plane (flat) lens."""
        is_feasible, msg = check_physical_feasibility(
            radius1=float('inf'),
            radius2=float('inf'),
            thickness=5.0,
            diameter=20.0,
        )
        assert is_feasible is True
        assert msg is None

    def test_with_parabolic_surface_parameters(self):
        """Test lens with parabolic surface flags and sag values."""
        is_feasible, msg = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=5.0,
            diameter=20.0,
            is_parabolic_1=True,
            parabolic_sag_1=0.5,
            is_parabolic_2=False,
            parabolic_sag_2=0.0,
        )
        assert isinstance(is_feasible, bool)

    def test_with_conic_constant_parameters(self):
        """Test lens with conic constant (K parameter)."""
        is_feasible, msg = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=5.0,
            diameter=20.0,
            conic_constant_1=-1.0,  # Parabolic
            conic_constant_2=0.0,   # Spherical
        )
        assert isinstance(is_feasible, bool)

    def test_with_clear_aperture_parameters(self):
        """Test lens with explicit clear aperture values."""
        is_feasible, msg = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=5.0,
            diameter=25.0,
            clear_aperture_1=20.0,
            clear_aperture_2=20.0,
        )
        assert isinstance(is_feasible, bool)

    def test_meniscus_lens_both_surfaces_same_sign(self):
        """Test meniscus lens where both radii have the same sign."""
        is_feasible, msg = check_physical_feasibility(
            radius1=100.0,
            radius2=50.0,  # Both positive
            thickness=5.0,
            diameter=20.0,
        )
        assert isinstance(is_feasible, bool)

    def test_all_parameters_specified_valid_lens(self):
        """Test with all optional parameters fully specified."""
        is_feasible, msg = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=5.0,
            diameter=20.0,
            is_parabolic_1=True,
            parabolic_sag_1=0.3,
            is_parabolic_2=False,
            parabolic_sag_2=0.0,
            clear_aperture_1=18.0,
            clear_aperture_2=18.0,
            conic_constant_1=0.0,
            conic_constant_2=0.0,
        )
        assert is_feasible is True
        assert msg is None

    def test_return_type_tuple_bool_optional_str(self):
        """Test that function always returns Tuple[bool, Optional[str]]."""
        result = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=5.0,
            diameter=20.0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert result[1] is None or isinstance(result[1], str)

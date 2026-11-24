import pytest
import math
from openlens.validation import check_physical_feasibility


class TestCheckPhysicalFeasibilityBasic:
    """Test basic return types and happy path."""

    def test_valid_biconvex_lens(self):
        """Test a standard valid biconvex lens."""
        is_feasible, message = check_physical_feasibility(
            radius1=200.0,
            radius2=-200.0,
            thickness=10.0,
            diameter=25.0,
        )
        assert is_feasible is True
        assert message is None

    def test_return_type_is_tuple(self):
        """Verify return type is always a 2-tuple."""
        result = check_physical_feasibility(50, -50, 5, 25)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert result[1] is None or isinstance(result[1], str)


class TestCheckPhysicalFeasibilityFlatSurfaces:
    """Test flat surfaces (infinite radii)."""

    def test_flat_flat_surfaces(self):
        """Test a flat-flat lens (plano-plano)."""
        is_feasible, message = check_physical_feasibility(
            radius1=float("inf"),
            radius2=float("inf"),
            thickness=5.0,
            diameter=25.0,
        )
        assert is_feasible is True
        assert message is None

    def test_flat_convex(self):
        """Test flat front, convex back."""
        is_feasible, message = check_physical_feasibility(
            radius1=float("inf"),
            radius2=-100.0,
            thickness=5.0,
            diameter=25.0,
        )
        assert is_feasible is True
        assert message is None


class TestCheckPhysicalFeasibilityHardErrors:
    """Test hard errors that make geometry impossible."""

    def test_aperture_overhangs_surface(self):
        """Test when diameter overhangs a surface (|R| < D/2)."""
        is_feasible, message = check_physical_feasibility(
            radius1=10.0, radius2=-10.0, thickness=5.0, diameter=30.0
        )
        assert is_feasible is False
        assert message is not None
        assert "Aperture overhangs a surface" in message
        assert "10.0mm" in message and "15.0mm" in message

    def test_aperture_overhangs_picks_first_radius(self):
        """Test that overhang message reports the first qualifying radius."""
        is_feasible, message = check_physical_feasibility(
            radius1=-8.0, radius2=100.0, thickness=5.0, diameter=20.0
        )
        assert is_feasible is False
        assert "Aperture overhangs" in message
        assert "8.0mm" in message

    def test_surfaces_intersect(self):
        """Test when surfaces intersect within the clear aperture."""
        is_feasible, message = check_physical_feasibility(
            radius1=50.0, radius2=-50.0, thickness=1.0, diameter=40.0
        )
        assert is_feasible is False
        assert "Surfaces intersect within the clear aperture" in message
        assert "<= 0" in message


class TestCheckPhysicalFeasibilitySoftWarnings:
    """Test soft warnings that result in False but allow usage."""

    def test_thickness_warning_alone(self):
        """Test thickness warning: thickness > 0.5 * min_radius."""
        is_feasible, message = check_physical_feasibility(
            radius1=float("inf"),
            radius2=-20.0,
            thickness=15.0,  # 15 > 0.5 * 20 = 10
            diameter=10.0,
        )
        assert is_feasible is False
        assert message is not None
        assert "Thickness" in message
        assert "very large compared to" in message

    def test_thickness_at_boundary_no_warning(self):
        """Test that thickness exactly at 0.5*min_radius does not warn."""
        is_feasible, message = check_physical_feasibility(
            radius1=float("inf"),
            radius2=-100.0,
            thickness=50.0,  # Exactly 0.5 * 100
            diameter=10.0,
        )
        assert is_feasible is True
        assert message is None

    def test_diameter_warning(self):
        """Test diameter warning: diameter > 0.8 * min_radius."""
        is_feasible, message = check_physical_feasibility(
            radius1=100.0, radius2=-100.0, thickness=5.0, diameter=85.0
        )
        assert is_feasible is False
        assert "Diameter" in message
        assert "large compared to" in message

    def test_diameter_at_boundary_no_warning(self):
        """Test that diameter exactly at 0.8*min_radius does not warn."""
        is_feasible, message = check_physical_feasibility(
            radius1=100.0, radius2=-100.0, thickness=5.0, diameter=80.0
        )
        assert is_feasible is True
        assert message is None

    def test_thin_lens_warning(self):
        """Test thin lens warning: thickness < 0.02 * diameter."""
        is_feasible, message = check_physical_feasibility(
            radius1=200.0,
            radius2=-200.0,
            thickness=0.9,  # 0.9 < 0.02 * 50 = 1.0
            diameter=50.0,
        )
        assert is_feasible is False
        assert "very thin" in message.lower()

    def test_thin_lens_at_boundary_no_warning(self):
        """Test that thickness exactly at 0.02*diameter does not warn."""
        is_feasible, message = check_physical_feasibility(
            radius1=200.0,
            radius2=-200.0,
            thickness=1.0,  # Exactly 0.02 * 50
            diameter=50.0,
        )
        assert is_feasible is True
        assert message is None


class TestCheckPhysicalFeasibilityMultipleWarnings:
    """Test combinations of warnings."""

    def test_multiple_warnings_present(self):
        """Test that multiple warnings are both present in message."""
        is_feasible, message = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=60.0,  # Thickness warning: 60 > 0.5*100
            diameter=85.0,  # Diameter warning: 85 > 0.8*100
        )
        assert is_feasible is False
        assert "Thickness" in message
        assert "Diameter" in message

    def test_hard_error_suppresses_soft_warnings_in_presentation(self):
        """Test that hard error prevents edge fragility warning."""
        is_feasible, message = check_physical_feasibility(
            radius1=50.0, radius2=-50.0, thickness=1.0, diameter=40.0
        )
        assert is_feasible is False
        assert "Surfaces intersect" in message


class TestCheckPhysicalFeasibilityEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_non_numeric_radius_raises_type_error(self):
        """Test that non-numeric radius raises TypeError."""
        with pytest.raises(TypeError):
            check_physical_feasibility(
                radius1=None, radius2=100.0, thickness=5.0, diameter=25.0
            )

    def test_very_large_radii(self):
        """Test with very large (nearly flat) radii."""
        is_feasible, message = check_physical_feasibility(
            radius1=1e6, radius2=-1e6, thickness=5.0, diameter=25.0
        )
        assert is_feasible is True
        assert message is None

    def test_concave_lens(self):
        """Test concave lens (negative lenses)."""
        is_feasible, message = check_physical_feasibility(
            radius1=-100.0, radius2=100.0, thickness=5.0, diameter=25.0
        )
        assert is_feasible is True
        assert message is None

    def test_parabolic_surface_parameters(self):
        """Test that parabolic parameters are accepted without error."""
        is_feasible, message = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=5.0,
            diameter=25.0,
            is_parabolic_1=True,
            parabolic_sag_1=0.5,
        )
        assert isinstance(is_feasible, bool)

    def test_conic_constant_parameters(self):
        """Test that conic constant parameters are accepted."""
        is_feasible, message = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=5.0,
            diameter=25.0,
            conic_constant_1=-0.5,
            conic_constant_2=0.0,
        )
        assert isinstance(is_feasible, bool)

    def test_clear_aperture_parameters(self):
        """Test that clear aperture parameters are accepted."""
        is_feasible, message = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=5.0,
            diameter=25.0,
            clear_aperture_1=20.0,
            clear_aperture_2=22.0,
        )
        assert isinstance(is_feasible, bool)


class TestCheckPhysicalFeasibilityMessageFormat:
    """Test message formatting specifics."""

    def test_heuristic_messages_use_proper_formatting(self):
        """Test that heuristic messages use .1f formatting for dimensions."""
        is_feasible, message = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=60.123,  # Will be formatted with .1f
            diameter=25.0,
        )
        assert is_feasible is False
        # Thickness should appear as 60.1
        assert "60.1mm" in message

    def test_overhang_message_includes_dimensions(self):
        """Test that overhang message includes specific radius and D/2."""
        is_feasible, message = check_physical_feasibility(
            radius1=10.0, radius2=-10.0, thickness=5.0, diameter=30.0
        )
        assert is_feasible is False
        assert "10.0mm" in message  # Radius
        assert "15.0mm" in message  # D/2


class TestCheckPhysicalFeasibilityWarningOrder:
    """Test that warnings are reported in the specified order."""

    def test_warnings_joined_with_space(self):
        """Test that multiple warnings are separated by single space."""
        is_feasible, message = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=60.0,
            diameter=85.0,
        )
        assert is_feasible is False
        # Both should be present, separated by space
        words = message.split()
        assert len(words) > 10  # Enough words for multiple messages

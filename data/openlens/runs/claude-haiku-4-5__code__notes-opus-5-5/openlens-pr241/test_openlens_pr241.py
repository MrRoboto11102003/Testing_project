import pytest
from openlens.validation import check_physical_feasibility


class TestCheckPhysicalFeasibility:
    """Tests for check_physical_feasibility function."""

    def test_basic_biconvex_valid(self):
        """Basic biconvex lens with good proportions should be valid."""
        is_feasible, message = check_physical_feasibility(
            radius1=200, radius2=-200, thickness=10, diameter=25
        )
        assert is_feasible is True
        assert message is None

    def test_flat_flat_valid(self):
        """Flat/flat lens (both radii infinite) should be valid."""
        is_feasible, message = check_physical_feasibility(
            radius1=float('inf'), radius2=float('inf'), thickness=5, diameter=25
        )
        assert is_feasible is True
        assert message is None

    def test_negative_radii_valid(self):
        """Negative radii (concave surfaces) should work correctly."""
        is_feasible, message = check_physical_feasibility(
            radius1=-100, radius2=-100, thickness=8, diameter=20
        )
        assert is_feasible is True
        assert message is None

    def test_aperture_overhang_hard_error(self):
        """When diameter/2 > |R|, aperture overhangs surface."""
        is_feasible, message = check_physical_feasibility(
            radius1=10, radius2=-10, thickness=5, diameter=30
        )
        assert is_feasible is False
        assert "Aperture overhangs a surface" in message
        assert "|R| = 10.0mm" in message
        assert "D/2 = 15.0mm" in message

    def test_aperture_overhang_checks_first_radius(self):
        """Overhang checks radius1 before radius2."""
        is_feasible, message = check_physical_feasibility(
            radius1=-8, radius2=100, thickness=5, diameter=20
        )
        assert is_feasible is False
        assert "|R| = 8.0mm" in message

    def test_intersecting_surfaces_hard_error(self):
        """When edge <= 0, surfaces intersect."""
        is_feasible, message = check_physical_feasibility(
            radius1=50, radius2=-50, thickness=1, diameter=40
        )
        assert is_feasible is False
        assert "Surfaces intersect within the clear aperture" in message
        assert "<= 0" in message

    def test_thickness_warning_alone(self):
        """Thickness > 0.5*min_radius produces soft warning."""
        is_feasible, message = check_physical_feasibility(
            radius1=float('inf'), radius2=-20, thickness=15, diameter=10
        )
        assert is_feasible is False
        assert "Thickness" in message
        assert "very large compared to" in message

    def test_diameter_warning_alone(self):
        """Diameter > 0.8*min_radius produces soft warning."""
        is_feasible, message = check_physical_feasibility(
            radius1=float('inf'), radius2=-100, thickness=5, diameter=85
        )
        assert is_feasible is False
        assert "Diameter" in message
        assert "large compared to" in message

    def test_thin_lens_warning_alone(self):
        """Thickness < 0.02*diameter produces soft warning."""
        is_feasible, message = check_physical_feasibility(
            radius1=float('inf'), radius2=-200, thickness=0.9, diameter=50
        )
        assert is_feasible is False
        assert "very thin" in message

    def test_thickness_boundary_no_warning(self):
        """thickness == 0.5*min_radius does not warn (strict >)."""
        is_feasible, message = check_physical_feasibility(
            radius1=float('inf'), radius2=-100, thickness=50, diameter=10
        )
        assert is_feasible is True
        assert message is None

    def test_diameter_boundary_no_warning(self):
        """diameter == 0.8*min_radius does not warn (strict >)."""
        is_feasible, message = check_physical_feasibility(
            radius1=float('inf'), radius2=-100, thickness=5, diameter=80
        )
        assert is_feasible is True
        assert message is None

    def test_thin_lens_boundary_no_warning(self):
        """thickness == 0.02*diameter does not warn (strict <)."""
        is_feasible, message = check_physical_feasibility(
            radius1=float('inf'), radius2=-200, thickness=1.0, diameter=50
        )
        assert is_feasible is True
        assert message is None

    def test_multiple_soft_warnings(self):
        """Multiple soft warnings are combined."""
        is_feasible, message = check_physical_feasibility(
            radius1=float('inf'), radius2=-20, thickness=15, diameter=25
        )
        assert is_feasible is False
        assert "Thickness" in message
        assert "Diameter" in message

    def test_hard_error_with_warnings(self):
        """Hard error appears first when combined with soft warnings."""
        is_feasible, message = check_physical_feasibility(
            radius1=10, radius2=-10, thickness=5, diameter=30
        )
        assert is_feasible is False
        assert "Aperture overhangs" in message

    def test_none_radius_raises_type_error(self):
        """None as radius raises TypeError."""
        with pytest.raises(TypeError):
            check_physical_feasibility(
                radius1=None, radius2=-100, thickness=5, diameter=20
            )

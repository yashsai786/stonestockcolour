import pytest
from src.domain.value_objects.color import RGBColor, LABColor, ColorDistance, ColorProfile

def test_rgb_color_validation():
    # Valid RGB values
    color = RGBColor(10, 20, 30)
    assert color.r == 10
    assert color.g == 20
    assert color.b == 30
    assert color.to_tuple() == (10, 20, 30)

    # Invalid RGB values (bounds check)
    with pytest.raises(ValueError):
        RGBColor(-5, 100, 255)
    with pytest.raises(ValueError):
        RGBColor(256, 100, 255)


def test_lab_color_validation():
    # Valid LAB
    lab = LABColor(50.0, -10.0, 25.0)
    assert lab.l == 50.0
    assert lab.a == -10.0
    assert lab.b == 25.0
    
    # Invalid L bounds
    with pytest.raises(ValueError):
        LABColor(105.0, 0.0, 0.0)
    # Invalid a bounds
    with pytest.raises(ValueError):
        LABColor(50.0, -135.0, 0.0)


def test_color_profile_creation():
    profile = ColorProfile(
        name="Beige",
        lab=LABColor(75.0, 4.0, 15.0),
        tolerance=15.0
    )
    assert profile.name == "Beige"
    assert profile.lab.l == 75.0
    assert profile.tolerance == 15.0

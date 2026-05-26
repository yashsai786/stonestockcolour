import pytest
from src.infrastructure.cv.cv_color_matcher_service import CVColorMatcherService
from src.domain.value_objects.color import LABColor, ColorProfile

class DummyColorProfileRepository:
    def __init__(self, profiles):
        self.profiles = profiles
    def load_profiles(self):
        return self.profiles

def test_cielab_graphic_arts_distance(mock_color_profiles):
    repo = DummyColorProfileRepository(mock_color_profiles)
    matcher = CVColorMatcherService(repo)

    # Test exact CIE94 distance calculation triggers properly
    match = matcher.match_color(LABColor(95.0, 0.0, 0.0))
    assert match.name == "Pure White"

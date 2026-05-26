import pytest
from src.domain.value_objects.color import LABColor, ColorProfile, DominantColorResult
from src.infrastructure.repositories.json_color_profile_repository import JsonColorProfileRepository
from src.infrastructure.cv.cv_color_matcher_service import CVColorMatcherService
from src.domain.exceptions.domain_exceptions import ColorMatchingException

class DummyColorProfileRepository:
    def __init__(self, profiles):
        self.profiles = profiles
    def load_profiles(self):
        return self.profiles
    def get_by_name(self, name):
        for p in self.profiles:
            if p.name == name:
                return p
        return None

def test_delta_e_matching_success(mock_color_profiles):
    repo = DummyColorProfileRepository(mock_color_profiles)
    matcher = CVColorMatcherService(repo)

    # Perfect match test
    query = LABColor(95.0, 0.0, 0.0)  # Standard CIELAB for Pure White
    matched = matcher.match_color(query)
    assert matched.name == "Pure White"

    # Close shade matching (should match Grey)
    query_grey = LABColor(48.0, 1.0, -1.0)
    matched_grey = matcher.match_color(query_grey)
    assert matched_grey.name == "Grey"


def test_delta_e_threshold_clamping(mock_color_profiles):
    repo = DummyColorProfileRepository(mock_color_profiles)
    # Use fallback mode (Euclidean distance) to verify the fallback logic
    matcher = CVColorMatcherService(repo, use_fallback=True)

    query = LABColor(52.0, -0.5, 0.5)
    matched = matcher.match_color(query)
    assert matched.name == "Grey"


def test_empty_profiles_raises_exception():
    repo = DummyColorProfileRepository([])
    matcher = CVColorMatcherService(repo)
    
    with pytest.raises(ColorMatchingException):
        matcher.match_color(LABColor(50.0, 0.0, 0.0))


def test_palette_matching(mock_color_profiles):
    repo = DummyColorProfileRepository(mock_color_profiles)
    matcher = CVColorMatcherService(repo)

    dominant = [
        DominantColorResult(lab=LABColor(94.0, 0.2, 0.1), percentage=70.0),
        DominantColorResult(lab=LABColor(51.0, -0.2, -0.1), percentage=30.0)
    ]

    matched_palette = matcher.match_palette(dominant)
    assert len(matched_palette) == 2
    assert matched_palette[0][0].name == "Pure White"
    assert matched_palette[0][1] == 70.0
    assert matched_palette[1][0].name == "Grey"
    assert matched_palette[1][1] == 30.0

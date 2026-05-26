import pytest
from src.application.services.match_color_handler import MatchColorHandler
from src.domain.events.domain_events import DominantColorCalculatedEvent
from src.domain.entities.stone_color_analysis import StoneColorAnalysis
from src.domain.value_objects.color import DominantColorResult, LABColor
from src.infrastructure.repositories.in_memory_analysis_repository import InMemoryAnalysisRepository
from src.infrastructure.cv.cv_color_matcher_service import CVColorMatcherService

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

def test_match_color_handler_execution(mock_color_profiles):
    repo = InMemoryAnalysisRepository()
    repo_profile = DummyColorProfileRepository(mock_color_profiles)
    matcher = CVColorMatcherService(repo_profile)

    analysis = StoneColorAnalysis(analysis_id="tx-match")
    repo.save(analysis)

    handler = MatchColorHandler(repo, matcher)

    # Dominant colors: 75% white, 25% grey
    dominant = [
        DominantColorResult(lab=LABColor(94.0, 0.2, 0.1), percentage=75.0),
        DominantColorResult(lab=LABColor(51.0, -0.2, -0.1), percentage=25.0)
    ]

    event = DominantColorCalculatedEvent(
        analysis_id="tx-match",
        dominant_colors=dominant
    )

    # Invoke
    handler(event)

    # Assertions
    updated_analysis = repo.get_by_id("tx-match")
    assert updated_analysis.status == "COMPLETED"
    assert updated_analysis.primary_color == "Pure White"
    assert updated_analysis.primary_percentage == 75.0
    assert updated_analysis.secondary_color == "Grey"
    assert updated_analysis.secondary_percentage == 25.0
    assert updated_analysis.confidence > 0.8

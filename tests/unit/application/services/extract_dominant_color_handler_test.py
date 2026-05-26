import pytest
import numpy as np
from src.application.services.extract_dominant_color_handler import ExtractDominantColorHandler
from src.domain.events.domain_events import HandRemovedEvent, DominantColorCalculatedEvent
from src.domain.entities.stone_color_analysis import StoneColorAnalysis
from src.infrastructure.repositories.in_memory_analysis_repository import InMemoryAnalysisRepository
from src.infrastructure.cv.cv_dominant_color_analyzer import CVDominantColorAnalyzer
from src.infrastructure.event_bus.in_memory_event_bus import InMemoryEventBus

def test_extract_dominant_color_handler_execution():
    repo = InMemoryAnalysisRepository()
    analyzer = CVDominantColorAnalyzer()
    event_bus = InMemoryEventBus()

    analysis = StoneColorAnalysis(analysis_id="tx-color")
    analysis.set_skin_removed(skin_mask=None)
    repo.save(analysis)

    published = []
    event_bus.publish = lambda ev: published.append(ev)

    handler = ExtractDominantColorHandler(repo, analyzer, event_bus)

    # 1000 pixels of white LAB color
    pixels = np.tile([95.0, 0.0, 0.0], (1000, 1))

    event = HandRemovedEvent(
        analysis_id="tx-color",
        image_arr=None,
        valid_pixels=pixels
    )

    # Invoke
    handler(event)

    # Check published events
    assert len(published) == 1
    next_event = published[0]
    assert isinstance(next_event, DominantColorCalculatedEvent)
    assert next_event.analysis_id == "tx-color"
    assert len(next_event.dominant_colors) > 0
    # The primary dominant color should be Pure White (approx 95 L)
    assert 90.0 <= next_event.dominant_colors[0].lab.l <= 100.0

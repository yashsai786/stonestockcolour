import pytest
from src.application.services.detect_slab_handler import DetectSlabHandler
from src.domain.events.domain_events import ImageUploadedEvent, StoneDetectedEvent
from src.domain.entities.stone_color_analysis import StoneColorAnalysis
from src.infrastructure.repositories.in_memory_analysis_repository import InMemoryAnalysisRepository
from src.infrastructure.cv.opencv_slab_detection_service import OpenCVSlabDetectionService
from src.infrastructure.event_bus.in_memory_event_bus import InMemoryEventBus

def test_detect_slab_handler_execution(synthetic_image_bytes):
    repo = InMemoryAnalysisRepository()
    slab_service = OpenCVSlabDetectionService()
    event_bus = InMemoryEventBus()

    # Pre-register our analysis aggregate
    analysis = StoneColorAnalysis(analysis_id="tx-slab")
    repo.save(analysis)

    published = []
    event_bus.publish = lambda ev: published.append(ev)

    handler = DetectSlabHandler(repo, slab_service, event_bus)

    event = ImageUploadedEvent(
        analysis_id="tx-slab",
        image_bytes=synthetic_image_bytes,
        filename="slab.png"
    )

    # Invoke handler
    handler(event)

    # Assertions
    updated_analysis = repo.get_by_id("tx-slab")
    assert updated_analysis.status == "SLAB_DETECTED"
    assert updated_analysis.slab_contour is not None
    assert updated_analysis.slab_mask is not None

    # Check published events
    assert len(published) == 1
    next_event = published[0]
    assert isinstance(next_event, StoneDetectedEvent)
    assert next_event.analysis_id == "tx-slab"
    assert next_event.image_arr is not None
    assert next_event.slab_mask is not None

import pytest
import numpy as np
from src.application.services.remove_hand_handler import RemoveHandHandler
from src.domain.events.domain_events import StoneDetectedEvent, HandRemovedEvent
from src.domain.entities.stone_color_analysis import StoneColorAnalysis
from src.infrastructure.repositories.in_memory_analysis_repository import InMemoryAnalysisRepository
from src.infrastructure.cv.hsv_skin_removal_service import HSVSkinRemovalService
from src.infrastructure.event_bus.in_memory_event_bus import InMemoryEventBus

def test_remove_hand_handler_execution(synthetic_slab_image_generator):
    repo = InMemoryAnalysisRepository()
    skin_service = HSVSkinRemovalService()
    event_bus = InMemoryEventBus()

    # Pre-register aggregate
    analysis = StoneColorAnalysis(analysis_id="tx-hand")
    analysis.set_slab_data(contour=None, mask=None)
    repo.save(analysis)

    published = []
    event_bus.publish = lambda ev: published.append(ev)

    handler = RemoveHandHandler(repo, skin_service, event_bus)

    # Generate synthetic image and slab mask (entire image is slab for simplicity)
    img = synthetic_slab_image_generator(add_hand=False)
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255

    event = StoneDetectedEvent(
        analysis_id="tx-hand",
        image_arr=img,
        slab_contour=None,
        slab_mask=mask
    )

    # Invoke
    handler(event)

    # Assertions
    updated_analysis = repo.get_by_id("tx-hand")
    assert updated_analysis.status == "HAND_REMOVED"
    assert updated_analysis.skin_mask is not None

    # Check published event
    assert len(published) == 1
    next_event = published[0]
    assert isinstance(next_event, HandRemovedEvent)
    assert next_event.analysis_id == "tx-hand"
    assert next_event.valid_pixels is not None
    assert len(next_event.valid_pixels) > 0
    # Should be (N, 3) matrix
    assert next_event.valid_pixels.ndim == 2
    assert next_event.valid_pixels.shape[1] == 3

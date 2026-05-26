from src.domain.events.domain_events import ImageUploadedEvent, StoneDetectedEvent
import uuid

def test_image_uploaded_event_creation():
    analysis_id = str(uuid.uuid4())
    event = ImageUploadedEvent(
        analysis_id=analysis_id,
        image_bytes=b"fakebytes",
        filename="slab.png"
    )
    
    assert event.analysis_id == analysis_id
    assert event.image_bytes == b"fakebytes"
    assert event.filename == "slab.png"
    assert event.event_id is not None
    assert event.occurred_at is not None


def test_stone_detected_event_creation():
    analysis_id = str(uuid.uuid4())
    event = StoneDetectedEvent(
        analysis_id=analysis_id,
        image_arr=None,
        slab_contour=None,
        slab_mask=None
    )
    
    assert event.analysis_id == analysis_id
    assert event.image_arr is None
    assert event.slab_contour is None
    assert event.slab_mask is None

from dataclasses import dataclass, field
from datetime import datetime
import uuid
from typing import Any, List
from src.domain.value_objects.color import DominantColorResult

@dataclass(frozen=True, kw_only=True)
class DomainEvent:
    """Base class for all domain events."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    occurred_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True, kw_only=True)
class ImageUploadedEvent(DomainEvent):
    """Fired when an image has been uploaded and ready for analysis."""
    analysis_id: str
    image_bytes: bytes
    filename: str


@dataclass(frozen=True, kw_only=True)
class StoneDetectedEvent(DomainEvent):
    """Fired when the largest stone contour has been detected."""
    analysis_id: str
    image_arr: Any  # ndarray of resized image
    slab_contour: Any  # contour points
    slab_mask: Any  # binary mask of slab


@dataclass(frozen=True, kw_only=True)
class HandRemovedEvent(DomainEvent):
    """Fired when the hands and background elements are masked out."""
    analysis_id: str
    image_arr: Any  # ndarray of resized image
    valid_pixels: Any  # (N, 3) BGR or LAB color pixels array


@dataclass(frozen=True, kw_only=True)
class DominantColorCalculatedEvent(DomainEvent):
    """Fired when dominant color clusters have been successfully extracted."""
    analysis_id: str
    dominant_colors: List[DominantColorResult]

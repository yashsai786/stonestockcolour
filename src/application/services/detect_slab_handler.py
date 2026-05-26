import logging
from src.domain.events.domain_events import ImageUploadedEvent, StoneDetectedEvent
from src.domain.repositories.analysis_repository import AnalysisRepository
from src.domain.services.slab_detection_service import SlabDetectionService
from src.domain.entities.stone_image import StoneImage
from src.application.ports.event_bus import EventBus
from src.domain.exceptions.domain_exceptions import DomainException

logger = logging.getLogger(__name__)

class DetectSlabHandler:
    """
    Subscribes to ImageUploadedEvent.
    Detects the stone slab in the image, masks background, and emits StoneDetectedEvent.
    """
    def __init__(
        self,
        repository: AnalysisRepository,
        slab_detection_service: SlabDetectionService,
        event_bus: EventBus
    ):
        self._repository = repository
        self._slab_detection_service = slab_detection_service
        self._event_bus = event_bus

    def __call__(self, event: ImageUploadedEvent) -> None:
        analysis_id = event.analysis_id
        analysis = self._repository.get_by_id(analysis_id)
        if not analysis:
            logger.error(f"Analysis {analysis_id} not found.")
            return

        try:
            # Instantiate StoneImage to decode and validate raw bytes
            stone_image = StoneImage(event.image_bytes, event.filename)
            
            # Early resize for faster contour detection (max 600px width/height)
            resized_img = stone_image.get_resized(max_dimension=600)
            
            # Detect slab
            contour, slab_mask = self._slab_detection_service.detect_slab(resized_img)
            
            # Update entity state
            analysis.set_slab_data(contour, slab_mask)
            self._repository.save(analysis)

            # Publish next event in the chain
            next_event = StoneDetectedEvent(
                analysis_id=analysis_id,
                image_arr=resized_img,
                slab_contour=contour,
                slab_mask=slab_mask
            )
            self._event_bus.publish(next_event)

        except DomainException as e:
            logger.error(f"Domain error in DetectSlabHandler for {analysis_id}: {str(e)}")
            analysis.set_failure(f"Slab detection failed: {str(e)}")
            self._repository.save(analysis)
        except Exception as e:
            logger.error(f"Unexpected error in DetectSlabHandler for {analysis_id}: {str(e)}")
            analysis.set_failure(f"Slab detection failed with unexpected error: {str(e)}")
            self._repository.save(analysis)
stream = None

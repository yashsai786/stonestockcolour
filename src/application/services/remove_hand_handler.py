import logging
import cv2
import numpy as np
from src.domain.events.domain_events import StoneDetectedEvent, HandRemovedEvent
from src.domain.repositories.analysis_repository import AnalysisRepository
from src.domain.services.skin_removal_service import SkinRemovalService
from src.application.ports.event_bus import EventBus
from src.domain.exceptions.domain_exceptions import DomainException, SkinRemovalException

logger = logging.getLogger(__name__)

class RemoveHandHandler:
    """
    Subscribes to StoneDetectedEvent.
    Detects and masks out skin/hands, extracts valid pixels,
    converts them to standard CIELAB color space, and emits HandRemovedEvent.
    """
    def __init__(
        self,
        repository: AnalysisRepository,
        skin_removal_service: SkinRemovalService,
        event_bus: EventBus
    ):
        self._repository = repository
        self._skin_removal_service = skin_removal_service
        self._event_bus = event_bus

    def __call__(self, event: StoneDetectedEvent) -> None:
        analysis_id = event.analysis_id
        analysis = self._repository.get_by_id(analysis_id)
        if not analysis:
            logger.error(f"Analysis {analysis_id} not found.")
            return

        try:
            image_arr = event.image_arr
            slab_mask = event.slab_mask

            # Generate the skin mask (1 where skin/hand is present, 0 elsewhere)
            skin_mask = self._skin_removal_service.remove_skin(image_arr)
            
            # Combine masks: valid pixels must be inside the slab AND NOT skin
            # Vectorized numpy bitwise operations
            valid_mask = cv2.bitwise_and(slab_mask, cv2.bitwise_not(skin_mask))
            
            # Update entity state
            analysis.set_skin_removed(skin_mask)
            self._repository.save(analysis)

            # Ensure we have at least some valid slab pixels left
            num_valid_pixels = np.sum(valid_mask > 0)
            if num_valid_pixels == 0:
                raise SkinRemovalException("No valid stone pixels remaining after hand and background masking.")

            # Convert entire image BGR -> LAB color space using OpenCV
            lab_img = cv2.cvtColor(image_arr, cv2.COLOR_BGR2Lab)
            
            # Extract only the pixels within the valid mask (shape: N, 3)
            raw_lab_pixels = lab_img[valid_mask > 0].astype(np.float32)

            # Convert OpenCV LAB format to standard CIELAB values:
            # L_std = L_cv * 100 / 255
            # a_std = a_cv - 128
            # b_std = b_cv - 128
            standard_lab_pixels = np.empty_like(raw_lab_pixels)
            standard_lab_pixels[:, 0] = raw_lab_pixels[:, 0] * (100.0 / 255.0)
            standard_lab_pixels[:, 1] = raw_lab_pixels[:, 1] - 128.0
            standard_lab_pixels[:, 2] = raw_lab_pixels[:, 2] - 128.0

            # Optimizations: Random Pixel Sampling
            # If we have too many pixels, sample randomly to speed up downstream clustering.
            # 10,000 pixels is highly representative yet clusters in <5ms.
            max_sample_size = 10000
            if len(standard_lab_pixels) > max_sample_size:
                indices = np.random.choice(len(standard_lab_pixels), max_sample_size, replace=False)
                sampled_pixels = standard_lab_pixels[indices]
            else:
                sampled_pixels = standard_lab_pixels

            # Publish next event in the chain
            next_event = HandRemovedEvent(
                analysis_id=analysis_id,
                image_arr=image_arr,
                valid_pixels=sampled_pixels
            )
            self._event_bus.publish(next_event)

        except DomainException as e:
            logger.error(f"Domain error in RemoveHandHandler for {analysis_id}: {str(e)}")
            analysis.set_failure(f"Skin/Hand removal failed: {str(e)}")
            self._repository.save(analysis)
        except Exception as e:
            logger.error(f"Unexpected error in RemoveHandHandler for {analysis_id}: {str(e)}")
            analysis.set_failure(f"Skin/Hand removal failed with unexpected error: {str(e)}")
            self._repository.save(analysis)

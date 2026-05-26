import pytest
import numpy as np
from src.infrastructure.cv.opencv_slab_detection_service import OpenCVSlabDetectionService
from src.domain.exceptions.domain_exceptions import SlabDetectionException

def test_detects_slab_correctly(synthetic_slab_image_generator):
    service = OpenCVSlabDetectionService()
    
    # Generate clean slab filling 64% of the image (0.2 to 0.8 in both dimensions)
    img = synthetic_slab_image_generator(
        width=400,
        height=300,
        slab_bounds=(0.2, 0.2, 0.8, 0.8),
        slab_color=(200, 200, 200),
        bg_color=(10, 10, 10)
    )

    contour, mask = service.detect_slab(img)
    
    assert contour is not None
    assert mask is not None
    assert mask.shape == (300, 400)
    
    # Center of the slab should be active (255)
    assert mask[150, 200] == 255
    # Border/Background should be black (0)
    assert mask[10, 10] == 0


def test_fallback_when_slab_too_small(synthetic_slab_image_generator):
    service = OpenCVSlabDetectionService()
    
    # Slab area is extremely small (<5% of image)
    img = synthetic_slab_image_generator(
        width=400,
        height=300,
        slab_bounds=(0.48, 0.48, 0.52, 0.52)
    )

    # Should trigger fallback to full image mask
    contour, mask = service.detect_slab(img)
    
    assert mask is not None
    # In fallback mode, the entire mask should be active (255)
    assert np.all(mask == 255)


def test_invalid_image_handling():
    service = OpenCVSlabDetectionService()
    
    # Null or empty array
    with pytest.raises(SlabDetectionException):
        service.detect_slab(np.array([]))

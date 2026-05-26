import pytest
import numpy as np
from src.infrastructure.cv.opencv_slab_detection_service import OpenCVSlabDetectionService

def test_opencv_slab_detection_success(synthetic_slab_image_generator):
    service = OpenCVSlabDetectionService()
    img = synthetic_slab_image_generator(slab_bounds=(0.1, 0.1, 0.9, 0.9))
    
    contour, mask = service.detect_slab(img)
    assert contour is not None
    assert mask is not None
    assert mask.shape == (300, 400)

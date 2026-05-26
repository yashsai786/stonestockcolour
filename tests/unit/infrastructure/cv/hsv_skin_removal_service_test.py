import pytest
import numpy as np
from src.infrastructure.cv.hsv_skin_removal_service import HSVSkinRemovalService

def test_hsv_skin_removal_success(synthetic_slab_image_generator):
    service = HSVSkinRemovalService()
    img = synthetic_slab_image_generator(add_hand=True)
    
    mask = service.remove_skin(img)
    assert mask is not None
    assert mask.shape == (300, 400)

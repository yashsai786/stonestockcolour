import pytest
import numpy as np
from src.infrastructure.cv.hsv_skin_removal_service import HSVSkinRemovalService
from src.domain.exceptions.domain_exceptions import SkinRemovalException

def test_removes_visible_hand(synthetic_slab_image_generator):
    service = HSVSkinRemovalService()
    
    # Image with overlapping pinkish-beige hand (HSV skin ranges)
    img_with_hand = synthetic_slab_image_generator(
        width=200,
        height=200,
        slab_color=(240, 240, 240),
        add_hand=True,
        hand_color=(150, 160, 230) # Skin color (BGR: 230, 160, 150 -> HSV: ~5, ~89, ~230)
    )

    mask = service.remove_skin(img_with_hand)
    
    assert mask is not None
    assert mask.shape == (200, 200)
    # The center of the image where the hand was drawn must be detected as skin (255)
    assert mask[100, 100] == 255


def test_handles_no_hand_images(synthetic_slab_image_generator):
    service = HSVSkinRemovalService()
    
    # Pure grey image with no skin tones
    img_no_hand = synthetic_slab_image_generator(add_hand=False)
    
    mask = service.remove_skin(img_no_hand)
    
    assert mask is not None
    # No skin should be detected
    assert np.sum(mask > 0) == 0


def test_empty_image_raises_exception():
    service = HSVSkinRemovalService()
    
    with pytest.raises(SkinRemovalException):
        service.remove_skin(np.array([]))

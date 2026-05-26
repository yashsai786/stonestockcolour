import pytest
from src.domain.entities.stone_image import StoneImage
from src.domain.exceptions.domain_exceptions import InvalidImageException

def test_stone_image_decodes_valid_bytes(synthetic_image_bytes):
    stone_image = StoneImage(synthetic_image_bytes, "test.png")
    
    assert stone_image.width == 400
    assert stone_image.height == 300
    assert stone_image.decoded_image is not None
    assert stone_image.decoded_image.shape == (300, 400, 3)


def test_stone_image_early_resizing(synthetic_image_bytes):
    stone_image = StoneImage(synthetic_image_bytes, "test.png")
    
    # Request resize down to 200px max dimension
    resized = stone_image.get_resized(max_dimension=200)
    
    # Width is larger than height, so it should scale width to 200 and height to 150
    assert resized.shape == (150, 200, 3)


def test_invalid_bytes_raises_exception():
    with pytest.raises(InvalidImageException):
        StoneImage(b"not an image", "test.png")

    with pytest.raises(InvalidImageException):
        StoneImage(b"", "empty.png")
